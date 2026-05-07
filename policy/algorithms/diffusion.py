import functools
from typing import Any, Literal, cast

import hydra_zen
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.training_utils import EMAModel
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from policy.datamodules.maniskill_datamodule import ManiSkillDataModule
from policy.utils import flatten_tensor_dict, sum_shapes
from policy.utils.typing_utils import DiffusionSchedulerProtocol, HydraConfigFor


class DiffusionPolicy(L.LightningModule):
    """Diffusion Policy as in Cheng et. al (IJRR)

    Reference:
        - Arxiv: https://arxiv.org/abs/2303.04137v4
        - Paper website: https://diffusion-policy.cs.columbia.edu/
        - Maniskill implementation: https://github.com/haosulab/ManiSkill/tree/main/examples/baselines/diffusion_policy
    """

    def __init__(
        self,
        network: HydraConfigFor[nn.Module],
        ema: HydraConfigFor[EMAModel],
        noise_scheduler: HydraConfigFor[DiffusionSchedulerProtocol],
        datamodule: ManiSkillDataModule,
        optimizer: HydraConfigFor[functools.partial[Optimizer]],
        lr_scheduler: HydraConfigFor[functools.partial[LRScheduler]] | None = None,
        act_horizon: int = 8,
        prediction_type: Literal["epsilon", "sample", "v_prediction"] = "epsilon",
    ):
        super().__init__()

        # TODO: are you sure? What if I need to log control_mode, obs_mode etc.? I should not rely on those in Rollout!
        self.save_hyperparameters(ignore=["datamodule"])

        self.network_config = network
        self.network: torch.nn.Module | None = None
        self.ema_config = ema
        self.ema: EMAModel | None = None

        self.noise_scheduler_config = noise_scheduler
        self.noise_scheduler: DiffusionSchedulerProtocol | None = hydra_zen.instantiate(
            self.noise_scheduler_config, prediction_type=prediction_type
        )

        self.optimizer_config = optimizer
        self.optimizer: Optimizer | None = None

        self.lr_scheduler_config = lr_scheduler
        self.lr_scheduler: LRScheduler | None = None

        self.datamodule = datamodule

        self.cond_horizon = self.datamodule.cond_horizon
        self.pred_horizon = self.datamodule.pred_horizon
        self.act_horizon = act_horizon

        if self.act_horizon > self.pred_horizon:
            raise ValueError(
                f"Action horizon ({self.act_horizon}) cannot be greater than prediction horizon ({self.pred_horizon}). "
                "The model cannot execute more timesteps (act_horizon) than its total prediction horizon (pred_horizon)."
            )

        if self.act_horizon < self.cond_horizon:
            raise ValueError(
                f"Action horizon ({self.act_horizon}) cannot be less than conditioning horizon ({self.cond_horizon}). "
                "The model needs to predict at least as many timesteps as it conditions on (including repredicting the past)."
            )

        if self.cond_horizon + self.act_horizon - 1 > self.pred_horizon:
            raise ValueError(
                f"Prediction horizon ({self.pred_horizon}) is too short! "
                f"It must be at least {self.cond_horizon + self.act_horizon - 1} "
                f"to contain the past actions ({self.cond_horizon - 1}) plus "
                f"the actions to execute ({self.act_horizon})."
            )

        self.act_dim = self.datamodule.act_dim
        cond_dim = self.datamodule.cond_dim
        if isinstance(cond_dim, dict):
            self.cond_dim = sum_shapes(cond_dim)
        else:
            self.cond_dim = cond_dim

        # TODO: Normalize observation/env_states
        #   - Should be pre-computed for the dataset, in the maniskill_datamodule, maybe saving them as <h5_file_path>.stats.json (one time only, to avoid repeated computation every time we train a model)
        #   - Then the diffusion policy can load them and apply the normalization using the TensorZNormalizer from utils/normalizer.py
        #   - The Diffusion should correctly select the file related to what we are looking for: cond_source, then select physix_env_states vs obs, and apply the normalization accordingly.
        #   - Thus Diffusion should have access to the datamodule's parameters about cond_source, physx_env_states, obs_mode, etc. to be able to do this correctly.
        #   - Can diffusion access such data? YES, since datamodule is passed in the init

    def configure_model(self) -> None:
        if self.network is not None:
            return

        # We suppose to flatten the conditioning tensor (like in FiLM + Unet), tho this could need to be generalized in the future
        external_cond_dim = self.cond_horizon * self.cond_dim
        self.network = hydra_zen.instantiate(
            self.network_config, input_dim=self.act_dim, external_cond_dim=external_cond_dim
        )

        if self.ema is not None:
            return

        self.ema = hydra_zen.instantiate(self.ema_config, parameters=self.network.parameters())

    def configure_optimizers(self) -> Optimizer | dict:

        # Optimizers and schedulers could actually be made in one shot, without partial,
        # however I prefer to follow the template prescription, just for coherence

        optimizer_partial = hydra_zen.instantiate(self.optimizer_config)
        optimizer = optimizer_partial(self.parameters())

        if self.lr_scheduler_config is not None:
            lr_scheduler_partial = hydra_zen.instantiate(self.lr_scheduler_config)
            lr_scheduler = lr_scheduler_partial(optimizer)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        else:
            return optimizer

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        pass  # We dont test on data, we only run simulation rollouts

    def _shared_step(self, batch: dict[str, Any], batch_idx: int, phase: str) -> torch.Tensor:
        """Main step logic, it doesn't differ between training and validation except for the
        logging.

        Shapes:
            batch["cond_seq"]: [B, cond_horizon, cond_dim]
            batch["act_seq"]: [B, pred_horizon, act_dim]
            returns: scalar loss tensor []
        """
        flatten_cond = flatten_tensor_dict(batch["cond_seq"])
        action_seq = batch["act_seq"]

        loss = self._compute_loss(flatten_cond, action_seq)

        self.log(f"{phase}/loss", loss, prog_bar=True, sync_dist=(phase == "val"))
        return loss

    def on_train_batch_end(self, outputs: torch.Tensor, batch: dict[str, Any], batch_idx: int):
        """Automatically step the EMA model after every training batch iteration."""
        if self.network is None:
            raise ValueError(
                "Network not initialized. Call configure_model() before on_train_batch_end."
            )

        if self.ema is None:
            raise ValueError(
                "EMA not initialized. Call configure_model() before on_train_batch_end."
            )

        self.ema.to(self.device)

        self.ema.step(self.network.parameters())

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Explicitly save the EMA model state since it's not an nn.Module."""
        super().on_save_checkpoint(checkpoint)

        if self.ema is not None:
            checkpoint["ema_state_dict"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Explicitly load the EMA model state."""
        super().on_load_checkpoint(checkpoint)

        self.configure_model()
        if self.ema is not None and "ema_state_dict" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state_dict"])

    def get_action(
        self,
        cond_seq: torch.Tensor,
        num_inference_steps: int | None = None,
        clamp_range: tuple | None = None,
    ):
        """Runs the reverse diffusion process to predict an action sequence from the current
        observation.

        Shapes:
            cond_seq: [B, cond_horizon * cond_dim] (flattened conditioning)
            returns: [B, act_horizon, act_dim] (denoised actions to execute)
        """
        if self.network is None:
            raise ValueError(
                "Network not initialized. Call configure_model() before getting action."
            )

        if self.noise_scheduler is None:
            raise ValueError(
                "Noise Scheduler not initialized. Call configure_model() before getting action."
            )

        if self.ema is None:
            raise ValueError(
                "EMA Model not initialized. Call configure_model() before getting action."
            )

        B = cond_seq.shape[0]

        self.ema.store(self.network.parameters())
        self.ema.copy_to(self.network.parameters())

        if num_inference_steps is None:
            num_inference_steps = int(self.noise_scheduler.config["num_train_timesteps"])

        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)

        with torch.no_grad():
            noisy_act_seq = torch.randn((B, self.pred_horizon, self.act_dim), device=self.device)

            for t in self.noise_scheduler.timesteps:
                t = int(t.item())

                latent_model_input = self.noise_scheduler.scale_model_input(noisy_act_seq, t)

                model_pred = self.network(
                    sample=latent_model_input,
                    timestep=t,
                    external_cond=cond_seq,
                )

                output = self.noise_scheduler.step(
                    model_output=model_pred,
                    timestep=t,
                    sample=noisy_act_seq,
                    return_dict=False,
                )

                noisy_act_seq = output[0]

        self.ema.restore(self.network.parameters())

        start = self.cond_horizon - 1
        end = start + self.act_horizon

        denoised_act_seq = noisy_act_seq[:, start:end]
        if clamp_range is not None:
            low, high = clamp_range
            denoised_act_seq = torch.clamp(denoised_act_seq, low, high)

        return denoised_act_seq

    def _compute_loss(self, cond_seq: torch.Tensor, act_seq: torch.Tensor) -> torch.Tensor:
        """Samples noise, adds it to the target sequence, and computes the reconstruction loss.

        Shapes:
            cond_seq: [B, cond_horizon * cond_dim] (flattened condition sequence)
            act_seq: [B, pred_horizon, act_dim] (target action chunk)
            returns: scalar loss tensor []
        """
        if self.network is None:
            raise ValueError(
                "Network not initialized. Call configure_model() before computing loss."
            )

        if self.noise_scheduler is None:
            raise ValueError(
                "Noise Scheduler not initialized. Call configure_model() before computing loss."
            )

        if self.ema is None:
            raise ValueError(
                "EMA Model not initialized. Call configure_model() before computing loss."
            )

        B = cond_seq.shape[0]

        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=self.device)

        timesteps = torch.randint(
            0,
            self.noise_scheduler.config["num_train_timesteps"],
            (B,),
            device=self.device,
            dtype=torch.int32,
        )
        timesteps = cast(torch.IntTensor, timesteps)

        noisy_act_seq = self.noise_scheduler.add_noise(act_seq, noise, timesteps)
        prediction = self.network(noisy_act_seq, timesteps, external_cond=cond_seq)

        pred_type = self.noise_scheduler.config.get("prediction_type", "epsilon")

        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = act_seq
        elif pred_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(act_seq, noise, timesteps)
        else:
            raise ValueError(f"Unsupported prediction_type: {pred_type}")

        loss = F.mse_loss(prediction, target)
        return loss
