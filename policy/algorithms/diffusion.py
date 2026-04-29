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

# TODO: Review whole template to ensure it works
# TODO: Compare with original maniskill code, line by line

# TODO: docstrings with types and shapes everywhere
# TODO: less comments within code? --> Try write a more CLEAR CODE


class DiffusionPolicy(L.LightningModule):
    def __init__(
        self,
        network: HydraConfigFor[nn.Module],
        noise_scheduler: HydraConfigFor[DiffusionSchedulerProtocol],
        ema: HydraConfigFor[EMAModel],
        datamodule: ManiSkillDataModule,
        optimizer: HydraConfigFor[functools.partial[Optimizer]],
        lr_scheduler: HydraConfigFor[functools.partial[LRScheduler]] | None = None,
        act_horizon: int = 8,
        prediction_type: Literal["epsilon", "sample", "v_prediction"] = "epsilon",
    ):
        """Implements a diffusion policy.

        parameters:
            - network: a Hydra config for the policy network architecture
            - noise_scheduler: a Hydra config for the noise scheduler
            - ema: a Hydra config for the EMA model
            - datamodule: the ManiSkillDataModule instance that will provide the training and validation
            - optimizer: a Hydra config for the optimizer
            - lr_scheduler: an optional Hydra config for the learning rate scheduler
            - act_horizon: the number of future actions to predict
            - prediction_type: the type of prediction the noise scheduler is configured for. Should be one of "epsilon", "sample", or "v_prediction".
        """
        super().__init__()

        # Saves all the arguments to self.hparams and logs them to W&B
        # datamodule is excluded because LightningDataModule is not JSON-serialisable as plain hyperparameters
        # Network and optimizer are included as they are passed as Hydra configs
        self.save_hyperparameters(ignore=["datamodule"])

        self.network_config = network
        self.network: torch.nn.Module | None = None
        self.ema_config = ema
        self.ema: EMAModel | None = None

        self.noise_scheduler_config = noise_scheduler
        self.noise_scheduler: DiffusionSchedulerProtocol | None = hydra_zen.instantiate(
            self.noise_scheduler_config, prediction_type=prediction_type
        )

        self.datamodule = datamodule
        self.cond_horizon = self.datamodule.cond_horizon
        self.pred_horizon = self.datamodule.pred_horizon

        self.optimizer_config = optimizer
        self.optimizer: Optimizer | None = None

        self.lr_scheduler_config = lr_scheduler
        self.lr_scheduler: LRScheduler | None = None

        self.act_horizon = act_horizon
        self.act_dim = self.datamodule.act_dim

        if self.cond_horizon + self.act_horizon - 1 > self.pred_horizon:
            raise ValueError(
                f"Prediction horizon ({self.pred_horizon}) is too short! "
                f"It must be at least {self.cond_horizon + self.act_horizon - 1} "
                f"to contain the past actions ({self.cond_horizon - 1}) plus "
                f"the actions to execute ({self.act_horizon})."
            )

        # The conditioning signal fed to the network is determined by the datamodule's
        # use_phsyx_env_states. We use cond_dim as the single source of truth so that the
        # policy never needs to know whether it comes from env_states, obs, or both.
        raw_cond_dim = self.datamodule.cond_dim
        if isinstance(raw_cond_dim, dict):
            # Recursive helper to sum the last element of every 'shape' tuple
            self.cond_dim = sum_shapes(raw_cond_dim)
        else:
            # Fallback in case it's already an integer
            self.cond_dim = raw_cond_dim

        # TODO: Normalize observation/env_states
        #   - Should be pre-computed for the dataset, in the maniskill_datamodule, maybe saving them as <h5_file_path>.stats.json (one time only, to avoid repeated computation every time we train a model)
        #   - Then the diffusion policy can load them and apply the normalization using the TensorZNormalizer from utils/normalizer.py
        #   - The Diffusion should correctly select the file related to what we are looking for: cond_source, then select physix_env_states vs obs, and apply the normalization accordingly.
        #   - Thus Diffusion should have access to the datamodule's parameters about cond_source, physx_env_states, obs_mode, etc. to be able to do this correctly.
        #   - Can diffusion access such data? YES, since datamodule is passed in the init

    def configure_model(self) -> None:
        """Lightning calls this before training starts to initialize weights safely."""
        if self.network is not None:
            return

        # Here our conditioning is the flatten observation sequence, so we compute the dimension accordingly
        external_cond_dim = self.cond_horizon * self.cond_dim

        # Use hydra_zen to instantiate, injecting computed dimensions
        self.network = hydra_zen.instantiate(
            self.network_config, input_dim=self.act_dim, external_cond_dim=external_cond_dim
        )

        if self.ema is not None:
            return

        # Once network is instantiated, add EMA as well
        self.ema = hydra_zen.instantiate(self.ema_config, parameters=self.network.parameters())

    def configure_optimizers(self) -> Optimizer | dict:
        """Creates the optimizer and LR scheduler."""

        # NOTE: Optimizers and schedulers could actually be made in one shot, without partial,
        # however I prefer to follow the template prescription, just for coherence

        # Instantiate the optimizer config
        optimizer_partial = hydra_zen.instantiate(self.optimizer_config)
        optimizer = optimizer_partial(self.parameters())

        # Instantiate the scheduler config, if provided
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

    def _compute_loss(self, cond_seq: torch.Tensor, act_seq: torch.Tensor) -> torch.Tensor:
        """Loss calculation logic."""

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

        # Accessing prediction_type directly is deprecated, use config instead
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config["num_train_timesteps"],
            (B,),
            device=self.device,
            dtype=torch.int32,
        )
        timesteps = cast(torch.IntTensor, timesteps)

        flatten_cond = flatten_tensor_dict(cond_seq)

        noisy_act_seq = self.noise_scheduler.add_noise(act_seq, noise, timesteps)
        prediction = self.network(noisy_act_seq, timesteps, external_cond=flatten_cond)

        pred_type = self.noise_scheduler.config.get("prediction_type", "epsilon")

        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = act_seq
        elif pred_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(act_seq, noise, timesteps)
        else:
            raise ValueError(f"Unsupported prediction_type: {pred_type}")

        return F.mse_loss(prediction, target)

    def _shared_step(self, batch, batch_idx, phase: str) -> torch.Tensor:
        "Main step logic, it doesn't differ between training and validation except for the logging."
        # Datamodule will return a dict, but we need the underlying tensors
        flatten_cond = flatten_tensor_dict(batch["cond_seq"])
        action_seq = batch["act_seq"]

        loss = self._compute_loss(flatten_cond, action_seq)
        self.log(f"{phase}/loss", loss, prog_bar=True, sync_dist=(phase == "val"))
        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        pass

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

        self.ema.step(self.network.parameters())

    def get_action(
        self,
        cond_seq: torch.Tensor,
        num_inference_steps: int | None = None,
        clamp_range: tuple | None = None,
    ):
        """Used during inference/evaluation in the environment."""
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

        # Store main network weights
        self.ema.store(self.network.parameters())

        # Temporarily copy EMA weights in the model
        self.ema.copy_to(self.network.parameters())

        if num_inference_steps is None:
            num_inference_steps = int(self.noise_scheduler.config["num_train_timesteps"])

        # Initialize the timesteps for the scheduler
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
