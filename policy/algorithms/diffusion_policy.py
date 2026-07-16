import functools
from typing import Any, Literal, cast

import h5py
import hydra_zen
import lightning as L
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.training_utils import EMAModel
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from policy.transforms import MinMaxNormalizer, ZScoreNormalizer
from policy.utils import (
    concat_leaf_tensors,
    flatten_and_concat_leaf_tensors,
    get_batch_size,
    get_total_dim,
    to_tensor,
)
from policy.utils.h5_utils import load_h5_data
from policy.utils.typing_utils import DiffusionSchedulerProtocol, HydraConfigFor, PolicyProtocol
from policy.utils.utils import stack_dicts


class DiffusionPolicy(L.LightningModule, PolicyProtocol):
    """Trains a diffusion policy to predict action sequences from observation histories.

    Diffusion Policy as in Cheng et. al (IJRR)

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
        optimizer: HydraConfigFor[functools.partial[Optimizer]],
        lr_scheduler: HydraConfigFor[functools.partial[LRScheduler]] | None = None,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        act_horizon: int = 8,
        obs_dim: dict | int = 48,
        act_dim: int = 4,
        prediction_type: Literal["epsilon", "sample", "v_prediction"] = "epsilon",
        normalizer: bool | dict | HydraConfigFor[nn.Module] | None = None,
        action_normalizer: bool | dict | HydraConfigFor[nn.Module] | None = None,
        flatten_obs: bool | None = None,
    ):
        super().__init__()

        self.save_hyperparameters()

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

        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.act_horizon = act_horizon

        if self.act_horizon > self.pred_horizon:
            raise ValueError(
                f"Action horizon ({self.act_horizon}) cannot be greater than prediction horizon ({self.pred_horizon}). "
                "The model cannot execute more timesteps (act_horizon) than its total prediction horizon (pred_horizon)."
            )

        if self.obs_horizon + self.act_horizon - 1 > self.pred_horizon:
            raise ValueError(
                f"Prediction horizon ({self.pred_horizon}) is too short! "
                f"It must be at least {self.obs_horizon + self.act_horizon - 1} "
                f"to contain the past actions ({self.obs_horizon - 1}) plus "
                f"the actions to execute ({self.act_horizon})."
            )

        self.act_dim = act_dim
        self.obs_dim = obs_dim

        # Initialize the normalizer
        self.normalizer: ZScoreNormalizer | MinMaxNormalizer | None = None

        norm_spec = normalizer

        if omegaconf.OmegaConf.is_config(norm_spec):
            norm_spec = omegaconf.OmegaConf.to_object(norm_spec)

        if isinstance(norm_spec, bool) and norm_spec:
            self.normalizer = ZScoreNormalizer(obs_dim)
        elif isinstance(norm_spec, dict) and "_target_" in norm_spec:
            self.normalizer = hydra_zen.instantiate(norm_spec, spec=obs_dim)

        # Initialize the action normalizer
        self.action_normalizer: ZScoreNormalizer | MinMaxNormalizer | None = None

        act_norm_spec = action_normalizer

        if omegaconf.OmegaConf.is_config(act_norm_spec):
            act_norm_spec = omegaconf.OmegaConf.to_object(act_norm_spec)

        if isinstance(act_norm_spec, bool) and act_norm_spec:
            self.action_normalizer = ZScoreNormalizer(act_dim)
        elif isinstance(act_norm_spec, dict) and "_target_" in act_norm_spec:
            self.action_normalizer = hydra_zen.instantiate(act_norm_spec, spec=act_dim)

        if flatten_obs is None:
            # Auto-detect if we are using a Transformer or a Unet
            network_target = self.network_config.get("_target_").lower()
            if "gpt" in network_target or "transformer" in network_target:
                self.flatten_obs = False
            elif "unet" in network_target:
                self.flatten_obs = True
            else:
                raise ValueError(
                    f"Cannot auto-detect network type from target: {network_target}. "
                    "Please specify flatten_obs explicitly."
                )
        else:
            self.flatten_obs = flatten_obs

        if self.flatten_obs:
            self.network_cond_dim = self.obs_horizon * get_total_dim(obs_dim)
        else:
            self.network_cond_dim = get_total_dim(obs_dim)

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" and self.normalizer and self.action_normalizer:
            self._configure_normalizers()

    def configure_model(self) -> None:
        if self.network is not None:
            return
        self.network = hydra_zen.instantiate(
            self.network_config, external_cond_dim=self.network_cond_dim
        )

        if self.ema is not None:
            return
        self.ema = hydra_zen.instantiate(self.ema_config, parameters=self.network.parameters())

    def configure_optimizers(self) -> Optimizer | dict:

        optimizer_partial = hydra_zen.instantiate(self.optimizer_config)
        optimizer = optimizer_partial(filter(lambda p: p.requires_grad, self.parameters()))

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "DiffusionPolicy does not support direct forward pass. Use get_action() instead."
        )

    def get_action(
        self,
        obs_seq: torch.Tensor | dict,
        num_inference_timesteps: int | None = None,
        output_clip_range: tuple | None = None,
    ):
        """Runs the reverse diffusion process to predict an action sequence from the current
        observation.

        Shapes:
            obs_seq: [B, obs_horizon * obs_dim] (flattened conditioning)
            returns: [B, act_horizon, act_dim] (denoised actions to execute)
        """
        if self.normalizer is not None:
            obs_seq = self.normalizer.normalize(obs_seq)

        obs_seq = self._prepare_network_cond(obs_seq)

        return self._run_diffusion_loop(obs_seq, num_inference_timesteps, output_clip_range)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, "train")

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

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        raise NotImplementedError(
            "DiffusionPolicy does not support test_step. Use simulation rollouts instead."
        )

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

    def _shared_step(self, batch: dict[str, Any], batch_idx: int, phase: str) -> torch.Tensor:
        """Main step logic, it doesn't differ between training and validation except for the
        logging.

        Shapes:
            batch["obs_seq"]: [B, obs_horizon, obs_dim]
            batch["act_seq"]: [B, pred_horizon, act_dim]
            returns: scalar loss tensor []
        """
        obs_seq = batch["obs_seq"]
        action_seq = batch["act_seq"]

        if self.normalizer is not None:
            obs_seq = self.normalizer.normalize(obs_seq)

        if self.action_normalizer is not None:
            action_seq = self.action_normalizer.normalize(action_seq)

        network_cond = self._prepare_network_cond(obs_seq)

        loss = self._compute_loss(network_cond, action_seq)

        self.log(f"{phase}/loss", loss, prog_bar=True, sync_dist=(phase == "val"))
        return loss

    def _compute_loss(self, obs_seq: torch.Tensor, act_seq: torch.Tensor) -> torch.Tensor:
        """Samples noise, adds it to the target sequence, and computes the reconstruction loss.

        Shapes:
            obs_seq: [B, obs_horizon * obs_dim] (flattened condition sequence)
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

        B = obs_seq.shape[0]

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
        prediction = self.network(noisy_act_seq, timesteps, obs=obs_seq)

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

    def _configure_normalizers(self) -> None:

        if self.normalizer is None:
            raise ValueError(
                "Normalizer is None. Make sure to set the normalizer before training."
            )

        if self.action_normalizer is None:
            raise ValueError(
                "Action normalizer is None. Make sure to set the action normalizer before training."
            )

        if self.normalizer.is_fit and self.action_normalizer.is_fit:
            return

        dm = getattr(self.trainer, "datamodule", None)
        if dm is None:
            raise ValueError(
                "Datamodule is not available in the trainer. Make sure to set the datamodule before training."
            )

        train_set = getattr(dm, "train_set", None)
        if train_set is None:
            raise ValueError("Training set is not available in the datamodule.")

        if train_set.lazy:
            if not self.normalizer.is_fit:

                def trajectory_obs_generator():
                    for traj in train_set.trajectories:
                        ep_id = traj["episode_id"]
                        h5_traj = train_set.h5_file[f"traj_{ep_id}"]
                        obs_node = h5_traj["obs"]
                        if isinstance(obs_node, h5py.Group):
                            obs_ep = load_h5_data(obs_node)
                        else:
                            obs_ep = obs_node[:]
                        yield to_tensor(obs_ep)

                self.normalizer.fit_incremental(trajectory_obs_generator())

            if not self.action_normalizer.is_fit:

                def trajectory_act_generator():
                    for traj in train_set.trajectories:
                        ep_id = traj["episode_id"]
                        h5_traj = train_set.h5_file[f"traj_{ep_id}"]
                        act_node = h5_traj["actions"]
                        if isinstance(act_node, h5py.Group):
                            act_ep = load_h5_data(act_node)
                        else:
                            act_ep = act_node[:]
                        yield to_tensor(act_ep)

                self.action_normalizer.fit_incremental(trajectory_act_generator())
        else:
            if not self.normalizer.is_fit:
                all_obs = [to_tensor(traj["obs"]) for traj in train_set.trajectories]
                stacked_obs = stack_dicts(all_obs)
                self.normalizer.fit(stacked_obs)

            if not self.action_normalizer.is_fit:
                all_act = [to_tensor(traj["actions"]) for traj in train_set.trajectories]
                stacked_act = stack_dicts(all_act)
                self.action_normalizer.fit(stacked_act)

    def _run_diffusion_loop(
        self,
        network_cond: torch.Tensor,
        num_inference_steps: int | None = None,
        output_clip_range: tuple | None = None,
    ):
        """Generic helper containing the actual reverse diffusion process loop."""
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

        B = get_batch_size(network_cond)

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
                    obs=network_cond,
                )

                output = self.noise_scheduler.step(
                    model_output=model_pred,
                    timestep=t,
                    sample=noisy_act_seq,
                    return_dict=False,
                )

                noisy_act_seq = output[0]

        self.ema.restore(self.network.parameters())

        start = self.obs_horizon - 1
        end = start + self.act_horizon

        denoised_act_seq = noisy_act_seq[:, start:end]
        if self.action_normalizer is not None:
            denoised_act_seq = self.action_normalizer.unnormalize(denoised_act_seq)
        if output_clip_range is not None:
            low, high = output_clip_range
            denoised_act_seq = torch.clamp(denoised_act_seq, low, high)

        return denoised_act_seq

    def _prepare_network_cond(self, obs_seq: dict | torch.Tensor) -> torch.Tensor:
        """Prepares the observation sequence for the network conditioning."""
        if self.flatten_obs:
            return flatten_and_concat_leaf_tensors(obs_seq, device=self.device)
        else:
            return concat_leaf_tensors(obs_seq, device=self.device)
