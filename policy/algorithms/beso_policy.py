import functools
from typing import Any

import hydra_zen
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import EDMEulerScheduler
from diffusers.training_utils import EMAModel
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from policy.utils import flatten_tensor_from_mapping, get_batch_size
from policy.utils.typing_utils import HydraConfigFor, PolicyProtocol


class BesoPolicy(L.LightningModule, PolicyProtocol):
    """Trains a Behavioral Cloning via Score-Based Diffusion (BESO) policy.

    Replaces custom k-diffusion samplers with Hugging Face's EDMEulerScheduler.
    """

    def __init__(
        self,
        network: HydraConfigFor[nn.Module],
        ema: HydraConfigFor[EMAModel],
        optimizer: HydraConfigFor[functools.partial[Optimizer]],
        lr_scheduler: HydraConfigFor[functools.partial[LRScheduler]] | None = None,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        act_horizon: int = 8,
        obs_dim: int = 48,
        act_dim: int = 4,
        sigma_data: float = 0.5,
        sigma_min: float = 0.001,
        sigma_max: float = 80.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Network points to the KarrasDenoiserWrapper config
        self.network_config = network
        self.network: torch.nn.Module | None = None
        self.ema_config = ema
        self.ema: EMAModel | None = None

        self.optimizer_config = optimizer
        self.lr_scheduler_config = lr_scheduler

        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.act_horizon = act_horizon
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # BESO/K-Diffusion Parameters
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # Instantiate Hugging Face EDM Scheduler for inference
        self.noise_scheduler = EDMEulerScheduler(
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            sigma_data=self.sigma_data,
            prediction_type="epsilon",  # Internally handled by HF, but our wrapper outputs x_0
        )

    def configure_model(self) -> None:
        if self.network is not None:
            return

        # Instantiate Wrapper -> Instantiates inner DiffusionGPT
        self.network = hydra_zen.instantiate(self.network_config)

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
                "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step", "frequency": 1},
            }
        return optimizer

    def _sample_lognormal_sigmas(
        self, batch_size: int, mean: float = -1.2, std: float = 1.2
    ) -> torch.Tensor:
        """Draws training sigmas from a log-normal distribution as per Karras et al."""
        rnd_normal = torch.randn(batch_size, device=self.device)
        return (rnd_normal * std + mean).exp()

    def _compute_loss(self, obs_seq: torch.Tensor, act_seq: torch.Tensor) -> torch.Tensor:
        """BESO Loss formulation.

        The wrapper handles the preconditioner scaling.
        """
        if self.network is None or self.ema is None:
            raise ValueError("Initialize network and EMA via configure_model() first.")

        B = obs_seq.shape[0]

        # Sample continuous noise levels
        sigmas = self._sample_lognormal_sigmas(B)

        # Add noise to target actions
        noise = torch.randn_like(act_seq)
        sigma_bd = sigmas.view(B, 1, 1)
        noisy_act_seq = act_seq + noise * sigma_bd

        # Predict denoised sequence (Karras wrapper returns predicted x_0)
        denoised_pred = self.network(sample=noisy_act_seq, timestep=sigmas, obs=obs_seq)

        # K-diffusion training computes MSE directly against the clean target
        loss = F.mse_loss(denoised_pred, act_seq)
        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        pass

    def _shared_step(self, batch: dict[str, Any], batch_idx: int, phase: str) -> torch.Tensor:
        flatten_obs = flatten_tensor_from_mapping(batch["obs_seq"])
        action_seq = batch["act_seq"]
        loss = self._compute_loss(flatten_obs, action_seq)
        self.log(f"{phase}/loss", loss, prog_bar=True, sync_dist=(phase == "val"))
        return loss

    def on_train_batch_end(self, outputs: torch.Tensor, batch: dict[str, Any], batch_idx: int):
        self.ema.to(self.device)
        self.ema.step(self.network.parameters())

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        if self.ema is not None:
            checkpoint["ema_state_dict"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        super().on_load_checkpoint(checkpoint)
        self.configure_model()
        if self.ema is not None and "ema_state_dict" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state_dict"])

    def get_action(
        self,
        obs_seq: torch.Tensor | dict,
        num_inference_steps: int = 20,
        clamp_range: tuple | None = None,
    ):
        """Runs the Karras EDM reverse diffusion process via Hugging Face."""
        B = get_batch_size(obs_seq)
        obs_seq = flatten_tensor_from_mapping(obs_seq, device=self.device)

        self.ema.store(self.network.parameters())
        self.ema.copy_to(self.network.parameters())

        # Set up the HF EDM Scheduler discrete timesteps (which map to sigmas)
        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)

        with torch.no_grad():
            # Initial noise scaled by the maximum noise level (sigma_max)
            noisy_act_seq = (
                torch.randn((B, self.pred_horizon, self.act_dim), device=self.device)
                * self.sigma_max
            )

            for t in self.noise_scheduler.timesteps:
                # In EDM, the timestep itself is the sigma value
                sigma = t.expand(B)

                # We do not need to call noise_scheduler.scale_model_input() here
                # because our KarrasDenoiserWrapper explicitly handles the c_in scaling internally!

                # Get model prediction (Our wrapper outputs the denoised x_0)
                model_pred = self.network(sample=noisy_act_seq, timestep=sigma, obs=obs_seq)

                # HF Scheduler step maps the predicted x_0 and current noise to the previous noise level
                output = self.noise_scheduler.step(
                    model_output=model_pred,
                    timestep=t,
                    sample=noisy_act_seq,
                    return_dict=False,
                )
                noisy_act_seq = output[0]

        self.ema.restore(self.network.parameters())

        # Extract the actions to execute
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        denoised_act_seq = noisy_act_seq[:, start:end]

        if clamp_range is not None:
            low, high = clamp_range
            denoised_act_seq = torch.clamp(denoised_act_seq, low, high)

        return denoised_act_seq
