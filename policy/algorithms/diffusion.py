from typing import Any

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from torch.optim.adamw import AdamW

from policy.datamodules.maniskill_datamodule import ManiSkillDataModule


class DiffusionPolicy(L.LightningModule):
    def __init__(
        self,
        network: nn.Module,
        datamodule: ManiSkillDataModule,
        obs_horizon: int,
        act_horizon: int,
        pred_horizon: int,
        action_dim: int,
        obs_dim: int,
        num_diffusion_iters: int = 100,
        lr: float = 1e-4,
        warmup_steps: int = 500,
        **kwargs,
    ):
        super().__init__()
        # TODO: are we sure about this? Maybe I should include network?
        # Saves all the arguments to self.hparams and logs them to W&B
        # network and datamodule are excluded because nn.Module / LightningDataModule
        # are not JSON-serialisable as plain hyperparameters.
        self.save_hyperparameters(ignore=["network", "datamodule"])

        self.obs_horizon = obs_horizon
        self.act_horizon = act_horizon
        self.pred_horizon = pred_horizon
        self.act_dim = action_dim
        self.obs_dim = obs_dim
        self.lr = lr
        self.warmup_steps = warmup_steps

        self.datamodule = datamodule

        # Assign the network first so that self.ema can reference its parameters.
        self.noise_pred_net = network

        self.ema = EMAModel(
            parameters=self.noise_pred_net.parameters(),
            decay=0.999,
            inv_gamma=1.0,
            power=0.75,
        )

        self.num_diffusion_iters = num_diffusion_iters
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

    def training_step(self, batch, batch_idx):
        """Lightning automatically calls this during trainer.fit()"""
        # Note: Depending on your dataloader dict keys, adjust these:
        obs_seq = batch["env_seq"]
        action_seq = batch["action_seq"]

        loss = self._compute_loss(obs_seq, action_seq)

        # Log to wandb/tensorboard automatically
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Lightning automatically calls this during validation."""
        obs_seq = batch["env_seq"]
        action_seq = batch["action_seq"]

        loss = self._compute_loss(obs_seq, action_seq)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self) -> Any:  # Override output for pyright
        """Lightning uses this to set up your optimizer."""
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)

        stepping_batches = self.trainer.estimated_stepping_batches

        if isinstance(stepping_batches, float):
            raise ValueError(
                "Training steps evaluated to infinity! "
                "Ensure you have set trainer.max_steps in your Hydra config."
            )

        lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=stepping_batches,
        )

        # Lightning parses this dict, so we tell Pyright to ignore the return type mismatch
        return {  # type: ignore[return-value]
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Automatically step the EMA model after every training iteration."""
        self.ema.step(self.noise_pred_net.parameters())

    def _compute_loss(self, obs_seq, action_seq):
        """Your original loss calculation logic."""
        B = obs_seq.shape[0]
        obs_cond = obs_seq.flatten(start_dim=1)

        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=self.device)

        timesteps = torch.randint(0, self.num_diffusion_iters, (B,), device=self.device).long()

        noisy_action_seq = self.noise_scheduler.add_noise(action_seq, noise, timesteps)  # type: ignore[argtype]
        noise_pred = self.noise_pred_net(noisy_action_seq, timesteps, global_cond=obs_cond)

        return F.mse_loss(noise_pred, noise)

    def get_action(self, obs_seq):
        """Used during inference/evaluation in the environment."""
        B = obs_seq.shape[0]

        # Temporarily copy EMA weights in the model
        self.ema.copy_to(self.noise_pred_net.parameters())

        with torch.no_grad():
            obs_cond = obs_seq.flatten(start_dim=1)
            noisy_action_seq = torch.randn(
                (B, self.pred_horizon, self.act_dim), device=self.device
            )

            for k in self.noise_scheduler.timesteps:
                noise_pred = self.noise_pred_net(
                    sample=noisy_action_seq,
                    timestep=k,
                    global_cond=obs_cond,
                )
                output = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,  # type: ignore[argtype]
                    sample=noisy_action_seq,
                )

                noisy_action_seq = output.prev_sample

        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return noisy_action_seq[:, start:end]
