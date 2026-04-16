import lightning as L
import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch.optim.adamw import AdamW

from policy.algorithms.networks.unet1d import ConditionalUnet1D


class DiffusionPolicy(L.LightningModule):
    def __init__(
        self,
        obs_horizon: int,
        act_horizon: int,
        pred_horizon: int,
        action_dim: int,
        obs_dim: int,
        diffusion_step_embed_dim: int = 256,
        unet_dims: list = [256, 512, 1024],
        n_groups: int = 8,
        num_diffusion_iters: int = 100,
        lr: float = 1e-4,  # Added for Lightning optimizer
    ):
        super().__init__()
        # Saves all the arguments to self.hparams and logs them to W&B
        self.save_hyperparameters()

        self.obs_horizon = obs_horizon
        self.act_horizon = act_horizon
        self.pred_horizon = pred_horizon
        self.act_dim = action_dim
        self.obs_dim = obs_dim
        self.lr = lr

        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.act_dim,
            global_cond_dim=self.obs_horizon * self.obs_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=unet_dims,
            n_groups=n_groups,
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

    def configure_optimizers(self):
        """Lightning uses this to set up your optimizer."""
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        # You can also return a learning rate scheduler here if you want
        return optimizer

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
