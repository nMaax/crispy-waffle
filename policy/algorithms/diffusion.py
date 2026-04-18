import functools
from typing import cast

import hydra_zen
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from torch.optim.optimizer import Optimizer

from policy.datamodules.maniskill_datamodule import ManiSkillDataModule
from policy.utils import flatten_tensor_dict, get_batch_size
from policy.utils.typing_utils import HydraConfigFor


class DiffusionPolicy(L.LightningModule):
    def __init__(
        self,
        network: HydraConfigFor[nn.Module],
        optimizer: HydraConfigFor[functools.partial[Optimizer]],
        datamodule: ManiSkillDataModule,
        act_horizon: int,
        action_dim: int,
        obs_dim: int,
        num_diffusion_iters: int = 100,
        lr: float = 1e-4,
        warmup_steps: int = 500,
        init_seed: int = 42,
    ):
        super().__init__()
        # Saves all the arguments to self.hparams and logs them to W&B
        # datamodule is excluded because LightningDataModule is not JSON-serialisable as plain hyperparameters.
        self.save_hyperparameters(ignore=["datamodule"])

        self.network_config = network
        self.network: torch.nn.Module | None = None
        self.ema: EMAModel | None = None

        self.optimizer_config = optimizer
        self.optimizer: Optimizer | None = None

        self.datamodule = datamodule
        self.obs_horizon = self.datamodule.obs_horizon
        self.pred_horizon = self.datamodule.pred_horizon

        self.act_horizon = act_horizon
        self.act_dim = action_dim
        self.obs_dim = obs_dim
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.init_seed = init_seed

        self.num_diffusion_iters = num_diffusion_iters
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

    def configure_model(self):
        """Lightning calls this before training starts to initialize weights safely."""
        if self.network is not None:
            return

        global_cond_dim = self.obs_horizon * self.obs_dim

        # Fork the RNG to guarantee reproducible weight initialization
        with torch.random.fork_rng():
            # TODO: review your approach to seeding overall, where do you set the seeds? Are you sure that one seed will cover everything and there aren't sneaky overwrites?
            # Do not worry about setting seed manually, we are inside the with block, so once this finishes the original RNG state will be restored.
            torch.manual_seed(self.init_seed)

            # Use hydra_zen to instantiate, injecting computed dimensions
            self.network = hydra_zen.instantiate(
                self.network_config, input_dim=self.act_dim, global_cond_dim=global_cond_dim
            )

        # Now that the network exists, we can create the EMA model
        self.ema = EMAModel(
            parameters=self.network.parameters(),
            decay=0.999,
            inv_gamma=1.0,
            power=0.75,
        )

    def configure_optimizers(self):
        # TODO: doesn't lighitng have configure lr scheduler?
        """Creates the optimizers."""

        # Instantiate the optimizer config into a functools.partial object.
        optimizer_partial = hydra_zen.instantiate(self.optimizer_config)

        # Call the functools.partial object, passing the parameters as an argument.
        optimizer = optimizer_partial(self.parameters())

        # This then returns the optimizer.
        return optimizer

    def _compute_loss(self, obs_seq, action_seq):
        """Loss calculation logic."""

        if self.network is None:
            raise ValueError(
                "Network not initialized. Call configure_model() before computing loss."
            )

        B = get_batch_size(obs_seq)
        obs_cond = flatten_tensor_dict(obs_seq)

        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=self.device)

        timesteps = torch.randint(
            0, self.num_diffusion_iters, (B,), device=self.device, dtype=torch.int32
        )
        timesteps = cast(torch.IntTensor, timesteps)

        noisy_action_seq = self.noise_scheduler.add_noise(action_seq, noise, timesteps)
        noise_pred = self.network(noisy_action_seq, timesteps, global_cond=obs_cond)

        return F.mse_loss(noise_pred, noise)

    def shared_step(self, batch, batch_idx, phase: str):
        loss = self._compute_loss(batch["env_seq"], batch["action_seq"])
        self.log(f"{phase}/loss", loss, prog_bar=True, sync_dist=(phase == "val"))
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "val")

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Automatically step the EMA model after every training iteration."""
        if self.network is None:
            raise ValueError(
                "Network not initialized. Call configure_model() before on_train_batch_end."
            )

        if self.ema is None:
            raise ValueError(
                "EMA not initialized. Call configure_model() before on_train_batch_end."
            )

        self.ema.step(self.network.parameters())

    def get_action(self, obs_seq):
        """Used during inference/evaluation in the environment."""

        if self.network is None:
            raise ValueError(
                "Network not initialized. Call configure_model() before getting action."
            )

        if self.ema is None:
            raise ValueError("EMA not initialized. Call configure_model() before getting action.")

        B = get_batch_size(obs_seq)

        # Temporarily copy EMA weights in the model
        self.ema.copy_to(self.network.parameters())

        with torch.no_grad():
            obs_cond = flatten_tensor_dict(obs_seq)
            noisy_action_seq = torch.randn(
                (B, self.pred_horizon, self.act_dim), device=self.device
            )

            for k in self.noise_scheduler.timesteps:
                noise_pred = self.network(
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
