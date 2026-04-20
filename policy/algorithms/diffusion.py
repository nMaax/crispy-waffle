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
from policy.utils import flatten_tensor_dict, get_batch_size, sum_shapes
from policy.utils.typing_utils import HydraConfigFor


class DiffusionPolicy(L.LightningModule):
    def __init__(
        self,
        network: HydraConfigFor[nn.Module],
        noise_scheduler: HydraConfigFor[DDPMScheduler],
        ema: HydraConfigFor[EMAModel],
        optimizer: HydraConfigFor[functools.partial[Optimizer]],
        datamodule: ManiSkillDataModule,
        act_horizon: int,
    ):
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
        self.noise_scheduler: DDPMScheduler | None = hydra_zen.instantiate(
            self.noise_scheduler_config
        )

        self.optimizer_config = optimizer
        self.optimizer: Optimizer | None = None

        self.datamodule = datamodule
        self.obs_horizon = self.datamodule.obs_horizon
        self.pred_horizon = self.datamodule.pred_horizon

        self.act_horizon = act_horizon

        # Extract these from the datamodule
        self.act_dim = self.datamodule.action_dim

        # TODO: not really that good to put env as obs, I am mixing terminology here, later re-order
        # Should also find a solution for naming convention, as sometimes you write act_dim and sometimes action_dim, obs_dim vs env_state_dim, etc.
        # Maybe we can just unify everything as obs and action, and then in the datamodule we can have some logic to extract the right dimensions from the right places, and then in the policy we just use obs_dim and action_dim without caring about where they come from?
        raw_obs_dim = (
            self.datamodule.env_state_dim
        )  # Grab the extracted shapes from the datamodule
        if isinstance(raw_obs_dim, dict):
            # Recursive helper to sum the last element of every 'shape' tuple
            self.obs_dim = sum_shapes(raw_obs_dim)
        else:
            # Fallback in case it's already an integer
            self.obs_dim = cast(int, raw_obs_dim)

    def configure_model(self):
        """Lightning calls this before training starts to initialize weights safely."""
        if self.network is not None:
            return

        # Here our conditioning is the flatten observation sequence, so we compute the dimension accordingly
        global_cond_dim = self.obs_horizon * self.obs_dim

        # Use hydra_zen to instantiate, injecting computed dimensions
        self.network = hydra_zen.instantiate(
            self.network_config, input_dim=self.act_dim, global_cond_dim=global_cond_dim
        )

        if self.ema is not None:
            return

        # Once network is instantiated, add EMA as well
        self.ema = hydra_zen.instantiate(self.ema_config, parameters=self.network.parameters())

    def configure_optimizers(self):
        """Creates the optimizers."""

        # TODO: Can Lighting configure a LR scheduler too?

        # NOTE: Optimizers and schedulers could actually be made in one shot, without partial, for how they are handles here
        # however I prefer to follow the template prescription, just for coherence

        # Instantiate the optimizer config into a functools.partial object
        optimizer_partial = hydra_zen.instantiate(self.optimizer_config)

        # Call the functools.partial object, passing the parameters as an argument
        optimizer = optimizer_partial(self.parameters())

        # This then returns the optimizer
        return optimizer

    def _compute_loss(self, obs_seq, action_seq):
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

        B = get_batch_size(obs_seq)
        obs_cond = flatten_tensor_dict(obs_seq)

        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=self.device)

        timesteps = torch.randint(
            0,
            self.noise_scheduler.num_diffusion_iters,
            (B,),
            device=self.device,
            dtype=torch.int32,
        )
        timesteps = cast(torch.IntTensor, timesteps)

        # Here we do noise-prediction (as in DDPM), not data prediction
        # TODO: Maybe in the future we could generalize noise pred vs action pred as hyperparameters in the DiffusionPolicy class?
        noisy_action_seq = self.noise_scheduler.add_noise(action_seq, noise, timesteps)
        noise_pred = self.network(noisy_action_seq, timesteps, global_cond=obs_cond)

        return F.mse_loss(noise_pred, noise)

    def _shared_step(self, batch, batch_idx, phase: str):
        "Main step logic, it doesn't differ between training and validation except for the logging."
        loss = self._compute_loss(batch["env_seq"], batch["action_seq"])
        self.log(f"{phase}/loss", loss, prog_bar=True, sync_dist=(phase == "val"))
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def on_train_batch_end(self, outputs, batch, batch_idx):
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

    def get_action(self, obs_seq):
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

        B = get_batch_size(obs_seq)

        # Temporarily copy EMA weights in the model
        self.ema.copy_to(self.network.parameters())

        with torch.no_grad():
            obs_cond = flatten_tensor_dict(obs_seq)
            noisy_action_seq = torch.randn(
                (B, self.pred_horizon, self.act_dim), device=self.device
            )

            for k in self.noise_scheduler.timesteps:
                k = cast(int, k.item())

                noise_pred = self.network(
                    sample=noisy_action_seq,
                    timestep=k,
                    global_cond=obs_cond,
                )
                output = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=noisy_action_seq,
                )

                noisy_action_seq = output.prev_sample

        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return noisy_action_seq[:, start:end]
