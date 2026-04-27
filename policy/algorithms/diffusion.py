import functools
from typing import Literal, cast

import hydra_zen
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.training_utils import EMAModel
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from policy.datamodules.maniskill_datamodule import ManiSkillDataModule
from policy.utils import flatten_tensor_dict, get_batch_size, sum_shapes
from policy.utils.typing_utils import DiffusionSchedulerProtocol, HydraConfigFor

# TODO: Major fixes
# - [ ] Double check we are passing action/seq pairs synchronized between train and eval (see if slicing use the right indices)
# - [ ] Normalize observation/env_states before feeding them, consider deltas_* actions should already leave in the [-1, +1] range
#   - Choose normalization formula coherent with DiffusionPolicy, e.g. MinMax instead of z-score, or even better make this a hyperparameter

# TODO: Minor details
#   - [ ] review whole template to ensure it works


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
        self.act_dim = self.datamodule.action_dim

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

    def configure_model(self):
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

    def configure_optimizers(self):
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

    def _compute_loss(self, cond_seq, action_seq):
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

        B = get_batch_size(cond_seq)

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

        noisy_action_seq = self.noise_scheduler.add_noise(action_seq, noise, timesteps)
        prediction = self.network(noisy_action_seq, timesteps, external_cond=flatten_cond)

        pred_type = self.noise_scheduler.config.get("prediction_type", "epsilon")

        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = action_seq
        elif pred_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(action_seq, noise, timesteps)
        else:
            raise ValueError(f"Unsupported prediction_type: {pred_type}")

        return F.mse_loss(prediction, target)

    def _shared_step(self, batch, batch_idx, phase: str):
        "Main step logic, it doesn't differ between training and validation except for the logging."
        loss = self._compute_loss(batch["cond_seq"], batch["action_seq"])
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

    def get_action(
        self, cond_seq, num_inference_steps: int | None = None, clamp_action: bool = True
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

        B = get_batch_size(cond_seq)

        # Store main network weights
        self.ema.store(self.network.parameters())

        # Temporarily copy EMA weights in the model
        self.ema.copy_to(self.network.parameters())

        if num_inference_steps is None:
            num_inference_steps = int(self.noise_scheduler.config["num_train_timesteps"])

        # Initialize the timesteps for the scheduler
        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)

        with torch.no_grad():
            flatten_cond = flatten_tensor_dict(cond_seq)
            noisy_action_seq = torch.randn(
                (B, self.pred_horizon, self.act_dim), device=self.device
            )

            for t in self.noise_scheduler.timesteps:
                t = int(t.item())

                latent_model_input = self.noise_scheduler.scale_model_input(noisy_action_seq, t)

                model_pred = self.network(
                    sample=latent_model_input,
                    timestep=t,
                    external_cond=flatten_cond,
                )

                output = self.noise_scheduler.step(
                    model_output=model_pred,
                    timestep=t,
                    sample=noisy_action_seq,
                    return_dict=False,
                )

                noisy_action_seq = output[0]

        self.ema.restore(self.network.parameters())

        start = self.cond_horizon - 1
        end = start + self.act_horizon

        denoised_action_seq = noisy_action_seq[:, start:end]
        if clamp_action:
            low = torch.as_tensor(
                self.env.action_space.low,
                device=denoised_action_seq.device,
                dtype=denoised_action_seq.dtype,
            )
            high = torch.as_tensor(
                self.env.action_space.high,
                device=denoised_action_seq.device,
                dtype=denoised_action_seq.dtype,
            )

            denoised_action_seq = torch.clamp(denoised_action_seq, low, high)

        return denoised_action_seq
