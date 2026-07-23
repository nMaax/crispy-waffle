import functools
from collections.abc import Mapping
from typing import Any, cast

import hydra_zen
import lightning as L
import omegaconf
import torch
import torch.nn as nn
from diffusers.training_utils import EMAModel
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from policy.transforms import MinMaxNormalizer, ZScoreNormalizer
from policy.utils import cat_dicts, get_total_dim
from policy.utils.typing_utils import (
    DiffusionSchedulerProtocol,
    DimSpec,
    HydraConfigFor,
    PolicyProtocol,
    TensorTree,
)


class BaseDiffusionAgent(L.LightningModule, PolicyProtocol):
    """Base class for diffusion-based imitation-learning agents.

    Subclasses must implement :meth:`_compute_loss` and
    :meth:`_run_diffusion_loop`.

    The :meth:`_shared_step` and :meth:`get_action`
    are provided as templates that subclasses
    may override to thread additional conditioning (e.g. goals).
    """

    def __init__(
        self,
        network: HydraConfigFor[nn.Module],
        optimizer: HydraConfigFor[functools.partial[Optimizer]],
        lr_scheduler: HydraConfigFor[functools.partial[LRScheduler]] | None = None,
        ema: HydraConfigFor[EMAModel] | None = None,
        noise_scheduler: HydraConfigFor[DiffusionSchedulerProtocol] | None = None,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        act_horizon: int = 8,
        obs_dim: DimSpec = 48,
        act_dim: int = 4,
        obs_normalizer: bool | HydraConfigFor[nn.Module] | None = None,
        act_normalizer: bool | HydraConfigFor[nn.Module] | None = None,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.network_config = network
        self.network: torch.nn.Module | None = None

        self.optimizer_config = optimizer
        self.optimizer: Optimizer | None = None

        self.lr_scheduler_config = lr_scheduler
        self.lr_scheduler: LRScheduler | None = None

        self.ema_config = ema
        self.ema: EMAModel | None = None

        self.noise_scheduler_config = noise_scheduler
        self.noise_scheduler: DiffusionSchedulerProtocol | None = None

        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.act_horizon = act_horizon
        self._validate_horizons()

        self.act_dim = act_dim
        self.obs_dim = obs_dim

        self._instantiate_normalizers(obs_normalizer, act_normalizer)

    def _validate_horizons(self) -> None:
        """Sanity-checks the observation / prediction / action horizons."""
        if self.act_horizon > self.pred_horizon:
            raise ValueError(
                f"Action horizon ({self.act_horizon}) cannot be greater than "
                f"prediction horizon ({self.pred_horizon}). The model cannot "
                "execute more timesteps (act_horizon) than its total prediction "
                "horizon (pred_horizon)."
            )

        if self.obs_horizon + self.act_horizon - 1 > self.pred_horizon:
            raise ValueError(
                f"Prediction horizon ({self.pred_horizon}) is too short! "
                f"It must be at least {self.obs_horizon + self.act_horizon - 1} "
                f"to contain the past actions ({self.obs_horizon - 1}) plus "
                f"the actions to execute ({self.act_horizon})."
            )

    def _instantiate_normalizers(
        self,
        obs_normalizer: bool | HydraConfigFor[nn.Module] | None,
        act_normalizer: bool | HydraConfigFor[nn.Module] | None,
    ) -> None:
        """Instantiates the observation and action normalizers from their specs.

        By convention a bare ``True`` yields a :class:`ZScoreNormalizer` for the
        observations and a :class:`MinMaxNormalizer` for the actions; otherwise use
        Hydra configs to specify a custom class.
        """
        self.obs_normalizer: nn.Module | None = self._build_normalizer(
            obs_normalizer, self.obs_dim, ZScoreNormalizer
        )
        self.act_normalizer: nn.Module | None = self._build_normalizer(
            act_normalizer, self.act_dim, MinMaxNormalizer
        )

    @staticmethod
    def _build_normalizer(
        spec: bool | HydraConfigFor[nn.Module] | None,
        dim: DimSpec,
        default_cls: type[nn.Module],
    ) -> nn.Module | None:
        """Instantiates a normalizer from a spec."""
        if omegaconf.OmegaConf.is_config(spec):
            spec = cast(Any, omegaconf.OmegaConf.to_object(spec))

        if isinstance(spec, bool) and spec:
            return default_cls(dim)
        if isinstance(spec, dict) and "_target_" in spec:
            return hydra_zen.instantiate(spec, spec=dim)
        return None

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" and self.obs_normalizer and self.act_normalizer:
            self._configure_normalizers()

    def _configure_normalizers(self) -> None:
        if self.obs_normalizer is None:
            raise ValueError(
                "Observation normalizer is None. Make sure to set the obs normalizer before training."
            )

        if self.act_normalizer is None:
            raise ValueError(
                "Action normalizer is None. Make sure to set the act normalizer before training."
            )

        dm = getattr(self.trainer, "datamodule", None)
        if dm is None:
            raise ValueError(
                "Datamodule is not available in the trainer. Make sure to set the datamodule before training."
            )

        train_set = getattr(dm, "train_set", None)
        if train_set is None:
            raise ValueError("Training set is not available in the datamodule.")

        if train_set.lazy:
            if not self.obs_normalizer.is_fit:

                def obs_generator():
                    for item in train_set:
                        yield item["obs_seq"]

                self.obs_normalizer.fit_incremental(obs_generator())

            if not self.act_normalizer.is_fit:

                def act_generator():
                    for item in train_set:
                        yield item["act_seq"]

                self.act_normalizer.fit_incremental(act_generator())
        else:
            if not self.obs_normalizer.is_fit:
                all_obs = [item["obs_seq"] for item in train_set]
                self.obs_normalizer.fit(cat_dicts(all_obs))

            if not self.act_normalizer.is_fit:
                all_act = [item["act_seq"] for item in train_set]
                self.act_normalizer.fit(cat_dicts(all_act))

    def configure_model(self) -> None:
        if self.network is not None:
            return

        cond_dims = self._get_cond_dims()
        self.network = hydra_zen.instantiate(self.network_config, cond_dims=cond_dims)

        if self.ema_config is not None:
            self.ema = hydra_zen.instantiate(self.ema_config, parameters=self.network.parameters())

        if self.noise_scheduler_config is not None:
            self.noise_scheduler = hydra_zen.instantiate(self.noise_scheduler_config)

    def _get_cond_dims(self) -> DimSpec:
        """Reports the per-timestep conditioning dimensionality passed to the network's
        ``cond_dims``.

        Widths here are *not* multiplied by ``obs_horizon`` -- each network knows its own horizon
        (via config) and is responsible for resolving how it consumes the time axis of each key.
        """
        return {"obs": get_total_dim(self.obs_dim)}

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
            f"{type(self).__name__} does not support a direct forward pass. Use get_action() instead."
        )

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, "train")

    def on_train_batch_end(
        self, outputs: torch.Tensor, batch: dict[str, Any], batch_idx: int
    ) -> None:
        """Automatically step the EMA model after every training batch iteration."""
        if self.network is None:
            raise ValueError(
                "Network not initialized. Call configure_model() before on_train_batch_end."
            )

        if self.ema is not None:
            self.ema.to(self.device)
            self.ema.step(self.network.parameters())

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        # Dummy step; actual evaluation is handled in simulation rollouts via RolloutEvaluationCallback
        pass

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
        obs_seq: torch.Tensor | Mapping[str, Any],
        num_inference_steps: int | None = None,
        output_clip_range: tuple | None = None,
    ):
        """Runs the reverse diffusion process to predict an action sequence from the current
        observation.

        Shapes:
            obs_seq: [B, obs_horizon * obs_dim] (flattened conditioning) or dict
            returns: [B, act_horizon, act_dim] (denoised actions to execute)
        """
        if self.obs_normalizer is not None:
            obs_seq = self.obs_normalizer.normalize(obs_seq)

        external_cond = self._build_external_cond(obs_seq)

        action = self._run_diffusion_loop(
            external_cond=external_cond,
            num_inference_steps=num_inference_steps,
            output_clip_range=output_clip_range,
        )

        return action

    def _shared_step(self, batch: dict[str, Any], batch_idx: int, phase: str) -> torch.Tensor:
        """Main step logic, it doesn't differ between training and validation except for the
        logging.

        Shapes:
            batch["obs_seq"]: [B, obs_horizon, obs_dim] or dict
            batch["act_seq"]: [B, pred_horizon, act_dim]
            returns: scalar loss tensor []
        """
        obs_seq = batch["obs_seq"]
        action_seq = batch["act_seq"]

        if self.obs_normalizer is not None:
            obs_seq = self.obs_normalizer.normalize(obs_seq)

        if self.act_normalizer is not None:
            action_seq = self.act_normalizer.normalize(action_seq)

        external_cond = self._build_external_cond(obs_seq)

        loss = self._compute_loss(external_cond, action_seq)

        self.log(f"{phase}/loss", loss, prog_bar=True, sync_dist=(phase == "val"))
        return loss

    def _build_external_cond(self, obs: TensorTree) -> dict[str, TensorTree]:
        """Prepares the network ``external_cond``. ``external_cond`` is a dict of tensors.

        Observations are packaged in their un-flattened shape (``[B, obs_horizon,
        dim]`` or a nested tree of such tensors); handling and flattening them is a responsibility
        of the receiving network.
        """
        return {"obs": obs}

    def _compute_loss(self, external_cond: TensorTree, act_seq: torch.Tensor) -> torch.Tensor:
        """Samples noise, adds it to the target sequence, and computes the reconstruction loss.

        Shapes:
            external_cond: network conditioning tree (e.g. ``{"obs": ...}``)
            act_seq: [B, pred_horizon, act_dim] (target action chunk)
            returns: scalar loss tensor []
        """
        raise NotImplementedError(f"{type(self).__name__} must implement _compute_loss().")

    def _run_diffusion_loop(
        self,
        external_cond: TensorTree,
        num_inference_steps: int | None = None,
        output_clip_range: tuple | None = None,
    ) -> torch.Tensor:
        """Reverse diffusion process loop.

        Shapes:
            external_cond: network conditioning tree (e.g. ``{"obs": ...}``)
            returns: [B, act_horizon, act_dim] (denoised actions to execute)
        """
        raise NotImplementedError(f"{type(self).__name__} must implement _run_diffusion_loop().")
