import functools
from collections.abc import Mapping
from typing import Any

import hydra_zen
import lightning as L
import torch
import torch.nn as nn
from diffusers.training_utils import EMAModel
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from policy.algorithms.networks.utils import derive_task_dim, resolve_proprio_dim
from policy.transforms import MinMaxNormalizer, ZScoreNormalizer
from policy.transforms.canonicalization.spec import canonical_normalization_mask
from policy.utils import as_batch, cat_dicts, get_total_dim, pop_leaf_key
from policy.utils.typing_utils import (
    DiffusionSchedulerProtocol,
    DimSpec,
    HydraConfigFor,
    PolicyProtocol,
    TensorTree,
    TokenizerProtocol,
)


class BaseDiffusionAgent(L.LightningModule, PolicyProtocol):
    """Base class for diffusion-based imitation-learning agents.

    Subclasses must implement :meth:`_compute_loss` and
    :meth:`_run_diffusion_loop`.

    The :meth:`_shared_step` and :meth:`get_action`
    are provided as templates that subclasses
    may override to provide additional conditioning (e.g. goals).
    """

    def __init__(
        self,
        decoder: HydraConfigFor[nn.Module],
        optimizer: HydraConfigFor[functools.partial[Optimizer]],
        encoder: HydraConfigFor[nn.Module] | None = None,
        tokenizer: HydraConfigFor[TokenizerProtocol] | None = None,
        relative_goal: bool = False,
        lr_scheduler: HydraConfigFor[functools.partial[LRScheduler]] | None = None,
        ema: HydraConfigFor[EMAModel] | None = None,
        noise_scheduler: HydraConfigFor[DiffusionSchedulerProtocol] | None = None,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        act_horizon: int = 8,
        obs_dim: DimSpec = 48,
        act_dim: int = 4,
        goal_horizon: int = 0,
        proprio_dim: int | None = None,
        task_dim: int | None = None,
        obs_normalizer: bool | HydraConfigFor[nn.Module] | None = None,
        act_normalizer: bool | HydraConfigFor[nn.Module] | None = None,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.decoder_config = decoder
        self.decoder: torch.nn.Module | None = None

        self.encoder_config = encoder
        self.encoder: torch.nn.Module | None = None

        self.tokenizer_config = tokenizer
        self.tokenizer: TokenizerProtocol | None = None
        self.relative_goal = relative_goal

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
        self.goal_horizon = goal_horizon
        self._validate_horizons()

        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self._proprio_dim = proprio_dim
        self._task_dim = task_dim

        self.obs_normalizer_config = obs_normalizer
        self.obs_normalizer: nn.Module | None = None

        self.act_normalizer_config = act_normalizer
        self.act_normalizer: nn.Module | None = None

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

    @property
    def goal_conditioned(self) -> bool:
        """Whether a goal is part of the conditioning at all."""
        return self.goal_horizon > 0

    @property
    def proprio_dim(self) -> int:
        """Width of the proprioception leaf, resolved against ``obs_dim`` on demand."""
        return resolve_proprio_dim(self.obs_dim, self._proprio_dim)

    @property
    def task_dim(self) -> int:
        """Width of the task (non-proprio) portion of an observation."""
        return derive_task_dim(self.obs_dim, self.proprio_dim, self._task_dim)

    def configure_model(self) -> None:
        if self.decoder is not None:
            return

        self._instantiate_tokenizer()

        self.encoder = (
            hydra_zen.instantiate(self.encoder_config, **self._encoder_extra_kwargs())
            if self.encoder_config is not None
            else None
        )

        self.obs_normalizer = self._instantiate_normalizer(
            config=self.obs_normalizer_config,
            spec=self._obs_normalizer_spec(),
            default_cls=ZScoreNormalizer,
            mask=self._obs_normalizer_mask(),
        )
        self.act_normalizer = self._instantiate_normalizer(
            config=self.act_normalizer_config,
            spec=self.act_dim,
            default_cls=MinMaxNormalizer,
            mask=None,
        )

        self.decoder = hydra_zen.instantiate(self.decoder_config, **self._decoder_extra_kwargs())

        if self.ema_config is not None:
            self.ema = hydra_zen.instantiate(self.ema_config, parameters=self._ema_parameters())

        if self.noise_scheduler_config is not None:
            self.noise_scheduler = hydra_zen.instantiate(self.noise_scheduler_config)

    def _instantiate_tokenizer(self) -> None:
        """Builds the tokenizer the encoder is sized from and the obs normalizer is fit in."""
        self.tokenizer = (
            hydra_zen.instantiate(
                self.tokenizer_config,
                task_dim=self._tokenizer_task_dim(),
                relative_goal=self.relative_goal,
            )
            if self.tokenizer_config is not None
            else None
        )
        self._validate_tokenizer()

    def _tokenizer_task_dim(self) -> Mapping[str, DimSpec]:
        """The task-only dim spec handed to the tokenizer, with proprioception split off."""
        if not isinstance(self.obs_dim, Mapping):
            raise TypeError(
                f"Tokenizers require a canonical dict obs_dim tree, got {type(self.obs_dim).__name__}. "
            )
        return {key: dim for key, dim in self.obs_dim.items() if key != "proprio"}

    def _validate_tokenizer(self) -> None:
        if self.encoder_config is not None and self.tokenizer is None:
            raise ValueError(
                f"{type(self).__name__} is configured with an encoder but no tokenizer. The "
                "encoder consumes tokens, so name one explicitly in the config (e.g. "
                "`tokenizer: state`)."
            )

        if self.tokenizer is None:
            return

        if not self.relative_goal and not self.tokenizer.supports_single_side:
            raise ValueError(
                f"{type(self.tokenizer).__name__} cannot be used with relative_goal=False: "
                "it only produces tokens for goal deltas, so absolute conditioning isn't supported."
            )

        if not self.goal_conditioned and not self.tokenizer.supports_single_side:
            raise ValueError(
                f"{type(self.tokenizer).__name__} cannot tokenize a standalone observation "
                "state (supports_single_side=False), so it cannot be used unconditioned."
            )

    def _instantiate_normalizer(
        self,
        config: bool | HydraConfigFor[nn.Module] | None,
        spec: DimSpec,
        default_cls: type[nn.Module],
        mask: TensorTree | None = None,
    ) -> nn.Module | None:
        """Instantiates a normalizer from its Hydra config, over the given dimension spec.

        config = ``True`` yields a :class:`ZScoreNormalizer` for the
        observations and a :class:`MinMaxNormalizer` for the actions; otherwise use
        Hydra configs to specify a custom class.
        """
        if isinstance(config, bool) and config:
            return default_cls(spec, mask=mask)
        if isinstance(config, Mapping):
            return hydra_zen.instantiate(config, spec=spec, mask=mask)
        return None

    def _obs_normalizer_spec(self) -> DimSpec:
        """The space the obs normalizer is fit in: tokens when tokenizing, else the raw tree."""
        if self.tokenizer is None:
            return self.obs_dim
        else:
            return {"proprio": self.proprio_dim, "task": self.tokenizer.token_spec}

    def _obs_normalizer_mask(self) -> TensorTree | None:
        """Channels of :meth:`_obs_normalizer_spec` that an affine rescale would destroy."""
        if self.tokenizer is not None:
            return {"task": self.tokenizer.categorical_mask}
        if isinstance(self.obs_dim, Mapping):
            return canonical_normalization_mask(self.obs_dim)
        return None

    def on_fit_start(self) -> None:
        """Fits the normalizers, after ``configure_model`` has built the tokenizer they need."""
        self._fit_normalizers()

    def _fit_normalizers(self) -> None:
        if self.obs_normalizer is None and self.act_normalizer is None:
            return

        dm = getattr(self.trainer, "datamodule", None)
        if dm is None:
            raise ValueError(
                "Datamodule is not available in the trainer. Make sure to set the datamodule before training."
            )

        train_set = getattr(dm, "train_set", None)
        if train_set is None:
            raise ValueError("Training set is not available in the datamodule.")

        if self.obs_normalizer is not None and not self.obs_normalizer.is_fit:
            if train_set.lazy:
                self.obs_normalizer.fit_incremental(
                    self._obs_normalizer_view(item) for item in train_set
                )
            else:
                self.obs_normalizer.fit(
                    cat_dicts([self._obs_normalizer_view(item) for item in train_set])
                )

        if self.act_normalizer is not None and not self.act_normalizer.is_fit:
            if train_set.lazy:
                self.act_normalizer.fit_incremental(item["act_seq"] for item in train_set)
            else:
                self.act_normalizer.fit(cat_dicts([item["act_seq"] for item in train_set]))

    def _obs_normalizer_view(self, item: dict[str, Any]) -> TensorTree:
        """Maps one training-set item into :meth:`_obs_normalizer_spec`'s space."""
        if self.tokenizer is None:
            return item["obs_seq"]
        return self._tokenize(as_batch(item["obs_seq"]))

    def _tokenize(self, obs: TensorTree) -> dict[str, TensorTree]:
        """Splits proprioception off and turns the task tree into raw, pre-normalization tokens."""
        if self.tokenizer is None:
            raise ValueError(
                f"{type(self).__name__} has no tokenizer; _tokenize() should not be reached."
            )

        proprio, task = pop_leaf_key(obs, "proprio", self.proprio_dim)
        if proprio is None:
            raise ValueError("Observation mapping must contain a 'proprio' key.")

        return {"proprio": proprio, "task": self.tokenizer.tokenize(task, None)}

    def _encoder_extra_kwargs(self) -> dict[str, Any]:
        """Extra kwargs threaded to encoder instantiation."""
        if self.tokenizer is None:
            raise ValueError("Cannot size an encoder without a tokenizer.")

        return {
            "proprio_dim": self.proprio_dim,
            "token_dim": self.tokenizer.output_dim,
            "tokens_per_step": self.tokenizer.tokens_per_step,
            "goal_conditioned": self.goal_conditioned,
            "relative_goal": self.relative_goal,
        }

    def _decoder_extra_kwargs(self) -> dict[str, Any]:
        """Extra kwargs threaded to decoder instantiation."""
        return {"cond_dims": self._get_cond_dims()}

    def _get_cond_dims(self) -> DimSpec:
        """Reports the per-timestep conditioning dimensionality passed to the decoder."""
        if self.encoder is not None:
            return self.encoder.cond_dims
        return {"obs": get_total_dim(self.obs_dim)}

    def _encode(self, external_cond: Mapping[str, TensorTree]) -> Mapping[str, TensorTree]:
        """Tokenizes, normalizes, then embeds."""
        if "obs" not in external_cond:
            raise ValueError("external_cond must contain an 'obs' entry.")

        if self.encoder is None:
            return external_cond

        return self.encoder(self._normalize_obs(self._tokenize(**external_cond)))

    def _ema_parameters(self) -> list[torch.nn.Parameter]:
        """Parameters tracked by EMA: whatever training actually updates."""
        if self.decoder is None:
            raise ValueError("Decoder not initialized. Call configure_model() first.")

        modules = [self.decoder] if self.encoder is None else [self.encoder, self.decoder]
        return [p for m in modules for p in m.parameters() if p.requires_grad]

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
        if self.decoder is None:
            raise ValueError(
                "Decoder not initialized. Call configure_model() before on_train_batch_end."
            )

        if self.ema is not None:
            self.ema.to(self.device)
            self.ema.step(self._ema_parameters())

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
    ) -> torch.Tensor:
        """Runs the reverse diffusion process to predict an action sequence from the current
        observation.

        Shapes:
            obs_seq: [B, obs_horizon * obs_dim] (flattened conditioning) or dict
            returns: [B, act_horizon, act_dim] (denoised actions to execute)
        """
        external_cond = self._build_external_cond(obs_seq)
        return self._run_diffusion_loop(
            external_cond=external_cond,
            num_inference_steps=num_inference_steps,
            output_clip_range=output_clip_range,
        )

    def _shared_step(self, batch: dict[str, Any], batch_idx: int, phase: str) -> torch.Tensor:
        """Main step logic, it doesn't differ between training and validation except for the
        logging.

        Shapes:
            batch["obs_seq"]: [B, obs_horizon, obs_dim] or dict
            batch["act_seq"]: [B, pred_horizon, act_dim]
            returns: scalar loss tensor []
        """
        action_seq = self._normalize_act(batch["act_seq"])
        external_cond = self._build_external_cond_from_batch(batch)

        loss = self._compute_loss(external_cond, action_seq)

        self.log(f"{phase}/loss", loss, prog_bar=True, sync_dist=(phase == "val"))
        return loss

    def _normalize_obs(self, obs: TensorTree) -> TensorTree:
        """Normalizes observations if a normalizer is configured; otherwise returns them as-is."""
        return self.obs_normalizer.normalize(obs) if self.obs_normalizer is not None else obs

    def _normalize_act(self, act: torch.Tensor) -> torch.Tensor:
        """Normalizes actions if a normalizer is configured; otherwise returns them as-is."""
        return self.act_normalizer.normalize(act) if self.act_normalizer is not None else act

    def _build_external_cond_from_batch(self, batch: dict[str, Any]) -> dict[str, TensorTree]:
        """Extracts the observation conditioning from a batch."""
        return self._build_external_cond(batch["obs_seq"])

    def _build_external_cond(self, obs: TensorTree) -> dict[str, TensorTree]:
        """Prepares the raw (un-normalized) network ``external_cond``."""
        return {"obs": obs}

    def _compute_loss(
        self, external_cond: Mapping[str, TensorTree], act_seq: torch.Tensor
    ) -> torch.Tensor:
        """Samples noise, adds it to the target sequence, and computes the reconstruction loss.

        Shapes:
            external_cond: network conditioning tree (e.g. ``{"obs": ...}``)
            act_seq: [B, pred_horizon, act_dim] (target action chunk)
            returns: scalar loss tensor []
        """
        raise NotImplementedError(f"{type(self).__name__} must implement _compute_loss().")

    def _run_diffusion_loop(
        self,
        external_cond: Mapping[str, TensorTree],
        num_inference_steps: int | None = None,
        output_clip_range: tuple | None = None,
    ) -> torch.Tensor:
        """Reverse diffusion process loop.

        Shapes:
            external_cond: network conditioning tree (e.g. ``{"obs": ...}``)
            returns: [B, act_horizon, act_dim] (denoised actions to execute)
        """
        raise NotImplementedError(f"{type(self).__name__} must implement _run_diffusion_loop().")
