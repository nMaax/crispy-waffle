from collections.abc import Mapping
from logging import getLogger as get_logger
from typing import Literal

import hydra_zen
import torch
import torch.nn as nn

from policy.algorithms.networks.encoder.spec import ConditioningContract
from policy.algorithms.networks.encoder.tokenizers import StateTokenizer
from policy.algorithms.networks.utils import derive_task_dim, resolve_proprio_dim
from policy.utils import get_ndim, map_leaves, merge_dicts, pop_leaf_key
from policy.utils.typing_utils import (
    DimSpec,
    HydraConfigFor,
    PoolingProtocol,
    TensorTree,
    TokenizerProtocol,
)

logger = get_logger(__name__)


class ConditioningEncoder(nn.Module):
    """Turns a canonicalized obs/goal tree into the conditioning tensors a downstream network
    consumes."""

    def __init__(
        self,
        obs_dim: DimSpec,
        proprio_dim: int | None = None,
        task_dim: int | None = None,
        goal_conditioned: bool = False,
        relative_goal: bool = False,
        decoder_type: Literal["film", "cross_attention"] = "film",
        tokenizer: HydraConfigFor[TokenizerProtocol] | None = None,
        embedder: HydraConfigFor[nn.Module] | None = None,
        pooling: HydraConfigFor[PoolingProtocol] | None = None,
    ):
        super().__init__()

        self.proprio_dim = resolve_proprio_dim(obs_dim, proprio_dim)
        self.task_dim = derive_task_dim(obs_dim, self.proprio_dim, task_dim)
        task_dim_spec: DimSpec = (
            {k: v for k, v in obs_dim.items() if k != "proprio"}
            if isinstance(obs_dim, Mapping)
            else self.task_dim
        )

        self.goal_conditioned = goal_conditioned
        self.relative_goal = relative_goal

        self.decoder_type = decoder_type

        # Tokenizer
        if isinstance(tokenizer, TokenizerProtocol):
            self.tokenizer: TokenizerProtocol = tokenizer
        elif tokenizer is not None:
            self.tokenizer = hydra_zen.instantiate(
                tokenizer, task_dim=task_dim_spec, relative_goal=self.relative_goal
            )
        else:
            logger.warning(
                "ConditioningEncoder built with tokenizer=None; silently defaulting to "
                "StateTokenizer. Prefer naming it explicitly in the Hydra config (e.g. "
                "`override tokenizer@encoder.tokenizer: state`) so the choice is visible there "
                "instead of hidden behind this fallback."
            )
            self.tokenizer = StateTokenizer(
                task_dim=task_dim_spec, relative_goal=self.relative_goal
            )
        self.tokens_per_step = self.tokenizer.tokens_per_step

        # Embedder
        if isinstance(embedder, nn.Module):
            self.embedder = embedder
        elif embedder is not None:
            self.embedder = hydra_zen.instantiate(embedder, input_dim=self.tokenizer.output_dim)
        else:
            self.embedder = None

        if self.embedder is not None:
            self.output_dim = self.embedder.output_dim
        else:
            self.output_dim = self.tokenizer.output_dim

        # Pooling
        if isinstance(pooling, nn.Module | PoolingProtocol):
            self.pooling = pooling
        elif pooling is not None:
            self.pooling = hydra_zen.instantiate(pooling, dim=self.output_dim)
        else:
            self.pooling = None

        # Validation and extra options
        self._validate_config()
        self.cond_dims = self._compute_cond_dims()

    def _validate_config(self) -> None:
        if self.relative_goal and not self.goal_conditioned:
            raise ValueError(
                "relative_goal=True requires goal_conditioned=True: there is no goal to "
                "difference the observations against otherwise."
            )

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

        if self.decoder_type == "cross_attention":
            if self.tokens_per_step is not None and self.tokens_per_step <= 1:
                raise ValueError(
                    "mode='cross_attention' requires a tokenizer with tokens_per_step > 1; "
                    f"got relative_goal={self.relative_goal!r}, tokens_per_step={self.tokens_per_step}. "
                    "Cross-attention needs a genuine per-object token sequence to attend over."
                )

        if self.decoder_type == "film" and self.tokens_per_step is None and not self.pools_objects:
            raise ValueError(
                "decoder_type='film' with a dynamic/variable number of object tokens requires "
                "pooling across objects (pools_objects=True, e.g. AttentionPooling(mode='objects'))."
            )

    def _compute_cond_dims(self) -> ConditioningContract:
        step_task_dim = (
            self.output_dim
            if self.pools_objects
            else self.output_dim * (self.tokens_per_step or 1)
        )
        goal_dim = step_task_dim if (self.goal_conditioned and not self.relative_goal) else 0

        if self.decoder_type == "cross_attention":
            return ConditioningContract(
                step_dim=self.proprio_dim,
                global_dim=goal_dim,
                context_dim=self.output_dim,
                context_key="context",
            )

        if self.pools_time:
            return ConditioningContract(
                step_dim=self.proprio_dim,
                global_dim=self.output_dim + goal_dim,
            )
        return ConditioningContract(
            step_dim=self.proprio_dim + step_task_dim,
            global_dim=goal_dim,
        )

    @property
    def is_multi_token(self) -> bool:
        """Whether the encoder processes multiple or dynamic tokens per timestep."""
        return self.tokens_per_step is None or self.tokens_per_step > 1

    @property
    def pooling_mode(self) -> Literal["all", "objects", "time"] | None:
        return self.pooling.mode if self.pooling is not None else None

    @property
    def pools_time(self) -> bool:
        return self.pooling.pools_time if self.pooling is not None else False

    @property
    def pools_objects(self) -> bool:
        return self.pooling.pools_objects if self.pooling is not None else False

    def forward(self, obs: TensorTree, goal: TensorTree | None = None) -> dict[str, TensorTree]:
        if not self.goal_conditioned:
            payload = self._build_obs(obs)
        elif goal is None:
            raise ValueError("goal_conditioned=True, but received goal=None.")
        elif self.relative_goal:
            payload = self._build_delta(obs, goal)
        else:
            payload = merge_dicts([self._build_obs(obs), self._build_goal(goal)])

        self.cond_dims.validate_payload(payload)
        return payload

    def _build_obs(self, obs: TensorTree) -> dict[str, TensorTree]:
        proprio, task_embedded = self._embed_states(obs)
        return self._package_task(proprio, task_embedded)

    def _build_goal(self, goal: TensorTree) -> dict[str, TensorTree]:
        _, goal_embedded = self._embed_states(goal, is_goal=True)
        return {"goal": goal_embedded}

    def _build_delta(self, obs: TensorTree, goal: TensorTree) -> dict[str, TensorTree]:
        """Conditions on the difference between the goal and each observation timestep."""
        obs_proprio, obs_task = pop_leaf_key(obs, "proprio", self.proprio_dim)
        goal_proprio, goal_task = pop_leaf_key(goal, "proprio", self.proprio_dim)
        if obs_proprio is None or goal_proprio is None:
            raise ValueError("Observation/goal mapping must contain a 'proprio' key.")

        # A 2D obs would not raise below: it would broadcast against the unsqueezed goal into
        # [B, B, F], silently mistaking the batch axis for the time axis. Checked generically
        # across dict-of-poses and flat-tensor task trees via the first leaf tensor found.
        if get_ndim(obs_task) != 3:
            raise ValueError(
                "relative_goal=True expects observations of shape [B, T, F], but got a "
                f"{get_ndim(obs_task)}D tree (shape shown after splitting off proprioception)."
            )

        # goal_task's own missing time axis is added here, on every leaf,
        # rather than left to the tokenizer.
        if get_ndim(goal_task) == 2:
            goal_task = map_leaves(lambda t: t.unsqueeze(1), goal_task)

        task_delta = self._embed_task(obs_task, goal_task)

        # The goal's own proprioception never enters conditioning: only the historical (observed)
        # proprioception is passed through raw, same as the absolute-conditioning path.
        return self._package_task(obs_proprio, task_delta)

    def _package_task(self, proprio: torch.Tensor, task: torch.Tensor) -> dict[str, TensorTree]:
        if self.decoder_type == "cross_attention":
            return {"obs": {"proprio": proprio}, "context": task}
        if self.pools_time:
            return {"obs": {"proprio": proprio}, "task": task}
        return {"obs": {"proprio": proprio, "task": task}}

    def _embed_states(
        self, states: TensorTree, *, is_goal: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Splits proprio/task and embeds the task components."""
        proprio, task = pop_leaf_key(states, "proprio", self.proprio_dim)
        if proprio is None:
            raise ValueError("Observation/goal mapping must contain a 'proprio' key.")

        if not self.tokenizer.supports_single_side:
            raise NotImplementedError(
                f"{type(self.tokenizer).__name__} tokenizes only goal-relative deltas "
                "(supports_single_side=False); there is no standalone embedding of a single "
                "state, so absolute conditioning isn't supported."
            )

        obs_task = None if is_goal else task
        goal_task = task if is_goal else None
        return proprio, self._embed_task(obs_task, goal_task)

    def _embed_task(
        self, obs_task: TensorTree | None, goal_task: TensorTree | None
    ) -> torch.Tensor:
        """Processes task components through tokenization, embedding, and pooling."""
        task = self.tokenizer.tokenize(obs_task, goal_task)

        assert isinstance(task, torch.Tensor), (
            f"Expected task to be a torch.Tensor, got {type(task)}"
        )

        had_no_time_axis = task.ndim == (3 if self.is_multi_token else 2)
        if had_no_time_axis:
            task = task.unsqueeze(1)

        task_embedded = self.embedder(task) if self.embedder is not None else task

        if self.pooling is not None:
            task_embedded = self.pooling(task_embedded)
        elif task_embedded.ndim == 4:
            b, t, k, d = task_embedded.shape
            # Cross-attention keeps K as a sequence axis (t-major, k-minor) for the network to
            # attend over; everything else folds it back into one wider per-timestep vector so
            # downstream FiLM conditioning keeps seeing "one vector per timestep".
            task_embedded = (
                task_embedded.reshape(b, t * k, d)
                if self.decoder_type == "cross_attention"
                else task_embedded.reshape(b, t, k * d)
            )

        # A pooling embedder already drops the time axis it was given, so there's nothing left to
        # squeeze; squeeze(1) would then operate on output_dim, which generally is not size 1
        # (thus squeeze would be a no-op); however we avoid it to keep it clean.
        if had_no_time_axis and not self.pools_time:
            task_embedded = task_embedded.squeeze(1)

        return task_embedded

    @torch.no_grad()
    def extract_embeddings(
        self, obs: TensorTree, goal: TensorTree | None = None
    ) -> dict[str, torch.Tensor]:
        """Extracts embedder outputs for observations (and optionally a goal).

        Helper for visualizing the embeddings. Always reports the absolute embeddings,
        independently of ``relative_goal``, so visualizations stay comparable across both modes.
        """
        _, obs_task_embeddings = self._embed_states(obs)
        res = {"obs_embeddings": obs_task_embeddings}
        if goal is not None:
            _, goal_task_embedding = self._embed_states(goal, is_goal=True)
            res["goal_embedding"] = goal_task_embedding
        return res
