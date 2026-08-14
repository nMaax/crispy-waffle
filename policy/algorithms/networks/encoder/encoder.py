from logging import getLogger as get_logger
from typing import Literal

import hydra_zen
import torch
import torch.nn as nn

from policy.algorithms.networks.encoder.spec import CondEntry, ConditioningSpec
from policy.algorithms.networks.encoder.tokenizers import StateTokenizer
from policy.utils import (
    derive_task_dim,
    get_ndim,
    map_leaves,
    merge_dicts,
    pop_leaf_key,
    resolve_proprio_dim,
)
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
        goal_conditioned: bool = True,
        relative_goal: bool = False,
        mode: Literal["film", "cross_attention"] | None = None,
        decoder_type: Literal["film", "cross_attention"] = "film",
        tokenizer: HydraConfigFor[TokenizerProtocol] | None = None,
        embedder: HydraConfigFor[nn.Module] | None = None,
        pooling: HydraConfigFor[PoolingProtocol] | None = None,
    ):
        super().__init__()

        self.proprio_dim = resolve_proprio_dim(obs_dim, proprio_dim)
        self.task_dim = derive_task_dim(obs_dim, self.proprio_dim, task_dim)

        self.goal_conditioned = goal_conditioned
        self.relative_goal = relative_goal

        self.decoder_type = mode if mode is not None else decoder_type
        self.mode = self.decoder_type

        # Tokenizer: unlike embedder/pooling below, this is never actually optional -- something
        # must always turn a task tree into tokens before embedding can happen, so tokenizer=None
        # just selects the default (StateTokenizer) rather than leaving self.tokenizer unset.
        if isinstance(tokenizer, TokenizerProtocol):
            self.tokenizer: TokenizerProtocol = tokenizer
        elif tokenizer is not None:
            self.tokenizer = hydra_zen.instantiate(
                tokenizer, task_dim=self.task_dim, relative_goal=self.relative_goal
            )
        else:
            logger.warning(
                "ConditioningEncoder built with tokenizer=None; silently defaulting to "
                "StateTokenizer. Prefer naming it explicitly in the Hydra config (e.g. "
                "`override tokenizer@encoder.tokenizer: state`) so the choice is visible there "
                "instead of hidden behind this fallback."
            )
            self.tokenizer = StateTokenizer(
                task_dim=self.task_dim, relative_goal=self.relative_goal
            )

        self.tokens_per_step: int = getattr(self.tokenizer, "tokens_per_step", 1)
        tokenizer_output_dim: int = getattr(self.tokenizer, "output_dim", self.task_dim)

        # Embedder: genuinely optional -- None means "pass raw tokens straight through" (identity).
        if isinstance(embedder, nn.Module):
            self.embedder: nn.Module | None = embedder
        elif embedder is not None:
            self.embedder = hydra_zen.instantiate(embedder, input_dim=tokenizer_output_dim)
        else:
            self.embedder = None

        self.output_dim: int = getattr(self.embedder, "output_dim", tokenizer_output_dim)

        # Pooling: genuinely optional -- None means "don't collapse the time/token axis".
        if isinstance(pooling, nn.Module | PoolingProtocol):
            self.pooling: PoolingProtocol | nn.Module | None = pooling
        elif pooling is not None:
            self.pooling = hydra_zen.instantiate(pooling, dim=self.output_dim)
        else:
            self.pooling = None

        # Validation
        self._validate_config()
        self.cond_dims: ConditioningSpec = self._compute_cond_dims()

    @property
    def pooling_mode(self) -> Literal["all", "objects", "time"] | None:
        return getattr(self.pooling, "mode", None) if self.pooling is not None else None

    @property
    def pools_time(self) -> bool:
        if self.pooling is None:
            return False
        return getattr(self.pooling, "pools_time", self.pooling_mode in ("all", "time"))

    @property
    def pools_objects(self) -> bool:
        if self.pooling is None:
            return False
        return getattr(self.pooling, "pools_objects", self.pooling_mode in ("all", "objects"))

    def _validate_config(self) -> None:
        if self.relative_goal and not self.goal_conditioned:
            raise ValueError(
                "relative_goal=True requires goal_conditioned=True: there is no goal to "
                "difference the observations against otherwise."
            )
        if not self.relative_goal and not self.tokenizer.supports_single_side:
            raise ValueError(
                f"{type(self.tokenizer).__name__} has no standalone tokenization of a single "
                "state (supports_single_side=False), so it cannot be used with relative_goal=False."
            )
        if not self.goal_conditioned and not self.tokenizer.supports_single_side:
            raise ValueError(
                f"{type(self.tokenizer).__name__} has no standalone tokenization of a single "
                "state (supports_single_side=False), so it cannot be used unconditioned."
            )

        if self.decoder_type == "cross_attention" and (
            not self.relative_goal or self.tokens_per_step <= 1
        ):
            raise ValueError(
                "mode='cross_attention' requires relative_goal=True and a tokenizer with "
                f"tokens_per_step > 1; got relative_goal={self.relative_goal!r}, "
                f"tokens_per_step={self.tokens_per_step}. "
                "Cross-attention needs a genuine per-object token sequence to attend over."
            )

    def _compute_cond_dims(self) -> ConditioningSpec:
        entries = dict(self._task_cond_entries())
        if self.goal_conditioned and not self.relative_goal:
            goal_width = (
                self.output_dim
                if (self.pools_time or self.pools_objects)
                else self.output_dim * self.tokens_per_step
            )
            entries["goal"] = CondEntry(width=goal_width, kind="global")
        return ConditioningSpec(entries)

    def _task_cond_entries(self) -> dict[str, CondEntry]:
        if self.decoder_type == "cross_attention":
            return {
                "obs": CondEntry(width=self.proprio_dim, kind="per_timestep"),
                "context": CondEntry(width=self.output_dim, kind="sequence"),
            }
        if self.pooling_mode == "time":
            return {
                "obs": CondEntry(width=self.proprio_dim, kind="per_timestep"),
                "task": CondEntry(width=self.output_dim, kind="sequence"),
            }
        if self.pools_time:
            # Pooling collapses the time axis, so "task" no longer shares "obs"'s
            # per-timestep width and must live outside it (mirrors "goal", which never has one).
            return {
                "obs": CondEntry(width=self.proprio_dim, kind="per_timestep"),
                "task": CondEntry(width=self.output_dim, kind="global"),
            }
        task_width = (
            self.output_dim if self.pools_objects else self.output_dim * self.tokens_per_step
        )
        return {
            "obs": CondEntry(
                width=self.proprio_dim + task_width,
                kind="per_timestep",
            )
        }

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

        tokens_per_step = self.tokens_per_step
        # With one token per timestep (the default), a time-axis-free input is 2D ([B, F]); with
        # K > 1 tokens per timestep, it's 3D ([B, K, F]) instead -- one rank higher either way.
        expected_ndim_with_time = 3 if tokens_per_step == 1 else 4
        had_no_time_axis = task.ndim == expected_ndim_with_time - 1
        if had_no_time_axis:
            task = task.unsqueeze(1)

        task_embedded = self.embedder(task) if self.embedder is not None else task

        if self.pooling is not None:
            task_embedded = self.pooling(task_embedded)
        elif tokens_per_step > 1:
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
