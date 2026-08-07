from collections.abc import Mapping
from typing import Any

import hydra_zen
import torch
import torch.nn as nn

from policy.algorithms.diffusion_policy import DiffusionPolicy
from policy.algorithms.tokenizers import FlattenStateTokenizer
from policy.utils import (
    derive_task_dim,
    get_ndim,
    map_leaves,
    merge_dicts,
    resolve_proprio_dim,
    split_leaf_key,
)
from policy.utils.typing_utils import (
    DimSpec,
    GoalConditionedPolicyProtocol,
    GoalDelta,
    HydraConfigFor,
    StateTokenizer,
    TensorTree,
)


class GoalConditionedDiffusionPolicy(DiffusionPolicy, GoalConditionedPolicyProtocol):
    """Goal-conditioned diffusion policy using diffusers noise schedulers.

    Three conditioning modes, selected by ``goal_delta``:

    - ``goal_delta=None`` (default, absolute): the network sees the observation embeddings and
      the goal embedding as separate conditioning entries, i.e. ``s_1, ..., s_T`` plus ``g``.
    - ``goal_delta="input"`` (relative): the goal is folded into the observation window, so the
      network sees one difference per observed timestep, differenced before the embedder,
      ``embed(g - s_t)``, and no standalone goal entry.
    - ``goal_delta="embedding"``: the same, but differenced after the embedder,
      ``embed(g) - embed(s_t)``. Identical to ``"input"`` for a bias-free linear embedder (the
      default identity included); the two diverge only for a nonlinear one.
      See :meth:`_build_delta_external_cond`.

    Proprioception never goes through the ``embedder``, which keeps embedders robot-agnostic; it is
    concatenated raw alongside the embedder outputs. ``exclude_proprio_from_goal=False`` adds the
    goal's proprioception to those outputs, next to the historical proprioception when concatenated
    with the embeddings.

    ``tokenizer`` selects how a state (or a goal) becomes the raw, pre-embedder tokens that
    ``embedder`` attends/pools over.
    """

    def __init__(
        self,
        *args,
        goal_horizon: int = 1,
        goal_delta: GoalDelta = None,
        proprio_dim: int | None = None,
        task_dim: int | None = None,
        embedder: HydraConfigFor[nn.Module] | None = None,
        tokenizer: HydraConfigFor[StateTokenizer] | None = None,
        exclude_proprio_from_goal: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.goal_horizon = goal_horizon
        self.goal_conditioned = goal_horizon > 0
        self.goal_delta = goal_delta

        if goal_delta is not None and not self.goal_conditioned:
            raise ValueError(
                f"goal_delta={goal_delta!r} requires goal_horizon > 0: there is no goal to "
                "difference the observations against when goal-conditioning is disabled."
            )

        proprio_dim, task_dim = self._validate_obs_dim(proprio_dim, task_dim)

        self.proprio_dim = proprio_dim
        self.task_dim = task_dim
        self.goal_dim = task_dim
        self.exclude_proprio_from_goal = exclude_proprio_from_goal

        self.embedder_config = embedder
        self.embedder: nn.Module | None = None

        self.tokenizer_config = tokenizer
        self.tokenizer: StateTokenizer | None = None

    def _validate_obs_dim(self, proprio_dim: int | None, task_dim: int | None) -> tuple[int, int]:
        proprio_dim = resolve_proprio_dim(self.obs_dim, proprio_dim)
        task_dim = derive_task_dim(self.obs_dim, proprio_dim, task_dim)
        return proprio_dim, task_dim

    def configure_model(self) -> None:
        if self.network is not None:
            return
        tokenizer = self._resolve_tokenizer()

        if self.goal_delta not in tokenizer.compatible_goal_deltas:
            raise ValueError(
                f"goal_delta={self.goal_delta!r} is not supported by "
                f"{type(tokenizer).__name__}; supported values: "
                f"{sorted(str(d) for d in tokenizer.compatible_goal_deltas)}."
            )
        if not self.goal_conditioned and not tokenizer.supports_single_side:
            raise ValueError(
                f"{type(tokenizer).__name__} has no standalone tokenization of a single "
                "state (supports_single_side=False), so it cannot be used with goal_horizon=0."
            )

        self.embedder = (
            hydra_zen.instantiate(self.embedder_config, input_dim=tokenizer.output_dim)
            if self.embedder_config is not None
            else nn.Identity()
        )
        super().configure_model()

    def _resolve_tokenizer(self) -> StateTokenizer:
        """Lazily constructs the tokenizer, caching it into ``self.tokenizer`` on first call --
        regardless of which caller is first -- so :meth:`_get_cond_dims` (which may run before
        :meth:`configure_model`, e.g. via :meth:`BaseDiffusionAgent.configure_model`'s own
        ``cond_dims=self._get_cond_dims()`` call) and :meth:`configure_model` itself share exactly
        one instance instead of each constructing (and discarding) their own.

        Mirrors how the embedder is instantiated: ``hydra_zen.instantiate(self.tokenizer_config,
        task_dim=self.task_dim)``, or a direct ``FlattenStateTokenizer`` when unconfigured (no
        instantiate call at all, exactly like ``embedder_config=None`` falls back to
        ``nn.Identity()``). Tests that patch ``hydra_zen.instantiate`` globally get a
        ``MagicMock`` here too in that case -- same as they already do for the embedder -- and
        must swap in a real tokenizer the same way they swap in a real embedder.
        """
        if self.tokenizer is not None:
            return self.tokenizer
        if self.tokenizer_config is None:
            self.tokenizer = FlattenStateTokenizer(task_dim=self.task_dim)
            return self.tokenizer

        tokenizer = hydra_zen.instantiate(self.tokenizer_config, task_dim=self.task_dim)
        self.tokenizer = tokenizer
        return tokenizer

    def _get_cond_dims(self) -> DimSpec:
        """Reports the per-timestep conditioning dimensionality passed to the network's
        ``cond_dims``.

        Mirrors :meth:`_build_external_cond`: each branch below has a builder counterpart named
        alike, and the two trees must stay in sync.
        """
        if not self.goal_conditioned:
            return self._obs_cond_dims()
        else:
            return self._goal_conditioned_cond_dims()

    def _goal_conditioned_cond_dims(self) -> dict[str, DimSpec]:
        if self.goal_delta is None:
            return self._absolute_cond_dims()
        else:
            return self._delta_cond_dims()

    def _absolute_cond_dims(self) -> dict[str, DimSpec]:
        return {**self._obs_cond_dims(), **self._goal_cond_dims()}

    def _obs_cond_dims(self) -> dict[str, DimSpec]:
        embed_dim = self._embedder_output_dim()
        if self._embedder_pools_time():
            # A pooling embedder collapses the time axis, so "task" no longer shares "obs"'s
            # per-timestep width and must live outside it (mirrors "goal", which never has one).
            return {"obs": {"proprio": self.proprio_dim}, "task": embed_dim}
        tokens_per_step = self._resolve_tokenizer().tokens_per_step
        return {"obs": {"proprio": self.proprio_dim, "task": embed_dim * tokens_per_step}}

    def _goal_cond_dims(self) -> dict[str, DimSpec]:
        # Only reachable when goal_delta is None (see _goal_conditioned_cond_dims), which every
        # tokenizer with tokens_per_step > 1 disallows via compatible_goal_deltas -- so embed_dim
        # here is never multiplied by a per-timestep token count in practice.
        embed_dim = self._embedder_output_dim()
        if self.exclude_proprio_from_goal:
            return {"goal": embed_dim}
        else:
            return {"goal": {"proprio": self.proprio_dim, "task": embed_dim}}

    def _delta_cond_dims(self) -> dict[str, DimSpec]:
        # The differences have the same width as the obs entries, so the goal adds none of its own.
        return self._obs_cond_dims()

    def _embedder_output_dim(self) -> int:
        """Lookup of the embedder's output dim.

        Reads config only, never an instantiated module, so that
        :meth:`_get_cond_dims` remains safe to call before :meth:`configure_model`.
        """
        if self.embedder_config is None:
            return self._resolve_tokenizer().output_dim

        return self.embedder_config.get("output_dim")

    def _embedder_pools_time(self) -> bool:
        """Whether the embedder collapses the time axis instead of returning one embedding per
        timestep."""
        if self.embedder_config is None:
            return False

        return self.embedder_config.get("pooling") is not None

    @torch.no_grad()
    def extract_embeddings(
        self,
        obs: torch.Tensor | dict,
        goal: torch.Tensor | dict | None = None,
    ):
        """Extracts embedder outputs for observations (and optionally a goal).

        Helper function for visualizing the embeddings. Always reports the absolute embeddings,
        independently of ``goal_delta``, so visualizations stay comparable across both modes.
        """
        if isinstance(obs, Mapping):
            obs = {k: v.to(self.device) for k, v in obs.items()}
        else:
            obs = obs.to(self.device)

        if goal is not None:
            if isinstance(goal, Mapping):
                goal = {k: v.to(self.device) for k, v in goal.items()}
            else:
                goal = goal.to(self.device)

        if self.obs_normalizer is not None:
            obs = self.obs_normalizer.normalize(obs)
            if goal is not None:
                goal = self.obs_normalizer.normalize(goal)

        _, obs_task_embeddings = self._embed_states(obs)
        res = {"obs_embeddings": obs_task_embeddings.cpu()}

        if goal is not None:
            _, goal_task_embedding = self._embed_states(goal, is_goal=True)
            res["goal_embedding"] = goal_task_embedding.cpu()

        return res

    def get_action(
        self,
        obs_seq: torch.Tensor | Mapping[str, Any],
        goal: torch.Tensor | Mapping[str, Any] | None = None,
        num_inference_steps: int | None = None,
        output_clip_range: tuple | None = None,
    ) -> torch.Tensor:
        """Runs the reverse diffusion process to predict an action sequence from the observation
        and goal.

        Shapes:
            obs_seq: [B, obs_horizon * obs_dim] or dict
            goal: [B, obs_dim] or dict
            returns: [B, act_horizon, act_dim] (denoised actions to execute)
        """
        if self.obs_normalizer is not None:
            obs_seq = self.obs_normalizer.normalize(obs_seq)
            if goal is not None:
                goal = self.obs_normalizer.normalize(goal)

        external_cond = self._build_external_cond(obs_seq, goal)

        return self._run_diffusion_loop(
            external_cond=external_cond,
            num_inference_steps=num_inference_steps,
            output_clip_range=output_clip_range,
        )

    def _shared_step(self, batch: dict[str, Any], batch_idx: int, phase: str) -> torch.Tensor:
        """Main step logic for training and validation."""
        obs_seq = batch["obs_seq"]
        action_seq = batch["act_seq"]
        goal = batch.get("goal", None)

        if not isinstance(obs_seq, torch.Tensor | Mapping):
            raise ValueError(
                f"Expected batch['obs_seq'] to be a torch.Tensor or Mapping, but got {type(obs_seq)}."
            )

        if goal is not None and not isinstance(goal, torch.Tensor | Mapping):
            raise ValueError(
                f"Expected batch['goal'] to be a torch.Tensor or Mapping, but got {type(goal)}."
            )

        if self.obs_normalizer is not None:
            obs_seq = self.obs_normalizer.normalize(obs_seq)
            if goal is not None:
                goal = self.obs_normalizer.normalize(goal)

        if self.act_normalizer is not None:
            action_seq = self.act_normalizer.normalize(action_seq)

        external_cond = self._build_external_cond(obs_seq, goal)

        loss = self._compute_loss(external_cond, action_seq)

        self.log(f"{phase}/loss", loss, prog_bar=True, sync_dist=(phase == "val"))
        return loss

    def _build_external_cond(
        self, obs: TensorTree, goal: TensorTree | None
    ) -> dict[str, TensorTree]:
        if not self.goal_conditioned:
            return self._build_obs_external_cond(obs)
        else:
            return self._build_goal_conditioned_external_cond(obs, goal)

    def _build_goal_conditioned_external_cond(
        self, obs: TensorTree, goal: TensorTree | None
    ) -> dict[str, TensorTree]:
        if goal is None:
            raise ValueError(
                f"{type(self).__name__} is configured with goal_horizon={self.goal_horizon} > 0, "
                "but received goal=None."
            )

        if self.goal_delta is None:
            return self._build_absolute_external_cond(obs, goal)
        else:
            return self._build_delta_external_cond(obs, goal)

    def _build_absolute_external_cond(
        self, obs: TensorTree, goal: TensorTree
    ) -> dict[str, TensorTree]:
        """Conditions on the observation embeddings and the goal embedding as separate entries."""
        return merge_dicts(
            [self._build_obs_external_cond(obs), self._build_goal_external_cond(goal)]
        )

    def _build_obs_external_cond(self, obs: TensorTree) -> dict[str, TensorTree]:
        proprio, task_embedded = self._embed_states(obs)
        if self._embedder_pools_time():
            return {"obs": {"proprio": proprio}, "task": task_embedded}
        return {"obs": {"proprio": proprio, "task": task_embedded}}

    def _build_goal_external_cond(self, goal: TensorTree) -> dict[str, TensorTree]:
        proprio, goal_embedded = self._embed_states(goal, is_goal=True)
        if self.exclude_proprio_from_goal:
            return {"goal": goal_embedded}
        else:
            return {"goal": {"proprio": proprio, "task": goal_embedded}}

    def _build_delta_external_cond(
        self, obs: TensorTree, goal: TensorTree
    ) -> dict[str, TensorTree]:
        """Conditions on the difference between the goal and each observation timestep."""
        obs_proprio, obs_task = split_leaf_key(obs, "proprio", self.proprio_dim)
        goal_proprio, goal_task = split_leaf_key(goal, "proprio", self.proprio_dim)
        if obs_proprio is None or goal_proprio is None:
            raise ValueError("Observation/goal mapping must contain a 'proprio' key.")

        # A 2D obs would not raise below: it would broadcast against the unsqueezed goal into
        # [B, B, F], silently mistaking the batch axis for the time axis. Checked generically
        # across dict-of-poses and flat-tensor task trees via the first leaf tensor found.
        if get_ndim(obs_task) != 3:
            raise ValueError(
                "goal_delta expects observations of shape [B, T, F], but got a "
                f"{get_ndim(obs_task)}D tree (shape shown after splitting off proprioception)."
            )

        if goal_proprio.ndim == obs_proprio.ndim - 1:
            goal_proprio = goal_proprio.unsqueeze(1)

        # goal_task's own missing time axis is inserted here, on every leaf, rather than left to
        # each tokenizer: a goal-relative ("input") tokenize() call sees both sides and can insert
        # it itself, but goal_delta="embedding" tokenizes each side separately -- a lone,
        # time-axis-free goal embedding would then broadcast its *batch* axis against obs's *time*
        # axis instead of erroring (the very trap the check above guards against for obs).
        if get_ndim(goal_task) == 2:
            goal_task = map_leaves(lambda t: t.unsqueeze(1), goal_task)

        task_delta = self._tokenize_delta(obs_task, goal_task)
        if self.exclude_proprio_from_goal:
            proprio = obs_proprio
        else:
            proprio = goal_proprio - obs_proprio

        if self._embedder_pools_time():
            return {"obs": {"proprio": proprio}, "task": task_delta}
        return {"obs": {"proprio": proprio, "task": task_delta}}

    def _embed_states(
        self, states: TensorTree, *, is_goal: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Splits proprio/task and embeds the task components."""
        proprio, task = split_leaf_key(states, "proprio", self.proprio_dim)
        if proprio is None:
            raise ValueError("Observation/goal mapping must contain a 'proprio' key.")

        if self.tokenizer is None:
            raise ValueError(
                "Tokenizer not initialized. Call configure_model() before using the embedder."
            )
        if not self.tokenizer.supports_single_side:
            raise NotImplementedError(
                f"{type(self.tokenizer).__name__} tokenizes only goal-relative deltas "
                "(supports_single_side=False); there is no standalone embedding of a single "
                "state, so absolute conditioning and extract_embeddings() aren't supported."
            )

        tokens = (
            self.tokenizer.tokenize(None, task) if is_goal else self.tokenizer.tokenize(task, None)
        )
        return proprio, self._embed_task(tokens)

    def _tokenize_delta(self, obs_task: TensorTree, goal_task: TensorTree) -> torch.Tensor:
        """Embeds the goal-state difference, differencing before or after the embedder."""
        if self.tokenizer is None:
            raise ValueError(
                "Tokenizer not initialized. Call configure_model() before using the embedder."
            )
        if self.goal_delta == "embedding":
            return self._embed_task(self.tokenizer.tokenize(None, goal_task)) - self._embed_task(
                self.tokenizer.tokenize(obs_task, None)
            )
        else:
            return self._embed_task(self.tokenizer.tokenize(obs_task, goal_task))

    def _embed_task(self, task: torch.Tensor) -> torch.Tensor:
        """Runs task components through the embedder."""
        if self.embedder is None:
            raise ValueError(
                "Embedder not initialized. Call configure_model() before using the embedder."
            )
        if self.tokenizer is None:
            raise ValueError(
                "Tokenizer not initialized. Call configure_model() before using the embedder."
            )

        tokens_per_step = self.tokenizer.tokens_per_step
        # With one token per timestep (the default), a time-axis-free input is 2D ([B, F]); with
        # K > 1 tokens per timestep, it's 3D ([B, K, F]) instead -- one rank higher either way.
        expected_ndim_with_time = 3 if tokens_per_step == 1 else 4
        had_no_time_axis = task.ndim == expected_ndim_with_time - 1
        if had_no_time_axis:
            task = task.unsqueeze(1)

        task_embedded = self.embedder(task)

        if tokens_per_step > 1 and not self._embedder_pools_time():
            # Fold the K per-object embeddings for each real timestep back into one wider
            # per-timestep vector, so downstream conditioning code (and ConditionalUnet1D, which
            # only ever flattens per-timestep conditioning) keep seeing "one vector per timestep"
            # without needing any changes of their own.
            b, t, k, d = task_embedded.shape
            task_embedded = task_embedded.reshape(b, t, k * d)

        # A pooling embedder already drops the time axis it was given, so there's nothing left to
        # squeeze; squeeze(1) would then operate on output_dim, which generally is not size 1
        # (thus squeeze would be a no-op); however we avoid it to keep it clean
        if had_no_time_axis and not self._embedder_pools_time():
            task_embedded = task_embedded.squeeze(1)

        return task_embedded
