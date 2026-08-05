from collections.abc import Mapping
from typing import Any, Literal

import hydra_zen
import torch
import torch.nn as nn

from policy.algorithms.diffusion_policy import DiffusionPolicy
from policy.utils import (
    derive_task_dim,
    merge_dicts,
    resolve_proprio_dim,
    split_proprio_task,
)
from policy.utils.typing_utils import (
    DimSpec,
    GoalConditionedPolicyProtocol,
    HydraConfigFor,
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

    Proprioception never goes through the embedder, which keeps embedders robot-agnostic; it is
    concatenated raw alongside the embedder outputs. ``exclude_proprio_from_goal=False`` adds the
    goal's proprioception to those outputs, next to the historical proprioception when concatenated
    with the embeddings.
    """

    def __init__(
        self,
        *args,
        goal_horizon: int = 1,
        proprio_dim: int | None = None,
        task_dim: int | None = None,
        embedder: HydraConfigFor[nn.Module] | None = None,
        exclude_proprio_from_goal: bool = True,
        goal_delta: Literal["input", "embedding"] | None = None,
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

    def _validate_obs_dim(self, proprio_dim: int | None, task_dim: int | None) -> tuple[int, int]:
        proprio_dim = resolve_proprio_dim(self.obs_dim, proprio_dim)
        task_dim = derive_task_dim(self.obs_dim, proprio_dim, task_dim)
        return proprio_dim, task_dim

    def configure_model(self) -> None:
        if self.network is not None:
            return
        self.embedder = (
            hydra_zen.instantiate(self.embedder_config, input_dim=self.task_dim)
            if self.embedder_config is not None
            else nn.Identity()
        )
        super().configure_model()

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
        return {"obs": {"proprio": self.proprio_dim, "task": embed_dim}}

    def _goal_cond_dims(self) -> dict[str, DimSpec]:
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
            return self.task_dim

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
            _, goal_task_embedding = self._embed_states(goal)
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
        proprio, goal_embedded = self._embed_states(goal)
        if self.exclude_proprio_from_goal:
            return {"goal": goal_embedded}
        else:
            return {"goal": {"proprio": proprio, "task": goal_embedded}}

    def _build_delta_external_cond(
        self, obs: TensorTree, goal: TensorTree
    ) -> dict[str, TensorTree]:
        """Conditions on the difference between the goal and each observation timestep."""
        obs_proprio, obs_task = split_proprio_task(obs, self.proprio_dim)
        goal_proprio, goal_task = split_proprio_task(goal, self.proprio_dim)

        # A 2D obs would not raise below: it would broadcast against the unsqueezed goal into
        # [B, B, F], silently mistaking the batch axis for the time axis.
        if obs_task.ndim != 3:
            raise ValueError(
                "goal_delta expects observations of shape [B, T, F], but got "
                f"{tuple(obs_task.shape)} (shape shown after splitting off proprioception)."
            )

        goal_had_no_time_axis = goal_task.ndim == 2
        if goal_had_no_time_axis:
            goal_task = goal_task.unsqueeze(1)
            goal_proprio = goal_proprio.unsqueeze(1)

        task_delta = self._embed_task_delta(goal_task, obs_task)
        if self.exclude_proprio_from_goal:
            proprio = obs_proprio
        else:
            proprio = goal_proprio - obs_proprio

        if self._embedder_pools_time():
            return {"obs": {"proprio": proprio}, "task": task_delta}
        return {"obs": {"proprio": proprio, "task": task_delta}}

    def _embed_states(self, states: TensorTree) -> tuple[torch.Tensor, torch.Tensor]:
        """Splits proprio/task and embeds the task components."""
        proprio, task = split_proprio_task(states, self.proprio_dim)
        return proprio, self._embed_task(task)

    def _embed_task_delta(self, goal_task: torch.Tensor, obs_task: torch.Tensor) -> torch.Tensor:
        """Embeds the goal-state difference, differencing before or after the embedder."""
        if self.goal_delta == "input":
            return self._embed_task(goal_task - obs_task)
        else:
            return self._embed_task(goal_task) - self._embed_task(obs_task)

    def _embed_task(self, task: torch.Tensor) -> torch.Tensor:
        """Runs task components through the embedder."""
        if self.embedder is None:
            raise ValueError(
                "Embedder not initialized. Call configure_model() before using the embedder."
            )

        had_no_time_axis = task.ndim == 2
        if had_no_time_axis:
            task = task.unsqueeze(1)

        task_embedded = self.embedder(task)

        # A pooling embedder already drops the time axis it was given, so there's nothing left to
        # squeeze; squeeze(1) would then operate on output_dim, which generally is not size 1
        # (thus squeeze would be a no-op); however we avoid it to keep it clean
        if had_no_time_axis and not self._embedder_pools_time():
            task_embedded = task_embedded.squeeze(1)

        return task_embedded
