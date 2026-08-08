from collections.abc import Mapping
from typing import Any

import torch

from policy.algorithms.beso_policy import BesoPolicy
from policy.utils import (
    concat_leaf_tensors,
    derive_task_dim,
    resolve_proprio_dim,
    resolve_task_width,
    split_leaf_key,
)
from policy.utils.typing_utils import DimSpec, TensorTree


class BesoPlusPlusPolicy(BesoPolicy):
    """BESO++DeltaInput: extends `BesoPolicy` with two repo-local additions, always both active,
    that have no counterpart in the upstream reference implementation.

    (https://github.com/intuitive-robots/beso): proprioception gets its own per-timestep network
    token, and the policy conditions on the per-timestep goal-state delta (``g - s_t``) instead of
    a standalone goal token. Mutually exclusive with classifier-free guidance
    (``goal_drop_prob``/``cfg_lambda`` must stay at their off defaults): CFG's "unconditional"
    goal-zeroing has no well-defined meaning once there's no standalone goal tensor to zero.

    See `BesoPolicy` for the paper-faithful core this extends.
    """

    def __init__(
        self,
        *args,
        proprio_dim: int | None = None,
        task_dim: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        goal_horizon = self.goal_horizon
        if goal_horizon != 1:
            raise ValueError(
                f"{type(self).__name__} requires goal_horizon=1 (got {goal_horizon=}): a single "
                "goal frame is differenced against every obs timestep; multi-frame goals would "
                "need an explicit broadcasting rule that isn't defined here."
            )

        goal_drop_prob = self.goal_drop_prob
        cfg_lambda = self.cfg_lambda
        if goal_drop_prob > 0.0 or cfg_lambda is not None:
            raise ValueError(
                f"{type(self).__name__} is mutually exclusive with classifier-free guidance "
                f"({goal_drop_prob=}, {cfg_lambda=}): CFG's 'unconditional' goal-zeroing has no "
                "well-defined meaning once there's no standalone goal tensor to zero. Leave "
                "goal_drop_prob=0.0 and cfg_lambda=None."
            )

        self.proprio_dim, self.task_dim = self._validate_obs_dim(proprio_dim, task_dim)

    def _validate_obs_dim(self, proprio_dim: int | None, task_dim: int | None) -> tuple[int, int]:
        proprio_dim = resolve_proprio_dim(self.obs_dim, proprio_dim)
        task_dim = derive_task_dim(self.obs_dim, proprio_dim, task_dim)
        return proprio_dim, task_dim

    def _network_extra_kwargs(self) -> dict[str, Any]:
        # g - s_t is folded into the "obs"/"task" conditioning entry (see
        # `_build_delta_external_cond`), so there's no separate goal-token block to allocate.
        return {"proprio_dim": self.proprio_dim, "use_proprio_token": True, "goal_horizon": 0}

    def _get_cond_dims(self) -> DimSpec:
        """Reports the per-timestep conditioning dimensionality passed to the network.

        The goal is folded into the "obs"/"task" entry instead (see `_build_delta_external_cond`),
        so no "goal" key is ever reported.
        """
        cond_dims = super()._get_cond_dims()
        return {key: value for key, value in cond_dims.items() if key != "goal"}

    def _build_external_cond(
        self, obs: TensorTree, goal: TensorTree | None
    ) -> dict[str, TensorTree]:
        if goal is None:
            raise ValueError(
                f"{type(self).__name__} is configured with goal_horizon={self.goal_horizon}, but "
                "received goal=None."
            )
        return self._build_delta_external_cond(obs, goal)

    def _build_delta_external_cond(
        self, obs: TensorTree, goal: TensorTree
    ) -> dict[str, TensorTree]:
        """Folds the goal into the obs conditioning as a per-timestep delta ``g - s_t``, computed
        over the task-only portion of the state -- proprio has no goal counterpart, so it stays
        absolute and unchanged.

        Produces no standalone ``"goal"`` entry: `_network_extra_kwargs` overrides the network's
        `goal_horizon` to 0 so `DiffusionGPT` doesn't expect one either.
        """
        obs_proprio, obs_task = split_leaf_key(obs, "proprio", self.proprio_dim)
        if obs_proprio is None:
            raise ValueError(
                f"{type(self).__name__} requires external_cond['obs'] to carry a 'proprio' key."
            )

        obs_task_flat = (
            concat_leaf_tensors(obs_task, dim=-1) if isinstance(obs_task, Mapping) else obs_task
        )
        goal_task_flat = self._extract_goal_task(goal)

        if goal_task_flat.ndim == obs_task_flat.ndim - 1:
            goal_task_flat = goal_task_flat.unsqueeze(1)

        task_delta = goal_task_flat - obs_task_flat

        return {"obs": {"proprio": obs_proprio, "task": task_delta}}

    def _extract_goal_task(self, goal: TensorTree) -> torch.Tensor:
        """Extracts the task-only portion of `goal`: a Mapping has its 'proprio' key (if any)
        discarded; a flat tensor is resolved via `resolve_task_width` (already task-width, or full
        obs-width, in which case the leading `proprio_dim` features are sliced off)."""
        if isinstance(goal, Mapping):
            _, goal_task = split_leaf_key(goal, "proprio", self.proprio_dim)
            return (
                concat_leaf_tensors(goal_task, dim=-1)
                if isinstance(goal_task, Mapping)
                else goal_task
            )

        return resolve_task_width(goal, self.proprio_dim, self.task_dim, label="goal width")
