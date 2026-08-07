"""State tokenizers: turn a canonicalized, proprio-already-split-off obs/goal task tree into raw
(pre-embedder) tokens.

See :class:`policy.utils.typing_utils.StateTokenizer` for the interface both classes below
implement.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import ClassVar

import torch
from mani_skill.utils.geometry.rotation_conversions import (
    quaternion_invert,
    quaternion_multiply,
    quaternion_to_axis_angle,
)

from policy.utils import concat_leaf_tensors
from policy.utils.typing_utils import GoalDelta, TensorTree, get_tensor

_DEFAULT_OBJECT_KEYS = ("a_pose", "b_pose", "tcp_pose")
_DEFAULT_TCP_KEY = "tcp_pose"


def _relative_pose(g: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    """Delta from pose ``o`` to pose ``g``.

    Returns ``[..., 6]`` (delta-position(3), rotation-vector(3)).
    """
    delta_pos = g[..., :3] - o[..., :3]
    q_rel = quaternion_multiply(g[..., 3:7], quaternion_invert(o[..., 3:7]))
    rotvec = quaternion_to_axis_angle(q_rel)
    return torch.cat([delta_pos, rotvec], dim=-1)


class FlattenStateTokenizer:
    """Tokenizes the entire state as one flat vector per timestep, assuming no proprioception is
    passed."""

    compatible_goal_deltas: ClassVar[frozenset[GoalDelta]] = frozenset(
        {None, "input", "embedding"}
    )
    supports_single_side: ClassVar[bool] = True

    tokens_per_step = 1

    def __init__(self, task_dim: int):
        self.output_dim = task_dim

    def tokenize(self, obs_task: TensorTree | None, goal_task: TensorTree | None) -> torch.Tensor:
        if obs_task is not None and goal_task is not None:
            obs_flat = (
                concat_leaf_tensors(obs_task, dim=-1)
                if isinstance(obs_task, Mapping)
                else obs_task
            )
            goal_flat = (
                concat_leaf_tensors(goal_task, dim=-1)
                if isinstance(goal_task, Mapping)
                else goal_task
            )
            if goal_flat.ndim == obs_flat.ndim - 1:
                goal_flat = goal_flat.unsqueeze(1)
            return goal_flat - obs_flat

        task = obs_task if obs_task is not None else goal_task
        if task is None:
            raise ValueError("tokenize() requires at least one of obs_task/goal_task.")
        return concat_leaf_tensors(task, dim=-1) if isinstance(task, Mapping) else task


class PerObjectStateTokenizer:
    """Tokenizes each object pose (e.g. ``objA``, ``objB``, ``TCP``) separately, one token per
    object per timestep (o_k,t).

    Per key ``k`` in ``object_keys`` and timestep ``t``::

        r_k,t = relative_pose(g_k, o_k,t)       # always present
        c_k,t = relative_pose(o_TCP,t, o_k,t)   # if include_tcp_relative
        n_k,t = ||r_k,t[:3]||                   # if include_position_norm
        token_k,t = concat(r_k,t , c_k,t , n_k,t)

    Only supports ``goal_delta="input"``.
    """

    R_DIM = 6
    C_DIM = 6
    NORM_DIM = 1

    compatible_goal_deltas: ClassVar[frozenset[GoalDelta]] = frozenset({"input"})
    supports_single_side: ClassVar[bool] = False

    def __init__(
        self,
        object_keys: Sequence[str] = _DEFAULT_OBJECT_KEYS,
        tcp_key: str = _DEFAULT_TCP_KEY,
        include_tcp_relative: bool = False,
        include_position_norm: bool = False,
        task_dim: int | None = None,  # for API consistency
    ):
        if tcp_key not in object_keys:
            raise ValueError(f"tcp_key {tcp_key!r} must be one of object_keys {object_keys!r}.")

        self.object_keys = tuple(object_keys)
        self.tcp_key = tcp_key
        self.include_tcp_relative = include_tcp_relative
        self.include_position_norm = include_position_norm

        self.output_dim = (
            self.R_DIM
            + (self.C_DIM if include_tcp_relative else 0)
            + (self.NORM_DIM if include_position_norm else 0)
        )
        self.tokens_per_step = len(self.object_keys)

    def tokenize(self, obs_task: TensorTree | None, goal_task: TensorTree | None) -> torch.Tensor:
        if obs_task is None or goal_task is None:
            raise ValueError(
                f"{type(self).__name__} tokenizes only goal-relative deltas; both obs_task and "
                f"goal_task must be provided ({self.supports_single_side=})."
            )
        if not isinstance(obs_task, Mapping) or not isinstance(goal_task, Mapping):
            raise TypeError(
                f"{type(self).__name__} requires dict-shaped task trees keyed by "
                f"{self.object_keys}, got {type(obs_task).__name__}/{type(goal_task).__name__}."
            )
        missing = [k for k in self.object_keys if k not in obs_task or k not in goal_task]
        if missing:
            raise KeyError(f"obs/goal task tree missing required pose keys: {missing}")

        tcp_obs = get_tensor(obs_task, self.tcp_key)  # [B, T, 7]
        tokens = []
        for key in self.object_keys:
            o_k = get_tensor(obs_task, key)  # [B, T, 7]
            g_k = get_tensor(goal_task, key)  # [B, 7] or [B, 1, 7]
            if g_k.ndim == o_k.ndim - 1:
                g_k = g_k.unsqueeze(1)

            r_k = _relative_pose(g_k, o_k)  # [B,T,6]
            token = r_k
            if self.include_tcp_relative:
                c_k = _relative_pose(tcp_obs, o_k)  # == 0 when key == self.tcp_key
                token = torch.cat([token, c_k], dim=-1)
            if self.include_position_norm:
                token = torch.cat([token, r_k[..., :3].norm(dim=-1, keepdim=True)], dim=-1)
            tokens.append(token)

        return torch.stack(tokens, dim=2)  # [B, T, K, output_dim]
