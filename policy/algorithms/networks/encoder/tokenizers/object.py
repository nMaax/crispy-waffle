from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import ClassVar

import torch

from policy.algorithms.networks.encoder.tokenizers.utils import relative_se3_pose
from policy.utils.typing_utils import TensorTree, get_tensor

_DEFAULT_OBJECT_KEYS = ("a_pose", "b_pose", "tcp_pose")


class ObjectTokenizer:
    """Tokenizes each object pose (e.g. ``a_pose``, ``b_pose``, ``tcp_pose``) separately, one token
    per object per timestep.

    - In relative mode (``relative_goal=True``): computes SE(3) pose deltas (6D: delta-pos + rotvec) for each object.
    - In absolute mode (``relative_goal=False``): passes raw object poses (7D) for each object.
    """

    POSE_DIM = 7
    RELATIVE_SE3_DIM = 6

    supports_single_side: ClassVar[bool] = True

    def __init__(
        self,
        object_keys: Sequence[str] = _DEFAULT_OBJECT_KEYS,
        relative_goal: bool = True,
        task_dim: int | None = None,  # for API consistency
    ):
        self.object_keys = tuple(object_keys)
        self.relative_goal = relative_goal
        self.output_dim = self.RELATIVE_SE3_DIM if relative_goal else self.POSE_DIM
        self.tokens_per_step = len(self.object_keys)

    def tokenize(self, obs_task: TensorTree | None, goal_task: TensorTree | None) -> torch.Tensor:
        if obs_task is not None and goal_task is not None:
            if not isinstance(obs_task, Mapping) or not isinstance(goal_task, Mapping):
                raise TypeError(
                    f"{type(self).__name__} requires dict-shaped task trees keyed by "
                    f"{self.object_keys}, got {type(obs_task).__name__}/{type(goal_task).__name__}."
                )
            missing = [k for k in self.object_keys if k not in obs_task or k not in goal_task]
            if missing:
                raise KeyError(f"obs/goal task tree missing required pose keys: {missing}")

            tokens = []
            for key in self.object_keys:
                o_k = get_tensor(obs_task, key)  # [B, T, 7]
                g_k = get_tensor(goal_task, key)  # [B, 7] or [B, 1, 7]
                if g_k.ndim == o_k.ndim - 1:
                    g_k = g_k.unsqueeze(1)

                r_k = relative_se3_pose(g_k, o_k)  # [B, T, 6]
                tokens.append(r_k)

            return torch.stack(tokens, dim=2)  # [B, T, K, 6]

        task = obs_task if obs_task is not None else goal_task
        if task is None:
            raise ValueError("tokenize() requires at least one of obs_task/goal_task.")
        if not isinstance(task, Mapping):
            raise TypeError(
                f"{type(self).__name__} requires dict-shaped task tree keyed by "
                f"{self.object_keys}, got {type(task).__name__}."
            )
        missing = [k for k in self.object_keys if k not in task]
        if missing:
            raise KeyError(f"task tree missing required pose keys: {missing}")

        tokens = []
        for key in self.object_keys:
            pose = get_tensor(task, key)  # [B, T, 7] or [B, 7]
            tokens.append(pose)

        return torch.stack(tokens, dim=2)  # [B, T, K, 7] or [B, K, 7]
