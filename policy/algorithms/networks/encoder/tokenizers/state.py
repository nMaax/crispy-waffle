from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

import torch

from policy.algorithms.networks.encoder.tokenizers.utils import relative_se3_pose
from policy.utils import concat_leaf_tensors
from policy.utils.typing_utils import TensorTree, get_tensor


class StateTokenizer:
    """Tokenizes the entire state as one flat vector per timestep, assuming no proprioception is
    passed.

    - In relative mode (``relative_goal=True``): computes SE(3) relative pose deltas for 7D pose components.
    - In absolute mode (``relative_goal=False``): passes raw flat states.
    """

    supports_single_side: ClassVar[bool] = True

    tokens_per_step = 1

    def __init__(self, task_dim: int, relative_goal: bool = True):
        self.task_dim = task_dim
        self.relative_goal = relative_goal
        if relative_goal and task_dim % 7 == 0:
            self.output_dim = (task_dim // 7) * 6
        else:
            self.output_dim = task_dim

    def tokenize(self, obs_task: TensorTree | None, goal_task: TensorTree | None) -> torch.Tensor:
        if obs_task is not None and goal_task is not None:
            return self._tokenize_relative(obs_task, goal_task)

        task = obs_task if obs_task is not None else goal_task
        if task is None:
            raise ValueError("tokenize() requires at least one of obs_task/goal_task.")
        return concat_leaf_tensors(task, dim=-1) if isinstance(task, Mapping) else task

    def _tokenize_relative(self, obs_task: TensorTree, goal_task: TensorTree) -> torch.Tensor:
        if isinstance(obs_task, Mapping) and isinstance(goal_task, Mapping):
            deltas = []
            for key in sorted(obs_task.keys()):
                if key in goal_task:
                    o = get_tensor(obs_task, key)
                    g = get_tensor(goal_task, key)
                    if g.ndim == o.ndim - 1:
                        g = g.unsqueeze(1)
                    deltas.append(relative_se3_pose(g, o))
            return torch.cat(deltas, dim=-1)

        obs_flat = (
            concat_leaf_tensors(obs_task, dim=-1) if isinstance(obs_task, Mapping) else obs_task
        )
        goal_flat = (
            concat_leaf_tensors(goal_task, dim=-1) if isinstance(goal_task, Mapping) else goal_task
        )
        if goal_flat.ndim == obs_flat.ndim - 1:
            goal_flat = goal_flat.unsqueeze(1)

        if obs_flat.shape[-1] % 7 == 0 and goal_flat.shape[-1] % 7 == 0:
            num_poses = obs_flat.shape[-1] // 7
            deltas = []
            for i in range(num_poses):
                o_i = obs_flat[..., i * 7 : (i + 1) * 7]
                g_i = goal_flat[..., i * 7 : (i + 1) * 7]
                deltas.append(relative_se3_pose(g_i, o_i))
            return torch.cat(deltas, dim=-1)

        return goal_flat - obs_flat
