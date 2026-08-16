from __future__ import annotations

from collections.abc import Mapping

import torch

from policy.algorithms.networks.encoder.tokenizers.base import BaseTokenizer
from policy.algorithms.networks.encoder.tokenizers.utils import relative_se3_pose
from policy.transforms.canonicalization.spec import RELATIVE_SE3_DIM
from policy.utils import concat_leaf_tensors, get_tensor, get_total_dim, match_shapes
from policy.utils.typing_utils import DimSpec, TensorTree


class StateTokenizer(BaseTokenizer):
    """Tokenizes the state mapping into one flat vector per timestep.

    - In relative mode (``relative_goal=True``): computes SE(3) relative pose deltas w.r.t. goal
      for named 7D pose components, preserving role indicators and differencing other keys.
    - In absolute mode (``relative_goal=False``): passes raw flat states.
    """

    def __init__(self, task_dim: DimSpec, relative_goal: bool = True):
        super().__init__(relative_goal=relative_goal)
        if relative_goal and isinstance(task_dim, Mapping):
            self.output_dim = sum(
                RELATIVE_SE3_DIM if key.endswith("_pose") else get_total_dim(dim)
                for key, dim in task_dim.items()
            )
        else:
            self.output_dim = get_total_dim(task_dim)

    @property
    def tokens_per_step(self) -> int:
        return 1

    def _tokenize_relative(
        self, obs_task: Mapping[str, TensorTree], goal_task: Mapping[str, TensorTree]
    ) -> torch.Tensor:
        deltas = []
        for key in sorted(obs_task.keys()):
            o = get_tensor(obs_task, key)
            g = match_shapes(get_tensor(goal_task, key), o)
            if key.endswith("_pose"):
                deltas.append(relative_se3_pose(g, o))
            elif key.endswith("_role"):
                deltas.append(o)
            else:
                deltas.append(g - o)
        return torch.cat(deltas, dim=-1)

    def _tokenize_absolute(self, task: Mapping[str, TensorTree]) -> torch.Tensor:
        return concat_leaf_tensors(task, dim=-1)
