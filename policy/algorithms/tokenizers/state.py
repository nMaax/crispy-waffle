from __future__ import annotations

from collections.abc import Mapping

import torch

from policy.algorithms.tokenizers.base import BaseTokenizer
from policy.algorithms.tokenizers.utils import relative_se3_pose
from policy.transforms.canonicalization.spec import RELATIVE_SE3_DIM
from policy.utils import concat_leaf_tensors, get_tensor, get_total_dim, match_shapes
from policy.utils.typing_utils import DimSpec, TensorTree

# The TCP frame is ablated out of the tokens, which carry only goal and role information.
TCP_KEY = "tcp_pose"


class StateTokenizer(BaseTokenizer):
    """Tokenizes the state mapping into one flat vector per timestep.

    - In relative mode (``relative_goal=True``): computes SE(3) relative pose deltas w.r.t. goal
      for named 7D pose components, preserving role indicators and differencing other keys.
    - In absolute mode (``relative_goal=False``): passes raw flat states.
    """

    def __init__(self, task_dim: DimSpec, relative_goal: bool = True):
        super().__init__(relative_goal=relative_goal)
        widths = self._channel_widths(task_dim)
        self.output_dim = sum(width for _, width in widths)
        self._categorical_mask = torch.cat(
            [
                torch.full((width,), not key.endswith("_role"), dtype=torch.bool)
                for key, width in widths
            ]
        )

    def _channel_widths(self, task_dim: DimSpec) -> list[tuple[str, int]]:
        """Per-key output widths, in the channel order the matching tokenize path emits."""
        if not isinstance(task_dim, Mapping):
            return [("task", get_total_dim(task_dim))]

        if self.relative_goal:
            # Mirrors _tokenize_relative: poses reduced to an SE(3) delta.
            return [
                (key, RELATIVE_SE3_DIM if key.endswith("_pose") else get_total_dim(task_dim[key]))
                for key in self._keys(task_dim)
            ]

        # Mirrors _tokenize_absolute's concat_leaf_tensors: full widths.
        return [(key, get_total_dim(task_dim[key])) for key in self._keys(task_dim)]

    @property
    def categorical_mask(self) -> torch.Tensor:
        return self._categorical_mask

    @property
    def tokens_per_step(self) -> int:
        return 1

    def _tokenize_relative(
        self, obs_task: Mapping[str, TensorTree], goal_task: Mapping[str, TensorTree]
    ) -> torch.Tensor:
        deltas = []
        for key in self._keys(obs_task):
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
        return concat_leaf_tensors({key: task[key] for key in self._keys(task)}, dim=-1)

    def _keys(self, task: Mapping[str, object]) -> list[str]:
        """Token keys in the order the matching tokenize path emits them, minus the TCP frame."""
        keys = [key for key in task if key != TCP_KEY]
        return sorted(keys) if self.relative_goal else keys
