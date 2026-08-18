from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from policy.algorithms.tokenizers.base import BaseTokenizer
from policy.algorithms.tokenizers.utils import relative_se3_pose
from policy.transforms.canonicalization.spec import (
    CATEGORICAL_KEYS,
    RELATIVE_SE3_DIM,
    dim_shape,
)
from policy.utils import get_tensor, match_shapes
from policy.utils.typing_utils import DimSpec, TensorTree


class StateTokenizer(BaseTokenizer):
    """Tokenizes the state mapping into one flat vector per timestep.

    - In relative mode (``relative_goal=True``): computes SE(3) relative pose deltas w.r.t. goal
      for named 7D pose components, preserving role indicators and differencing other keys.
    - In absolute mode (``relative_goal=False``): passes raw flat states.


    ``obj_valid`` is ignored: this tokenizer has no way to express an absent object, so it suits
    fixed-population tasks only. Use ``GraphTokenizer`` for the clutter environments.
    """

    def __init__(self, task_dim: Mapping[str, DimSpec], relative_goal: bool = True):
        super().__init__(relative_goal=relative_goal)
        self._spec_ndims = self._task_leaf_ndims(task_dim)
        widths = self._channel_widths(task_dim)
        self.output_dim = sum(width for _, width in widths)
        self._categorical_mask = torch.cat(
            [
                torch.full((width,), key not in CATEGORICAL_KEYS, dtype=torch.bool)
                for key, width in widths
            ]
        )

    def _channel_widths(self, task_dim: Mapping[str, DimSpec]) -> list[tuple[str, int]]:
        """Per-key output widths, in the channel order the matching tokenize path emits."""
        keys = sorted(task_dim.keys()) if self.relative_goal else list(task_dim.keys())
        return [(key, self._width(key, task_dim[key])) for key in keys]

    def _width(self, key: str, dim: DimSpec) -> int:
        """Mirrors _tokenize_relative branch for branch, so output_dim cannot disagree with it."""
        shape = dim_shape(dim)
        if not self.relative_goal:
            return math.prod(shape)
        if key.endswith("_pose"):
            # A 7D pose collapses to a 6D SE(3) delta, slot axis intact.
            return math.prod(shape[:-1]) * RELATIVE_SE3_DIM
        if key in CATEGORICAL_KEYS:
            return math.prod(shape)
        raise ValueError(self._no_reduction_message(key))

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
        for key in sorted(obs_task.keys()):
            o = get_tensor(obs_task, key)
            g = match_shapes(get_tensor(goal_task, key), o)
            if key.endswith("_pose"):
                delta = relative_se3_pose(g, o)
            elif key in CATEGORICAL_KEYS:
                delta = o
            else:
                raise ValueError(self._no_reduction_message(key))
            deltas.append(self._flatten(key, delta))
        return torch.cat(deltas, dim=-1)

    def _tokenize_absolute(self, task: Mapping[str, TensorTree]) -> torch.Tensor:
        return torch.cat(
            [self._flatten(key, get_tensor(task, key)) for key in task.keys()], dim=-1
        )

    def _flatten(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        """Folds a key's own trailing axes (e.g. a pool's slot axis) into the feature axis.

        Counted from the right, so it holds whether or not the leading time axis is present: a
        standalone goal arrives as [B, *] where an observation window arrives as [B, T, *].
        """
        spec_ndim = self._spec_ndims[key]
        return tensor.flatten(start_dim=tensor.ndim - spec_ndim) if spec_ndim > 1 else tensor

    @staticmethod
    def _no_reduction_message(key: str) -> str:
        return (
            f"Task key {key!r} is neither a pose nor categorical, so it has no defined "
            "goal-relative reduction."
        )

    @staticmethod
    def _task_leaf_ndims(task_dim: Mapping[str, DimSpec]) -> dict[str, int]:
        return {key: len(dim_shape(dim)) for key, dim in task_dim.items()}
