from collections.abc import Mapping
from typing import Any, overload

import numpy as np
import torch

from policy.utils.typing_utils import RawTree, TensorTree


class DictFlattener:
    """Recursively flattens a nested dictionary of tensors/arrays and concatenates them along the
    last dimension."""

    @overload
    def __call__(self, obs: torch.Tensor) -> torch.Tensor: ...

    @overload
    def __call__(self, obs: np.ndarray) -> np.ndarray: ...

    @overload
    def __call__(self, obs: Mapping[str, TensorTree]) -> torch.Tensor: ...

    @overload
    def __call__(self, obs: Mapping[str, RawTree]) -> torch.Tensor | np.ndarray: ...

    def __call__(
        self, obs: Mapping[str, Any] | torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        if not isinstance(obs, Mapping):
            return obs

        leaves = self._get_leaves(obs)
        if not leaves:
            raise ValueError("Dictionary must contain at least one leaf tensor/array.")

        first = leaves[0]
        if isinstance(first, torch.Tensor):
            tensor_leaves = [x for x in leaves if isinstance(x, torch.Tensor)]
            return torch.cat(tensor_leaves, dim=-1)
        elif isinstance(first, np.ndarray):
            array_leaves = [x for x in leaves if isinstance(x, np.ndarray)]
            return np.concatenate(array_leaves, axis=-1)
        else:
            raise TypeError(f"Unsupported leaf type: {type(first)}")

    def _get_leaves(
        self, data: Mapping[str, Any]
    ) -> list[torch.Tensor | np.ndarray]:
        leaves: list[torch.Tensor | np.ndarray] = []
        for value in data.values():
            if isinstance(value, Mapping):
                leaves.extend(self._get_leaves(value))
            elif isinstance(value, torch.Tensor | np.ndarray):
                leaves.append(value)
            else:
                raise TypeError(f"Unsupported value type in dictionary: {type(value)}")
        return leaves
