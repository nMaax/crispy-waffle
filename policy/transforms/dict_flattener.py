from collections.abc import Mapping

import numpy as np
import torch


class DictFlattener:
    """Recursively flattens a nested dictionary of tensors/arrays and concatenates them along the
    last dimension."""

    def __call__(self, obs: dict | torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        if not isinstance(obs, Mapping):
            return obs

        leaves = self._get_leaves(obs)
        if not leaves:
            raise ValueError("Dictionary must contain at least one leaf tensor/array.")

        first = leaves[0]
        if isinstance(first, torch.Tensor):
            return torch.cat(leaves, dim=-1)
        elif isinstance(first, np.ndarray):
            return np.concatenate(leaves, axis=-1)
        else:
            raise TypeError(f"Unsupported leaf type: {type(first)}")

    def _get_leaves(self, data: Mapping) -> list:
        leaves = []
        for value in data.values():
            if isinstance(value, Mapping):
                leaves.extend(self._get_leaves(value))
            else:
                leaves.append(value)
        return leaves
