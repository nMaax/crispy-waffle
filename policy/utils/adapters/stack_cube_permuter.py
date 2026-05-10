from collections.abc import Callable
from typing import Any

import torch

from policy.utils import get_batch_size

IndexSelector = Callable[[torch.Tensor | dict[str, Any]], torch.Tensor]


class CubesPermuter:
    """Tricks a policy trained on StackCube-v1 into stacking Cube B on Cube A by swapping their
    identities in the observation space."""

    def __init__(self, selector: IndexSelector | list[int] | torch.Tensor | None = None):
        """
        Args:
            selector: A callable that takes the observation and returns indices to swap.
                      Can also be a static list of indices or a PyTorch tensor.
        """
        if selector is None or selector == "all":
            self.selector = lambda obs: torch.arange(get_batch_size(obs))
        elif selector == "even":
            self.selector = lambda obs: torch.arange(0, get_batch_size(obs), 2)
        elif selector == "odd":
            self.selector = lambda obs: torch.arange(1, get_batch_size(obs), 2)
        elif isinstance(selector, list | torch.Tensor):
            static_indices = torch.as_tensor(selector, dtype=torch.long)
            self.selector = lambda obs: static_indices
        else:
            self.selector = selector

    def apply(self, obs: torch.Tensor | dict[str, Any]) -> torch.Tensor | dict[str, Any]:
        indices = self.selector(obs)

        if indices.dtype == torch.bool:
            indices = indices.nonzero(as_tuple=True)[0]

        indices = indices.to(torch.long)

        if len(indices) == 0:
            return obs

        if isinstance(obs, dict):
            return self._apply_to_dict(obs, indices)
        else:
            return self._apply_to_tensor(obs, indices)

    def _apply_to_tensor(self, obs: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Simply swaps the relevant parts of the observation tensor to permute the identities."""
        swapped = obs.clone()

        a_pose = swapped[indices, ..., 25:32].clone()
        b_pose = swapped[indices, ..., 32:39].clone()
        tcp_to_a = swapped[indices, ..., 39:42].clone()
        tcp_to_b = swapped[indices, ..., 42:45].clone()
        a_to_b = swapped[indices, ..., 45:48].clone()

        swapped[indices, ..., 25:32] = b_pose
        swapped[indices, ..., 32:39] = a_pose
        swapped[indices, ..., 39:42] = tcp_to_b
        swapped[indices, ..., 42:45] = tcp_to_a
        swapped[indices, ..., 45:48] = -a_to_b

        return swapped

    def _apply_to_dict(self, obs_dict: dict[str, Any], indices: torch.Tensor) -> dict[str, Any]:
        swapped = obs_dict.copy()

        a_pose = swapped["cubeA_pose"][indices].clone()
        b_pose = swapped["cubeB_pose"][indices].clone()
        tcp_to_a = swapped["tcp_to_cubeA_pos"][indices].clone()
        tcp_to_b = swapped["tcp_to_cubeB_pos"][indices].clone()
        a_to_b = swapped["cubeA_to_cubeB_pos"][indices].clone()

        swapped["cubeA_pose"][indices] = b_pose
        swapped["cubeB_pose"][indices] = a_pose
        swapped["tcp_to_cubeA_pos"][indices] = tcp_to_b
        swapped["tcp_to_cubeB_pos"][indices] = tcp_to_a
        swapped["cubeA_to_cubeB_pos"][indices] = -a_to_b

        return swapped
