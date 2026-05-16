from collections.abc import Callable
from typing import Any

import torch

from policy.utils.typing_utils import AdapterProtocol

IndexSelector = Callable[[torch.Tensor | dict[str, Any]], torch.Tensor]


class StackCubeSwappedToStackCubeAdapter(AdapterProtocol):
    """Tricks a policy trained on StackCube-v1 into stacking Cube B on Cube A by swapping their
    identities in the observation space."""

    def __init__(
        self,
    ):
        """
        Args:
            selector: A callable that takes the observation and returns indices to swap.
                      Can also be a static list of indices or a PyTorch tensor.
        """

    def apply(self, obs: torch.Tensor | dict[str, Any]) -> torch.Tensor | dict[str, Any]:

        if isinstance(obs, dict):
            return self._apply_to_dict(obs)
        else:
            return self._apply_to_tensor(obs)

    def _apply_to_tensor(self, obs: torch.Tensor) -> torch.Tensor:

        # StackCube-v1: [0:18 (proprio), 18:25 (TCP pose), 25:32 (Cube A pose), 32:39 (Cube B pose), 39:42 (TCP to A), 42:45 (TCP to B), 45:48 (A to B)]

        proprio = obs[..., 0:25].clone()
        a_pose = obs[..., 25:32].clone()
        b_pose = obs[..., 32:39].clone()
        tcp_to_a = obs[..., 39:42].clone()
        tcp_to_b = obs[..., 42:45].clone()
        a_to_b = obs[..., 45:48].clone()

        swapped = torch.zeros((*obs.shape[:-1], 48), dtype=obs.dtype, device=obs.device)
        swapped[..., 0:25] = proprio
        swapped[..., 25:32] = b_pose
        swapped[..., 32:39] = a_pose
        swapped[..., 39:42] = tcp_to_b
        swapped[..., 42:45] = tcp_to_a
        swapped[..., 45:48] = -a_to_b

        return swapped

    def _apply_to_dict(self, obs: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError(
            "StackCubeSwappedToStackCubeAdapter does not support dict observations yet."
        )
