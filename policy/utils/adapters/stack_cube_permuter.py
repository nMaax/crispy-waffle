from typing import Any

import torch


class StackCubeObservationPermuter:
    """Tricks a policy trained on StackCube-v1 into stacking Cube B on Cube A by swapping their
    identities in the observation space.

    Supports both dictionaries of batched tensors and batched tensors themselves of arbitrary
    dimensionality grater than 2.
    """

    def __init__(self, swap_env_indices: list[int] | torch.Tensor):
        self.swap_indices = torch.as_tensor(swap_env_indices, dtype=torch.long)

    def apply(self, obs: torch.Tensor | dict[str, Any]) -> torch.Tensor | dict[str, Any]:
        if len(self.swap_indices) == 0:
            return obs

        if isinstance(obs, dict):
            return self._apply_to_dict(obs)
        else:
            return self._apply_to_tensor(obs)

    def _apply_to_tensor(self, obs: torch.Tensor) -> torch.Tensor:
        swapped = obs.clone()

        a_pose = swapped[self.swap_indices, ..., 25:32].clone()
        b_pose = swapped[self.swap_indices, ..., 32:39].clone()
        tcp_to_a = swapped[self.swap_indices, ..., 39:42].clone()
        tcp_to_b = swapped[self.swap_indices, ..., 42:45].clone()
        a_to_b = swapped[self.swap_indices, ..., 45:48].clone()

        swapped[self.swap_indices, ..., 25:32] = b_pose
        swapped[self.swap_indices, ..., 32:39] = a_pose

        swapped[self.swap_indices, ..., 39:42] = tcp_to_b
        swapped[self.swap_indices, ..., 42:45] = tcp_to_a

        swapped[self.swap_indices, ..., 45:48] = -a_to_b

        return swapped

    def _apply_to_dict(self, obs_dict: dict[str, Any]) -> dict[str, Any]:
        swapped = obs_dict.copy()

        a_pose = swapped["cubeA_pose"][self.swap_indices].clone()
        b_pose = swapped["cubeB_pose"][self.swap_indices].clone()
        tcp_to_a = swapped["tcp_to_cubeA_pos"][self.swap_indices].clone()
        tcp_to_b = swapped["tcp_to_cubeB_pos"][self.swap_indices].clone()
        a_to_b = swapped["cubeA_to_cubeB_pos"][self.swap_indices].clone()

        swapped["cubeA_pose"][self.swap_indices] = b_pose
        swapped["cubeB_pose"][self.swap_indices] = a_pose
        swapped["tcp_to_cubeA_pos"][self.swap_indices] = tcp_to_b
        swapped["tcp_to_cubeB_pos"][self.swap_indices] = tcp_to_a
        swapped["cubeA_to_cubeB_pos"][self.swap_indices] = -a_to_b

        return swapped
