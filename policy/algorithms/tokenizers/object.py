from __future__ import annotations

from collections.abc import Mapping

import torch

from policy.algorithms.tokenizers.base import BaseTokenizer
from policy.algorithms.tokenizers.utils import relative_se3_pose
from policy.transforms.canonicalization.spec import (
    POSE_DIM,
    RELATIVE_SE3_DIM,
    ROLE_DIM,
    TCP_SLOT,
)
from policy.utils import get_tensor, match_shapes
from policy.utils.typing_utils import DimSpec, TensorTree


class ObjectTokenizer(BaseTokenizer):
    """Tokenizes each object entity in a state as a standalone token.

    - In relative mode (``relative_goal=True``):
        1. SE(3) pose relative to TCP (6D)
        2. SE(3) pose delta from current state to goal (6D)
        3. One-hot role indicator [is_tcp, is_pick, is_target, is_clutter] (4D)
    - In absolute mode (``relative_goal=False``):
        1. SE(3) pose relative to TCP (6D)
        2. Absolute SE(3) pose (7D)
        3. One-hot role indicator [is_tcp, is_pick, is_target, is_clutter] (4D)

    ``obj_valid`` is ignored: this tokenizer has no way to express an absent object, so it suits
    fixed-population tasks only. Use ``GraphTokenizer`` for the clutter environments.
    """

    def __init__(
        self,
        task_dim: DimSpec,
        relative_goal: bool = True,
    ):
        super().__init__(relative_goal=relative_goal)
        self._tokens_per_step = self._num_slots(task_dim)
        self.output_dim = (
            RELATIVE_SE3_DIM + RELATIVE_SE3_DIM + ROLE_DIM
            if relative_goal
            else RELATIVE_SE3_DIM + POSE_DIM + ROLE_DIM
        )
        # The role one-hot is the trailing ROLE_DIM block of every token, in both modes.
        self._normalization_mask = torch.cat(
            [
                torch.ones(self.output_dim - ROLE_DIM, dtype=torch.bool),
                torch.zeros(ROLE_DIM, dtype=torch.bool),
            ]
        )

    @property
    def token_spec(self) -> DimSpec:
        return self.output_dim

    @property
    def normalization_mask(self) -> torch.Tensor:
        return self._normalization_mask

    @property
    def tokens_per_step(self) -> int:
        return self._tokens_per_step

    def _tokenize_relative(
        self, obs_task: Mapping[str, TensorTree], goal_task: Mapping[str, TensorTree]
    ) -> torch.Tensor:
        obs_pose = get_tensor(obs_task, "obj_pose")
        goal_pose = match_shapes(get_tensor(goal_task, "obj_pose"), obs_pose)
        role = match_shapes(get_tensor(obs_task, "obj_role"), obs_pose)

        rel_to_tcp = relative_se3_pose(obs_pose, self._tcp_pose(obs_pose))
        goal_delta = relative_se3_pose(goal_pose, obs_pose)

        return torch.cat([rel_to_tcp, goal_delta, role], dim=-1)

    def _tokenize_absolute(self, task: Mapping[str, TensorTree]) -> torch.Tensor:
        pose = get_tensor(task, "obj_pose")
        role = match_shapes(get_tensor(task, "obj_role"), pose)

        rel_to_tcp = relative_se3_pose(pose, self._tcp_pose(pose))

        return torch.cat([rel_to_tcp, pose, role], dim=-1)

    @staticmethod
    def _tcp_pose(pose: torch.Tensor) -> torch.Tensor:
        """Broadcasts the TCP slot back over the slot axis as every slot's reference frame."""
        return pose[..., TCP_SLOT : TCP_SLOT + 1, :].expand_as(pose)

    @staticmethod
    def _num_slots(task_dim: DimSpec) -> int:
        if not isinstance(task_dim, Mapping):
            raise TypeError(
                f"ObjectTokenizer expects a canonical dict task_dim, got {type(task_dim).__name__}."
            )
        return int(task_dim["obj_pose"][0])  # type: ignore[index]
