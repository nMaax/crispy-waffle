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

    Each token's ``"tokens"`` leaf carries geometry only; role is emitted separately as a
    ``"role"`` leaf, for a downstream embedder to inject additively (like a positional embedding)
    rather than concatenated into the token.

    - In relative mode (``relative_goal=True``), ``"tokens"`` is:
        1. SE(3) pose relative to TCP (6D, omitted when ``include_rel_to_tcp=False``)
        2. SE(3) pose delta from current state to goal (6D)
    - In absolute mode (``relative_goal=False``), ``"tokens"`` is:
        1. SE(3) pose relative to TCP (6D, omitted when ``include_rel_to_tcp=False``)
        2. Absolute SE(3) pose (7D)

    ``obj_valid`` is ignored: this tokenizer has no way to express an absent object, so it suits
    fixed-population tasks only. Use ``GraphTokenizer`` for the clutter environments.
    """

    def __init__(
        self,
        task_dim: DimSpec,
        relative_goal: bool = True,
        include_rel_to_tcp: bool = True,
    ):
        super().__init__(relative_goal=relative_goal)
        self._tokens_per_step = self._num_slots(task_dim)
        self.include_rel_to_tcp = include_rel_to_tcp
        rel_to_tcp_dim = RELATIVE_SE3_DIM if include_rel_to_tcp else 0
        self.output_dim = (
            rel_to_tcp_dim + RELATIVE_SE3_DIM if relative_goal else rel_to_tcp_dim + POSE_DIM
        )
        self._normalization_mask = {
            "tokens": torch.ones(self.output_dim, dtype=torch.bool),
            "role": torch.zeros(ROLE_DIM, dtype=torch.bool),
        }

    @property
    def normalization_mask(self) -> dict[str, torch.Tensor]:
        return self._normalization_mask

    @property
    def token_spec(self) -> DimSpec:
        return {"tokens": self.output_dim, "role": ROLE_DIM}

    @property
    def tokens_per_step(self) -> int:
        return self._tokens_per_step

    def _tokenize_relative(
        self, obs_task: Mapping[str, TensorTree], goal_task: Mapping[str, TensorTree]
    ) -> dict[str, torch.Tensor]:
        obs_pose = get_tensor(obs_task, "obj_pose")
        goal_pose = match_shapes(get_tensor(goal_task, "obj_pose"), obs_pose)
        role = match_shapes(get_tensor(obs_task, "obj_role"), obs_pose)

        goal_delta = relative_se3_pose(goal_pose, obs_pose)

        if self.include_rel_to_tcp:
            rel_to_tcp = relative_se3_pose(obs_pose, self._tcp_pose(obs_pose))
            tokens = torch.cat([rel_to_tcp, goal_delta], dim=-1)
        else:
            tokens = goal_delta

        return {"tokens": tokens, "role": role}

    def _tokenize_absolute(self, task: Mapping[str, TensorTree]) -> dict[str, torch.Tensor]:
        pose = get_tensor(task, "obj_pose")
        role = match_shapes(get_tensor(task, "obj_role"), pose)

        if self.include_rel_to_tcp:
            rel_to_tcp = relative_se3_pose(pose, self._tcp_pose(pose))
            tokens = torch.cat([rel_to_tcp, pose], dim=-1)
        else:
            tokens = pose

        return {"tokens": tokens, "role": role}

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
