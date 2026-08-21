from __future__ import annotations

from collections.abc import Mapping

import torch

from policy.algorithms.tokenizers.base import BaseTokenizer
from policy.algorithms.tokenizers.utils import relative_se3_pose
from policy.transforms.canonicalization.spec import (
    RELATIVE_SE3_DIM,
    ROLE_DIM,
)
from policy.utils import get_tensor
from policy.utils.typing_utils import DimSpec, TensorTree


class GraphTokenizer(BaseTokenizer):
    """Turns the observed window and the goal into one scene graph per sample."""

    supports_single_side = False

    def __init__(self, task_dim: DimSpec, relative_goal: bool = True):
        super().__init__(relative_goal=relative_goal)

        # For base-tokenizer contract
        self._tokens_per_step = self._num_slots(task_dim)
        self.output_dim = RELATIVE_SE3_DIM

        # This is only for graphs instead
        self.edge_dim = RELATIVE_SE3_DIM

    @property
    def tokens_per_step(self) -> int:
        return self._tokens_per_step

    @property
    def token_spec(self) -> DimSpec:
        return {
            "nodes": self.output_dim,
            "role": ROLE_DIM,
            "valid": self._tokens_per_step,
            "edge_feat": self.edge_dim,
        }

    @property
    def normalization_mask(self) -> dict[str, torch.Tensor]:
        return {
            "nodes": torch.ones(self.output_dim, dtype=torch.bool),
            "role": torch.zeros(ROLE_DIM, dtype=torch.bool),
            "valid": torch.zeros(self._tokens_per_step, dtype=torch.bool),
            "edge_feat": torch.ones(self.edge_dim, dtype=torch.bool),
        }

    def _tokenize_relative(
        self, obs_task: Mapping[str, TensorTree], goal_task: Mapping[str, TensorTree]
    ) -> dict[str, torch.Tensor]:
        pose = self._concat_time(obs_task, goal_task, "obj_pose")
        role = self._concat_time(obs_task, goal_task, "obj_role")
        valid = self._concat_time(obs_task, goal_task, "obj_valid")

        # The goal occupies the last concatenated timestep (see `_concat_time`): every slot's
        # goal-timestep pose is that same slot's own target, so broadcast it back over time to
        # get each node's delta to its own goal.
        goal_pose = pose[:, -1:, :, :].expand_as(pose)
        nodes = relative_se3_pose(goal_pose, pose)

        return {
            "nodes": nodes,
            "role": role,
            "valid": valid,
            "edge_feat": self._edge_features(pose),
        }

    def _tokenize_absolute(self, task: Mapping[str, TensorTree]) -> dict[str, torch.Tensor]:
        raise NotImplementedError(
            "GraphTokenizer builds one graph spanning observations and goal together, so it "
            "cannot tokenize a single side (supports_single_side=False)."
        )

    @staticmethod
    def _edge_features(pose: torch.Tensor) -> torch.Tensor:
        """Pairwise SE(3) delta between every node pair, as ``[B, S, S, 6]`` with S = T_all * K.

        Entry ``[i, j]`` is the delta from node ``i`` (query) to node ``j`` (key).
        """
        flat = pose.flatten(1, 2)  # [B, S, 7]
        seq = flat.shape[1]
        query = flat.unsqueeze(2).expand(-1, -1, seq, -1)  # [B, S, S, 7]
        key = flat.unsqueeze(1).expand(-1, seq, -1, -1)  # [B, S, S, 7]
        return relative_se3_pose(key, query)

    @staticmethod
    def _concat_time(
        obs_task: Mapping[str, TensorTree], goal_task: Mapping[str, TensorTree], key: str
    ) -> torch.Tensor:
        """Appends the goal frames after the observed ones along the time axis."""
        return torch.cat([get_tensor(obs_task, key), get_tensor(goal_task, key)], dim=1)

    @staticmethod
    def _num_slots(task_dim: DimSpec) -> int:
        if not isinstance(task_dim, Mapping):
            raise TypeError(
                f"GraphTokenizer expects a canonical dict task_dim, got {type(task_dim).__name__}."
            )
        return int(task_dim["obj_pose"][0])  # type: ignore[index]
