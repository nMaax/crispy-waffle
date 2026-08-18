from __future__ import annotations

from collections.abc import Mapping

import torch

from policy.algorithms.tokenizers.base import BaseTokenizer
from policy.algorithms.tokenizers.utils import relative_se3_pose
from policy.transforms.canonicalization.spec import (
    RELATIVE_SE3_DIM,
    ROLE_DIM,
    TCP_SLOT,
)
from policy.utils import get_tensor
from policy.utils.typing_utils import DimSpec, TensorTree


class GraphTokenizer(BaseTokenizer):
    """Turns the observed window and the goal into one scene graph per sample.

    Nodes are pool slots (the TCP at slot 0, then every scene object) laid out ``[B, T_all, K, D]``
    with ``T_all = obs_horizon + goal_horizon``: the goal frames are simply the trailing
    timesteps, so a goal node is an ordinary node the topology can point at.

    Node features are ``[pose relative to the TCP (6), role one-hot (4)]``; all *pairwise*
    geometry lives on the edges instead, as the SE(3) delta between the two endpoints. Edges are
    emitted for every pair -- which pairs actually connect is topology the embedder applies as an
    attention mask, not something this parameterless tokenizer decides.

    ``valid`` carries the per-slot activity flag through, so an absent clutter object can be
    excluded from attention rather than attended to at a parked, off-table pose.
    """

    supports_single_side = False

    def __init__(self, task_dim: DimSpec, relative_goal: bool = True):
        super().__init__(relative_goal=relative_goal)
        self._tokens_per_step = self._num_slots(task_dim)
        self.output_dim = RELATIVE_SE3_DIM + ROLE_DIM
        self.edge_dim = RELATIVE_SE3_DIM

    @property
    def tokens_per_step(self) -> int:
        return self._tokens_per_step

    @property
    def token_spec(self) -> DimSpec:
        return {
            "nodes": self.output_dim,
            "valid": self._tokens_per_step,
            "edge_feat": self.edge_dim,
        }

    @property
    def normalization_mask(self) -> dict[str, torch.Tensor]:
        return {
            "nodes": torch.cat(
                [
                    torch.ones(self.output_dim - ROLE_DIM, dtype=torch.bool),
                    torch.zeros(ROLE_DIM, dtype=torch.bool),
                ]
            ),
            "valid": torch.zeros(self._tokens_per_step, dtype=torch.bool),
            "edge_feat": torch.ones(self.edge_dim, dtype=torch.bool),
        }

    def _tokenize_relative(
        self, obs_task: Mapping[str, TensorTree], goal_task: Mapping[str, TensorTree]
    ) -> dict[str, torch.Tensor]:
        pose = self._concat_time(obs_task, goal_task, "obj_pose")
        role = self._concat_time(obs_task, goal_task, "obj_role")
        valid = self._concat_time(obs_task, goal_task, "obj_valid")

        tcp_pose = pose[..., TCP_SLOT : TCP_SLOT + 1, :].expand_as(pose)
        nodes = torch.cat([relative_se3_pose(pose, tcp_pose), role], dim=-1)

        return {"nodes": nodes, "valid": valid, "edge_feat": self._edge_features(pose)}

    def _tokenize_absolute(self, task: Mapping[str, TensorTree]) -> dict[str, torch.Tensor]:
        raise NotImplementedError(
            "GraphTokenizer builds one graph spanning observations and goal together, so it "
            "cannot tokenize a single side (supports_single_side=False)."
        )

    @staticmethod
    def _edge_features(pose: torch.Tensor) -> torch.Tensor:
        """Pairwise SE(3) delta between every node pair, as ``[B, S, S, 6]`` with S = T_all * K.

        Entry ``[i, j]`` is the delta from node ``i`` (the query) to node ``j`` (the key), in the
        same t-major/k-minor flattening the embedder attends over.
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
