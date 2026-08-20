from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from policy.algorithms.tokenizers.base import BaseTokenizer
from policy.algorithms.tokenizers.utils import relative_se3_pose
from policy.transforms.canonicalization.spec import (
    POSE_DIM,
    RELATIVE_SE3_DIM,
    ROLE_DIM,
)
from policy.utils import get_tensor, match_shapes
from policy.utils.typing_utils import DimSpec, TensorTree

TCP_POSE_KEY = "tcp_pose"


class ObjectTokenizer(BaseTokenizer):
    """Tokenizes each object entity in a state as a standalone token.

    - In relative mode (``relative_goal=True``):
        1. SE(3) pose relative to TCP (6D, omitted when ``include_rel_to_tcp=False``)
        2. SE(3) pose delta from current state to goal (6D)
        3. One-hot role indicator [is_pick, is_target, is_clutter] (3D)
    - In absolute mode (``relative_goal=False``):
        1. SE(3) pose relative to TCP (6D, omitted when ``include_rel_to_tcp=False``)
        2. Absolute SE(3) pose (7D)
        3. One-hot role indicator [is_pick, is_target, is_clutter] (3D)
    """

    def __init__(
        self,
        object_keys: Sequence[str] | None = None,
        relative_goal: bool = True,
        include_rel_to_tcp: bool = True,
        task_dim: DimSpec | None = None,  # for API consistency
    ):
        super().__init__(relative_goal=relative_goal)
        self.object_keys = tuple(object_keys) if object_keys is not None else None
        self.include_rel_to_tcp = include_rel_to_tcp
        rel_to_tcp_dim = RELATIVE_SE3_DIM if include_rel_to_tcp else 0
        self.output_dim = (
            rel_to_tcp_dim + RELATIVE_SE3_DIM + ROLE_DIM
            if relative_goal
            else rel_to_tcp_dim + POSE_DIM + ROLE_DIM
        )
        # The role one-hot is the trailing ROLE_DIM block of every token, in both modes.
        self._categorical_mask = torch.cat(
            [
                torch.ones(self.output_dim - ROLE_DIM, dtype=torch.bool),
                torch.zeros(ROLE_DIM, dtype=torch.bool),
            ]
        )

    @property
    def categorical_mask(self) -> torch.Tensor:
        return self._categorical_mask

    @property
    def tokens_per_step(self) -> int | None:
        return len(self.object_keys) if self.object_keys is not None else None

    def _keys(self, task_dict: Mapping[str, TensorTree]) -> list[str]:
        if self.object_keys is not None:
            return list(self.object_keys)
        keys = sorted(
            [k for k in task_dict.keys() if k.startswith("obj_") and k.endswith("_pose")],
            key=lambda k: int(k.split("_")[1]) if k.split("_")[1].isdigit() else k,
        )
        if not keys:
            raise KeyError(
                f"No canonical object pose keys ('obj_i_pose') found in task dict. "
                f"Available keys: {list(task_dict.keys())}. Ensure Canonicalizer transform was applied."
            )
        # The TCP is an entity like any other object, carrying its own goal delta.
        return [TCP_POSE_KEY, *keys]

    def _tokenize_relative(
        self, obs_task: Mapping[str, TensorTree], goal_task: Mapping[str, TensorTree]
    ) -> torch.Tensor:
        keys = self._keys(obs_task)
        tcp_pose = get_tensor(obs_task, TCP_POSE_KEY)

        tokens = []
        for key in keys:
            o_k = get_tensor(obs_task, key)
            g_k = match_shapes(get_tensor(goal_task, key), o_k)
            role = match_shapes(get_tensor(obs_task, key.replace("_pose", "_role")), o_k)

            goal_delta = relative_se3_pose(g_k, o_k)

            if self.include_rel_to_tcp:
                rel_to_tcp = relative_se3_pose(o_k, tcp_pose)
                tokens.append(torch.cat([rel_to_tcp, goal_delta, role], dim=-1))
            else:
                tokens.append(torch.cat([goal_delta, role], dim=-1))

        return torch.stack(tokens, dim=2)

    def _tokenize_absolute(self, task: Mapping[str, TensorTree]) -> torch.Tensor:
        keys = self._keys(task)
        tcp_pose = get_tensor(task, TCP_POSE_KEY)

        tokens = []
        for key in keys:
            o_k = get_tensor(task, key)
            role = match_shapes(get_tensor(task, key.replace("_pose", "_role")), o_k)

            if self.include_rel_to_tcp:
                rel_to_tcp = relative_se3_pose(o_k, tcp_pose)
                tokens.append(torch.cat([rel_to_tcp, o_k, role], dim=-1))
            else:
                tokens.append(torch.cat([o_k, role], dim=-1))

        stack_dim = 2 if tokens[0].ndim >= 3 else 1
        return torch.stack(tokens, dim=stack_dim)
