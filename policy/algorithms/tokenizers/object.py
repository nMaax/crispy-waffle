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
        1. SE(3) pose delta from current state to goal (6D)
        2. One-hot role indicator [is_tcp, is_pick, is_target, is_clutter] (4D), if
           ``include_role``
    - In absolute mode (``relative_goal=False``):
        1. Absolute SE(3) pose (7D)
        2. One-hot role indicator [is_tcp, is_pick, is_target, is_clutter] (4D), if
           ``include_role``

    The role one-hot is constant per slot in every fixed-role env, so it carries information only
    where the pick/target assignment varies per episode (``*RandomPick-v1``).
    """

    def __init__(
        self,
        object_keys: Sequence[str] | None = None,
        relative_goal: bool = True,
        include_role: bool = True,
        task_dim: DimSpec | None = None,  # for API consistency
    ):
        super().__init__(relative_goal=relative_goal)
        self.object_keys = tuple(object_keys) if object_keys is not None else None
        self.include_role = include_role
        role_dim = ROLE_DIM if include_role else 0
        self.output_dim = (RELATIVE_SE3_DIM if relative_goal else POSE_DIM) + role_dim
        # The role one-hot is the trailing block of every token, in both modes.
        self._categorical_mask = torch.cat(
            [
                torch.ones(self.output_dim - role_dim, dtype=torch.bool),
                torch.zeros(role_dim, dtype=torch.bool),
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

        tokens = []
        for key in keys:
            o_k = get_tensor(obs_task, key)
            g_k = match_shapes(get_tensor(goal_task, key), o_k)

            parts = [relative_se3_pose(g_k, o_k)]
            if self.include_role:
                parts.append(self._role(obs_task, key, o_k))

            tokens.append(torch.cat(parts, dim=-1))

        return torch.stack(tokens, dim=2)

    def _tokenize_absolute(self, task: Mapping[str, TensorTree]) -> torch.Tensor:
        keys = self._keys(task)

        tokens = []
        for key in keys:
            o_k = get_tensor(task, key)

            parts = [o_k]
            if self.include_role:
                parts.append(self._role(task, key, o_k))

            tokens.append(torch.cat(parts, dim=-1))

        stack_dim = 2 if tokens[0].ndim >= 3 else 1
        return torch.stack(tokens, dim=stack_dim)

    def _role(
        self, task: Mapping[str, TensorTree], key: str, o_k: torch.Tensor
    ) -> torch.Tensor:
        return match_shapes(get_tensor(task, key.replace("_pose", "_role")), o_k)
