from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import ClassVar

import torch

from policy.algorithms.networks.encoder.tokenizers.utils import relative_se3_pose
from policy.transforms.canonicalization.spec import (
    POSE_DIM,
    RELATIVE_SE3_DIM,
    ROLE_DIM,
)
from policy.utils.typing_utils import DimSpec, TensorTree, get_tensor


class ObjectTokenizer:
    """Tokenizes each object entity into an enriched token containing:
    1. SE(3) pose relative to TCP (6D)
    2. SE(3) pose delta from current state to goal (6D, or raw 7D pose in absolute mode)
    3. One-hot role indicator [is_pick, is_target, is_clutter] (3D)

    Total token width:
    - In relative mode (``relative_goal=True``): 6 + 6 + 3 = 15D
    - In absolute mode (``relative_goal=False``): 6 + 7 + 3 = 16D
    """

    supports_single_side: ClassVar[bool] = True

    def __init__(
        self,
        object_keys: Sequence[str] | None = None,
        relative_goal: bool = True,
        task_dim: DimSpec | None = None,  # for API consistency
    ):
        self.object_keys = tuple(object_keys) if object_keys is not None else None
        self.relative_goal = relative_goal
        self.output_dim = (
            RELATIVE_SE3_DIM + RELATIVE_SE3_DIM + ROLE_DIM
            if relative_goal
            else RELATIVE_SE3_DIM + POSE_DIM + ROLE_DIM
        )

    @property
    def tokens_per_step(self) -> int | None:
        return len(self.object_keys) if self.object_keys is not None else None

    def tokenize(self, obs_task: TensorTree | None, goal_task: TensorTree | None) -> torch.Tensor:
        if obs_task is None and goal_task is None:
            raise ValueError("tokenize() requires at least one of obs_task/goal_task.")

        task_dict = obs_task if isinstance(obs_task, Mapping) else goal_task
        if not isinstance(task_dict, Mapping):
            raise TypeError(
                f"{type(self).__name__} requires dict-shaped task trees, got "
                f"{type(obs_task).__name__}/{type(goal_task).__name__}."
            )

        if self.object_keys is not None:
            keys = list(self.object_keys)
        else:
            keys = sorted(
                [k for k in task_dict.keys() if k.startswith("obj_") and k.endswith("_pose")],
                key=lambda k: int(k.split("_")[1]) if k.split("_")[1].isdigit() else k,
            )
            if not keys:
                raise KeyError(
                    f"No canonical object pose keys ('obj_i_pose') found in task dict. "
                    f"Available keys: {list(task_dict.keys())}. Ensure Canonicalizer transform was applied."
                )

        if self.object_keys is not None:
            if obs_task is not None and isinstance(obs_task, Mapping):
                missing = [k for k in self.object_keys if k not in obs_task]
                if missing:
                    raise KeyError(f"obs task tree missing required pose keys: {missing}")
            if goal_task is not None and isinstance(goal_task, Mapping):
                missing = [k for k in self.object_keys if k not in goal_task]
                if missing:
                    raise KeyError(f"goal task tree missing required pose keys: {missing}")

        tcp_pose = get_tensor(task_dict, "tcp_pose")

        tokens = []
        for key in keys:
            role_key = key.replace("_pose", "_role")

            if isinstance(obs_task, Mapping):
                o_k = get_tensor(obs_task, key)
            elif isinstance(goal_task, Mapping):
                o_k = get_tensor(goal_task, key)
            else:
                raise TypeError(f"{type(self).__name__} requires Mapping obs_task or goal_task.")

            # Relative pose to TCP
            rel_to_tcp = relative_se3_pose(o_k, tcp_pose)

            # SE(3) pose delta from current state to goal (or raw pose)
            if self.relative_goal:
                if goal_task is not None and isinstance(goal_task, Mapping) and key in goal_task:
                    g_k = get_tensor(goal_task, key)
                    if g_k.ndim < o_k.ndim:
                        g_k = g_k.unsqueeze(1)
                    goal_delta = relative_se3_pose(g_k, o_k)
                else:
                    goal_delta = torch.zeros(
                        *o_k.shape[:-1],
                        RELATIVE_SE3_DIM,
                        dtype=o_k.dtype,
                        device=o_k.device,
                    )
            else:
                goal_delta = o_k

            # Role indicator
            role = get_tensor(task_dict, role_key)
            if role.ndim < o_k.ndim:
                role = role.unsqueeze(1)
            if role.shape[:-1] != o_k.shape[:-1]:
                role = role.expand(*o_k.shape[:-1], ROLE_DIM)

            tokens.append(torch.cat([rel_to_tcp, goal_delta, role], dim=-1))

        stack_dim = 2 if tokens[0].ndim >= 3 else 1
        return torch.stack(tokens, dim=stack_dim)
