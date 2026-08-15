from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import ClassVar

import torch

from policy.algorithms.networks.encoder.tokenizers.utils import relative_se3_pose
from policy.utils.typing_utils import TensorTree, get_tensor


class ObjectTokenizer:
    """Tokenizes each object entity into an enriched token containing:
    1. SE(3) pose relative to TCP (6D)
    2. SE(3) pose delta from current state to goal (6D, or raw 7D pose in absolute mode)
    3. One-hot role indicator [is_pick, is_target, is_clutter] (3D)

    Total token width:
    - In relative mode (``relative_goal=True``): 6 + 6 + 3 = 15D
    - In absolute mode (``relative_goal=False``): 6 + 7 + 3 = 16D
    """

    POSE_DIM = 7
    RELATIVE_SE3_DIM = 6
    ROLE_DIM = 3

    supports_single_side: ClassVar[bool] = True

    def __init__(
        self,
        object_keys: Sequence[str] | None = None,
        relative_goal: bool = True,
        task_dim: int | None = None,  # for API consistency
    ):
        self.object_keys = tuple(object_keys) if object_keys is not None else None
        self.relative_goal = relative_goal
        self.output_dim = (
            self.RELATIVE_SE3_DIM + self.RELATIVE_SE3_DIM + self.ROLE_DIM
            if relative_goal
            else self.RELATIVE_SE3_DIM + self.POSE_DIM + self.ROLE_DIM
        )

    @property
    def tokens_per_step(self) -> int | None:
        return len(self.object_keys) if self.object_keys is not None else None

    def _resolve_object_keys(self, task: Mapping[str, TensorTree]) -> list[str]:
        if self.object_keys is not None:
            return list(self.object_keys)
        # Check if task has obj_0_pose, obj_1_pose, ...
        obj_keys = [k for k in task.keys() if k.startswith("obj_") and k.endswith("_pose")]
        if obj_keys:
            return sorted(obj_keys, key=lambda k: int(k.split("_")[1]) if k.split("_")[1].isdigit() else k)

        core_keys = [k for k in ("a_pose", "b_pose") if k in task]
        clutter_keys = sorted(
            [k for k in task.keys() if k.startswith("clutter_") and k.endswith("_pose")],
            key=lambda k: int(k.split("_")[1]) if k.split("_")[1].isdigit() else k,
        )
        other_keys = sorted(
            k
            for k in task.keys()
            if k not in core_keys
            and k not in clutter_keys
            and k != "tcp_pose"
            and not k.endswith("_role")
            and k.endswith("_pose")
        )
        resolved = core_keys + clutter_keys + other_keys
        if not resolved:
            resolved = [
                k for k in sorted(task.keys()) if k != "tcp_pose" and not k.endswith("_role")
            ]
        return resolved

    def _get_role_tensor(
        self, key: str, task: Mapping[str, TensorTree], sample: torch.Tensor
    ) -> torch.Tensor:
        role_key = key.replace("_pose", "_role")
        if role_key in task:
            role = get_tensor(task, role_key)
            while role.ndim < sample.ndim:
                role = role.unsqueeze(1)
            if role.shape[:-1] != sample.shape[:-1]:
                role = role.expand(*sample.shape[:-1], self.ROLE_DIM)
            return role
        if key == "a_pose":
            role = torch.tensor([1.0, 0.0, 0.0], dtype=sample.dtype, device=sample.device)
        elif key == "b_pose":
            role = torch.tensor([0.0, 1.0, 0.0], dtype=sample.dtype, device=sample.device)
        elif key == "tcp_pose":
            role = torch.tensor([0.0, 0.0, 0.0], dtype=sample.dtype, device=sample.device)
        else:
            role = torch.tensor([0.0, 0.0, 1.0], dtype=sample.dtype, device=sample.device)
        return role.expand(*sample.shape[:-1], self.ROLE_DIM)

    def _get_tcp_pose(
        self, task: Mapping[str, TensorTree], sample_obj_pose: torch.Tensor
    ) -> torch.Tensor:
        if "tcp_pose" in task:
            tcp = get_tensor(task, "tcp_pose")
            while tcp.ndim < sample_obj_pose.ndim:
                tcp = tcp.unsqueeze(1)
            if tcp.shape[:-1] != sample_obj_pose.shape[:-1]:
                tcp = tcp.expand(*sample_obj_pose.shape[:-1], 7)
            return tcp
        id_pose = torch.tensor(
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            dtype=sample_obj_pose.dtype,
            device=sample_obj_pose.device,
        )
        return id_pose.expand(*sample_obj_pose.shape[:-1], 7)

    def tokenize(self, obs_task: TensorTree | None, goal_task: TensorTree | None) -> torch.Tensor:
        if obs_task is None and goal_task is None:
            raise ValueError("tokenize() requires at least one of obs_task/goal_task.")

        task_dict = obs_task if isinstance(obs_task, Mapping) else goal_task
        if not isinstance(task_dict, Mapping):
            raise TypeError(
                f"{type(self).__name__} requires dict-shaped task trees, got "
                f"{type(obs_task).__name__}/{type(goal_task).__name__}."
            )

        keys = self._resolve_object_keys(task_dict)
        if not keys:
            raise KeyError("No object pose keys found in task tree.")

        if self.object_keys is not None:
            if obs_task is not None and isinstance(obs_task, Mapping):
                missing = [k for k in self.object_keys if k not in obs_task]
                if missing:
                    raise KeyError(f"obs task tree missing required pose keys: {missing}")
            if goal_task is not None and isinstance(goal_task, Mapping):
                missing = [k for k in self.object_keys if k not in goal_task]
                if missing:
                    raise KeyError(f"goal task tree missing required pose keys: {missing}")

        first_key = keys[0]
        first_pose = get_tensor(task_dict, first_key)
        tcp_pose = self._get_tcp_pose(task_dict, first_pose)

        tokens = []
        for key in keys:
            if isinstance(obs_task, Mapping):
                o_k = get_tensor(obs_task, key)
            elif isinstance(goal_task, Mapping):
                o_k = get_tensor(goal_task, key)
            else:
                raise TypeError(f"{type(self).__name__} requires Mapping obs_task or goal_task.")

            tcp_k = tcp_pose
            if tcp_k.shape[:-1] != o_k.shape[:-1]:
                while tcp_k.ndim < o_k.ndim:
                    tcp_k = tcp_k.unsqueeze(1)
                tcp_k = tcp_k.expand(*o_k.shape[:-1], 7)

            rel_to_tcp = relative_se3_pose(o_k, tcp_k)

            if self.relative_goal:
                if goal_task is not None and isinstance(goal_task, Mapping) and key in goal_task:
                    g_k = get_tensor(goal_task, key)
                    while g_k.ndim < o_k.ndim:
                        g_k = g_k.unsqueeze(1)
                    goal_delta = relative_se3_pose(g_k, o_k)
                else:
                    goal_delta = torch.zeros(
                        *o_k.shape[:-1],
                        self.RELATIVE_SE3_DIM,
                        dtype=o_k.dtype,
                        device=o_k.device,
                    )
            else:
                goal_delta = o_k

            role = self._get_role_tensor(key, task_dict, o_k)
            token_k = torch.cat([rel_to_tcp, goal_delta, role], dim=-1)
            tokens.append(token_k)

        stack_dim = 2 if first_pose.ndim >= 3 else 1
        return torch.stack(tokens, dim=stack_dim)
