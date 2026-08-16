from collections.abc import Mapping

import torch

from policy.transforms.canonicalization.spec import (
    ROLE_CLUTTER,
    ROLE_PICK,
    ROLE_TARGET,
    canonical_dim_spec,
)
from policy.transforms.canonicalization.utils import match_shape, role_tensor
from policy.utils import get_subtree, get_tensor
from policy.utils.typing_utils import TensorTree


class Canonicalizer:
    """Standardizes different pick-and-place tasks into a unified dictionary format."""

    dim_spec = staticmethod(canonical_dim_spec)

    def __init__(self, env_id: str):
        self.task_id = env_id

        self._parsers = {
            "StackCube-v1": self._parse_pool_dict,
            "StackCubeLockedRotation-v1": self._parse_pool_dict,
            "StackCubeRestrictedSpawn-v1": self._parse_pool_dict,
            "PlaceCubeLeft-v1": self._parse_pool_dict,
            "PlaceCubeLeftLockedRotation-v1": self._parse_pool_dict,
            "PlaceCubeRight-v1": self._parse_pool_dict,
            "PlaceCubeRightLockedRotation-v1": self._parse_pool_dict,
            "StackCubeClutter-v1": self._parse_pool_dict,
            "StackCubeClutterLockedRotation-v1": self._parse_pool_dict,
            "StackCubeClutterRandomPick-v1": self._parse_stack_cube_clutter_random_pick_dict,
            "StackCubeClutterRandomPickLockedRotation-v1": (
                self._parse_stack_cube_clutter_random_pick_dict
            ),
            "StackCubeSwapped-v1": self._parse_stack_cube_swapped_dict,
            "StackCubeSwappedLockedRotation-v1": self._parse_stack_cube_swapped_dict,
        }

    def __call__(self, obs: TensorTree) -> dict[str, torch.Tensor]:
        if not isinstance(obs, Mapping):
            raise TypeError(
                f"Canonicalizer expects a mapping observation, got {type(obs).__name__}."
            )
        parser = self._parsers[self.task_id]
        return parser(obs)

    def _parse_stack_cube_swapped_dict(
        self, obs: Mapping[str, TensorTree]
    ) -> dict[str, torch.Tensor]:
        return self._parse_pool_dict(obs, pick_key="cubeB", target_key="cubeA")

    def _parse_stack_cube_clutter_random_pick_dict(
        self, obs: Mapping[str, TensorTree]
    ) -> dict[str, torch.Tensor]:
        return self._parse_pool_dict(obs, is_random_pick=True)

    def _parse_pool_dict(
        self,
        obs: Mapping[str, TensorTree],
        *,
        pick_key: str = "cubeA",
        target_key: str = "cubeB",
        is_random_pick: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Parses any stack_cube-family task into the standardized object-pool format."""
        agent = get_subtree(obs, "agent")
        extra = get_subtree(obs, "extra")

        qpos = get_tensor(agent, "qpos")
        qvel = get_tensor(agent, "qvel")
        proprio = torch.cat([qpos, qvel], dim=-1)
        tcp_pose = get_tensor(extra, "tcp_pose")

        res: dict[str, torch.Tensor] = {"proprio": proprio, "tcp_pose": tcp_pose}
        next_idx = 0

        if not is_random_pick:
            pick_pose = get_tensor(extra, f"{pick_key}_pose")
            target_pose = get_tensor(extra, f"{target_key}_pose")
            res[f"obj_{next_idx}_pose"] = pick_pose
            res[f"obj_{next_idx}_role"] = role_tensor(ROLE_PICK, pick_pose)
            next_idx += 1
            res[f"obj_{next_idx}_pose"] = target_pose
            res[f"obj_{next_idx}_role"] = role_tensor(ROLE_TARGET, target_pose)
            next_idx += 1

        # [obj_0_pose, obj_1_pose, ...] -> [0, 1, ...]
        obj_indices = sorted(
            int(k.split("_")[1])
            for k in extra.keys()
            if k.startswith("obj_") and k.endswith("_pose") and k.split("_")[1].isdigit()
        )

        for i in obj_indices:
            active_k = f"obj_{i}_active"
            # Check if object i is active or not, if not, skip it and don't pass it to the algorithm.
            if active_k in extra and not bool(torch.any(get_tensor(extra, active_k))):
                continue

            pose_tensor = get_tensor(extra, f"obj_{i}_pose")
            res[f"obj_{next_idx}_pose"] = pose_tensor

            if is_random_pick:
                # Every pool member (cubeA, cubeB, and clutter alike) carries its own per-episode
                # role. Read it directly instead of guessing.
                is_pick_t = match_shape(get_tensor(extra, f"obj_{i}_is_pick"), pose_tensor)
                is_target_t = match_shape(get_tensor(extra, f"obj_{i}_is_target"), pose_tensor)
                # An object is clutter if it is neither pick nor target
                is_clutter_t = 1.0 - torch.clamp(is_pick_t + is_target_t, 0.0, 1.0)
                res[f"obj_{next_idx}_role"] = torch.cat(
                    [is_pick_t, is_target_t, is_clutter_t], dim=-1
                )
            else:
                # obj_i here is always decorative clutter: the fixed pair above already claimed
                # the pick/target roles.
                res[f"obj_{next_idx}_role"] = role_tensor(ROLE_CLUTTER, pose_tensor)

            next_idx += 1

        if next_idx == 0:
            raise KeyError("No object pose keys found in observation extra subtree.")

        return res
