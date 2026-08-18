from collections.abc import Mapping

import torch

from policy.transforms.canonicalization.spec import (
    ROLE_CLUTTER,
    ROLE_PICK,
    ROLE_TARGET,
    ROLE_TCP,
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
            "StackCubeSwapped-v1": self._parse_pool_dict,
            "StackCubeSwappedLockedRotation-v1": self._parse_pool_dict,
        }

    def __call__(self, obs: TensorTree) -> dict[str, torch.Tensor]:
        if not isinstance(obs, Mapping):
            raise TypeError(
                f"Canonicalizer expects a mapping observation, got {type(obs).__name__}."
            )
        parser = self._parsers[self.task_id]
        return parser(obs)

    def _parse_stack_cube_clutter_random_pick_dict(
        self, obs: Mapping[str, TensorTree]
    ) -> dict[str, torch.Tensor]:
        return self._parse_pool_dict(obs, is_random_pick=True)

    def _parse_pool_dict(
        self,
        obs: Mapping[str, TensorTree],
        *,
        is_random_pick: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Parses any stack_cube-family task into the standardized object-pool format."""
        agent = get_subtree(obs, "agent")
        extra = get_subtree(obs, "extra")

        qpos = get_tensor(agent, "qpos")
        qvel = get_tensor(agent, "qvel")
        proprio = torch.cat([qpos, qvel], dim=-1)

        tcp_pose = get_tensor(extra, "tcp_pose")
        poses: list[torch.Tensor] = [tcp_pose]
        roles: list[torch.Tensor] = [role_tensor(ROLE_TCP, tcp_pose)]
        # The TCP is always present, so its validity is unconditional.
        valids: list[torch.Tensor] = [torch.ones_like(tcp_pose[..., :1])]

        if not is_random_pick:
            # cubeA is always the pick and cubeB the target, by key name -- the swapped tasks
            # deliberately keep that labelling so the policy has to re-identify from the goal.
            for key, role in (("cubeA", ROLE_PICK), ("cubeB", ROLE_TARGET)):
                pose = get_tensor(extra, f"{key}_pose")
                poses.append(pose)
                roles.append(role_tensor(role, pose))
                valids.append(torch.ones_like(pose[..., :1]))

        # [obj_0_pose, obj_1_pose, ...] -> [0, 1, ...]
        obj_indices = sorted(
            int(k.split("_")[1])
            for k in extra.keys()
            if k.startswith("obj_") and k.endswith("_pose") and k.split("_")[1].isdigit()
        )

        for i in obj_indices:
            pose = get_tensor(extra, f"obj_{i}_pose")
            poses.append(pose)

            if is_random_pick:
                # Every pool member (cubeA, cubeB, and clutter alike) carries its own per-episode
                # role. Read it directly instead of guessing.
                is_pick_t = match_shape(get_tensor(extra, f"obj_{i}_is_pick"), pose)
                is_target_t = match_shape(get_tensor(extra, f"obj_{i}_is_target"), pose)
                # An object is clutter if it is neither pick nor target
                is_clutter_t = 1.0 - torch.clamp(is_pick_t + is_target_t, 0.0, 1.0)
                is_tcp_t = torch.zeros_like(is_pick_t)
                roles.append(torch.cat([is_tcp_t, is_pick_t, is_target_t, is_clutter_t], dim=-1))
            else:
                # obj_i here is always decorative clutter: the fixed pair above already claimed
                # the pick/target roles.
                roles.append(role_tensor(ROLE_CLUTTER, pose))

            active_k = f"obj_{i}_active"
            valids.append(
                match_shape(get_tensor(extra, active_k), pose)
                if active_k in extra
                else torch.ones_like(pose[..., :1])
            )

        if not obj_indices and is_random_pick:
            raise KeyError("No object pose keys found in observation extra subtree.")

        return {
            "proprio": proprio,
            "obj_pose": torch.stack(poses, dim=-2),
            "obj_role": torch.stack(roles, dim=-2),
            "obj_valid": torch.cat(valids, dim=-1),
        }
