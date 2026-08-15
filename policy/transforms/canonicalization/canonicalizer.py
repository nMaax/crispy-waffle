from collections.abc import Mapping

import torch

from policy.utils.typing_utils import TensorTree, get_subtree, get_tensor


class Canonicalizer:
    """Standardizes different pick-and-place tasks into a unified dictionary format.

    Standardized dict format:
    [proprio, tcp_pose, a_pose, b_pose]
    """

    DIM_SPEC: dict[str, int] = {
        "proprio": 18,
        "tcp_pose": 7,
        "obj_0_pose": 7,
        "obj_0_role": 3,
        "obj_1_pose": 7,
        "obj_1_role": 3,
    }

    def __init__(self, env_id: str):
        self.task_id = env_id

        self._parsers = {
            "StackCube-v1": self._parse_stack_cube_dict,
            "StackCubeLockedRotation-v1": self._parse_stack_cube_locked_rotation_dict,
            "StackCubeSwapped-v1": self._parse_stack_cube_swapped_dict,
            "StackCubeSwappedLockedRotation-v1": self._parse_stack_cube_swapped_locked_rotation_dict,
            "StackCubeRestrictedSpawn-v1": self._parse_stack_cube_restricted_spawn_dict,
            "PlaceSphere-v1": self._parse_place_sphere_dict,
            "PlaceSphereRestrictedSpawn-v1": self._parse_place_sphere_restricted_spawn_dict,
            "PlaceCubeLeft-v1": self._parse_place_cube_left_dict,
            "PlaceCubeLeftLockedRotation-v1": self._parse_place_cube_left_locked_rotation_dict,
            "PlaceCubeRight-v1": self._parse_place_cube_right_dict,
            "PlaceCubeRightLockedRotation-v1": self._parse_place_cube_right_locked_rotation_dict,
            "StackCubeClutter-v1": self._parse_stack_cube_clutter_dict,
            "StackCubeClutterLockedRotation-v1": self._parse_stack_cube_clutter_locked_rotation_dict,
            "StackCubeClutterRandomPick-v1": self._parse_stack_cube_clutter_random_pick_dict,
            "StackCubeClutterRandomPickLockedRotation-v1": self._parse_stack_cube_clutter_random_pick_locked_rotation_dict,
        }

    def __call__(self, obs: TensorTree) -> dict[str, torch.Tensor]:
        if not isinstance(obs, Mapping):
            raise TypeError(
                f"Canonicalizer expects a mapping observation, got {type(obs).__name__}."
            )
        parser = self._parsers[self.task_id]
        return parser(obs)

    # Dictionary parsers for ManiSkill native state_dict observations
    def _parse_place_cube_left_locked_rotation_dict(
        self, obs: Mapping[str, TensorTree]
    ) -> dict[str, torch.Tensor]:
        return self._parse_stack_cube_dict(obs)

    def _parse_place_cube_left_dict(
        self, obs: Mapping[str, TensorTree]
    ) -> dict[str, torch.Tensor]:
        return self._parse_stack_cube_dict(obs)

    def _parse_place_cube_right_locked_rotation_dict(
        self, obs: Mapping[str, TensorTree]
    ) -> dict[str, torch.Tensor]:
        return self._parse_stack_cube_dict(obs)

    def _parse_place_cube_right_dict(
        self, obs: Mapping[str, TensorTree]
    ) -> dict[str, torch.Tensor]:
        return self._parse_stack_cube_dict(obs)

    def _parse_stack_cube_restricted_spawn_dict(
        self, obs: Mapping[str, TensorTree]
    ) -> dict[str, torch.Tensor]:
        return self._parse_stack_cube_dict(obs)

    def _parse_stack_cube_swapped_locked_rotation_dict(
        self, obs: Mapping[str, TensorTree]
    ) -> dict[str, torch.Tensor]:
        return self._parse_stack_cube_dict(obs)

    def _parse_stack_cube_swapped_dict(
        self, obs: Mapping[str, TensorTree]
    ) -> dict[str, torch.Tensor]:
        return self._parse_stack_cube_dict(obs)

    def _parse_stack_cube_locked_rotation_dict(
        self, obs: Mapping[str, TensorTree]
    ) -> dict[str, torch.Tensor]:
        return self._parse_stack_cube_dict(obs)

    def _parse_stack_cube_dict(self, obs: Mapping[str, TensorTree]) -> dict[str, torch.Tensor]:
        agent = get_subtree(obs, "agent")
        extra = get_subtree(obs, "extra")

        qpos = get_tensor(agent, "qpos")
        qvel = get_tensor(agent, "qvel")
        proprio = torch.cat([qpos, qvel], dim=-1)
        tcp_pose = get_tensor(extra, "tcp_pose")
        cube_a_pose = get_tensor(extra, "cubeA_pose")
        cube_b_pose = get_tensor(extra, "cubeB_pose")

        role_pick = torch.tensor(
            [1.0, 0.0, 0.0], dtype=cube_a_pose.dtype, device=cube_a_pose.device
        ).expand(*cube_a_pose.shape[:-1], 3)
        role_target = torch.tensor(
            [0.0, 1.0, 0.0], dtype=cube_b_pose.dtype, device=cube_b_pose.device
        ).expand(*cube_b_pose.shape[:-1], 3)
        # [0.0, 0.0, 1.0] is reserved for clutter objects, which are absent in this task
        # TCP is identifies by [0, 0, 0, 0], which is mathematically equivalent to not passing it at all

        return {
            "proprio": proprio,
            "tcp_pose": tcp_pose,
            "obj_0_pose": cube_a_pose,
            "obj_0_role": role_pick,
            "obj_1_pose": cube_b_pose,
            "obj_1_role": role_target,
        }

    def _parse_stack_cube_clutter_locked_rotation_dict(
        self, obs: Mapping[str, TensorTree]
    ) -> dict[str, torch.Tensor]:
        return self._parse_stack_cube_clutter_dict(obs)

    def _parse_stack_cube_clutter_dict(
        self, obs: Mapping[str, TensorTree]
    ) -> dict[str, torch.Tensor]:
        # StackCubeClutter-v1 and StackCubeClutterLockedRotation-v1 have the same observation structure
        # as StackCube-v1 (indicating A and B explicitly by the keys). The StackCube parser can then be
        # reused for both tasks.
        res = self._parse_stack_cube_dict(obs)
        extra = get_subtree(obs, "extra")

        # [obj_0_pose, obj_1_pose, obj_2_pose, ...] -> [0, 1, 2, ...]
        obj_indices = sorted(
            int(k.split("_")[1])
            for k in extra.keys()
            if k.startswith("obj_") and k.endswith("_pose") and k.split("_")[1].isdigit()
        )
        next_idx = sum(1 for k in res if k.startswith("obj_") and k.endswith("_pose"))
        for i in obj_indices:
            active_k = f"obj_{i}_active"
            if active_k in extra and not bool(torch.any(get_tensor(extra, active_k))):
                continue

            pose_tensor = get_tensor(extra, f"obj_{i}_pose")
            res[f"obj_{next_idx}_pose"] = pose_tensor
            res[f"obj_{next_idx}_role"] = torch.tensor(
                [0.0, 0.0, 1.0], dtype=pose_tensor.dtype, device=pose_tensor.device
            ).expand(*pose_tensor.shape[:-1], 3)
            next_idx += 1
        return res

    def _parse_stack_cube_clutter_random_pick_locked_rotation_dict(
        self, obs: Mapping[str, TensorTree]
    ) -> dict[str, torch.Tensor]:
        return self._parse_stack_cube_clutter_random_pick_dict(obs)

    def _parse_stack_cube_clutter_random_pick_dict(
        self, obs: Mapping[str, TensorTree]
    ) -> dict[str, torch.Tensor]:
        agent = get_subtree(obs, "agent")
        extra = get_subtree(obs, "extra")

        qpos = get_tensor(agent, "qpos")
        qvel = get_tensor(agent, "qvel")
        proprio = torch.cat([qpos, qvel], dim=-1)
        tcp_pose = get_tensor(extra, "tcp_pose")

        obj_indices = sorted(
            int(k.split("_")[1])
            for k in extra.keys()
            if k.startswith("obj_") and k.endswith("_pose") and k.split("_")[1].isdigit()
        )

        if not obj_indices:
            raise KeyError("No obj_*_pose keys found in observation extra subtree.")

        res: dict[str, torch.Tensor] = {
            "proprio": proprio,
            "tcp_pose": tcp_pose,
        }

        active_idx = 0
        for i in obj_indices:
            active_k = f"obj_{i}_active"

            # Skip slots that are marked inactive by the environment
            if active_k in extra and not bool(torch.any(get_tensor(extra, active_k))):
                continue

            pose_k = f"obj_{i}_pose"
            pose_tensor = get_tensor(extra, pose_k)
            out_pose_k = f"obj_{active_idx}_pose"
            res[out_pose_k] = pose_tensor

            is_pick_k = f"obj_{i}_is_pick"
            is_target_k = f"obj_{i}_is_target"
            if is_pick_k in extra and is_target_k in extra:
                is_pick_t = get_tensor(extra, is_pick_k).to(
                    dtype=pose_tensor.dtype, device=pose_tensor.device
                )
                is_target_t = get_tensor(extra, is_target_k).to(
                    dtype=pose_tensor.dtype, device=pose_tensor.device
                )
                while is_pick_t.ndim < pose_tensor.ndim:
                    is_pick_t = is_pick_t.unsqueeze(-1)
                while is_target_t.ndim < pose_tensor.ndim:
                    is_target_t = is_target_t.unsqueeze(-1)
                if is_pick_t.shape[:-1] != pose_tensor.shape[:-1]:
                    is_pick_t = is_pick_t.expand(*pose_tensor.shape[:-1], 1)
                if is_target_t.shape[:-1] != pose_tensor.shape[:-1]:
                    is_target_t = is_target_t.expand(*pose_tensor.shape[:-1], 1)

                # # An object is clutter if it is neither pick nor target
                is_clutter_t = 1.0 - torch.clamp(is_pick_t + is_target_t, 0.0, 1.0)
                role = torch.cat([is_pick_t, is_target_t, is_clutter_t], dim=-1)
                res[f"obj_{active_idx}_role"] = role

            active_idx += 1

        return res

    def _parse_place_sphere_restricted_spawn_dict(
        self, obs: Mapping[str, TensorTree]
    ) -> dict[str, torch.Tensor]:
        return self._parse_place_sphere_dict(obs)

    def _parse_place_sphere_dict(self, obs: Mapping[str, TensorTree]) -> dict[str, torch.Tensor]:
        agent = get_subtree(obs, "agent")
        extra = get_subtree(obs, "extra")

        qpos = get_tensor(agent, "qpos")
        qvel = get_tensor(agent, "qvel")
        proprio = torch.cat([qpos, qvel], dim=-1)
        tcp_pose = get_tensor(extra, "tcp_pose")

        # Sphere pose is directly extra["obj_pose"]
        sphere_pose = get_tensor(extra, "obj_pose")
        bin_pos = get_tensor(extra, "bin_pos")

        fake_quat = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], dtype=sphere_pose.dtype, device=sphere_pose.device
        )
        fake_quat_B = fake_quat.expand(*bin_pos.shape[:-1], 4)
        bin_pose = torch.cat([bin_pos, fake_quat_B], dim=-1)

        role_pick = torch.tensor(
            [1.0, 0.0, 0.0], dtype=sphere_pose.dtype, device=sphere_pose.device
        ).expand(*sphere_pose.shape[:-1], 3)
        role_target = torch.tensor(
            [0.0, 1.0, 0.0], dtype=bin_pose.dtype, device=bin_pose.device
        ).expand(*bin_pose.shape[:-1], 3)

        return {
            "proprio": proprio,
            "tcp_pose": tcp_pose,
            "obj_0_pose": sphere_pose,
            "obj_0_role": role_pick,
            "obj_1_pose": bin_pose,
            "obj_1_role": role_target,
        }
