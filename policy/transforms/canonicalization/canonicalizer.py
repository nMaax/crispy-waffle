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
        "a_pose": 7,
        "b_pose": 7,
    }

    def __init__(self, env_id: str):
        self.task_id = env_id

        self._parsers = {
            "StackCube-v1": self._parse_stack_cube_dict,
            "StackCubeLockedRotation-v1": self._parse_stack_cube_locked_rotation_dict,
            "StackCubeRestrictedSpawn-v1": self._parse_stack_cube_restricted_spawn_dict,
            "StackCubeSwapped-v1": self._parse_stack_cube_swapped_dict,
            "StackCubeSwappedLockedRotation-v1": self._parse_stack_cube_swapped_locked_rotation_dict,
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
    def _parse_place_cube_left_dict(
        self, obs: Mapping[str, TensorTree]
    ) -> dict[str, torch.Tensor]:
        return self._parse_stack_cube_dict(obs)

    def _parse_place_cube_left_locked_rotation_dict(
        self, obs: Mapping[str, TensorTree]
    ) -> dict[str, torch.Tensor]:
        return self._parse_stack_cube_dict(obs)

    def _parse_place_cube_right_dict(
        self, obs: Mapping[str, TensorTree]
    ) -> dict[str, torch.Tensor]:
        return self._parse_stack_cube_dict(obs)

    def _parse_place_cube_right_locked_rotation_dict(
        self, obs: Mapping[str, TensorTree]
    ) -> dict[str, torch.Tensor]:
        return self._parse_stack_cube_dict(obs)

    def _parse_stack_cube_locked_rotation_dict(
        self, obs: Mapping[str, TensorTree]
    ) -> dict[str, torch.Tensor]:
        return self._parse_stack_cube_dict(obs)

    def _parse_stack_cube_swapped_dict(
        self, obs: Mapping[str, TensorTree]
    ) -> dict[str, torch.Tensor]:
        return self._parse_stack_cube_dict(obs)

    def _parse_stack_cube_swapped_locked_rotation_dict(
        self, obs: Mapping[str, TensorTree]
    ) -> dict[str, torch.Tensor]:
        return self._parse_stack_cube_dict(obs)

    def _parse_stack_cube_restricted_spawn_dict(
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

        return {
            "proprio": proprio,
            "tcp_pose": tcp_pose,
            "a_pose": cube_a_pose,
            "b_pose": cube_b_pose,
        }

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

        return {
            "proprio": proprio,
            "tcp_pose": tcp_pose,
            "a_pose": sphere_pose,
            "b_pose": bin_pose,
        }

    def _parse_stack_cube_clutter_locked_rotation_dict(
        self, obs: Mapping[str, TensorTree]
    ) -> dict[str, torch.Tensor]:
        return self._parse_stack_cube_clutter_dict(obs)

    def _parse_stack_cube_clutter_dict(
        self, obs: Mapping[str, TensorTree]
    ) -> dict[str, torch.Tensor]:
        res = self._parse_stack_cube_dict(obs)
        extra = get_subtree(obs, "extra")
        if isinstance(extra, Mapping):
            for i in range(10):
                pose_k = f"obj_{i}_pose"
                if pose_k in extra:
                    res[f"clutter_{i}_pose"] = get_tensor(extra, pose_k)
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

        obj_indices = []
        if isinstance(extra, Mapping):
            for k in extra.keys():
                if k.startswith("obj_") and k.endswith("_pose"):
                    idx = int(k.split("_")[1])
                    obj_indices.append(idx)
        obj_indices.sort()

        if not obj_indices:
            raise KeyError("No obj_*_pose keys found in observation extra subtree.")

        poses = [get_tensor(extra, f"obj_{i}_pose") for i in obj_indices]
        stacked_poses = torch.stack(poses, dim=-2)
        pool_size = len(obj_indices)

        has_flags = (
            isinstance(extra, Mapping)
            and f"obj_{obj_indices[0]}_is_pick" in extra
            and f"obj_{obj_indices[0]}_is_target" in extra
        )

        if has_flags:
            is_pick_list = [get_tensor(extra, f"obj_{i}_is_pick") for i in obj_indices]
            is_target_list = [get_tensor(extra, f"obj_{i}_is_target") for i in obj_indices]

            stacked_pick = torch.stack(is_pick_list, dim=-1)
            stacked_target = torch.stack(is_target_list, dim=-1)

            while stacked_pick.ndim < stacked_poses.ndim - 1:
                stacked_pick = stacked_pick.unsqueeze(1)
                stacked_target = stacked_target.unsqueeze(1)
            if stacked_pick.shape[:-1] != stacked_poses.shape[:-2]:
                stacked_pick = stacked_pick.expand(*stacked_poses.shape[:-2], pool_size)
                stacked_target = stacked_target.expand(*stacked_poses.shape[:-2], pool_size)

            pick_idx = torch.argmax(stacked_pick.to(torch.int64), dim=-1)
            target_idx = torch.argmax(stacked_target.to(torch.int64), dim=-1)

            pick_pose = torch.gather(
                stacked_poses,
                -2,
                pick_idx.unsqueeze(-1).unsqueeze(-1).expand(*pick_idx.shape, 1, 7),
            ).squeeze(-2)
            target_pose = torch.gather(
                stacked_poses,
                -2,
                target_idx.unsqueeze(-1).unsqueeze(-1).expand(*target_idx.shape, 1, 7),
            ).squeeze(-2)
        else:
            pick_pose = poses[0]
            target_pose = poses[1]

        res = {
            "proprio": proprio,
            "tcp_pose": tcp_pose,
            "a_pose": pick_pose,
            "b_pose": target_pose,
        }

        if isinstance(extra, Mapping):
            clutter_count = 0
            for i in obj_indices:
                pose_k = f"obj_{i}_pose"
                if pose_k in extra:
                    res[f"clutter_{clutter_count}_pose"] = get_tensor(extra, pose_k)
                    clutter_count += 1

        return res
