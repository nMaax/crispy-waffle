from collections.abc import Mapping

import torch

from policy.utils.typing_utils import TensorTree, get_subtree, get_tensor


class PnPCanonicalizer:
    """Standardizes different pick-and-place tasks into a unified dictionary format.

    Standardized dict format:
    [proprio, tcp_pose, a_pose, b_pose, tcp_to_a, tcp_to_b, a_to_b]
    """

    DIM_SPEC: dict[str, int] = {
        "proprio": 18,
        "tcp_pose": 7,
        "a_pose": 7,
        "b_pose": 7,
        "tcp_to_a": 3,
        "tcp_to_b": 3,
        "a_to_b": 3,
    }

    def __init__(self, env_id: str):
        self.task_id = env_id

        self._parsers = {
            "StackCube-v1": self._parse_stack_cube_dict,
            "StackCubeLockedRotation-v1": self._parse_stack_cube_locked_rotation_dict,
            "StackCubeRestrictedSpawn-v1": self._parse_stack_cube_restricted_spawn_dict,
            "StackCubeSwapped-v1": self._parse_stack_cube_swapped_dict,
            "PlaceSphere-v1": self._parse_place_sphere_dict,
            "PlaceCubeLeft-v1": self._parse_place_cube_left_dict,
        }

    def __call__(self, obs: TensorTree) -> dict[str, torch.Tensor]:
        if not isinstance(obs, Mapping):
            raise TypeError(
                f"PnPCanonicalizer expects a mapping observation, got {type(obs).__name__}."
            )
        parser = self._parsers[self.task_id]
        return parser(obs)

    # Dictionary parsers for ManiSkill native state_dict observations
    def _parse_place_cube_left_dict(
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

        tcp_to_a = get_tensor(extra, "tcp_to_cubeA_pos")
        tcp_to_b = get_tensor(extra, "tcp_to_cubeB_pos")
        a_to_b = get_tensor(extra, "cubeA_to_cubeB_pos")

        return {
            "proprio": proprio,
            "tcp_pose": tcp_pose,
            "a_pose": cube_a_pose,
            "b_pose": cube_b_pose,
            "tcp_to_a": tcp_to_a,
            "tcp_to_b": tcp_to_b,
            "a_to_b": a_to_b,
        }

    def _parse_place_sphere_dict(self, obs: Mapping[str, TensorTree]) -> dict[str, torch.Tensor]:
        agent = get_subtree(obs, "agent")
        extra = get_subtree(obs, "extra")

        qpos = get_tensor(agent, "qpos")
        qvel = get_tensor(agent, "qvel")
        proprio = torch.cat([qpos, qvel], dim=-1)
        tcp_pose = get_tensor(extra, "tcp_pose")

        # Sphere pose is directly extra["obj_pose"]
        sphere_pose = get_tensor(extra, "obj_pose")
        sphere_pos = sphere_pose[..., :3]
        bin_pos = get_tensor(extra, "bin_pos")

        fake_quat = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], dtype=sphere_pose.dtype, device=sphere_pose.device
        )
        fake_quat_B = fake_quat.expand(*bin_pos.shape[:-1], 4)
        bin_pose = torch.cat([bin_pos, fake_quat_B], dim=-1)

        tcp_to_a = get_tensor(extra, "tcp_to_obj_pos")
        tcp_to_b = bin_pos - tcp_pose[..., :3]
        a_to_b = bin_pos - sphere_pos

        return {
            "proprio": proprio,
            "tcp_pose": tcp_pose,
            "a_pose": sphere_pose,
            "b_pose": bin_pose,
            "tcp_to_a": tcp_to_a,
            "tcp_to_b": tcp_to_b,
            "a_to_b": a_to_b,
        }
