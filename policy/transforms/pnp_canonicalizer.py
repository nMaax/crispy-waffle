import torch

from policy.utils.typing_utils import NestedTensorMapping


class PnPCanonicalizer:
    """Standardizes different pick-and-place tasks into a unified dictionary format.

    Standardized dict format:
    [proprio, tcp_pose, a_pose, b_pose, a_to_b, tcp_to_a, tcp_to_b]
    """

    def __init__(self, env_id: str):
        self.task_id = env_id

        self._parsers = {
            "StackCube-v1": self._parse_stack_cube_dict,
            "StackCubeRestrictedSpawn-v1": self._parse_stack_cube_restricted_spawn_dict,
            "StackCubeSwapped-v1": self._parse_stack_cube_swapped_dict,
            "PlaceSphere-v1": self._parse_place_sphere_dict,
            "PlaceSphereWristcam-v1": self._parse_place_sphere_wristcam_dict,
            "PlaceCubeLeft-v1": self._parse_place_cube_left_dict,
        }

    def __call__(self, obs: NestedTensorMapping) -> dict[str, torch.Tensor]:
        parser = self._parsers[self.task_id]
        return parser(obs)

    # Dictionary parsers for ManiSkill native state_dict observations
    def _parse_place_cube_left_dict(self, obs: NestedTensorMapping) -> dict[str, torch.Tensor]:
        return self._parse_stack_cube_dict(obs)

    def _parse_stack_cube_dict(self, obs: NestedTensorMapping) -> dict[str, torch.Tensor]:
        agent = obs["agent"]
        extra = obs["extra"]

        # proprio is qpos and qvel concatenated
        proprio = torch.cat([agent["qpos"], agent["qvel"]], dim=-1)
        tcp_pose = extra["tcp_pose"]
        cube_a_pose = extra["cubeA_pose"]
        cube_b_pose = extra["cubeB_pose"]

        return {
            "proprio": proprio,
            "tcp_pose": tcp_pose,
            "a_pose": cube_a_pose,
            "b_pose": cube_b_pose,
            "a_to_b": cube_a_pose[..., :3] - cube_b_pose[..., :3],
            "tcp_to_a": tcp_pose[..., :3] - cube_a_pose[..., :3],
            "tcp_to_b": tcp_pose[..., :3] - cube_b_pose[..., :3],
        }

    def _parse_stack_cube_swapped_dict(self, obs: NestedTensorMapping) -> dict[str, torch.Tensor]:
        return self._parse_stack_cube_dict(obs)

    def _parse_stack_cube_restricted_spawn_dict(
        self, obs: NestedTensorMapping
    ) -> dict[str, torch.Tensor]:
        return self._parse_stack_cube_dict(obs)

    def _parse_place_sphere_dict(self, obs: NestedTensorMapping) -> dict[str, torch.Tensor]:
        agent = obs["agent"]
        extra = obs["extra"]

        proprio = torch.cat([agent["qpos"], agent["qvel"]], dim=-1)
        tcp_pose = extra["tcp_pose"]

        # Sphere pose is directly extra["obj_pose"]
        sphere_pose = extra["obj_pose"]
        sphere_pos = sphere_pose[..., :3]
        bin_pos = extra["bin_pos"]

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
            "a_to_b": sphere_pos - bin_pos,
            "tcp_to_a": tcp_pose[..., :3] - sphere_pos,
            "tcp_to_b": tcp_pose[..., :3] - bin_pos,
        }

    def _parse_place_sphere_wristcam_dict(
        self, obs: NestedTensorMapping
    ) -> dict[str, torch.Tensor]:
        return self._parse_place_sphere_dict(obs)
