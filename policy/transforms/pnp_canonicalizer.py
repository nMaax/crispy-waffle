import torch


class PnPCanonicalizer:
    """Standardizes different pick-and-place tasks into a unified vector or dictionary format.

    Standardized dict format:
    [proprio, tcp_pose, a_pose, b_pose, a_to_b, tcp_to_a, tcp_to_b]
    """

    def __init__(self, env_id: str, as_dict: bool = False):
        self.task_id = env_id
        self.as_dict = as_dict

        self._parsers = {
            "StackCube-v1": self._parse_stack_cube,
            "StackCubeRestrictedSpawn-v1": self._parse_stack_cube_restricted_spawn,
            "StackCubeSwapped-v1": self._parse_stack_cube_swapped,
            "PlaceSphere-v1": self._parse_place_sphere,
            "PlaceSphereWristcam-v1": self._parse_place_sphere_wristcam,
            "PlaceCubeLeft-v1": self._parse_place_cube_left,
        }

        self._dict_parsers = {
            "StackCube-v1": self._parse_stack_cube_dict,
            "StackCubeRestrictedSpawn-v1": self._parse_stack_cube_restricted_spawn_dict,
            "StackCubeSwapped-v1": self._parse_stack_cube_swapped_dict,
            "PlaceSphere-v1": self._parse_place_sphere_dict,
            "PlaceSphereWristcam-v1": self._parse_place_sphere_wristcam_dict,
            "PlaceCubeLeft-v1": self._parse_place_cube_left_dict,
        }

    # Should decouple from AdapterProtocol and rather make a TransformProtocol
    def __call__(self, obs: dict | torch.Tensor) -> dict | torch.Tensor:
        if isinstance(obs, dict):
            parser = self._dict_parsers[self.task_id]
            components = parser(obs)
        else:
            parser = self._parsers[self.task_id]
            components = parser(obs)

        if self.as_dict:
            return components
        else:
            # Concatenate list of components along last dimension
            return torch.cat(list(components.values()), dim=-1)

    def _parse_place_cube_left(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        return self._parse_stack_cube(obs)

    def _parse_stack_cube(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        proprio = obs[..., 0:18].clone()
        tcp_pose = obs[..., 18:25].clone()
        cube_a_pose = obs[..., 25:32].clone()
        cube_b_pose = obs[..., 32:39].clone()

        return {
            "proprio": proprio,
            "tcp_pose": tcp_pose,
            "a_pose": cube_a_pose,
            "b_pose": cube_b_pose,
            "a_to_b": cube_a_pose[..., :3] - cube_b_pose[..., :3],
            "tcp_to_a": tcp_pose[..., :3] - cube_a_pose[..., :3],
            "tcp_to_b": tcp_pose[..., :3] - cube_b_pose[..., :3],
        }

    def _parse_stack_cube_swapped(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        return self._parse_stack_cube(obs)

    def _parse_stack_cube_restricted_spawn(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        return self._parse_stack_cube(obs)

    def _parse_place_sphere(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        proprio = obs[..., 0:18].clone()
        tcp_pose = obs[..., 19:26].clone()
        sphere_pos = obs[..., 29:32].clone()
        bin_pos = obs[..., 26:29].clone()

        fake_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=obs.dtype, device=obs.device)

        fake_quat_A = fake_quat.expand(*sphere_pos.shape[:-1], 4)
        fake_quat_B = fake_quat.expand(*bin_pos.shape[:-1], 4)

        sphere_pose = torch.cat([sphere_pos, fake_quat_A], dim=-1)
        bin_pose = torch.cat([bin_pos, fake_quat_B], dim=-1)

        return {
            "proprio": proprio,
            "tcp_pose": tcp_pose,
            "a_pose": sphere_pose,
            "b_pose": bin_pose,
            "a_to_b": sphere_pose[..., :3] - bin_pose[..., :3],
            "tcp_to_a": tcp_pose[..., :3] - sphere_pose[..., :3],
            "tcp_to_b": tcp_pose[..., :3] - bin_pose[..., :3],
        }

    def _parse_place_sphere_wristcam(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        return self._parse_place_sphere(obs)

    # Dictionary parsers for ManiSkill native state_dict observations
    def _parse_place_cube_left_dict(self, obs: dict) -> dict[str, torch.Tensor]:
        return self._parse_stack_cube_dict(obs)

    def _parse_stack_cube_dict(self, obs: dict) -> dict[str, torch.Tensor]:
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

    def _parse_stack_cube_swapped_dict(self, obs: dict) -> dict[str, torch.Tensor]:
        return self._parse_stack_cube_dict(obs)

    def _parse_stack_cube_restricted_spawn_dict(self, obs: dict) -> dict[str, torch.Tensor]:
        return self._parse_stack_cube_dict(obs)

    def _parse_place_sphere_dict(self, obs: dict) -> dict[str, torch.Tensor]:
        agent = obs["agent"]
        extra = obs["extra"]

        proprio = torch.cat([agent["qpos"], agent["qvel"]], dim=-1)
        tcp_pose = extra["tcp_pose"]

        # Sphere pose is directly extra["obj_pose"]
        sphere_pose = extra["obj_pose"]
        sphere_pos = sphere_pose[..., :3]
        bin_pos = extra["bin_pos"]

        fake_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=sphere_pose.dtype, device=sphere_pose.device)
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

    def _parse_place_sphere_wristcam_dict(self, obs: dict) -> dict[str, torch.Tensor]:
        return self._parse_place_sphere_dict(obs)
