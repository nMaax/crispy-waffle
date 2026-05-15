import torch


class PnPCanonicalizer:
    """Standardizes different pick-and-place tasks into a unified vector format.

    [A, B, TCP, TCP-to-A, TCP-to-B, A-to-B]
    """

    def __init__(self, env_id: str):
        self.task_id = env_id
        self._parsers = {
            "StackCube-v1": self._parse_stack_cube,
            "StackCubeSwapped-v1": self._parse_stack_cube_swapped,
            "PlaceSphere-v1": self._parse_place_sphere,
            "PlaceSphereWristcam-v1": self._parse_place_sphere_wristcam,
        }

    # Should decouple from AdapterProtocol and rather make a TransformProtocol
    def _call__(self, obs: dict | torch.Tensor) -> torch.Tensor:
        if isinstance(obs, dict):
            raise NotImplementedError(
                "Dict observations not supported yet. Only tensor observations."
            )

        parser = self._parsers[self.task_id]
        components = parser(obs)
        components = list(components.values())

        return torch.cat(components, dim=-1)

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
        }

    def _parse_stack_cube_swapped(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:

        proprio = obs[..., 0:18].clone()
        tcp_pose = obs[..., 18:25].clone()
        cube_a_pose = obs[..., 25:32].clone()
        cube_b_pose = obs[..., 32:39].clone()

        return {
            "proprio": proprio,
            "tcp_pose": tcp_pose,
            "a_pose": cube_b_pose,
            "b_pose": cube_a_pose,
        }

    def _parse_place_sphere(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:

        proprio = obs[..., 0:18].clone()
        tcp_pose = obs[..., 19:26].clone()
        sphere_pos = obs[
            ..., 29:32
        ].clone()  # Actually PlaceSphere also provides quat for spehre, but we can just ignore it thanks to sphere symmetrical geometry
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
        }

    def _parse_place_sphere_wristcam(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        return self._parse_place_sphere(obs)
