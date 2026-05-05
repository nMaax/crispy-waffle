from typing import Any

import torch


class PlaceSphereToStackCubeAdapter:
    """Tricks a policy trained on StackCube-v1 into solving PlaceSphere-v1.

    Maps the 'Sphere' state to 'Cube A' and the 'Bin' state to 'Cube B'. Injects a Z-offset to
    account for the difference between stacking on a 4cm tall cube versus placing inside a shallow
    bin.
    """

    # 0.04 (StackCube expected offset) - 0.0225 (PlaceSphere physical target offset)
    SPHERE_X_OFFSET = 0.04  # 0.01
    SPHERE_Y_OFFSET = 0.08  # 0.02
    SPHERE_Z_OFFSET = 0.01  # 0.00

    BASKET_X_OFFSET = 0.040 + 0.008 + 0.05
    BASKET_Y_OFFSET = 0.024 + 0.052
    BASKET_Z_OFFSET = 0.018

    FAKE_QUAT = [1.0, 0.0, 0.0, 0.0]

    def apply(self, obs: torch.Tensor | dict[str, Any]) -> torch.Tensor | dict[str, Any]:

        if isinstance(obs, dict):
            return self._apply_to_dict(obs)
        else:
            return self._apply_to_tensor(obs)

    def _apply_to_tensor(self, obs: torch.Tensor) -> torch.Tensor:
        """Projects a 39-dim PlaceSphere state into a 48-dim StackCube state.

        Works across arbitrary batch dimensions [..., 39] -> [..., 48].
        """

        proprioception = obs[..., 0:18].clone()  # Proprioception
        tcp_pose = obs[..., 19:26].clone()  # TCP
        sphere_pos = obs[..., 29:32].clone()  # Cube A
        basket_pos = obs[..., 26:29].clone()  # Cube B

        sphere_pos[..., 0] += self.SPHERE_X_OFFSET
        sphere_pos[..., 1] += self.SPHERE_Y_OFFSET
        sphere_pos[..., 2] += self.SPHERE_Z_OFFSET

        basket_pos[..., 0] += self.BASKET_X_OFFSET
        basket_pos[..., 1] += self.BASKET_Y_OFFSET
        basket_pos[..., 2] += self.BASKET_Z_OFFSET

        tcp_to_sphere_pos = obs[..., 36:39].clone()  # TCP to Cube A
        tcp_to_basket = basket_pos - tcp_pose[..., 0:3]  # TCP to Cube B
        sphere_to_basket = basket_pos - sphere_pos  # Cube A to Cube B

        swapped = torch.zeros((*obs.shape[:-1], 48), dtype=obs.dtype, device=obs.device)

        swapped[..., 0:18] = proprioception

        # We completely drop `is_grasped` (index 18) as StackCube doesn't use it
        swapped[..., 18:25] = tcp_pose

        fake_quat_A = torch.tensor(self.FAKE_QUAT, dtype=obs.dtype, device=obs.device)
        fake_quat_A = fake_quat_A.expand(*sphere_pos.shape[:-1], 4)
        obj_pose = torch.cat([sphere_pos, fake_quat_A], dim=-1)

        swapped[..., 25:32] = obj_pose

        fake_quat_B = torch.tensor(self.FAKE_QUAT, dtype=obs.dtype, device=obs.device)
        fake_quat_B = fake_quat_B.expand(*sphere_pos.shape[:-1], 4)
        bin_pose = torch.cat([basket_pos, fake_quat_B], dim=-1)
        swapped[..., 32:39] = bin_pose

        swapped[..., 39:42] = tcp_to_sphere_pos
        swapped[..., 42:45] = tcp_to_basket
        swapped[..., 45:48] = sphere_to_basket

        return swapped

    def _apply_to_dict(self, obs_dict: dict[str, Any]) -> dict[str, Any]:
        adapted = obs_dict.copy()

        # ManiSkill observation dicts often nest these inside an "extra" key
        target_dict = adapted["extra"] if "extra" in adapted else adapted

        # Extract PlaceSphere specific states
        obj_pose = target_dict["obj_pose"].clone()
        bin_pos = target_dict["bin_pos"].clone()
        tcp_to_obj_pos = target_dict["tcp_to_obj_pos"].clone()
        tcp_pose = target_dict["tcp_pose"].clone()

        # Apply Z-Height Shift to the Bin
        bin_pos[..., 2] -= self.SPHERE_Z_OFFSET

        # Reconstruct missing Cube B Pose (Bin only has position, StackCube expects 7-dim Pose)
        # We append a neutral quaternion [1, 0, 0, 0] to the 3D position
        batch_shape = bin_pos.shape[:-1]
        fake_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=bin_pos.device, dtype=bin_pos.dtype)
        fake_quat = fake_quat.expand(*batch_shape, 4)
        cubeB_pose = torch.cat([bin_pos, fake_quat], dim=-1)

        # Calculate missing relative vectors that StackCube expects
        tcp_to_cubeB_pos = bin_pos - tcp_pose[..., 0:3]
        cubeA_to_cubeB_pos = bin_pos - obj_pose[..., 0:3]

        # Inject StackCube expected keys into the dictionary
        target_dict["cubeA_pose"] = obj_pose
        target_dict["cubeB_pose"] = cubeB_pose
        target_dict["tcp_to_cubeA_pos"] = tcp_to_obj_pos
        target_dict["tcp_to_cubeB_pos"] = tcp_to_cubeB_pos
        target_dict["cubeA_to_cubeB_pos"] = cubeA_to_cubeB_pos

        # Remove PlaceSphere specific keys so they don't bloat the flatten_tensor_dict output
        for key in ["obj_pose", "bin_pos", "tcp_to_obj_pos", "is_grasped"]:
            target_dict.pop(key, None)

        return adapted
