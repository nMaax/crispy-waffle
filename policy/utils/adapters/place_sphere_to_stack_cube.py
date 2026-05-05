from typing import Any

import torch


class PlaceSphereToStackCubeAdapter:
    """Tricks a policy trained on StackCube-v1 into solving PlaceSphere-v1.

    Maps the 'Sphere' state to 'Cube A' and the 'Bin' state to 'Cube B'. Injects a Z-offset to
    account for the difference between stacking on a 4cm tall cube versus placing inside a shallow
    bin.
    """

    # NOTE: the following offset have been found by repeated experiments,
    # we guess that other than representing natural physical offset due to geometrical
    # differences between the actors (cubes vs sphere + basket) in the two environments,
    # they also likely compesate for some intrinsic bias in the ground truth motionplanning
    # episodes where, for example, the cube used to be gripped not exactly at its center, but slightly off-centered,
    # which would cause the TCP w.r.t. the sphere to be slightly off-centered as well;
    # or, the cube used to be placed not exactly at the center of the other cube, but slightly off-centered as well;
    # while such behaviour did not have bad consequences in a cube grabbing task, for a sphere this means, for instance,
    # slipping away from the gripped, or not being collocated within the basket boundaries with precision.
    # A more appropriate and scientific solution for this would be to compute such offsets deterministically
    # from the training dataset, computing for example the average TCP-to-sphere position at the instance of grabbing across
    # the training episodes, and the average sphere-to-basket position at the end of the training episodes.
    # Or, even better, fix the original dataset by chirurgically adjusting the recorded TCP and object positions to be perfectly centered
    # and collocated, and then retrain the policy on this fixed dataset.

    SPHERE_X_OFFSET = 0.04
    SPHERE_Y_OFFSET = 0.08
    SPHERE_Z_OFFSET = 0.01

    BASKET_X_OFFSET = 0.098
    BASKET_Y_OFFSET = 0.076
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

        sphere_pose = adapted["obj_pose"].clone()
        basket_pos = adapted["bin_pos"].clone()
        tcp_to_sphere_pos = adapted["tcp_to_obj_pos"].clone()
        tcp_pose = adapted["tcp_pose"].clone()

        sphere_pose[..., 0] += self.SPHERE_X_OFFSET
        sphere_pose[..., 1] += self.SPHERE_Y_OFFSET
        sphere_pose[..., 2] += self.SPHERE_Z_OFFSET

        basket_pos[..., 0] += self.BASKET_X_OFFSET
        basket_pos[..., 1] += self.BASKET_Y_OFFSET
        basket_pos[..., 2] += self.BASKET_Z_OFFSET

        batch_shape = basket_pos.shape[:-1]
        fake_quat = torch.tensor(self.FAKE_QUAT, device=basket_pos.device, dtype=basket_pos.dtype)
        fake_quat = fake_quat.expand(*batch_shape, 4)
        cubeB_pose = torch.cat([basket_pos, fake_quat], dim=-1)

        tcp_to_cubeB_pos = basket_pos - tcp_pose[..., 0:3]
        cubeA_to_cubeB_pos = basket_pos - sphere_pose[..., 0:3]

        adapted["cubeA_pose"] = sphere_pose
        adapted["cubeB_pose"] = cubeB_pose
        adapted["tcp_to_cubeA_pos"] = tcp_to_sphere_pos
        adapted["tcp_to_cubeB_pos"] = tcp_to_cubeB_pos
        adapted["cubeA_to_cubeB_pos"] = cubeA_to_cubeB_pos

        for key in ["obj_pose", "bin_pos", "tcp_to_obj_pos", "is_grasped"]:
            adapted.pop(key, None)

        return adapted
