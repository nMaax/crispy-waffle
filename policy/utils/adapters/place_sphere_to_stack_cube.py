from typing import Any

import torch


class PlaceSphereToStackCubeAdapter:
    """Tricks a policy trained on StackCube-v1 into solving PlaceSphere-v1."""

    # NOTE: the following offset have been found by repeated experiments on checkpoint 2026-05-03/18-01-47/checkpoints/step_035000,
    # however we noted different checkpoints, also within the same training run, lead to different biases (e.g. sometimes southewest, sometimes norteast)
    # on the point where gripper tries to grab the cube. I think then I should modify the training loss of the policy with a high penalty for not pointing to the
    # exact center of the cube when grabbing (note also that motionplanning data is extremely precise, it grabs the cube at the center basically all times, tho it places the faces of the two cubes slightly\
    # offsetted)
    # On top of that, there is also a bian on the X value of the spawning position of the sphere, that tend to be OOD w.r.t the spawn X value of Cube A
    # so I either need to scale the training data to more general situations, or force the sphereto spawn within an in-distribution X value

    SPHERE_X_OFFSET = 0.07
    SPHERE_Y_OFFSET = 0.06
    SPHERE_Z_OFFSET = 0.01

    BASKET_X_OFFSET = 0.058
    BASKET_Y_OFFSET = 0.056
    BASKET_Z_OFFSET = 0.018

    FAKE_QUAT_A = [0.694755, 0, 0, 0.08153171]  # Cube A quaternon median
    FAKE_QUAT_B = [0.69651794, 0, 0, 0.05261808]  # Cube B quaternon median

    def apply(self, obs: torch.Tensor | dict[str, Any]) -> torch.Tensor | dict[str, Any]:

        if isinstance(obs, dict):
            return self._apply_to_dict(obs)
        else:
            return self._apply_to_tensor(obs)

    def _apply_to_tensor(self, obs: torch.Tensor) -> torch.Tensor:
        """Projects a 39-dim PlaceSphere state into a 48-dim StackCube state."""

        # TODO: add small scheme that describes both tensors and the meaning at each index/slice

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

        fake_quat_A = torch.tensor(self.FAKE_QUAT_A, dtype=obs.dtype, device=obs.device)
        fake_quat_A = fake_quat_A.expand(*sphere_pos.shape[:-1], 4)
        obj_pose = torch.cat([sphere_pos, fake_quat_A], dim=-1)

        swapped[..., 25:32] = obj_pose

        fake_quat_B = torch.tensor(self.FAKE_QUAT_B, dtype=obs.dtype, device=obs.device)
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
