from typing import Any

import torch


class PlaceSphereToStackCubeAdapter:
    """Tricks a policy trained on StackCube-v1 into solving PlaceSphere-v1."""

    SPHERE_X_OFFSET = 0.0
    SPHERE_Y_OFFSET = 0.0
    SPHERE_Z_OFFSET = 0.0

    BASKET_X_OFFSET = 0.0
    BASKET_Y_OFFSET = 0.0
    BASKET_Z_OFFSET = 0.018  # Cube B height (2cm) - basket border height (2mm)

    FAKE_QUAT_A = [1, 0, 0, 0]
    # Or [0.694755, 0, 0, 0.08153171]  # Cube A quaternon entry-wise median, # XXX: but it doesn't sum to 1!

    FAKE_QUAT_B = [1, 0, 0, 0]
    # Or [0.69651794, 0, 0, 0.05261808]  # Cube B quaternon entry-wise median, # XXX: but it doesn't sum to 1!

    def apply(self, obs: torch.Tensor | dict[str, Any]) -> torch.Tensor | dict[str, Any]:

        if isinstance(obs, dict):
            return self._apply_to_dict(obs)
        else:
            return self._apply_to_tensor(obs)

    def _apply_to_tensor(self, obs: torch.Tensor) -> torch.Tensor:
        """Projects a 39-dim PlaceSphere state into a 48-dim StackCube state."""

        # PlaceSphere-v1: [0:18 (proprio), 18:19 (is_grasped), 19:26 (TCP pose), 26:29 (bin pos), 29:36 (obj pose), 36:39 (TCP to obj pos)
        # StackCube-v1: [0:18 (proprio), 18:25 (TCP pose), 25:32 (Cube A pose), 32:39 (Cube B pose), 39:42 (TCP to A), 42:45 (TCP to B), 45:48 (A to B)]

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

        # NOTE: When using `obs[..., 36:39]` in place of the difference `sphere_pos - tcp_pos` we achieve
        # final success (also under seed 4803 and specific X,Y,Z offsets).
        # We explain this due to two main factors:
        #   1. obs[.., 36:39] is ignoring the sphere offset we introduced above
        #   2. the model is shifting its attention on the tensor entries before and after grabbing the cube, more specifically:
        #       a. before grabbing the cube it focuses on the absolute position, which is shifted if X_OFFSET
        #           (thus wrong! A truly general policy should grab air with this modification, but our does not)
        #       b. after grabbing the cube it shifts the attention to relative distances vectors
        #   If the absolute position is offsetted we indeed indicate a position where no sphere exists, but we also force the model to grab in a zone out of its distribution
        #   if we are precise enough we can effectively "guide" the model to find this "so-distant" sphere
        #   then once the robot starts to move the sphere it will focus more on the relative distacne instead of the absolute, which in such case will be
        #   CORRECT (it is not offsetted!)

        tcp_to_sphere_pos = sphere_pos - tcp_pose[..., 0:3]  # TCP to Cube A
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

        raise NotImplementedError(
            "Dict observation adaptation is not implemented yet. Only tensor observations are supported."
        )
