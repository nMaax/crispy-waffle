from typing import Any

import torch

from policy.utils.typing_utils import AdapterProtocol


class PlaceSphereToStackCubeAdapter(AdapterProtocol):
    """Tricks a policy trained on StackCube-v1 into solving PlaceSphere-v1."""

    BIN_Z_OFFSET = 0.018  # Cube B height (2cm) - bin border height (2mm)

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
        """Maps a 39-dim PlaceSphere state into a 48-dim StackCube state."""

        # PlaceSphere-v1: [0:18 (proprio), 18:19 (is_grasped), 19:26 (TCP pose), 26:29 (bin pos), 29:36 (obj pose), 36:39 (TCP to obj pos)
        # StackCube-v1: [0:18 (proprio), 18:25 (TCP pose), 25:32 (Cube A pose), 32:39 (Cube B pose), 39:42 (TCP to A), 42:45 (TCP to B), 45:48 (A to B)]

        proprioception = obs[..., 0:18].clone()
        tcp_pose = obs[..., 19:26].clone()
        sphere_pos = obs[..., 29:32].clone()
        bin_pos = obs[..., 26:29].clone()

        bin_pos[..., 2] += self.BIN_Z_OFFSET

        tcp_to_sphere_pos = sphere_pos - tcp_pose[..., 0:3]
        tcp_to_bin = bin_pos - tcp_pose[..., 0:3]
        sphere_to_bin = bin_pos - sphere_pos

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
        bin_pose = torch.cat([bin_pos, fake_quat_B], dim=-1)
        swapped[..., 32:39] = bin_pose

        swapped[..., 39:42] = tcp_to_sphere_pos
        swapped[..., 42:45] = tcp_to_bin
        swapped[..., 45:48] = sphere_to_bin

        return swapped

    def _apply_to_dict(self, obs_dict: dict[str, Any]) -> dict[str, Any]:

        raise NotImplementedError(
            "Dict observation adaptation is not implemented yet. Only tensor observations are supported."
        )
