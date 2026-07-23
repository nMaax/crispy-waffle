import torch
from mani_skill.envs.utils import randomization
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from .place_sphere_env import PlaceSphereEnv


@register_env("PlaceSphereRestrictedSpawn-v1", max_episode_steps=50, override=True)
class PlaceSphereRestrictedSpawnEnv(PlaceSphereEnv):
    """PlaceSphere-v1 but with the Sphere and Bin spawned within a restricted in-distribution range
    taken from StackCube-v1."""

    # In-distribution ranges (<1*std) for sphere (Cube A) and bin (Cube B) spawn in StackCube
    SPHERE_X_RANGE = (-0.08243, 0.07583)
    SPHERE_Y_RANGE = (-0.12941, 0.13140)
    BIN_X_RANGE = (-0.08909, 0.08211)
    BIN_Y_RANGE = (-0.13197, 0.13146)

    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)

        with torch.device(self.device):
            b = len(env_idx)

            gripper_clearance = 0.025
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + gripper_clearance

            region = (
                [
                    min(self.SPHERE_X_RANGE[0], self.BIN_X_RANGE[0]),
                    min(self.SPHERE_Y_RANGE[0], self.BIN_Y_RANGE[0]),
                ],
                [
                    max(self.SPHERE_X_RANGE[1], self.BIN_X_RANGE[1]),
                    max(self.SPHERE_Y_RANGE[1], self.BIN_Y_RANGE[1]),
                ],
            )

            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )

            sphere_xy = sampler.sample(radius, 100)
            bin_xy = sampler.sample(radius, 100, verbose=False)

            current_obj_pose = self.obj.pose
            new_obj_p = current_obj_pose.p[env_idx].clone()
            new_obj_p[:, :2] = sphere_xy
            new_obj_q = current_obj_pose.q[env_idx]
            self.obj.set_pose(Pose.create_from_pq(p=new_obj_p, q=new_obj_q))  # type: ignore

            current_bin_pose = self.bin.pose
            new_bin_p = current_bin_pose.p[env_idx].clone()
            new_bin_p[:, :2] = bin_xy
            new_bin_q = current_bin_pose.q[env_idx]
            self.bin.set_pose(Pose.create_from_pq(p=new_bin_p, q=new_bin_q))  # type: ignore
