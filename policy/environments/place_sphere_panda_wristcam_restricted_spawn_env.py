import random

import torch
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from .place_sphere_panda_wristcam_env import PlaceSphereWristcamEnv


@register_env("PlaceSphereWristcamRestrictedSpawnEnv-v1", max_episode_steps=50)
class PlaceSphereWristcamRestrictedSpawnEnv(PlaceSphereWristcamEnv):
    """PlaceSphere-v1 but with the Sphere and Bin spawned within a restricted in-distribution range
    taken from StackCube-v1."""

    # In-distribution ranges (<1*std) for sphere (Cube A) and bin (Cube B) spawn in StackCube
    SPHERE_X_RANGE = (-0.08243, 0.07583)
    SPHERE_Y_RANGE = (-0.12941, 0.13140)
    BIN_X_RANGE = (-0.08909, 0.08211)
    BIN_Y_RANGE = (-0.13197, 0.13146)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)

        sphere_x = random.uniform(*self.SPHERE_X_RANGE)
        sphere_y = random.uniform(*self.SPHERE_Y_RANGE)

        bin_x = random.uniform(*self.BIN_X_RANGE)
        bin_y = random.uniform(*self.BIN_Y_RANGE)

        # TODO: I should also check for collisions, and place the objects sufficiently away

        with torch.device(self.device):
            current_obj_pose = self.obj.pose

            new_obj_p = current_obj_pose.p.clone()
            new_obj_p[:, 0] = sphere_x
            new_obj_p[:, 1] = sphere_y
            self.obj.set_pose(Pose.create(new_obj_p, current_obj_pose.q))  # type: ignore

            current_bin_pose = self.bin.pose

            new_bin_p = current_bin_pose.p.clone()
            new_bin_p[:, 0] = bin_x
            new_bin_p[:, 1] = bin_y
            self.bin.set_pose(Pose.create(new_bin_p, current_bin_pose.q))  # type: ignore
