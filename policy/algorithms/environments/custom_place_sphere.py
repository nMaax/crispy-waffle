import random

import torch
from mani_skill.envs.tasks.tabletop.place_sphere import PlaceSphereEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose


@register_env("CustomPlaceSphere-v1", max_episode_steps=200)
class CustomPlaceSphereEnv(PlaceSphereEnv):
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)

        # In-distribution ranges (<1*std) for sphere (Cube A) and bin (Cube B)
        sphere_x_range = (-0.08243, 0.07583)
        sphere_y_range = (-0.12941, 0.13140)
        bin_x_range = (-0.08909, 0.08211)
        bin_y_range = (-0.13197, 0.13146)

        sphere_x = random.uniform(*sphere_x_range)
        sphere_y = random.uniform(*sphere_y_range)

        bin_x = random.uniform(*bin_x_range)
        bin_y = random.uniform(*bin_y_range)

        with torch.device(self.device):
            current_obj_pose = self.obj.pose

            new_obj_p = current_obj_pose.p.clone()
            new_obj_p[:, 0] = sphere_x
            new_obj_p[:, 1] = sphere_y
            self.obj.set_pose(Pose.create(new_obj_p, current_obj_pose.q))

            current_bin_pose = self.bin.pose

            new_bin_p = current_bin_pose.p.clone()
            new_bin_p[:, 0] = bin_x
            new_bin_p[:, 1] = bin_y

            self.bin.set_pose(Pose.create(new_bin_p, current_bin_pose.q))
