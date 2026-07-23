import torch
from mani_skill.envs.utils import randomization
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from policy.environments.stack_cube_env import StackCubeEnv


@register_env("StackCubeLockedRotation-v1", max_episode_steps=50, override=True)
class StackCubeLockedRotationEnv(StackCubeEnv):
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # Reuse the parent's placement logic, then re-roll only the cube
        # orientations so z-axis rotation is locked (parent locks x/y only).
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=True)
            self.cubeA.set_pose(Pose.create_from_pq(p=self.cubeA.pose.p[env_idx], q=qs))

            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=True)
            self.cubeB.set_pose(Pose.create_from_pq(p=self.cubeB.pose.p[env_idx], q=qs))
