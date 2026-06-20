import torch
from mani_skill.envs.utils import randomization
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from policy.environments.stack_cube_env import StackCubeEnv


@register_env("StackCubeRestrictedSpawn-v1", max_episode_steps=50, override=True)
class StackCubeRestrictedSpawnEnv(StackCubeEnv):
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.02

            xy = torch.zeros((b, 2))

            region = ([-0.05, -0.05], [0.05, 0.05])

            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.001
            cubeA_xy = xy + sampler.sample(radius, 100)
            cubeB_xy = xy + sampler.sample(radius, 100, verbose=False)

            xyz[:, :2] = cubeA_xy
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False)
            self.cubeA.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            xyz[:, :2] = cubeB_xy
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False)
            self.cubeB.set_pose(Pose.create_from_pq(p=xyz, q=qs))
