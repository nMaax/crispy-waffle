import sapien
import torch
from mani_skill.envs.utils import randomization
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose

from policy.algorithms.environments import PlaceSphereWristcamEnv


@register_env("PlaceSphereWristcamWithCubes-v1", max_episode_steps=50)
class PlaceSphereWristcamWithCubesEnv(PlaceSphereWristcamEnv):
    """Sanity check environment: uses PlaceSphere-v1 but physically spawns the StackCube-v1 Cube A and Cube B."""

    CUBE_HALF_SIZE = 0.02
    SPAWN_REGION = ([-0.1, -0.2], [0.1, 0.2])  # [-0.1, 0.1] x [-0.2, 0.2]

    def _load_scene(self, options: dict):
        """Borrow logic from StackCube, spawning cubes instead of sphere and bin."""

        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        cubeA = actors.build_cube(
            self.scene,
            half_size=self.CUBE_HALF_SIZE,
            color=[1, 0, 0, 1],  # Red
            name="cubeA",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )
        self.obj = cubeA

        cubeB = actors.build_cube(
            self.scene,
            half_size=self.CUBE_HALF_SIZE,
            color=[0, 1, 0, 1],  # Green
            name="cubeB",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[1, 0, 0.1]),
        )
        self.bin = cubeB

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Borrow logic from StackCube, spawning cubes instead of sphere and bin."""

        with torch.device(self.device):
            b = len(env_idx)

            self.table_scene.initialize(env_idx)

            xy_offset = torch.rand((b, 2)) * 0.2 - 0.1  # Uniform within [-0.1, 0.1)

            sampler = randomization.UniformPlacementSampler(
                bounds=self.SPAWN_REGION, batch_size=b, device=self.device
            )
            radius = (
                torch.linalg.norm(torch.tensor([self.CUBE_HALF_SIZE, self.CUBE_HALF_SIZE])) + 0.001
            )

            cubeA_xyz = torch.zeros((b, 3))
            cubeA_xyz[:, :2] = xy_offset + sampler.sample(radius, 100)
            cubeA_xyz[:, 2] = self.CUBE_HALF_SIZE

            cubeA_qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )

            cubeA_pose = Pose.create_from_pq(p=cubeA_xyz, q=cubeA_qs)
            self.obj.set_pose(cubeA_pose)

            cubeB_xyz = torch.zeros((b, 3))
            cubeB_xyz[:, :2] = xy_offset + sampler.sample(radius, 100)
            cubeB_xyz[:, 2] = self.CUBE_HALF_SIZE

            cubeB_qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )

            cubeB_pose = Pose.create_from_pq(p=cubeB_xyz, q=cubeB_qs)
            self.bin.set_pose(cubeB_pose)

    def _get_obs_extra(self, info: dict):
        """Same as StackCube observations.

        This will be compatible with Adapters taking StackCube-like tensors only, e.g. comtible
        with NoOpAdapter, but not PlaceSphereToStackCubeAdapter
        """

        if self.agent is None:
            raise ValueError("Agent is not initialized yet. Cannot compute observations.")

        if not hasattr(self.agent, "tcp") or self.agent.tcp is None:
            raise ValueError("Agent does not have a TCP. Cannot compute observations.")

        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                cubeA_pose=self.obj.pose.raw_pose,
                cubeB_pose=self.bin.pose.raw_pose,
                tcp_to_cubeA_pos=self.obj.pose.p - self.agent.tcp.pose.p,
                tcp_to_cubeB_pos=self.bin.pose.p - self.agent.tcp.pose.p,
                cubeA_to_cubeB_pos=self.bin.pose.p - self.obj.pose.p,
            )
        return obs

    def evaluate(self):
        return {"success": torch.zeros(len(self.obj.pose.p), dtype=torch.bool, device=self.device)}

    def compute_dense_reward(self, info: dict, **kwargs):
        return torch.zeros(len(self.obj.pose.p), device=self.device)
