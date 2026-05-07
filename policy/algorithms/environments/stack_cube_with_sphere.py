import numpy as np
import sapien
import torch
from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
from mani_skill.envs.utils import randomization
from mani_skill.utils import common
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("StackCubeWithSphere-v1", max_episode_steps=50)
class StackCubeWithSphereEnv(StackCubeEnv):
    """Sanity check environment: uses StackCube-v1 but physically spawns the PlaceSphere-v1 Sphere and Bin."""

    SPHERE_RADIUS = 0.02

    BIN_INNER_SIDE_HALF_LEN = 0.02
    BIN_SHORT_SIDE_HALF_SIZE = 0.0025

    BLOCK_HALF_SIZES = [
        BIN_SHORT_SIDE_HALF_SIZE,
        2 * BIN_SHORT_SIDE_HALF_SIZE + BIN_INNER_SIDE_HALF_LEN,
        2 * BIN_SHORT_SIDE_HALF_SIZE + BIN_INNER_SIDE_HALF_LEN,
    ]
    EDGE_BLOCK_HALF_SIZES = [
        BIN_SHORT_SIDE_HALF_SIZE,
        2 * BIN_SHORT_SIDE_HALF_SIZE + BIN_INNER_SIDE_HALF_LEN,
        2 * BIN_SHORT_SIDE_HALF_SIZE,
    ]

    SPAWN_REGION = ([-0.1, -0.2], [0.1, 0.2])  # [-0.1, 0.1] x [-0.2, 0.2]

    # NOTE: PlaceSphere DOES support "panda_wristcam", so no issues here!
    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _build_bin(self):
        """Helper to build the bin out of 5 boces: 1 for the bottom and 4 for the edges."""

        builder = self.scene.create_actor_builder()
        dx = self.BLOCK_HALF_SIZES[1] - self.BLOCK_HALF_SIZES[0]
        dy = self.BLOCK_HALF_SIZES[1] - self.BLOCK_HALF_SIZES[0]
        dz = self.EDGE_BLOCK_HALF_SIZES[2] + self.BLOCK_HALF_SIZES[0]

        poses = [
            sapien.Pose([0, 0, 0]),
            sapien.Pose([-dx, 0, dz]),
            sapien.Pose([dx, 0, dz]),
            sapien.Pose([0, -dy, dz]),
            sapien.Pose([0, dy, dz]),
        ]
        half_sizes = [
            (self.BLOCK_HALF_SIZES[1], self.BLOCK_HALF_SIZES[2], self.BLOCK_HALF_SIZES[0]),
            self.EDGE_BLOCK_HALF_SIZES,
            self.EDGE_BLOCK_HALF_SIZES,
            (
                self.EDGE_BLOCK_HALF_SIZES[1],
                self.EDGE_BLOCK_HALF_SIZES[0],
                self.EDGE_BLOCK_HALF_SIZES[2],
            ),
            (
                self.EDGE_BLOCK_HALF_SIZES[1],
                self.EDGE_BLOCK_HALF_SIZES[0],
                self.EDGE_BLOCK_HALF_SIZES[2],
            ),
        ]
        for pose, half_size in zip(poses, half_sizes):
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size)

        return builder.build_kinematic(name="bin")

    def _load_scene(self, options: dict):
        """Borrow logic from PlaceSphere, spawning sphere and bin instead of cubes."""

        self.cube_half_size = common.to_tensor([0.02] * 3, device=self.device)

        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        sphere = actors.build_sphere(
            self.scene,
            radius=self.SPHERE_RADIUS,
            color=np.array([12, 42, 160, 255]) / 255,  # Blue
            name="cubeA",
            body_type="dynamic",
        )
        self.cubeA = sphere

        bin = self._build_bin()
        self.cubeB = bin

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Borrow logic from PlaceSphere, spawning sphere and bin instead of cubes."""

        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xy_offset = torch.rand((b, 2)) * 0.2 - 0.1

            sampler = randomization.UniformPlacementSampler(
                bounds=self.SPAWN_REGION, batch_size=b, device=self.device
            )
            radius_bound = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.001

            sphere_xy = xy_offset + sampler.sample(radius_bound, 100)
            bin_xy = xy_offset + sampler.sample(radius_bound, 100, verbose=False)

            sphere_xyz = torch.zeros((b, 3))
            sphere_xyz[:, :2] = sphere_xy
            sphere_xyz[:, 2] = self.SPHERE_RADIUS

            sphere_qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False)
            self.cubeA.set_pose(Pose.create_from_pq(p=sphere_xyz, q=sphere_qs))

            bin_xyz = torch.zeros((b, 3))
            bin_xyz[:, :2] = bin_xy
            bin_xyz[:, 2] = self.BLOCK_HALF_SIZES[0]

            bin_qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False)
            self.cubeB.set_pose(Pose.create_from_pq(p=bin_xyz, q=bin_qs))

    def _get_obs_extra(self, info: dict):
        """Same as StackCube observations.

        This will be compatible with Adapters taking StackCube-like tensors only, e.g. compatible
        with NoOpAdapter, but not PlaceSphereToStackCubeAdapter
        """

        if self.agent is None:
            raise ValueError("Agent is not initialized yet. Cannot compute observations.")

        if not hasattr(self.agent, "tcp") or self.agent.tcp is None:
            raise ValueError("Agent does not have a TCP. Cannot compute observations.")

        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            fake_bin_p = self.cubeB.pose.p.clone()

            # NOTE: We must inject the Z-offset here so the policy doesn't try to
            # smash the sphere through the table. The bin is at 0.0025, but the policy
            # expects the target base (Cube B) to be at 0.020.
            fake_bin_p[:, 2] += 0.0175  # 0.020 (Expected) - 0.0025 (Actual)

            fake_cubeB_pose = torch.cat([fake_bin_p, self.cubeB.pose.q], dim=-1)

            obs.update(
                cubeA_pose=self.cubeA.pose.raw_pose,
                cubeB_pose=fake_cubeB_pose,
                tcp_to_cubeA_pos=self.cubeA.pose.p - self.agent.tcp.pose.p,
                tcp_to_cubeB_pos=fake_bin_p - self.agent.tcp.pose.p,
                cubeA_to_cubeB_pos=fake_bin_p - self.cubeA.pose.p,
            )
        return obs

    def evaluate(self):
        return {
            "success": torch.zeros(len(self.cubeA.pose.p), dtype=torch.bool, device=self.device)
        }

    def compute_dense_reward(self, info: dict, **kwargs):
        return torch.zeros(len(self.cubeA.pose.p), device=self.device)
