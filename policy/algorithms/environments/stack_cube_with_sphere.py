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
    """
    Sanity check environment: Uses StackCube-v1 spatial boundaries and observations,
    but physically spawns the PlaceSphere-v1 actors (Sphere and Bin).
    """

    # --- PlaceSphere Constants ---
    radius = 0.02
    inner_side_half_len = 0.02
    short_side_half_size = 0.0025
    block_half_size = [
        short_side_half_size,
        2 * short_side_half_size + inner_side_half_len,
        2 * short_side_half_size + inner_side_half_len,
    ]
    edge_block_half_size = [
        short_side_half_size,
        2 * short_side_half_size + inner_side_half_len,
        2 * short_side_half_size,
    ]

    def _build_bin(self):
        """Helper to build the bin exactly as PlaceSphere does."""
        builder = self.scene.create_actor_builder()
        dx = self.block_half_size[1] - self.block_half_size[0]
        dy = self.block_half_size[1] - self.block_half_size[0]
        dz = self.edge_block_half_size[2] + self.block_half_size[0]

        poses = [
            sapien.Pose([0, 0, 0]),
            sapien.Pose([-dx, 0, dz]),
            sapien.Pose([dx, 0, dz]),
            sapien.Pose([0, -dy, dz]),
            sapien.Pose([0, dy, dz]),
        ]
        half_sizes = [
            [self.block_half_size[1], self.block_half_size[2], self.block_half_size[0]],
            self.edge_block_half_size,
            self.edge_block_half_size,
            [
                self.edge_block_half_size[1],
                self.edge_block_half_size[0],
                self.edge_block_half_size[2],
            ],
            [
                self.edge_block_half_size[1],
                self.edge_block_half_size[0],
                self.edge_block_half_size[2],
            ],
        ]
        for pose, half_size in zip(poses, half_sizes):
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size)

        return builder.build_kinematic(name="bin")

    def _load_scene(self, options: dict):
        # 1. Define the cube half size (StackCube's reward/evaluate functions still expect this variable to exist)
        self.cube_half_size = common.to_tensor([0.02] * 3, device=self.device)

        # 2. Load the table manually (bypassing super()._load_scene)
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # 3. Load Sphere as cubeA
        self.cubeA = actors.build_sphere(
            self.scene,
            radius=self.radius,
            color=np.array([12, 42, 160, 255]) / 255,
            name="cubeA",
            body_type="dynamic",
        )

        # 4. Load Bin as cubeB
        self.cubeB = self._build_bin()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # --- STACK CUBE SPATIAL SAMPLING ---
            xyz = torch.zeros((b, 3))
            xy = torch.rand((b, 2)) * 0.2 - 0.1
            region = [[-0.1, -0.2], [0.1, 0.2]]
            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )
            radius_bound = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.001

            # These are guaranteed to be in the "comfort zone" of the policy
            cubeA_xy = xy + sampler.sample(radius_bound, 100)
            cubeB_xy = xy + sampler.sample(radius_bound, 100, verbose=False)

            # --- ACTOR PLACEMENT ---
            # Place the Sphere
            xyz[:, :2] = cubeA_xy
            xyz[:, 2] = self.radius  # 0.02
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False)
            self.cubeA.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            # Place the Bin
            xyz[:, :2] = cubeB_xy
            xyz[:, 2] = self.block_half_size[0]  # 0.0025 (Flush with table)
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False)
            self.cubeB.set_pose(Pose.create_from_pq(p=xyz, q=qs))

    def _get_obs_extra(self, info: dict):
        """Generates the StackCube observation dict.

        CRITICAL: We must inject the Z-offset here so the policy doesn't try to
        smash the sphere through the table. The bin is at 0.0025, but the policy
        expects the target base (Cube B) to be at 0.020.
        """
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            # Apply the mathematical Z-offset you discovered earlier
            fake_bin_p = self.cubeB.pose.p.clone()
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
        # Dummy evaluate to prevent evaluation crashes
        return {
            "success": torch.zeros(len(self.cubeA.pose.p), dtype=torch.bool, device=self.device)
        }

    def compute_dense_reward(self, info: dict, **kwargs):
        return torch.zeros(len(self.cubeA.pose.p), device=self.device)
