import torch
from mani_skill.envs.tasks.tabletop.place_sphere import PlaceSphereEnv
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose


@register_env("PlaceSphereWithCubes-v1", max_episode_steps=50)
class PlaceSphereWithCubesEnv(PlaceSphereEnv):
    def _load_scene(self, options: dict):

        # XXX: Cube half size??

        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.obj = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[1, 0, 0, 1],
            name="cubeA",
            body_type="dynamic",
            # initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )

        self.bin = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[0, 1, 0, 1],
            name="cubeB",
            body_type="kinematic",
            # initial_pose=sapien.Pose(p=[1, 0, 0.1]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # XXX: in StackCube they make a region from a radius??

            # Spawn Cube A using PlaceSphere's original X/Y bounds
            xyz = torch.zeros((b, 3))
            xyz[..., 0] = (torch.rand((b, 1)) * 0.05 - 0.1)[..., 0]  # [-0.1, -0.05]
            xyz[..., 1] = (torch.rand((b, 1)) * 0.2 - 0.1)[..., 0]  # [-0.1, 0.1]
            xyz[..., 2] = 0.02  # Cube resting on table (half-size)
            q = [1, 0, 0, 0]
            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.obj.set_pose(obj_pose)

            # Spawn Cube B using PlaceSphere's original X/Y bounds
            pos = torch.zeros((b, 3))
            pos[:, 0] = torch.rand((b, 1))[..., 0] * 0.1  # [0, 0.1]
            pos[:, 1] = torch.rand((b, 1))[..., 0] * 0.2 - 0.1  # [-0.1, 0.1]
            pos[:, 2] = 0.02  # Cube resting on table (half-size)
            bin_pose = Pose.create_from_pq(p=pos, q=q)
            self.bin.set_pose(bin_pose)

    def _get_obs_extra(self, info: dict):
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
