import torch
from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
from mani_skill.utils.registration import register_env


@register_env("StackCubeSwapped-v1", max_episode_steps=50)
class StackCubeSwappedEnv(StackCubeEnv):
    """The goal is to pick up the green cube (Cube B) and stack it on top of the red cube (Cube A)
    and let go."""

    def evaluate(self):
        pos_A = self.cubeA.pose.p
        pos_B = self.cubeB.pose.p

        offset = pos_B - pos_A

        xy_flag = (
            torch.linalg.norm(offset[..., :2], axis=1)
            <= torch.linalg.norm(self.cube_half_size[:2]) + 0.005
        )
        z_flag = torch.abs(offset[..., 2] - self.cube_half_size[..., 2] * 2) <= 0.005
        is_cubeB_on_cubeA = torch.logical_and(xy_flag, z_flag)

        is_cubeB_static = self.cubeB.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_cubeB_grasped = self.agent.is_grasping(self.cubeB)

        success = is_cubeB_on_cubeA * is_cubeB_static * (~is_cubeB_grasped)

        return {
            "is_cubeB_grasped": is_cubeB_grasped,
            "is_cubeB_on_cubeA": is_cubeB_on_cubeA,
            "is_cubeB_static": is_cubeB_static,
            "success": success.bool(),
        }

    def compute_dense_reward(self, obs, action, info):
        # reaching reward (robot to Cube B)
        tcp_pose = self.agent.tcp.pose.p
        cubeB_pos = self.cubeB.pose.p
        cubeB_to_tcp_dist = torch.linalg.norm(tcp_pose - cubeB_pos, axis=1)
        reward = 2 * (1 - torch.tanh(5 * cubeB_to_tcp_dist))

        # grasp and place reward (Cube B onto Cube A)
        cubeA_pos = self.cubeA.pose.p
        goal_xyz = torch.hstack(
            [cubeA_pos[:, 0:2], (cubeA_pos[:, 2] + self.cube_half_size[2] * 2)[:, None]]
        )
        cubeB_to_goal_dist = torch.linalg.norm(goal_xyz - cubeB_pos, axis=1)
        place_reward = 1 - torch.tanh(5.0 * cubeB_to_goal_dist)

        reward[info["is_cubeB_grasped"]] = (4 + place_reward)[info["is_cubeB_grasped"]]

        # ungrasp and static reward
        gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(self.device)
        is_cubeB_grasped = info["is_cubeB_grasped"]
        ungrasp_reward = torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        ungrasp_reward[~is_cubeB_grasped] = 1.0

        v = torch.linalg.norm(self.cubeB.linear_velocity, axis=1)
        av = torch.linalg.norm(self.cubeB.angular_velocity, axis=1)
        static_reward = 1 - torch.tanh(v * 10 + av)

        reward[info["is_cubeB_on_cubeA"]] = (6 + (ungrasp_reward + static_reward) / 2.0)[
            info["is_cubeB_on_cubeA"]
        ]

        reward[info["success"]] = 8

        return reward
