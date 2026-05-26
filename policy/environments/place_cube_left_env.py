import torch
from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
from mani_skill.utils.registration import register_env


@register_env("PlaceCubeLeft-v1", max_episode_steps=50)
class PlaceCubeLeftEnv(StackCubeEnv):
    def evaluate(self):
        pos_A = self.cubeA.pose.p
        pos_B = self.cubeB.pose.p

        target_y_offset = 0.08

        is_on_table = torch.abs(pos_A[:, 2] - pos_B[:, 2]) < 0.01

        is_x_aligned = torch.abs(pos_A[:, 0] - pos_B[:, 0]) < 0.02

        is_y_left = torch.abs(pos_A[:, 1] - (pos_B[:, 1] + target_y_offset)) < 0.02

        is_placed = is_on_table & is_x_aligned & is_y_left

        is_obj_static = self.cubeA.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_cubeA_grasped = self.agent.is_grasping(self.cubeA)

        success = is_placed & is_obj_static & (~is_cubeA_grasped)

        return {
            "is_on_table": is_on_table,
            "is_x_aligned": is_x_aligned,
            "is_y_left": is_y_left,
            "is_placed": is_placed,
            "is_obj_static": is_obj_static,
            "is_cubeA_grasped": is_cubeA_grasped,
            "success": success.bool(),
        }

    def compute_dense_reward(self, obs, action, info):
        # reaching reward
        tcp_pose = self.agent.tcp.pose.p
        cubeA_pos = self.cubeA.pose.p
        cubeA_to_tcp_dist = torch.linalg.norm(tcp_pose - cubeA_pos, axis=1)
        reward = 2 * (1 - torch.tanh(5 * cubeA_to_tcp_dist))

        # grasp and place reward
        cubeB_pos = self.cubeB.pose.p
        target_pos = cubeB_pos.clone()
        target_pos[:, 1] += 0.08  # Target is 8cm to the left
        target_pos[:, 2] = cubeB_pos[:, 2]  # Keep Z at table height

        cubeA_to_target_dist = torch.linalg.norm(target_pos - cubeA_pos, axis=1)
        place_reward = 1 - torch.tanh(5.0 * cubeA_to_target_dist)

        reward[info["is_cubeA_grasped"]] = (4 + place_reward)[info["is_cubeA_grasped"]]

        # ungrasp and static reward
        gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(self.device)
        is_cubeA_grasped = info["is_cubeA_grasped"]
        ungrasp_reward = (
            torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        )
        ungrasp_reward[~is_cubeA_grasped] = 1.0
        v = torch.linalg.norm(self.cubeA.linear_velocity, axis=1)
        av = torch.linalg.norm(self.cubeA.angular_velocity, axis=1)
        static_reward = 1 - torch.tanh(v * 10 + av)
        reward[info["is_placed"]] = (6 + (ungrasp_reward + static_reward) / 2.0)[
            info["is_placed"]
        ]

        reward[info["success"]] = 8

        return reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 8
