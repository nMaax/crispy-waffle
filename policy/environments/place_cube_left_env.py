import torch
from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
from mani_skill.utils.registration import register_env


@register_env("PlaceCubeLeft-v1", max_episode_steps=50)
class PlaceCubeLeftEnv(StackCubeEnv):
    Y_OFFSET = 0.08

    def evaluate(self):
        pos_A = self.cubeA.pose.p
        pos_B = self.cubeB.pose.p

        is_on_table = torch.abs(pos_A[:, 2] - pos_B[:, 2]) < 0.01

        is_x_aligned = torch.abs(pos_A[:, 0] - pos_B[:, 0]) < 0.02

        is_y_left = torch.abs(pos_A[:, 1] - (pos_B[:, 1] + self.Y_OFFSET)) < 0.02

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
        ungrasp_reward = torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        ungrasp_reward[~is_cubeA_grasped] = 1.0
        v = torch.linalg.norm(self.cubeA.linear_velocity, axis=1)
        av = torch.linalg.norm(self.cubeA.angular_velocity, axis=1)
        static_reward = 1 - torch.tanh(v * 10 + av)
        reward[info["is_placed"]] = (6 + (ungrasp_reward + static_reward) / 2.0)[info["is_placed"]]

        reward[info["success"]] = 8

        return reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 8

    def generate_heuristic_goal(self) -> torch.Tensor | dict:
        """Generates a heuristic goal state based on the current observation.

        Heuristic:
        - Cube A is on the left of Cube B (same x,z, offset y by some small distance)
        - Orientations of Cube A and TCP match Cube B
        - TCP is positioned exactly at Cube A
        """

        # Observation structure is just like StackCube (assuming Panda robot with 18-dim proprioception):
        # 0:18   - Robot proprioception (qpos, qvel) - set to 0
        # 18:25  - TCP Pose (7)
        # 25:32  - Cube A Pose (7)
        # 32:39  - Cube B Pose (7)
        # 39:42  - tcp_to_cubeA_pos (3)
        # 42:45  - tcp_to_cubeB_pos (3)
        # 45:48  - cubeA_to_cubeB_pos (3)

        obs = torch.as_tensor(self.get_obs(), device=self.device)
        goal = torch.zeros_like(obs)

        # Goal: Cube A is on the left (+y) of Cube B
        cube_B_pose = obs[..., 32:39]
        cube_B_pos = cube_B_pose[..., :3]
        cube_B_quat = cube_B_pose[..., 3:7]

        goal_cube_B_pos = cube_B_pos.clone()
        goal_cube_B_quat = cube_B_quat.clone()

        goal_cube_A_pos = cube_B_pos.clone()
        goal_cube_A_pos[..., 1] += self.Y_OFFSET  # This will be roughly 8cm
        goal_cube_A_quat = cube_B_quat.clone()  # Keep same orientation for simplicity

        # Goal: TCP is at Cube A's position, slightly above
        goal_tcp_pos = goal_cube_A_pos.clone()
        goal_tcp_pos[..., 2] += 0.03  # Just 3cm above the cube
        goal_tcp_quat = goal_cube_A_quat.clone()

        # Fill goal state
        goal[..., 18:21] = goal_tcp_pos
        goal[..., 21:25] = goal_tcp_quat
        goal[..., 25:28] = goal_cube_A_pos
        goal[..., 28:32] = goal_cube_A_quat
        goal[..., 32:35] = goal_cube_B_pos
        goal[..., 35:39] = goal_cube_B_quat

        # Relative positions
        # tcp_to_cubeA_pos = cubeA.p - tcp.p
        goal[..., 39:42] = goal_cube_A_pos - goal_tcp_pos
        # tcp_to_cubeB_pos = cubeB.p - tcp.p
        goal[..., 42:45] = goal_cube_B_pos - goal_tcp_pos
        # cubeA_to_cubeB_pos = cubeB.p - cubeA.p
        goal[..., 45:48] = goal_cube_B_pos - goal_cube_A_pos

        return goal
