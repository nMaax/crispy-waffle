import torch
from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
from mani_skill.utils.registration import register_env


@register_env("StackCubeSwapped-v1", max_episode_steps=50)
class StackCubeSwappedEnv(StackCubeEnv):
    """The goal is to pick up the green cube (Cube B) and stack it on top of the red cube (Cube A)
    and let go."""

    # NOTE: Stack cube swapped only modifies the success and reward conditions. It doesn't actually swap any cube
    # as it would be pointless. The task would be the same.

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

    def generate_heuristic_goal(self) -> torch.Tensor | dict:
        """Generates a heuristic goal state based on the current observation.

        Heuristic:
        - Cube B is on top of Cube A (same x,y, offset z by cube size)
        - Orientations of Cube B and TCP match Cube A
        - TCP is positioned exactly at Cube B
        """

        # Observation structure is just like in StackCube (assuming Panda robot with 18-dim proprioception):
        # 0:18   - Robot proprioception (qpos, qvel) - set to 0
        # 18:25  - TCP Pose (7)
        # 25:32  - Cube A Pose (7)
        # 32:39  - Cube B Pose (7)
        # 39:42  - tcp_to_cubeA_pos (3)
        # 42:45  - tcp_to_cubeB_pos (3)
        # 45:48  - cubeA_to_cubeB_pos (3)

        obs = torch.as_tensor(self.get_obs(), device=self.device)
        goal = torch.zeros_like(obs)

        # Goal: Cube A is stacked on top of Cube B
        cube_A_pose = obs[..., 25:32]
        cube_A_pos = cube_A_pose[..., :3]
        cube_A_quat = cube_A_pose[..., 3:7]

        goal_cube_A_pos = cube_A_pos.clone()
        goal_cube_A_quat = cube_A_quat.clone()

        goal_cube_B_pos = cube_A_pos.clone()
        goal_cube_B_pos[..., 2] += self.cube_half_size[2] * 2
        goal_cube_B_quat = cube_A_quat.clone()  # Keep same orientation for simplicity

        # Goal: TCP is at Cube B's position
        goal_tcp_pos = goal_cube_B_pos.clone()
        goal_tcp_quat = goal_cube_B_quat.clone()

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
