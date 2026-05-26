import torch
from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv as ManiSkillStackCubeEnv
from mani_skill.utils.registration import register_env


@register_env("StackCube-v1", max_episode_steps=50, override=True)
class StackCubeEnv(ManiSkillStackCubeEnv):
    def generate_heuristic_goal(self) -> torch.Tensor | dict:
        """Generates a heuristic goal state based on the current observation.

        Heuristic:
        - Cube A is on top of Cube B (same x,y, offset z by cube size)
        - Orientations of Cube A and TCP match Cube B
        - TCP is positioned exactly at Cube A
        """

        # In StackCube, the goal is to stack Cube A (red) on Cube B (green)
        #
        # Observation structure (assuming Panda robot with 18-dim proprioception):
        # 0:18   - Robot proprioception (qpos, qvel) - set to 0
        # 18:25  - TCP Pose (7)
        # 25:32  - Cube A Pose (7)
        # 32:39  - Cube B Pose (7)
        # 39:42  - tcp_to_cubeA_pos (3)
        # 42:45  - tcp_to_cubeB_pos (3)
        # 45:48  - cubeA_to_cubeB_pos (3)

        obs = torch.as_tensor(self.get_obs(), device=self.device)
        goal = torch.zeros_like(obs)

        # Goal: Cube A is stacked (+z) on top of Cube B
        cube_B_pose = obs[..., 25:32]
        cube_B_pos = cube_B_pose[..., :3]
        cube_B_quat = cube_B_pose[..., 3:7]

        goal_cube_B_pos = cube_B_pos.clone()
        goal_cube_B_quat = cube_B_quat.clone()

        goal_cube_A_pos = cube_B_pos.clone()
        goal_cube_A_pos[..., 2] += self.cube_half_size[2] * 2
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
