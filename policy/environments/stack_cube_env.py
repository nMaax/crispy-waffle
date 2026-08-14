from typing import Any, cast

import torch
from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv as ManiSkillStackCubeEnv
from mani_skill.utils.registration import register_env


@register_env("StackCube-v1", max_episode_steps=250, override=True)
class StackCubeEnv(ManiSkillStackCubeEnv):
    def generate_heuristic_goal(self) -> dict[str, Any]:
        """Generates a heuristic goal state based on the current observation.

        Heuristic:
        - Cube A is on top of Cube B (same x,y, offset z by cube size)
        - Orientations of Cube A and TCP match Cube B
        - TCP is positioned exactly at Cube A
        """
        obs_dict = cast(dict[str, Any], self._get_obs_state_dict(info={}))

        cube_half_size = torch.as_tensor(self.cube_half_size, device=self.device)

        cube_A_pose: torch.Tensor = obs_dict["extra"]["cubeA_pose"]
        cube_A_quat = cube_A_pose[..., 3:7]

        cube_B_pose: torch.Tensor = obs_dict["extra"]["cubeB_pose"]
        cube_B_pos = cube_B_pose[..., :3]
        cube_B_quat = cube_B_pose[..., 3:7]

        goal_cube_B_pos = cube_B_pos.clone()
        goal_cube_B_quat = cube_B_quat.clone()

        goal_cube_A_pos = cube_B_pos.clone()
        goal_cube_A_pos[..., 2] += cube_half_size[2] * 2
        goal_cube_A_quat = cube_A_quat.clone()

        # Goal: TCP is at Cube A's position, slightly above
        goal_tcp_pos = goal_cube_A_pos.clone()
        goal_tcp_pos[..., 2] += 0.03  # Just 3cm above the cube
        goal_tcp_quat = obs_dict["extra"]["tcp_pose"][..., 3:7].clone()

        agent_qpos = torch.zeros_like(obs_dict["agent"]["qpos"])
        agent_qvel = torch.zeros_like(obs_dict["agent"]["qvel"])

        goal_tcp_pose = torch.cat([goal_tcp_pos, goal_tcp_quat], dim=-1)
        goal_cubeA_pose = torch.cat([goal_cube_A_pos, goal_cube_A_quat], dim=-1)
        goal_cubeB_pose = torch.cat([goal_cube_B_pos, goal_cube_B_quat], dim=-1)

        tcp_to_cubeA_pos = goal_cube_A_pos - goal_tcp_pos
        tcp_to_cubeB_pos = goal_cube_B_pos - goal_tcp_pos
        cubeA_to_cubeB_pos = goal_cube_B_pos - goal_cube_A_pos

        return {
            "agent": {
                "qpos": agent_qpos,
                "qvel": agent_qvel,
            },
            "extra": {
                "tcp_pose": goal_tcp_pose,
                "cubeA_pose": goal_cubeA_pose,
                "cubeB_pose": goal_cubeB_pose,
                "tcp_to_cubeA_pos": tcp_to_cubeA_pos,
                "tcp_to_cubeB_pos": tcp_to_cubeB_pos,
                "cubeA_to_cubeB_pos": cubeA_to_cubeB_pos,
            },
        }
