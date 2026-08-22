from typing import Any, cast

import torch
from mani_skill.envs.tasks.tabletop.stack_cube_clutter_random_pick import (
    StackCubeClutterRandomPickEnv as ManiSkillStackCubeClutterRandomPickEnv,
)
from mani_skill.utils.registration import register_env


@register_env("StackCubeClutterRandomPick-v1", max_episode_steps=250, override=True)
class StackCubeClutterRandomPickEnv(ManiSkillStackCubeClutterRandomPickEnv):
    def generate_heuristic_goal(self) -> dict[str, Any]:
        """Generates a heuristic goal state based on the current observation.

        Heuristic:
        - Pick object is placed on top of Target object (same x,y, offset z by rest heights)
        - Orientations of Pick object and TCP match Target object
        - TCP is positioned at the final Pick object position
        - All other objects remain in their initial resting positions
        """
        info = self.evaluate()
        obs_dict = cast(dict[str, Any], self._get_obs_state_dict(info=info))
        extra = obs_dict["extra"]
        pool_size = len(self.pool_objects)

        pick_idx = info["pick_idx"]
        target_pos = info["target_pose"][..., :3]
        target_quat = info["target_pose"][..., 3:7]

        pick_rest_z = info["pick_rest_z"]
        target_rest_z = info["target_rest_z"]

        goal_pick_pos = target_pos.clone()
        goal_pick_pos[..., 2] += pick_rest_z + target_rest_z
        goal_pick_quat = target_quat.clone()

        goal_tcp_pos = goal_pick_pos.clone()
        goal_tcp_pos[..., 2] += 0.03
        goal_tcp_quat = extra["tcp_pose"][..., 3:7].clone()

        goal_tcp_pose = torch.cat([goal_tcp_pos, goal_tcp_quat], dim=-1)
        goal_pick_pose = torch.cat([goal_pick_pos, goal_pick_quat], dim=-1)

        agent_qpos = torch.zeros_like(obs_dict["agent"]["qpos"])
        agent_qvel = torch.zeros_like(obs_dict["agent"]["qvel"])

        goal_extra: dict[str, Any] = dict(extra)
        goal_extra["tcp_pose"] = goal_tcp_pose
        for i in range(pool_size):
            is_pick_slot = (pick_idx == i).unsqueeze(-1)
            goal_extra[f"obj_{i}_pose"] = torch.where(
                is_pick_slot, goal_pick_pose, extra[f"obj_{i}_pose"]
            )

        return {
            "agent": {
                "qpos": agent_qpos,
                "qvel": agent_qvel,
            },
            "extra": goal_extra,
        }
