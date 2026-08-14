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

        # Find pick and target slot indices from observations
        pick_slot = 0
        target_slot = 1
        for i in range(pool_size):
            is_pick = extra.get(f"obj_{i}_is_pick")
            if is_pick is not None and torch.any(is_pick):
                pick_slot = i
            is_target = extra.get(f"obj_{i}_is_target")
            if is_target is not None and torch.any(is_target):
                target_slot = i

        target_pose = extra[f"obj_{target_slot}_pose"]

        target_pos = target_pose[..., :3]
        target_quat = target_pose[..., 3:7]

        pick_rest_z = self.pool_rest_z[pick_slot]
        target_rest_z = self.pool_rest_z[target_slot]

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
        goal_extra[f"obj_{pick_slot}_pose"] = goal_pick_pose

        return {
            "agent": {
                "qpos": agent_qpos,
                "qvel": agent_qvel,
            },
            "extra": goal_extra,
        }
