from typing import Any, cast

from mani_skill.envs.tasks.tabletop.stack_cube_clutter import (
    StackCubeClutterEnv as ManiSkillStackCubeClutterEnv,
)
from mani_skill.utils.registration import register_env

from policy.environments.stack_cube_env import StackCubeEnv


@register_env("StackCubeClutter-v1", max_episode_steps=250, override=True)
class StackCubeClutterEnv(ManiSkillStackCubeClutterEnv):
    def generate_heuristic_goal(self) -> dict[str, Any]:
        """Generates a heuristic goal state based on the current observation.

        Heuristic:
        - Cube A is on top of Cube B (same x,y, offset z by cube size)
        - Orientations of Cube A and TCP match Cube B
        - TCP is positioned at Cube A
        - Clutter objects remain in place
        """
        obs_dict = cast(dict[str, Any], self._get_obs_state_dict(info={}))
        base_goal = StackCubeEnv.generate_heuristic_goal(cast(Any, self))

        # Preserve clutter object poses and active flags
        for i in range(self.NUM_CLUTTER_SLOTS):
            pose_key = f"obj_{i}_pose"
            active_key = f"obj_{i}_active"
            if pose_key in obs_dict.get("extra", {}):
                base_goal["extra"][pose_key] = obs_dict["extra"][pose_key]
            if active_key in obs_dict.get("extra", {}):
                base_goal["extra"][active_key] = obs_dict["extra"][active_key]

        return base_goal
