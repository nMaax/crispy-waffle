from mani_skill.envs.tasks.tabletop.place_cube_right_locked_rotation import (
    PlaceCubeRightLockedRotationEnv as ManiSkillPlaceCubeRightLockedRotationEnv,
)
from mani_skill.utils.registration import register_env

from policy.environments.place_cube_left_env import PlaceCubeLeftEnv


@register_env("PlaceCubeRightLockedRotation-v1", max_episode_steps=250, override=True)
class PlaceCubeRightLockedRotationEnv(ManiSkillPlaceCubeRightLockedRotationEnv):
    generate_heuristic_goal = PlaceCubeLeftEnv.generate_heuristic_goal
