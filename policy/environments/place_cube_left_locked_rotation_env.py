from mani_skill.envs.tasks.tabletop.place_cube_left_locked_rotation import (
    PlaceCubeLeftLockedRotationEnv as ManiSkillPlaceCubeLeftLockedRotationEnv,
)
from mani_skill.utils.registration import register_env

from policy.environments.place_cube_left_env import PlaceCubeLeftEnv
from policy.environments.stack_cube_env import StackCubeEnv


@register_env("PlaceCubeLeftLockedRotation-v1", max_episode_steps=250, override=True)
class PlaceCubeLeftLockedRotationEnv(ManiSkillPlaceCubeLeftLockedRotationEnv):
    STATE_SCHEMA = StackCubeEnv.STATE_SCHEMA
    generate_heuristic_goal = PlaceCubeLeftEnv.generate_heuristic_goal
