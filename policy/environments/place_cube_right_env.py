from mani_skill.envs.tasks.tabletop.place_cube_right import (
    PlaceCubeRightEnv as ManiSkillPlaceCubeRightEnv,
)
from mani_skill.utils.registration import register_env

from policy.environments.place_cube_left_env import PlaceCubeLeftEnv
from policy.environments.stack_cube_env import StackCubeEnv


@register_env("PlaceCubeRight-v1", max_episode_steps=250, override=True)
class PlaceCubeRightEnv(ManiSkillPlaceCubeRightEnv):
    STATE_SCHEMA = StackCubeEnv.STATE_SCHEMA
    # The heuristic goal is offset-agnostic: it reads ``TARGET_Y_OFFSET`` from the
    # upstream task, which is negative (right) here and positive (left) there.
    generate_heuristic_goal = PlaceCubeLeftEnv.generate_heuristic_goal
