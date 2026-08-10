from mani_skill.envs.tasks.tabletop.stack_cube_swapped_locked_rotation import (
    StackCubeSwappedLockedRotationEnv as ManiSkillStackCubeSwappedLockedRotationEnv,
)
from mani_skill.utils.registration import register_env

from policy.environments.stack_cube_env import StackCubeEnv
from policy.environments.stack_cube_swapped_env import StackCubeSwappedEnv


@register_env("StackCubeSwappedLockedRotation-v1", max_episode_steps=250, override=True)
class StackCubeSwappedLockedRotationEnv(ManiSkillStackCubeSwappedLockedRotationEnv):
    STATE_SCHEMA = StackCubeEnv.STATE_SCHEMA
    generate_heuristic_goal = StackCubeSwappedEnv.generate_heuristic_goal
