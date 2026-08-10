from mani_skill.envs.tasks.tabletop.stack_cube_locked_rotation import (
    StackCubeLockedRotationEnv as ManiSkillStackCubeLockedRotationEnv,
)
from mani_skill.utils.registration import register_env

from policy.environments.stack_cube_env import StackCubeEnv


@register_env("StackCubeLockedRotation-v1", max_episode_steps=250, override=True)
class StackCubeLockedRotationEnv(ManiSkillStackCubeLockedRotationEnv):
    STATE_SCHEMA = StackCubeEnv.STATE_SCHEMA
    generate_heuristic_goal = StackCubeEnv.generate_heuristic_goal
