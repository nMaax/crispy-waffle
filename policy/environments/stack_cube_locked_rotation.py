from mani_skill.envs.tasks.tabletop.stack_cube_locked_rotation import (
    StackCubeLockedRotationEnv as ManiSkillStackCubeLockedRotationEnv,
)
from mani_skill.utils.registration import register_env

from policy.environments.stack_cube_env import StackCubeEnv


@register_env("StackCubeLockedRotation-v1", max_episode_steps=50, override=True)
class StackCubeLockedRotationEnv(ManiSkillStackCubeLockedRotationEnv):
    # Same observation layout as StackCube (see StackCubeEnv.STATE_SCHEMA).
    STATE_SCHEMA = StackCubeEnv.STATE_SCHEMA
