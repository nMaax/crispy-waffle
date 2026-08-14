from mani_skill.envs.tasks.tabletop.stack_cube_clutter_locked_rotation import (
    StackCubeClutterLockedRotationEnv as ManiSkillStackCubeClutterLockedRotationEnv,
)
from mani_skill.utils.registration import register_env

from policy.environments.stack_cube_clutter_env import StackCubeClutterEnv


@register_env("StackCubeClutterLockedRotation-v1", max_episode_steps=250, override=True)
class StackCubeClutterLockedRotationEnv(ManiSkillStackCubeClutterLockedRotationEnv):
    generate_heuristic_goal = StackCubeClutterEnv.generate_heuristic_goal
