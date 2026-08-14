from mani_skill.envs.tasks.tabletop.stack_cube_clutter_random_pick_locked_rotation import (
    StackCubeClutterRandomPickLockedRotationEnv as ManiSkillStackCubeClutterRandomPickLockedRotationEnv,
)
from mani_skill.utils.registration import register_env

from policy.environments.stack_cube_clutter_random_pick_env import (
    StackCubeClutterRandomPickEnv,
)


@register_env("StackCubeClutterRandomPickLockedRotation-v1", max_episode_steps=250, override=True)
class StackCubeClutterRandomPickLockedRotationEnv(
    ManiSkillStackCubeClutterRandomPickLockedRotationEnv
):
    generate_heuristic_goal = StackCubeClutterRandomPickEnv.generate_heuristic_goal
