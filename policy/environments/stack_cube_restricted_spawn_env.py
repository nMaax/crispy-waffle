from mani_skill.envs.tasks.tabletop.stack_cube_restricted_spawn import (
    StackCubeRestrictedSpawnEnv as ManiSkillStackCubeRestrictedSpawnEnv,
)
from mani_skill.utils.registration import register_env

from policy.environments.stack_cube_env import StackCubeEnv


@register_env("StackCubeRestrictedSpawn-v1", max_episode_steps=250, override=True)
class StackCubeRestrictedSpawnEnv(ManiSkillStackCubeRestrictedSpawnEnv):
    generate_heuristic_goal = StackCubeEnv.generate_heuristic_goal
