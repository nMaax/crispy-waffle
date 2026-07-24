import warnings

from mani_skill.envs.tasks.tabletop.place_sphere_restricted_spawn import (
    PlaceSphereRestrictedSpawnEnv as ManiSkillPlaceSphereRestrictedSpawnEnv,
)
from mani_skill.utils.registration import register_env

from .place_sphere_env import PlaceSphereEnv


@register_env("PlaceSphereRestrictedSpawn-v1", max_episode_steps=50, override=True)
class PlaceSphereRestrictedSpawnEnv(ManiSkillPlaceSphereRestrictedSpawnEnv):
    STATE_SCHEMA = PlaceSphereEnv.STATE_SCHEMA

    def __init__(self, *args, robot_uids="panda_wristcam", **kwargs):
        if robot_uids != "panda_wristcam":
            warnings.warn(
                f"Initializing {self.__class__.__name__} with robot_uids='{robot_uids}'. "
                "Note that PlaceSphere environments in this codebase were trained with "
                "robot_uids='panda_wristcam'; results may differ when using a different robot UID.",
                UserWarning,
                stacklevel=2,
            )
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
