from mani_skill.envs.tasks.tabletop.place_sphere import (
    PlaceSphereEnv as ManiSkillPlaceSphereEnv,
)
from mani_skill.utils.registration import register_env


@register_env("PlaceSphere-v1", max_episode_steps=50, override=True)
class PlaceSphereEnv(ManiSkillPlaceSphereEnv):
    STATE_SCHEMA = {
        "agent": {
            "qpos": (0, 9),
            "qvel": (9, 18),
        },
        "extra": {
            "is_grasped": (18, 19),
            "tcp_pose": (19, 26),
            "bin_pos": (26, 29),
            "obj_pose": (29, 36),
            "tcp_to_obj_pos": (36, 39),
        },
    }
