import warnings

from mani_skill.envs.tasks.tabletop.place_sphere import PlaceSphereEnv
from mani_skill.utils.registration import register_env


@register_env("PlaceSphereWristcam-v1", max_episode_steps=50)
class PlaceSphereWristcamEnv(PlaceSphereEnv):
    """PlaceSphere-v1 but with the "panda_wristcam" robot instead of "panda"."""

    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

        # NOTE: Officially PlaceSphere does not support "panda_wristcam", but "panda" and "fetch" only
        # however if I pass it, it will still accept it. So I just notify it and move on with my life
        if robot_uids not in self.SUPPORTED_ROBOTS:
            warnings.warn(
                f"Unsupported robot_uids: {robot_uids}. Supported: {self.SUPPORTED_ROBOTS}. "
                "However it may still work, so we let it pass."
            )
