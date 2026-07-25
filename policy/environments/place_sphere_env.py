import warnings
from typing import Any, cast

import torch
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

    def generate_heuristic_goal(self) -> dict[str, Any]:
        """Generates a heuristic goal state based on the current observation.

        Heuristic:
        - Sphere (obj_pose) is placed in the bin (bin_pos)
        - TCP is positioned at the sphere, slightly above it
        - is_grasped is True (or 1.0)
        """
        obs_dict = cast(dict[str, Any], self._get_obs_state_dict(info={}))

        bin_pos: torch.Tensor = obs_dict["extra"]["bin_pos"]
        obj_pose: torch.Tensor = obs_dict["extra"]["obj_pose"]
        obj_quat = obj_pose[..., 3:7]

        goal_obj_pos = bin_pos.clone()
        goal_obj_quat = obj_quat.clone()

        goal_tcp_pos = goal_obj_pos.clone()
        goal_tcp_pos[..., 2] += 0.03  # 3cm above
        goal_tcp_quat = goal_obj_quat.clone()

        agent_qpos = torch.zeros_like(obs_dict["agent"]["qpos"])
        agent_qvel = torch.zeros_like(obs_dict["agent"]["qvel"])

        goal_obj_pose = torch.cat([goal_obj_pos, goal_obj_quat], dim=-1)
        goal_tcp_pose = torch.cat([goal_tcp_pos, goal_tcp_quat], dim=-1)
        is_grasped = torch.ones_like(obs_dict["extra"]["is_grasped"])
        tcp_to_obj_pos = goal_obj_pos - goal_tcp_pos

        return {
            "agent": {
                "qpos": agent_qpos,
                "qvel": agent_qvel,
            },
            "extra": {
                "is_grasped": is_grasped,
                "tcp_pose": goal_tcp_pose,
                "bin_pos": bin_pos.clone(),
                "obj_pose": goal_obj_pose,
                "tcp_to_obj_pos": tcp_to_obj_pos,
            },
        }
