from __future__ import annotations

import torch
from mani_skill.utils.geometry.rotation_conversions import (
    quaternion_invert,
    quaternion_multiply,
    quaternion_to_axis_angle,
)

from policy.transforms.canonicalization.spec import POSE_DIM


def relative_se3_pose(g: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    """Computes the 6D SE(3) pose delta (delta-position(3) + rotation-vector(3)) from 7D pose ``o``
    to 7D pose ``g``."""
    if g.shape[-1] != POSE_DIM or o.shape[-1] != POSE_DIM:
        raise ValueError(
            f"relative_se3_pose() expects last dim of goal g and observation o to be {POSE_DIM} (pos + quat), "
            f"got {g.shape=} and {o.shape=}."
        )
    delta_pos = g[..., :3] - o[..., :3]
    q_rel = quaternion_multiply(g[..., 3:7], quaternion_invert(o[..., 3:7]))
    rotvec = quaternion_to_axis_angle(q_rel)
    return torch.cat([delta_pos, rotvec], dim=-1)
