from __future__ import annotations

from collections.abc import Mapping

import torch

from policy.utils import get_total_dim
from policy.utils.typing_utils import DimSpec

# Dimensionality constants
POSE_DIM: int = 7  # 3D position + 4D quaternion [x, y, z, qw, qx, qy, qz]
RELATIVE_SE3_DIM: int = 6  # 3D position delta + 3D axis-angle rotation vector
ROLE_DIM: int = 4  # 4D one-hot vector [is_tcp, is_pick, is_target, is_clutter]
PROPRIO_DIM: int = 18  # 9D joint positions (qpos) + 9D joint velocities (qvel)

# Role one-hot vectors
Role = tuple[float, float, float, float]
ROLE_TCP: Role = (1.0, 0.0, 0.0, 0.0)
ROLE_PICK: Role = (0.0, 1.0, 0.0, 0.0)
ROLE_TARGET: Role = (0.0, 0.0, 1.0, 0.0)
ROLE_CLUTTER: Role = (0.0, 0.0, 0.0, 1.0)


def canonical_dim_spec(num_objects: int = 2) -> dict[str, int]:
    """Generates the canonical dimension specification dictionary for a given object count."""
    spec: dict[str, int] = {"proprio": PROPRIO_DIM, "tcp_pose": POSE_DIM, "tcp_role": ROLE_DIM}
    for i in range(num_objects):
        spec[f"obj_{i}_pose"] = POSE_DIM
        spec[f"obj_{i}_role"] = ROLE_DIM
    return spec


def canonical_normalization_mask(spec: DimSpec) -> dict[str, torch.Tensor]:
    """Marks the normalizable channels of a canonical observation spec.

    Role keys are one-hot indicators, so an affine rescale would destroy them (a constant one-hot
    z-scores to exactly zero); every other canonical key is normalizable in full.
    """
    if not isinstance(spec, Mapping):
        raise TypeError(
            f"canonical_normalization_mask expects a canonical dict spec, got {type(spec).__name__}."
        )

    return {
        key: torch.full((get_total_dim(dim),), not key.endswith("_role"), dtype=torch.bool)
        for key, dim in spec.items()
    }
