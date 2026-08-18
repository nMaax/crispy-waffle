from __future__ import annotations

from collections.abc import Mapping

import torch

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

# The TCP occupies pool slot 0, ahead of every scene object.
TCP_SLOT: int = 0


def canonical_dim_spec(num_objects: int = 2) -> dict[str, DimSpec]:
    """Generates the canonical dimension specification dictionary for a given object count.

    ``num_objects`` counts scene objects only; the pool is one wider to hold the TCP at slot 0.
    """
    num_slots = num_objects + 1
    return {
        "proprio": PROPRIO_DIM,
        "obj_pose": (num_slots, POSE_DIM),
        "obj_role": (num_slots, ROLE_DIM),
        "obj_valid": (num_slots,),
    }


# Roles are one-hot indicators and validity is a boolean flag
CATEGORICAL_KEYS: frozenset[str] = frozenset({"obj_role", "obj_valid"})


def dim_shape(dim: DimSpec) -> tuple[int, ...]:
    """The full leaf shape a dim spec describes, e.g. ``18 -> (18,)``, ``(3, 7) -> (3, 7)``."""
    if isinstance(dim, int):
        return (dim,)
    if isinstance(dim, torch.Tensor):
        return tuple(dim.shape)
    if isinstance(dim, Mapping):
        raise TypeError("dim_shape does not accept a nested spec.")
    return tuple(int(d) for d in dim)


def canonical_normalization_mask(spec: DimSpec) -> dict[str, torch.Tensor]:
    """Marks the normalizable channels of a canonical observation spec."""
    if not isinstance(spec, Mapping):
        raise TypeError(
            f"canonical_normalization_mask expects a canonical dict spec, got {type(spec).__name__}."
        )

    return {
        key: torch.full(dim_shape(dim), key not in CATEGORICAL_KEYS, dtype=torch.bool)
        for key, dim in spec.items()
    }
