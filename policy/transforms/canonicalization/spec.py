from __future__ import annotations

# Dimensionality constants
POSE_DIM: int = 7  # 3D position + 4D quaternion [x, y, z, qw, qx, qy, qz]
RELATIVE_SE3_DIM: int = 6  # 3D position delta + 3D axis-angle rotation vector
ROLE_DIM: int = 3  # 3D one-hot vector [is_pick, is_target, is_clutter]
PROPRIO_DIM: int = 18  # 9D joint positions (qpos) + 9D joint velocities (qvel)

# Role one-hot vectors
ROLE_PICK: tuple[float, float, float] = (1.0, 0.0, 0.0)
ROLE_TARGET: tuple[float, float, float] = (0.0, 1.0, 0.0)
ROLE_CLUTTER: tuple[float, float, float] = (0.0, 0.0, 1.0)


def canonical_dim_spec(num_objects: int = 2) -> dict[str, int]:
    """Generates the canonical dimension specification dictionary for a given object count."""
    spec: dict[str, int] = {"proprio": PROPRIO_DIM, "tcp_pose": POSE_DIM}
    for i in range(num_objects):
        spec[f"obj_{i}_pose"] = POSE_DIM
        spec[f"obj_{i}_role"] = ROLE_DIM
    return spec
