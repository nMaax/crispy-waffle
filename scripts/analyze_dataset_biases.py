"""Dataset analysis script for ManiSkill HDF5 trajectory data.

Dynamically extracts object positions, orientations, grasp/place offsets, and
observation/action statistics using the environment's STATE_SCHEMA.
Strict execution: fails fast with clear errors if env_id or schema is missing.
"""

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np

LIFT_THRESHOLD: float = 0.025


def fetch_env_schema(env_id: str) -> dict[str, Any]:
    """Fetch STATE_SCHEMA from policy.environments or Gymnasium registry for env_id.

    Raises:
        KeyError: If env_id is not registered in Gymnasium.
        AttributeError: If the environment class lacks a STATE_SCHEMA.
    """
    from gymnasium.envs.registration import registry

    import policy.environments  # noqa: F401

    if env_id not in registry:
        raise KeyError(
            f"Environment '{env_id}' is not registered in Gymnasium registry. "
            "Ensure it is imported in policy.environments."
        )

    spec = registry[env_id]
    entry_point = spec.entry_point

    cls: Any = None
    if callable(entry_point):
        cls = entry_point()
    elif isinstance(entry_point, str):
        module_name, class_name = entry_point.split(":")
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name, None)

    if cls is None:
        raise ValueError(f"Could not resolve environment class for '{env_id}'.")

    if not hasattr(cls, "STATE_SCHEMA"):
        cls_name = getattr(cls, "__name__", str(cls))
        raise AttributeError(
            f"Environment class '{cls_name}' for '{env_id}' does not define a 'STATE_SCHEMA'."
        )

    return getattr(cls, "STATE_SCHEMA")


def flatten_state_schema(schema: dict[str, Any]) -> dict[str, slice]:
    """Convert nested STATE_SCHEMA dict into flat mapping of feature_name -> slice."""
    flat: dict[str, slice] = {}

    def _recurse(d: dict[str, Any], prefix: str = "") -> None:
        for k, v in d.items():
            key_name = f"{prefix}{k}" if prefix else k
            if (
                isinstance(v, tuple)
                and len(v) == 2
                and isinstance(v[0], int)
                and isinstance(v[1], int)
            ):
                start, end = v
                length = end - start
                if length == 7 and ("pose" in key_name.lower() or "Pose" in key_name):
                    base_name = key_name.replace("_pose", "").replace("Pose", "")
                    flat[f"{base_name}_pos"] = slice(start, start + 3)
                    flat[f"{base_name}_quat"] = slice(start + 3, end)
                else:
                    flat[key_name] = slice(start, end)
            elif isinstance(v, dict):
                sub_prefix = "" if k in ("agent", "extra") else f"{key_name}_"
                _recurse(v, sub_prefix)

    _recurse(schema)
    if not flat:
        raise ValueError("Flattened STATE_SCHEMA is empty.")
    return flat


def load_raw_trajectory_data(
    h5_path: Path, env_id: str | None = None
) -> tuple[dict[str, np.ndarray], dict[str, slice]]:
    """Extract raw observation slices and full step data from HDF5 dataset."""
    if not h5_path.exists():
        raise FileNotFoundError(f"Dataset not found at {h5_path}")

    # Determine env_id from argument or dataset metadata json
    if not env_id:
        json_path = h5_path.with_suffix(".json")
        if json_path.exists():
            with open(json_path) as f:
                meta = json.load(f)
                env_id = meta.get("env_info", {}).get("env_id")

    if not env_id:
        raise ValueError(
            f"Could not infer env_id for dataset {h5_path}. Please pass --env_id explicitly."
        )

    raw_schema = fetch_env_schema(env_id)
    schema = flatten_state_schema(raw_schema)

    max_schema_dim = max(slc.stop for slc in schema.values())

    # Inspect dataset observation dimension
    with h5py.File(h5_path, "r") as f:
        first_key = next((k for k in f.keys() if k.startswith("traj_")), None)
        if first_key is None:
            raise ValueError(f"No trajectory keys found in {h5_path}")
        traj_group = f[first_key]
        if not isinstance(traj_group, h5py.Group):
            raise ValueError(f"Invalid dataset group at {first_key}")
        obs_dim = np.asarray(traj_group["obs"]).shape[-1]

    if obs_dim < max_schema_dim:
        raise ValueError(
            f"Dataset observation dimension ({obs_dim}) is smaller than schema requirement ({max_schema_dim})."
        )

    data: dict[str, list[np.ndarray]] = {key: [] for key in schema}
    grasp_offsets: list[np.ndarray] = []
    place_offsets: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_inputs: list[np.ndarray] = []

    # Detect object position and relative offset keys from schema
    obj_z_idx: int | None = None
    tcp_rel_key: str | None = None
    target_rel_key: str | None = None

    for k, slc in schema.items():
        if ("cube_a" in k.lower() or "obj" in k.lower()) and "pos" in k.lower():
            obj_z_idx = slc.start + 2
            break

    for k in schema:
        if "tcp_to" in k.lower():
            tcp_rel_key = k
            break

    for k in schema:
        if "to_" in k.lower() and k != tcp_rel_key:
            target_rel_key = k
            break

    with h5py.File(h5_path, "r") as f:
        for traj_key in f.keys():
            if not traj_key.startswith("traj_"):
                continue

            traj_group = f[traj_key]
            if not isinstance(traj_group, h5py.Group):
                continue

            obs = np.asarray(traj_group["obs"])
            actions = np.asarray(traj_group["actions"])

            all_inputs.append(obs)
            all_actions.append(actions)

            for key, slc in schema.items():
                data[key].append(obs[0, slc])

            if obj_z_idx is not None and tcp_rel_key is not None:
                obj_z = obs[:, obj_z_idx]
                lifted_mask = obj_z > LIFT_THRESHOLD
                if np.any(lifted_mask):
                    lifted_obs = obs[lifted_mask]
                    grasp_offsets.append(np.mean(lifted_obs[:, schema[tcp_rel_key]], axis=0))

            if target_rel_key is not None:
                place_offsets.append(obs[-1, schema[target_rel_key]])

    result = {key: np.array(vals) for key, vals in data.items()}
    result["actions"] = np.concatenate(all_actions, axis=0) if all_actions else np.empty((0,))
    result["inputs"] = np.concatenate(all_inputs, axis=0) if all_inputs else np.empty((0,))
    if grasp_offsets:
        result["grasp_offsets"] = np.array(grasp_offsets)
    if place_offsets:
        result["place_offsets"] = np.array(place_offsets)

    return result, schema


def print_stat_summary(name: str, data: np.ndarray) -> None:
    """Print mean, median, and std for a feature vector dataset."""
    if data.size == 0:
        print(f"--- {name} (empty data) ---")
        return

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    median = np.median(data, axis=0)

    print(f"--- {name} ---")
    print(f"Mean:   {mean}")
    print(f"Median: {median}")
    print(f"Std:    {std}")

    if data.ndim > 1 and data.shape[1] >= 2:
        print(f"Proposed In-Dist Dim 0: [{mean[0] - std[0]:.5f}, {mean[0] + std[0]:.5f}]")
        print(f"Proposed In-Dist Dim 1: [{mean[1] - std[1]:.5f}, {mean[1] + std[1]:.5f}]")
    print()


def print_extended_stats(
    name: str, data: np.ndarray, max_dims_to_print: int = 60
) -> None:
    """Print global and feature-wise statistics for high-dimensional data."""
    if data.size == 0:
        print(f"No data available for: {name}")
        return

    print("=========================================")
    print(f" STATS FOR: {name.upper()}")
    print("=========================================")
    print(f"Shape: {data.shape}")
    print(f"Global Mean: {np.mean(data):.5f}")
    print(f"Global Std:  {np.std(data):.5f}")
    print(f"Global Min:  {np.min(data):.5f}")
    print(f"Global Max:  {np.max(data):.5f}")
    print("\nPer-Dimension Breakdown:")

    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=0)

    dims_to_print = min(data.shape[1] if data.ndim > 1 else 1, max_dims_to_print)

    for i in range(dims_to_print):
        m = means[i] if data.ndim > 1 else means
        s = stds[i] if data.ndim > 1 else stds
        mn = mins[i] if data.ndim > 1 else mins
        mx = maxs[i] if data.ndim > 1 else maxs
        print(f"  Dim {i:2d} -> Mean: {m:.4f} | Std: {s:.4f} | Min: {mn:.4f} | Max: {mx:.4f}")

    if data.ndim > 1 and data.shape[1] > max_dims_to_print:
        print(f"  ... and {data.shape[1] - max_dims_to_print} more dimensions.")
    print()


def plot_feature_distributions(
    data_dict: dict[str, np.ndarray], schema: dict[str, slice]
) -> None:
    """Plot histograms for initial 3D positions and 4D quaternions found in schema."""
    pos_items = [
        (k, data_dict[k])
        for k in schema
        if "pos" in k.lower()
        and k in data_dict
        and data_dict[k].ndim > 1
        and data_dict[k].shape[1] == 3
    ]
    quat_items = [
        (k, data_dict[k])
        for k in schema
        if "quat" in k.lower()
        and k in data_dict
        and data_dict[k].ndim > 1
        and data_dict[k].shape[1] == 4
    ]

    if pos_items:
        fig, axes_raw = plt.subplots(
            1, len(pos_items), figsize=(4 * len(pos_items), 4), squeeze=False
        )
        axes = axes_raw[0]
        colors = ["red", "green", "blue"]
        labels = ["X", "Y", "Z"]

        for i, (name, arr) in enumerate(pos_items):
            ax = axes[i]
            for j in range(3):
                ax.hist(arr[:, j], bins=30, alpha=0.5, color=colors[j], label=labels[j])
            ax.set_title(name)
            ax.legend()
        plt.suptitle("Initial Position Distributions")
        plt.tight_layout()
        plt.show()

    if quat_items:
        fig, axes_raw = plt.subplots(
            1, len(quat_items), figsize=(4 * len(quat_items), 4), squeeze=False
        )
        axes = axes_raw[0]
        colors = ["red", "green", "blue", "purple"]
        labels = ["W", "X", "Y", "Z"]

        for i, (name, arr) in enumerate(quat_items):
            ax = axes[i]
            for j in range(4):
                ax.hist(arr[:, j], bins=30, alpha=0.5, color=colors[j], label=labels[j])
            ax.set_title(name)
            ax.legend()
        plt.suptitle("Initial Quaternion Distributions")
        plt.tight_layout()
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze initial object poses, offsets, and dataset statistics dynamically."
    )
    parser.add_argument(
        "--h5_path",
        type=Path,
        default=(
            Path.home()
            / ".maniskill/demos/StackCube-v1/motionplanning"
            / "trajectory.state.pd_ee_delta_pos.physx_cuda.h5"
        ),
        help="Path to HDF5 trajectory dataset.",
    )
    parser.add_argument(
        "--env_id",
        type=str,
        default=None,
        help="Environment ID (e.g. 'StackCube-v1', 'PlaceSphere-v1'). Inferred from dataset json if omitted.",
    )
    args = parser.parse_args()

    raw_data, schema = load_raw_trajectory_data(args.h5_path, env_id=args.env_id)

    print("=== EXTRACTED FEATURE SUMMARIES ===")
    for key in schema:
        if key in raw_data:
            print_stat_summary(key, raw_data[key])

    if "grasp_offsets" in raw_data and raw_data["grasp_offsets"].size > 0:
        print(f"Mean Grasp Offset: {np.mean(raw_data['grasp_offsets'], axis=0)}")
    if "place_offsets" in raw_data and raw_data["place_offsets"].size > 0:
        print(f"Mean Place Offset: {np.mean(raw_data['place_offsets'], axis=0)}\n")

    print_extended_stats("Action", raw_data["actions"])
    print_extended_stats("Input (Observation)", raw_data["inputs"])

    plot_feature_distributions(raw_data, schema)


if __name__ == "__main__":
    main()
