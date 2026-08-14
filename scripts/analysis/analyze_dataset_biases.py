"""Summarises episodes statistics over a recorded demonstration dataset.

Reads a ManiSkill HDF5 trajectory file with dictionary observations, and reports the spread of
initial object poses, the typical grasp and place offsets, and per-dimension statistics.

Pass several `--env-id` values to compare datasets across tasks; each gets its own figure and
report.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np

from policy.utils.h5_utils import load_h5_data
from scripts.utils import cli, theme
from scripts.utils.episodes import default_demo_path, ensure_local_dataset, trajectory_keys
from scripts.utils.figures import figure_path, save_figure
from scripts.utils.report import Report

SCRIPT_NAME = Path(__file__).stem

# Height above the table above which the manipulated object counts as lifted.
LIFT_THRESHOLD: float = 0.025

# Per-dimension breakdowns get long; observations here are ~50-dimensional.
MAX_DIMS_REPORTED = 60


def flatten_obs_dict(obs: dict[str, Any], prefix: str = "") -> dict[str, np.ndarray]:
    """Flattens a nested observation dict into leaf arrays with split pos/quat for 7D poses."""
    flat: dict[str, np.ndarray] = {}
    for key, value in obs.items():
        name = f"{prefix}{key}" if prefix else key
        if isinstance(value, dict):
            # agent/extra are grouping keys
            flat.update(
                flatten_obs_dict(value, prefix="" if key in ("agent", "extra") else f"{name}_")
            )
        elif isinstance(value, np.ndarray):
            if value.shape[-1] == 7 and "pose" in name.lower():
                base = name.replace("_pose", "").replace("Pose", "")
                flat[f"{base}_pos"] = value[..., :3]
                flat[f"{base}_quat"] = value[..., 3:7]
            else:
                flat[name] = value
    return flat


def load_dataset(h5_path: Path, env_id: str) -> tuple[dict[str, np.ndarray], list[str]]:
    """Reads every trajectory once, collecting initial features, offsets and full step data."""
    if not h5_path.exists():
        raise FileNotFoundError(f"Dataset not found at {h5_path}")

    initial: dict[str, list[np.ndarray]] = {}
    grasp_offsets: list[np.ndarray] = []
    place_offsets: list[np.ndarray] = []
    actions: list[np.ndarray] = []

    with h5py.File(h5_path, "r") as handle:
        keys = trajectory_keys(handle)
        if not keys:
            raise ValueError(f"No `traj_*` groups found in {h5_path}")

        for key in keys:
            group = handle[key]
            if not isinstance(group, h5py.Group):
                continue

            obs_group = group["obs"]
            if not isinstance(obs_group, h5py.Group):
                raise TypeError(f"Expected 'obs' in {key} to be an HDF5 group, got {type(obs_group)}")

            obs_tree = load_h5_data(obs_group)
            if not isinstance(obs_tree, dict):
                raise TypeError(f"Expected observation group to load as dict, got {type(obs_tree)}")

            flat_obs = flatten_obs_dict(obs_tree)
            actions.append(np.asarray(group["actions"]))

            for name, arr in flat_obs.items():
                if name not in initial:
                    initial[name] = []
                initial[name].append(arr[0])

            # Detect pick object and grasp offset
            tcp_pos = flat_obs.get("tcp_pos")
            obj_pos = next((v for k, v in flat_obs.items() if ("cubeA_pos" in k or "obj_0_pos" in k or "obj_pos" in k)), None)
            if tcp_pos is not None and obj_pos is not None:
                lifted_mask = obj_pos[:, 2] > (obj_pos[0, 2] + LIFT_THRESHOLD)
                if np.any(lifted_mask):
                    grasp_offsets.append(np.mean(obj_pos[lifted_mask] - tcp_pos[lifted_mask], axis=0))

            target_pos = next((v for k, v in flat_obs.items() if "cubeB_pos" in k or "bin_pos" in k), None)
            if obj_pos is not None and target_pos is not None:
                place_offsets.append(obj_pos[-1] - target_pos[-1])

    data = {name: np.array(values) for name, values in initial.items()}
    data["actions"] = np.concatenate(actions, axis=0) if actions else np.empty((0,))
    # Flatten all numeric observation leaves across all timesteps for general stats
    all_obs_flat = []
    for k, v in flat_obs.items():
        if np.issubdtype(v.dtype, np.number) and v.ndim > 1:
            all_obs_flat.append(v.reshape(len(v), -1))
    data["observations"] = np.concatenate(all_obs_flat, axis=-1) if all_obs_flat else np.empty((0,))
    if grasp_offsets:
        data["grasp_offsets"] = np.array(grasp_offsets)
    if place_offsets:
        data["place_offsets"] = np.array(place_offsets)
    data["_num_episodes"] = np.array(len(keys))
    schema_names = list(initial.keys())
    return data, schema_names


def report_feature_spread(report: Report, schema_names: list[str], data: dict) -> None:
    """Per-feature mean/std of the initial state, i.e. how varied the task setup is."""
    report.section("Initial-state spread (across episodes)")
    rows = []
    for name in schema_names:
        values = data.get(name)
        if values is None or values.size == 0:
            continue
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        rows.append(
            [
                name,
                np.array2string(mean, precision=4, suppress_small=True),
                np.array2string(std, precision=4, suppress_small=True),
            ]
        )
    report.table(["feature", "mean", "std"], rows)
    report.note(
        "A near-zero std means that feature is effectively constant across the dataset -- the "
        "policy can learn it as a bias rather than from the observation."
    )


def report_dimension_stats(report: Report, name: str, values: np.ndarray) -> None:
    """Global and per-dimension statistics for a stacked [steps, dims] array."""
    report.section(f"{name} statistics")
    if values.size == 0:
        report.note("(no data)")
        return

    report.kv("shape", values.shape)
    report.kv("global mean", f"{np.mean(values):.5f}")
    report.kv("global std", f"{np.std(values):.5f}")
    report.kv("global min", f"{np.min(values):.5f}")
    report.kv("global max", f"{np.max(values):.5f}")

    if values.ndim == 1:
        return

    limit = min(values.shape[1], MAX_DIMS_REPORTED)
    means, stds = np.mean(values, axis=0), np.std(values, axis=0)
    mins, maxs = np.min(values, axis=0), np.max(values, axis=0)
    report.table(
        ["dim", "mean", "std", "min", "max"],
        [
            [i, f"{means[i]:.4f}", f"{stds[i]:.4f}", f"{mins[i]:.4f}", f"{maxs[i]:.4f}"]
            for i in range(limit)
        ],
    )
    if values.shape[1] > limit:
        report.note(f"... and {values.shape[1] - limit} further dimensions not shown.")


def plot_distributions(
    data: dict[str, np.ndarray],
    schema_names: list[str],
    env_id: str,
    save_path: Path,
    *,
    show: bool,
    dpi: int,
) -> Path | None:
    """Histograms of the initial positions and quaternions, one panel per feature."""

    def panels(suffix: str, width: int) -> list[tuple[str, np.ndarray]]:
        return [
            (name, data[name])
            for name in schema_names
            if suffix in name.lower()
            and name in data
            and data[name].ndim > 1
            and data[name].shape[1] == width
        ]

    position_panels = panels("pos", 3)
    quaternion_panels = panels("quat", 4)
    groups = [
        ("position", position_panels, ["x", "y", "z"]),
        ("quaternion", quaternion_panels, ["w", "x", "y", "z"]),
    ]
    groups = [group for group in groups if group[1]]
    if not groups:
        return None

    num_columns = max(len(panels_) for _, panels_, _ in groups)
    fig, axes = plt.subplots(
        len(groups),
        num_columns,
        figsize=(4.2 * num_columns, 3.8 * len(groups)),
        squeeze=False,
    )

    for row, (group_name, group_panels, component_labels) in enumerate(groups):
        for column in range(num_columns):
            ax = axes[row, column]
            if column >= len(group_panels):
                ax.axis("off")
                continue

            name, values = group_panels[column]
            for component, label in enumerate(component_labels):
                ax.hist(
                    values[:, component],
                    bins=30,
                    alpha=0.55,
                    color=theme.SERIES[component],
                    label=label,
                )
            ax.set_title(f"{name} ({group_name})", fontsize=10, color=theme.TEXT_SECONDARY)
            ax.set_xlabel("value", fontsize=9)
            ax.set_ylabel("episodes", fontsize=9)
            theme.style_axes(ax)
            ax.legend(facecolor=theme.SURFACE, edgecolor=theme.GRID, fontsize=8)

    theme.set_title(
        fig,
        "Initial-state distributions",
        [("env", env_id), ("episodes", str(int(data["_num_episodes"])))],
    )
    save_figure(fig, save_path, show=show, dpi=dpi)
    print(f"Figure saved: {save_path}")
    return save_path


def analyse_env(env_id: str, args: argparse.Namespace) -> None:
    """Runs the whole analysis for one environment's dataset."""
    dataset_path = ensure_local_dataset(args.dataset_path or default_demo_path(env_id))
    data, schema_names = load_dataset(dataset_path, env_id)

    report = Report(
        "Dataset bias summary",
        [
            ("env", env_id),
            ("dataset", str(dataset_path)),
            ("episodes", str(int(data["_num_episodes"]))),
        ],
    )

    report_feature_spread(report, schema_names, data)

    report.section("Mean offsets")
    if "grasp_offsets" in data and data["grasp_offsets"].size:
        report.kv(
            "grasp (tcp -> object)",
            np.array2string(np.mean(data["grasp_offsets"], axis=0), precision=5),
        )
    if "place_offsets" in data and data["place_offsets"].size:
        report.kv(
            "place (final relative)",
            np.array2string(np.mean(data["place_offsets"], axis=0), precision=5),
        )

    report_dimension_stats(report, "Action", data["actions"])
    report_dimension_stats(report, "Observation", data["observations"])

    save_path = figure_path(
        SCRIPT_NAME, "initial-distributions", env_id=env_id, out_dir=args.out_dir
    )
    plotted = plot_distributions(data, schema_names, env_id, save_path, show=args.show, dpi=args.dpi)
    report.emit(plotted or save_path, save=not args.no_report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        parents=[cli.output_args()],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--env-id",
        type=str,
        nargs="+",
        default=["StackCubeLockedRotation-v1"],
        help="Environments whose datasets to analyse, one report each. "
        "(default: StackCubeLockedRotation-v1)",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Explicit .h5 to read. Only valid with a single --env-id; otherwise each env uses "
        "its conventional demo path.",
    )
    args = parser.parse_args()

    if args.dataset_path is not None and len(args.env_id) > 1:
        parser.error("--dataset-path applies to a single dataset; pass one --env-id with it.")
    return args


def main() -> None:
    args = parse_args()
    theme.apply_theme()

    for index, env_id in enumerate(args.env_id):
        if len(args.env_id) > 1:
            print(f"\n{'#' * 88}\n# Env {index + 1}/{len(args.env_id)}: {env_id}\n{'#' * 88}")
        analyse_env(env_id, args)


if __name__ == "__main__":
    main()
