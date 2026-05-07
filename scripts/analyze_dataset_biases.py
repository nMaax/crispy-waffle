# pyright: reportIndexIssue=false, reportOperatorIssue=false, reportArgumentType=false, reportCallIssue=false
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

STACKCUBE_STATE_MAPPING = {
    "cube_a_pos": slice(25, 28),
    "cube_a_quat": slice(28, 32),
    "cube_b_pos": slice(32, 35),
    "cube_b_quat": slice(35, 39),
    "tcp_to_cube_a": slice(39, 42),
    "cube_a_to_cube_b": slice(45, 48),
}

LIFT_THRESHOLD = 0.025

# Obtained via interaction with Sapien (render="human", clicking on the objects and copying the poses)
SPHERE_INITIAL_POSE = ([-0.0660682, 0.0243616, 0.02], [1, 1.38415e-06, -1.49853e-06, 4.63763e-12])
BIN_INITIAL_POSE = ([0.0410624, 0.0479882, 0.0025], [1, 0, 0, 0])


def load_raw_trajectory_data(h5_path: Path) -> dict[str, np.ndarray]:
    """Extracts raw observation slices from the HDF5 dataset."""
    data = {key: [] for key in STACKCUBE_STATE_MAPPING.keys()}
    grasp_offsets = []
    place_offsets = []

    if not h5_path.exists():
        raise FileNotFoundError(f"Dataset not found at {h5_path}")

    with h5py.File(h5_path, "r") as f:
        for traj_key in f.keys():
            if not traj_key.startswith("traj_"):
                continue

            obs = f[traj_key]["obs"][:]

            for key, slc in STACKCUBE_STATE_MAPPING.items():
                if "pos" in key or "quat" in key:
                    data[key].append(obs[0, slc])

            cube_a_z = obs[:, 27]
            lifted_mask = cube_a_z > LIFT_THRESHOLD
            if np.any(lifted_mask):
                grasp_offsets.append(
                    np.mean(obs[lifted_mask, STACKCUBE_STATE_MAPPING["tcp_to_cube_a"]], axis=0)
                )

            place_offsets.append(obs[-1, STACKCUBE_STATE_MAPPING["cube_a_to_cube_b"]])

    return {
        "cube_a_pos": np.array(data["cube_a_pos"]),
        "cube_b_pos": np.array(data["cube_b_pos"]),
        "cube_a_quat": np.array(data["cube_a_quat"]),
        "cube_b_quat": np.array(data["cube_b_quat"]),
        "grasp_offsets": np.array(grasp_offsets),
        "place_offsets": np.array(place_offsets),
    }


def print_stat_summary(name: str, data: np.ndarray):
    """Prints formatted mean, median, and std for a 3D vector."""
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    print(f"--- {name} ---")
    print(f"Mean:   {mean}")
    print(f"Median: {np.median(data, axis=0)}")
    print(f"Std:    {std}")

    if data.shape[1] >= 2:
        print(f"Proposed In-Dist X: [{mean[0] - std[0]:.5f}, {mean[0] + std[0]:.5f}]")
        print(f"Proposed In-Dist Y: [{mean[1] - std[1]:.5f}, {mean[1] + std[1]:.5f}]")
    print()


def plot_comparison(data_list: list[tuple[np.ndarray, list, str]], mode: str = "pos"):
    """Generates histograms comparing extracted data vs reference poses.

    mode: 'pos' for XYZ or 'quat' for WXYZ
    """
    fig, axs = plt.subplots(1, len(data_list), figsize=(12, 5))
    labels = ["X", "Y", "Z"] if mode == "pos" else ["W", "X", "Y", "Z"]
    colors = ["red", "green", "blue", "purple"] if mode == "quat" else ["red", "green", "blue"]

    for i, (data, ref, title) in enumerate(data_list):
        for j, (label, color) in enumerate(zip(labels, colors)):
            axs[i].hist(data[:, j], bins=30, alpha=0.5, color=color, label=label)
            axs[i].axvline(ref[j], color=color, linestyle="--", label=f"Ref {label}")

        axs[i].set_title(title)
        axs[i].legend()

    plt.suptitle(f"Initial {'Position' if mode == 'pos' else 'Quaternion'} Distributions")
    plt.tight_layout()
    plt.show()


def main():
    h5_filepath = (
        Path.home()
        / ".maniskill/demos/StackCube-v1/motionplanning"
        / "trajectory.state.pd_ee_delta_pos.physx_cuda.h5"
    )

    print(f"Analyzing dataset: {h5_filepath}\n")

    try:
        raw_data = load_raw_trajectory_data(h5_filepath)
    except Exception as e:
        print(f"Error: {e}")
        return

    print_stat_summary("Cube A Position", raw_data["cube_a_pos"])
    print_stat_summary("Cube B Position", raw_data["cube_b_pos"])

    print(f"Mean Grasp Offset (TCP->A): {np.mean(raw_data['grasp_offsets'], axis=0)}")
    print(f"Mean Place Offset (A->B):   {np.mean(raw_data['place_offsets'], axis=0)}\n")

    plot_comparison(
        [
            (raw_data["cube_a_pos"], SPHERE_INITIAL_POSE[0], "Cube A vs Sphere"),
            (raw_data["cube_b_pos"], BIN_INITIAL_POSE[0], "Cube B vs Bin"),
        ],
        mode="pos",
    )

    plot_comparison(
        [
            (raw_data["cube_a_quat"], SPHERE_INITIAL_POSE[1], "Cube A vs Sphere"),
            (raw_data["cube_b_quat"], BIN_INITIAL_POSE[1], "Cube B vs Bin"),
        ],
        mode="quat",
    )


if __name__ == "__main__":
    main()
