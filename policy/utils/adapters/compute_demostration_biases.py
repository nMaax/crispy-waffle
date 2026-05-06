# pyright: reportIndexIssue=false, reportOperatorIssue=false, reportArgumentType=false, reportCallIssue=false
import random
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

# Under seed: 4803
sphere_initial_pose = ([-0.0660682, 0.0243616, 0.02], [1, 1.38415e-06, -1.49853e-06, 4.63763e-12])
bin_initial_pose = ([0.0410624, 0.0479882, 0.0025], [1, 0, 0, 0])


def compute_demonstration_biases(h5_path: str | Path):
    print(f"Analyzing dataset: {h5_path}\n")

    all_grasp_offsets = []
    all_place_offsets = []
    all_initial_positions_a = []
    all_initial_positions_b = []
    all_initial_quaternions_a = []
    all_initial_quaternions_b = []

    with h5py.File(h5_path, "r") as f:
        for traj_key in f.keys():
            if not traj_key.startswith("traj_"):
                continue

            traj = f[traj_key]

            # obs shape: (num_steps, 48)
            obs = traj["obs"][:]

            all_initial_positions_a.append(tuple(obs[0, 25:28]))
            all_initial_positions_b.append(tuple(obs[0, 32:35]))

            all_initial_quaternions_a.append(tuple(obs[0, 28:32]))
            all_initial_quaternions_b.append(tuple(obs[0, 35:39]))

            # Calculate Grasp Offset (TCP to Cube A)
            # Cube A pose is at indices [25:32]. The Z coordinate is index 27.
            # A resting cube is at Z ~ 0.02. We consider it "grasped and lifted" if Z > 0.025
            cubeA_z = obs[:, 27]
            lifted_mask = cubeA_z > 0.025

            if np.any(lifted_mask):
                # tcp_to_cubeA_pos is at indices [39:42]
                # Average the offset only during the frames where the cube is being carried
                avg_grasp_offset = np.mean(obs[lifted_mask, 39:42], axis=0)
                all_grasp_offsets.append(avg_grasp_offset)

            # Placement Offset (Cube A to Cube B)
            # The last frame of the episode represents the final stacked state
            # cubeA_to_cubeB_pos is at indices [45:48]
            final_place_offset = obs[-1, 45:48]
            all_place_offsets.append(final_place_offset)

    if not all_grasp_offsets or not all_place_offsets:
        raise ValueError("Could not find valid grasping/placement frames in the dataset.")

    all_initial_positions_a = np.array(all_initial_positions_a)
    all_initial_positions_b = np.array(all_initial_positions_b)

    all_initial_quaternions_a = np.array(all_initial_quaternions_a)
    all_initial_quaternions_b = np.array(all_initial_quaternions_b)

    all_grasp_offsets = np.array(all_grasp_offsets)
    all_place_offsets = np.array(all_place_offsets)

    mean_initial_position_a = np.mean(all_initial_positions_a, axis=0)
    median_initial_position_a = np.median(all_initial_positions_a, axis=0)
    mean_initial_position_b = np.mean(all_initial_positions_b, axis=0)
    median_initial_position_b = np.median(all_initial_positions_b, axis=0)

    std_initial_position_a = np.std(all_initial_positions_a, axis=0)
    std_initial_position_b = np.std(all_initial_positions_b, axis=0)

    mean_initial_quaternion_a = np.mean(all_initial_quaternions_a, axis=0)
    median_initial_quaternion_a = np.quantile(all_initial_quaternions_a, 0.5, axis=0)
    q25_initial_quaternion_a = np.quantile(all_initial_quaternions_a, 0.25, axis=0)
    q75_initial_quaternion_a = np.quantile(all_initial_quaternions_a, 0.75, axis=0)

    mean_initial_quaternion_b = np.mean(all_initial_quaternions_b, axis=0)
    median_initial_quaternion_b = np.quantile(all_initial_quaternions_b, 0.5, axis=0)
    q25_initial_quaternion_b = np.quantile(all_initial_quaternions_b, 0.25, axis=0)
    q75_initial_quaternion_b = np.quantile(all_initial_quaternions_b, 0.75, axis=0)

    mean_grasp_bias = np.mean(all_grasp_offsets, axis=0)
    mean_place_bias = np.mean(all_place_offsets, axis=0)

    print()
    print("Mean and Median Initial Position (Cube A):")
    print(mean_initial_position_a, median_initial_position_a)
    print()
    print("Mean and Median Initial Position (Cube B):")
    print(mean_initial_position_b, median_initial_position_b)
    print()

    print("Std Initial Position (Cube A):")
    print(std_initial_position_a)
    print()
    print("Std Initial Position (Cube B):")
    print(std_initial_position_b)
    print()
    # Propose in-distribution coordinates within 1 std (ignore z)
    print("Proposed in-distribution (within 1 std) coordinates for Cube A (x, y):")
    print(
        f"x: [{mean_initial_position_a[0] - std_initial_position_a[0]:.5f}, {mean_initial_position_a[0] + std_initial_position_a[0]:.5f}]"
    )
    print(
        f"y: [{mean_initial_position_a[1] - std_initial_position_a[1]:.5f}, {mean_initial_position_a[1] + std_initial_position_a[1]:.5f}]"
    )
    print()
    print("Proposed in-distribution (within 1 std) coordinates for Cube B (x, y):")
    print(
        f"x: [{mean_initial_position_b[0] - std_initial_position_b[0]:.5f}, {mean_initial_position_b[0] + std_initial_position_b[0]:.5f}]"
    )
    print(
        f"y: [{mean_initial_position_b[1] - std_initial_position_b[1]:.5f}, {mean_initial_position_b[1] + std_initial_position_b[1]:.5f}]"
    )
    print()

    rand_x_a = random.uniform(
        mean_initial_position_a[0] - std_initial_position_a[0],
        mean_initial_position_a[0] + std_initial_position_a[0],
    )
    rand_y_a = random.uniform(
        mean_initial_position_a[1] - std_initial_position_a[1],
        mean_initial_position_a[1] + std_initial_position_a[1],
    )
    print(f"Random in-distribution (Cube A): x={rand_x_a:.5f}, y={rand_y_a:.5f}")
    print()

    rand_x_b = random.uniform(
        mean_initial_position_b[0] - std_initial_position_b[0],
        mean_initial_position_b[0] + std_initial_position_b[0],
    )
    rand_y_b = random.uniform(
        mean_initial_position_b[1] - std_initial_position_b[1],
        mean_initial_position_b[1] + std_initial_position_b[1],
    )
    print(f"Random in-distribution (Cube B): x={rand_x_b:.5f}, y={rand_y_b:.5f}")
    print()

    print("Mean Initial Quaternion (Cube A orientation at episode start):")
    print(mean_initial_quaternion_a)
    print("Median and IQR Initial Quaternion (Cube A orientation at episode start):")
    print(median_initial_quaternion_a, q25_initial_quaternion_a, q75_initial_quaternion_a)

    print()

    print("Mean Initial Quaternion (Cube B orientation at episode start):")
    print(mean_initial_quaternion_b)
    print("Median and IQR Initial Quaternion (Cube B orientation at episode start):")
    print(median_initial_quaternion_b, q25_initial_quaternion_b, q75_initial_quaternion_b)

    print()

    print("Mean Grasp offset (TCP-to-CubeA):")
    print(mean_grasp_bias)
    print()
    print("Mean Place offset (CubeA-to-CubeB):")
    print(mean_place_bias)

    sphere_pos, sphere_quat = sphere_initial_pose
    bin_pos, bin_quat = bin_initial_pose

    # Plot initial positions
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for i, (data, ref, title) in enumerate(
        [
            (all_initial_positions_a, sphere_pos, "Cube A vs Sphere"),
            (all_initial_positions_b, bin_pos, "Cube B vs Bin"),
        ]
    ):
        axs[i].hist(data[:, 0], bins=30, alpha=0.5, color="red", label="X")
        axs[i].hist(data[:, 1], bins=30, alpha=0.5, color="green", label="Y")
        axs[i].hist(data[:, 2], bins=30, alpha=0.5, color="blue", label="Z")
        axs[i].axvline(ref[0], color="red", linestyle="--", label="Ref X")
        axs[i].axvline(ref[1], color="green", linestyle="--", label="Ref Y")
        axs[i].axvline(ref[2], color="blue", linestyle="--", label="Ref Z")
        axs[i].set_title(title)
        axs[i].legend()
    plt.suptitle("Initial Position Distributions vs Reference")
    plt.tight_layout()
    plt.show()

    # Plot initial quaternions
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for i, (data, ref, title) in enumerate(
        [
            (all_initial_quaternions_a, sphere_quat, "Cube A vs Sphere"),
            (all_initial_quaternions_b, bin_quat, "Cube B vs Bin"),
        ]
    ):
        for j, (label, color) in enumerate(
            zip(["w", "x", "y", "z"], ["purple", "red", "green", "blue"])
        ):
            axs[i].hist(data[:, j], bins=30, alpha=0.5, color=color, label=label)
            axs[i].axvline(ref[j], color=color, linestyle="--", label=f"Ref {label}")
        axs[i].set_title(title)
        axs[i].legend()
    plt.suptitle("Initial Quaternion Distributions vs Reference")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    h5_filepath = (
        Path.home()
        / ".maniskill/demos/StackCube-v1/motionplanning"
        / "trajectory.state.pd_ee_delta_pos.physx_cuda.h5"
    )
    compute_demonstration_biases(h5_filepath)

    # *** Results ***
    #
    # Mean and Median Initial Position (Cube A):
    # [-0.00330096  0.0009971   0.02000021] [-0.00098172 -0.00440562  0.02      ]
    #
    # Mean and Median Initial Position (Cube B):
    # [-0.0034901  -0.00025871  0.02000021] [-0.00089826 -0.0026188   0.02      ]
    #
    # Mean Initial Quaternion (Cube A orientation at episode start):
    # [0.6189906  0.         0.         0.03655364]
    # Median and IQR Initial Quaternion (Cube A orientation at episode start):
    # [0.694755   0.         0.         0.08153171] [ 0.34263188  0.          0.         -0.67736864] [0.9146744 0.        0.        0.752867 ]
    #
    # Mean Initial Quaternion (Cube B orientation at episode start):
    # [0.6237932  0.         0.         0.03685375]
    # Median and IQR Initial Quaternion (Cube B orientation at episode start):
    # [0.69651794 0.         0.         0.05261808] [ 0.34458047  0.          0.         -0.65658665] [0.9170049 0.        0.        0.7571442]
    #
    # Mean Grasp offset (TCP-to-CubeA):
    # [-0.00032287  0.00020988 -0.00015297]
    #
    # Mean Place offset (CubeA-to-CubeB):
    # [ 0.00596169 -0.00024662 -0.03997973]
    #
    # SIGNIFICANT BIAS IN THE X VALUES, WHILE STACK CUBE EXPECTS EVERYTHING CENTERED AT (0, 0)
    # PUSH SPHERE PLACE ITS SPHERE AT 6cm AWAY ON THE X AXIS
    #
    # NO SIGNIFICANT BIAS IN GRAPSING OFFSET
    # BUT A BIAS IN PLACEMENT, WITH THE CUBE BEING PLACED SLIGHTLY OFF-CENTER.
