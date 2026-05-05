# pyright: reportIndexIssue=false, reportOperatorIssue=false, reportArgumentType=false, reportCallIssue=false
from pathlib import Path

import h5py
import numpy as np


def compute_demonstration_biases(h5_path: str | Path):
    print(f"Analyzing dataset: {h5_path}\n")

    all_grasp_offsets = []
    all_place_offsets = []

    with h5py.File(h5_path, "r") as f:
        for traj_key in f.keys():
            if not traj_key.startswith("traj_"):
                continue

            traj = f[traj_key]

            # obs shape: (num_steps, 48)
            obs = traj["obs"][:]

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

            # Calculate Placement Offset (Cube A to Cube B)
            # The last frame of the episode represents the final stacked state
            # cubeA_to_cubeB_pos is at indices [45:48]
            final_place_offset = obs[-1, 45:48]
            all_place_offsets.append(final_place_offset)

    if not all_grasp_offsets or not all_place_offsets:
        raise ValueError("Could not find valid grasping/placement frames in the dataset.")

    all_grasp_offsets = np.array(all_grasp_offsets)
    all_place_offsets = np.array(all_place_offsets)

    print("Grasp offsets (TCP to Cube A) across trajectories:")
    print(all_grasp_offsets)
    print("\n")
    print("Place offsets (Cube A to Cube B) across trajectories:")
    print(all_place_offsets)

    print("\n---\n")

    mean_grasp_bias = np.mean(all_grasp_offsets, axis=0)
    mean_place_bias = np.mean(all_place_offsets, axis=0)

    print("Mean Grasp offset (TCP-to-CubeA):")
    print(mean_grasp_bias)
    print()
    print("Mean Place offset (CubeA-to-CubeB):")
    print(mean_place_bias)


if __name__ == "__main__":
    h5_filepath = (
        Path.home()
        / ".maniskill/demos/StackCube-v1/motionplanning"
        / "trajectory.state.pd_ee_delta_pos.physx_cuda.h5"
    )
    compute_demonstration_biases(h5_filepath)

    # Result:
    #
    # Mean Grasp offset (TCP-to-CubeA):
    # [-0.00032287  0.00020988 -0.00015297]
    #
    # Mean Place offset (CubeA-to-CubeB):
    # [ 0.00596169 -0.00024662 -0.03997973]
    #
    # NO SIGNIFICANT BIAS IN GRAPSING
    # BUT A SIGNIFICANT BIAS IN PLACEMENT, WITH THE CUBE BEING PLACED SLIGHTLY OFF-CENTER.
