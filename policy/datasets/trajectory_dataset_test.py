import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
import torch

from .trajectory_dataset import TrajectoryDataset


@pytest.fixture
def dummy_trajectory_data(tmp_path: Path) -> Path:
    """Creates a temporary, minimal HDF5/JSON dataset to reliably test dataset logic without
    requiring actual downloaded ManiSkill data."""
    json_path = tmp_path / "dummy_dataset.json"
    h5_path = tmp_path / "dummy_dataset.h5"

    # Create JSON Metadata
    metadata = {
        "env_info": {
            "env_id": "MockEnv-v0",
            "env_kwargs": {
                "obs_mode": "state",
                "control_mode": "pd_joint_delta_pos",
                "sim_backend": "physx_cpu",
            },
        },
        "episodes": [
            {"episode_id": 0, "elapsed_steps": 10, "success": True},
            {"episode_id": 1, "elapsed_steps": 5, "success": False},
        ],
    }
    with open(json_path, "w") as f:
        json.dump(metadata, f)

    # Create HDF5 Data
    with h5py.File(h5_path, "w") as f:
        # Episode 0: Success, Length 10
        g0 = f.create_group("traj_0")
        g0.create_dataset("actions", data=np.ones((10, 2), dtype=np.float32))
        g0.create_dataset("obs", data=np.ones((10, 3), dtype=np.float32) * 2)
        g0.create_dataset("env_states", data=np.ones((10, 4), dtype=np.float32) * 3)

        # Episode 1: Failure, Length 5
        g1 = f.create_group("traj_1")
        g1.create_dataset("actions", data=np.ones((5, 2), dtype=np.float32) * 4)
        g1.create_dataset("obs", data=np.ones((5, 3), dtype=np.float32) * 5)
        g1.create_dataset("env_states", data=np.ones((5, 4), dtype=np.float32) * 6)

    return h5_path


def check_sequence_length(data: Any, expected_len: int) -> None:
    """Helper function to recursively assert sequence length (first dimension) of all leaf tensors
    or arrays."""
    if isinstance(data, dict):
        for v in data.values():
            check_sequence_length(v, expected_len)
    elif isinstance(data, torch.Tensor | np.ndarray):
        assert data.shape[0] == expected_len, (
            f"Expected first dimension to be {expected_len}, got {data.shape[0]}."
        )


class TestManiSkillDataset:
    def test_initialization_and_file_handling(self, tmp_path: Path):
        """Ensures the dataset fails gracefully with missing or bad files."""
        bad_extension = tmp_path / "bad_file.txt"
        bad_extension.touch()

        with pytest.raises(ValueError, match="Invalid file extension"):
            TrajectoryDataset(
                dataset_file=bad_extension,
                obs_horizon=2,
                pred_horizon=4,
            )

        missing_file = tmp_path / "missing.h5"
        with pytest.raises(FileNotFoundError, match="not found"):
            TrajectoryDataset(
                dataset_file=missing_file,
                obs_horizon=2,
                pred_horizon=4,
            )

    def test_filtering_and_loading(self, dummy_trajectory_data: Path):
        """Tests that success_only and load_count correctly filter the JSON episodes."""

        # Test success_only
        dataset_success = TrajectoryDataset(
            dataset_file=dummy_trajectory_data,
            obs_horizon=2,
            pred_horizon=4,
            success_only=True,
        )
        # Should only load trajectory 0 (Length 10)
        assert len(dataset_success.trajectories) == 1
        assert dataset_success.trajectories[0]["episode_id"] == 0
        assert len(dataset_success) == 10  # 10 windows for length 10

        # Test load_count
        dataset_count = TrajectoryDataset(
            dummy_trajectory_data,
            obs_horizon=2,
            pred_horizon=4,
            load_count=1,
            success_only=False,
        )
        assert len(dataset_count.trajectories) == 1
        assert dataset_count.trajectories[0]["episode_id"] == 0

    def test_lazy_vs_eager_parity(self, dummy_trajectory_data: Path):
        """Ensures lazy=True and lazy=False return the exact same tensors and structure."""
        kwargs = dict(
            dataset_file=dummy_trajectory_data,
            obs_horizon=2,
            pred_horizon=4,
        )

        dataset_eager = TrajectoryDataset(**kwargs, lazy=False)  # type: ignore
        dataset_lazy = TrajectoryDataset(**kwargs, lazy=True)  # type: ignore

        assert len(dataset_eager) == len(dataset_lazy)

        # Compare specific temporal windows
        indices_to_check = [0, len(dataset_eager) // 2, len(dataset_eager) - 1]

        for idx in indices_to_check:
            eager_item = dataset_eager[idx]
            lazy_item = dataset_lazy[idx]

            # Recursively check tensor values
            def check_tensors_match(a, b):
                if isinstance(a, dict):
                    assert isinstance(b, dict)
                    for k in a.keys():
                        check_tensors_match(a[k], b[k])
                else:
                    assert torch.allclose(a, b)

            check_tensors_match(eager_item["obs_seq"], lazy_item["obs_seq"])
            check_tensors_match(eager_item["act_seq"], lazy_item["act_seq"])

    def test_temporal_windowing(self, dummy_trajectory_data: Path):
        """Tests if _compute_trajectory_slices generates the right temporal boundaries."""
        obs_horizon = 2
        pred_horizon = 4

        dataset = TrajectoryDataset(
            dataset_file=dummy_trajectory_data,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            success_only=True,  # Episode 0, length 10
        )

        # We expect exactly L (10) slices
        assert len(dataset.slices) == 10

        # Test the first slice (t=0)
        traj_idx, obs_start, obs_end, act_start, act_end, L = dataset.slices[0]
        assert obs_start == -1  # 0 - 2 + 1
        assert obs_end == 1
        assert act_start == obs_start  # Aligns with obs_start
        assert act_end == act_start + pred_horizon

        # Test the last slice (t=9)
        traj_idx, obs_start, obs_end, act_start, act_end, L = dataset.slices[-1]
        assert obs_end == 10  # 9 + 1
        assert act_start == obs_start

    def test_slice_and_pad(self, dummy_trajectory_data: Path):
        """Tests the padding logic applied to HDF5 sequences directly."""
        dataset = TrajectoryDataset(
            dataset_file=dummy_trajectory_data,
            obs_horizon=2,
            pred_horizon=4,
            # Let's say actions have 2 dims: pad first with 0s, second with edge
            action_right_pad_as_zero_mask=[True, False],
        )

        # Mock some dummy data (Length 5, 2 Dims)
        dummy_data = np.array(
            [[10, 100], [11, 101], [12, 102], [13, 103], [14, 104]]  # Last valid frame
        )

        L = len(dummy_data)

        # Exceed boundaries to force padding on the right: valid[3:5] + pad[3 frames]
        padded = dataset._slice_and_pad(
            dummy_data,
            start=3,
            end=8,
            L=L,
            right_pad_as_zero_mask=dataset.action_right_pad_as_zero_mask,
        )

        assert padded.shape == (5, 2)

        # Check valid frames
        assert np.array_equal(padded[0], [13, 103])
        assert np.array_equal(padded[1], [14, 104])

        # Check padding frames (indexes 2, 3, 4)
        for i in range(2, 5):
            # Dim 0 is zero padded (True in mask)
            assert padded[i, 0] == 0
            # Dim 1 is edge padded (False in mask), so it copies the last valid frame (104)
            assert padded[i, 1] == 104

    def test_real_local_dataset(self):
        """Integration test using actual local files if they exist."""
        # Check standard path
        real_file = (
            Path.home()
            / ".maniskill/demos/StackCube-v1/motionplanning/trajectory.state.pd_ee_delta_pose.physx_cpu.h5"
        )

        if not real_file.exists():
            # Fallback check to any other potential h5 file
            paths = list(
                (Path.home() / ".maniskill/demos/StackCube-v1/motionplanning").glob("*.h5")
            )
            if paths:
                real_file = paths[0]
            else:
                pytest.skip(f"Real dataset not found at {real_file}. Skipping integration test.")

        dataset = TrajectoryDataset(
            dataset_file=real_file,
            obs_horizon=2,
            pred_horizon=16,
            success_only=True,
            lazy=True,
        )

        assert len(dataset) > 0
        sample = dataset[0]

        assert "obs_seq" in sample
        assert "act_seq" in sample

        # Check sequence shapes using our recursive helper
        check_sequence_length(sample["obs_seq"], 2)
        check_sequence_length(sample["act_seq"], 16)
