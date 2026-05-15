import json
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest
from torch.utils.data import DataLoader

from policy.datamodules.trajectory_datamodule import TrajectoryDataModule
from policy.datasets import DummyDataset, TrajectoryDataset
from tests.datamodules.test_datamodule import DataModuleTests


@pytest.mark.parametrize("datamodule_config", ["trajectory_datamodule"], indirect=True)
class TestManiSkillDataModule(DataModuleTests[TrajectoryDataModule]):
    """Test suite for the ManiSkillDataModule."""


@pytest.fixture
def datamodule_factory(tmp_path: Path):
    """A factory fixture to generate a customized ManiSkillDataModule backed by a temporary
    HDF5/JSON dataset on the fly."""

    def _create_datamodule(
        num_episodes: int = 10,
        episode_length: int = 5,
        obs_mode: str = "state",
        control_mode: str = "pd_ee_delta_pos",
        sim_backend: str = "physx_cpu",
        val_split: float = 0.2,
        seed: int = 42,
        **kwargs,
    ) -> TrajectoryDataModule:
        json_path = tmp_path / f"dummy_dataset_{obs_mode}_{control_mode}.json"
        h5_path = tmp_path / f"dummy_dataset_{obs_mode}_{control_mode}.h5"

        # Create JSON Metadata
        episodes = []
        for i in range(num_episodes):
            episodes.append({"episode_id": i, "elapsed_steps": episode_length, "success": True})

        metadata = {
            "env_info": {
                "env_id": "MockEnv-v0",
                "env_kwargs": {
                    "obs_mode": obs_mode,
                    "control_mode": control_mode,
                    "sim_backend": sim_backend,
                },
            },
            "episodes": episodes,
        }
        with open(json_path, "w") as f:
            json.dump(metadata, f)

        # Create HDF5 Data
        act_dim, obs_dim, env_state_dim = 4, 3, 5
        with h5py.File(h5_path, "w") as f:
            for i in range(num_episodes):
                g = f.create_group(f"traj_{i}")
                g.create_dataset(
                    "actions", data=np.ones((episode_length, act_dim), dtype=np.float32)
                )
                g.create_dataset("obs", data=np.ones((episode_length, obs_dim), dtype=np.float32))
                g.create_dataset(
                    "env_states", data=np.ones((episode_length, env_state_dim), dtype=np.float32)
                )

        return TrajectoryDataModule(
            dataset_file=h5_path,
            val_split=val_split,
            seed=seed,
            **kwargs,
        )

    return _create_datamodule


class TestManiSkillDataModuleLogic:
    """Test suite for the internal logic, splitting, and configuration of the DataModule."""

    def test_setup_and_splitting(self, datamodule_factory):
        """Verifies train/val splits respect the val_split ratio correctly."""
        # 10 episodes total, val_split = 0.2 -> 8 train, 2 val
        dm = datamodule_factory(num_episodes=10, val_split=0.2)
        dm.setup()

        assert isinstance(dm.train_set, TrajectoryDataset)
        assert isinstance(dm.val_set, TrajectoryDataset)

        # Check episode distribution (not temporal windows, but source episodes)
        assert len(dm.train_set.episodes) == 8
        assert len(dm.val_set.episodes) == 2

    def test_split_reproducibility(self, datamodule_factory):
        """Verifies that the same seed produces the exact same train/val split, and a different
        seed produces a different split."""
        dm_1 = datamodule_factory(num_episodes=20, val_split=0.2, seed=42)
        dm_1.setup()

        dm_2 = datamodule_factory(num_episodes=20, val_split=0.2, seed=42)
        dm_2.setup()

        dm_diff = datamodule_factory(num_episodes=20, val_split=0.2, seed=999)
        dm_diff.setup()

        def get_episode_ids(dataset):
            return [ep["episode_id"] for ep in dataset.episodes]

        # Same seed should match perfectly
        assert get_episode_ids(dm_1.train_set) == get_episode_ids(dm_2.train_set)

        # Different seed should shuffle differently
        assert get_episode_ids(dm_1.train_set) != get_episode_ids(dm_diff.train_set)

    @patch("policy.datamodules.trajectory_datamodule.rank_zero_warn")
    def test_json_metadata_parsing(self, mock_warn, datamodule_factory):
        """Tests parsing logic for physx backends and observation modes."""

        # 'auto' backend falls back to physx_cpu and warns
        dm_auto = datamodule_factory(sim_backend="auto", obs_mode="state")
        assert dm_auto.physx_backend == "physx_cpu"
        mock_warn.assert_called_with(
            "Dataset specifies 'auto' sim_backend. Defaulting to 'physx_cpu'."
        )

    @patch("policy.datamodules.trajectory_datamodule.rank_zero_warn")
    def test_infer_padding_masks_absolute_mode(self, mock_warn, datamodule_factory):
        """Absolute modes should default to None (edge padding) and warn if overridden."""
        dm_abs = datamodule_factory(
            control_mode="pd_joint_pos",
            action_left_pad_as_zero_mask=[True, True, True, True],  # Trying to override
        )
        left_mask, right_mask = dm_abs._infer_padding_masks()

        assert left_mask is None
        assert right_mask is None
        mock_warn.assert_called()
        assert "is absolute. The mask will be ignored" in mock_warn.call_args[0][0]

    def test_infer_padding_masks_delta_modes(self, datamodule_factory):
        """Delta/vel modes should default to zero padding except for the last dim (gripper)."""
        dm_delta = datamodule_factory(control_mode="pd_joint_delta_pos")
        left_mask, right_mask = dm_delta._infer_padding_masks()

        # We mocked act_dim to be 4 in the factory.
        # So we expect [True, True, True, False]
        expected_mask = np.array([True, True, True, False], dtype=bool)

        assert np.array_equal(left_mask, expected_mask)
        assert np.array_equal(right_mask, expected_mask)

    def test_infer_padding_masks_explicit_override(self, datamodule_factory):
        """Explicit overrides should be respected even if in delta/vel mode."""
        custom_mask = [False, False, True, True]
        dm_override = datamodule_factory(
            control_mode="pd_joint_delta_pos",
            action_left_pad_as_zero_mask=custom_mask,
            action_right_pad_as_zero_mask=custom_mask,
        )
        left_mask, right_mask = dm_override._infer_padding_masks()

        expected_mask = np.array(custom_mask, dtype=bool)
        assert np.array_equal(left_mask, expected_mask)
        assert np.array_equal(right_mask, expected_mask)


class TestManiSkillDataLoaders:
    """Test suite specifically for PyTorch DataLoader creation and batch generation."""

    def test_train_val_dataloaders(self, datamodule_factory):
        """Verifies dataloaders return batches with correct shapes."""
        batch_size = 2
        obs_horizon = 2
        pred_horizon = 4

        dm = datamodule_factory(
            batch_size=batch_size,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            obs_mode="state",
        )
        dm.setup()

        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()

        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)

        # Fetch one batch
        batch = next(iter(train_loader))

        assert "obs_seq" in batch
        assert "act_seq" in batch

        # Check shapes: (batch_size, horizon, dimension)
        # obs_dim = 3, act_dim = 4 (from factory mock)
        assert batch["obs_seq"].shape == (batch_size, obs_horizon, 3)
        assert batch["act_seq"].shape == (batch_size, pred_horizon, 4)

    def test_test_dataloader_is_dummy(self, datamodule_factory):
        """Verifies the test dataloader correctly yields the DummyDataset."""
        dm = datamodule_factory()
        test_loader = dm.test_dataloader()

        assert isinstance(test_loader.dataset, DummyDataset)
        assert len(test_loader) == 1

        batch = next(iter(test_loader))
        assert batch == {}
