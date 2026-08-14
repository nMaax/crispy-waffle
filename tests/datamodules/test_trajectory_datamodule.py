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

        kwargs.setdefault("canonicalize", False)
        kwargs.setdefault("as_dict", False)
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
        dm.setup("test")
        test_loader = dm.test_dataloader()

        assert isinstance(test_loader.dataset, DummyDataset)
        assert len(test_loader) == 1

        batch = next(iter(test_loader))
        assert batch == {}


class TestManiSkillDataModuleHFFetch:
    """Tests for the optional HF Hub dataset auto-fetch path (`hf_dataset_repo`)."""

    def _make_dataset_file(self, tmp_path: Path, monkeypatch) -> Path:
        fake_home = tmp_path / "home"
        monkeypatch.setattr(Path, "home", lambda: fake_home)
        return fake_home / ".maniskill" / "demos" / "StackCube-v1" / "motionplanning" / "trajectory.h5"

    def test_prepare_data_noop_when_hf_dataset_repo_none(self, tmp_path, monkeypatch):
        """Default behavior (hf_dataset_repo=None) never touches huggingface_hub."""
        dataset_file = self._make_dataset_file(tmp_path, monkeypatch)
        dataset_file.parent.mkdir(parents=True)
        dataset_file.write_bytes(b"")
        dataset_file.with_suffix(".json").write_text(
            json.dumps({"env_info": {}, "episodes": []})
        )

        dm = TrajectoryDataModule(dataset_file=dataset_file, seed=1)
        with patch("huggingface_hub.hf_hub_download") as mock_download:
            dm.prepare_data()
        mock_download.assert_not_called()

    def test_construction_defers_validation_when_hf_dataset_repo_set(self, tmp_path, monkeypatch):
        """With hf_dataset_repo set, missing local files must not raise at construction time."""
        dataset_file = self._make_dataset_file(tmp_path, monkeypatch)  # files don't exist yet
        dm = TrajectoryDataModule(dataset_file=dataset_file, seed=1, hf_dataset_repo="org/demos")
        assert not hasattr(dm, "env_id")

    def test_prepare_data_downloads_and_validates(self, tmp_path, monkeypatch):
        dataset_file = self._make_dataset_file(tmp_path, monkeypatch)

        def fake_download(filename, local_dir, **_kwargs):
            target = Path(local_dir) / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.suffix == ".json":
                target.write_text(
                    json.dumps(
                        {
                            "env_info": {
                                "env_id": "StackCube-v1",
                                "env_kwargs": {
                                    "obs_mode": "state",
                                    "control_mode": "pd_ee_delta_pos",
                                    "sim_backend": "physx_cpu",
                                },
                            },
                            "episodes": [],
                        }
                    )
                )
            else:
                target.write_bytes(b"")

        dm = TrajectoryDataModule(dataset_file=dataset_file, seed=1, hf_dataset_repo="org/demos")
        with patch(
            "huggingface_hub.hf_hub_download", side_effect=fake_download
        ) as mock_download:
            dm.prepare_data()

        assert mock_download.call_count == 2
        assert dm.env_id == "StackCube-v1"
        assert dataset_file.exists()
        assert dataset_file.with_suffix(".json").exists()

    def test_prepare_data_raises_when_download_leaves_files_missing(self, tmp_path, monkeypatch):
        dataset_file = self._make_dataset_file(tmp_path, monkeypatch)
        dm = TrajectoryDataModule(dataset_file=dataset_file, seed=1, hf_dataset_repo="org/demos")
        with patch("huggingface_hub.hf_hub_download"):  # no-op: doesn't materialize files
            with pytest.raises(FileNotFoundError):
                dm.prepare_data()

    def test_prepare_data_raises_valueerror_outside_maniskill_convention(
        self, tmp_path, monkeypatch
    ):
        self._make_dataset_file(tmp_path, monkeypatch)  # only to patch Path.home
        outside_file = tmp_path / "elsewhere" / "trajectory.h5"
        outside_file.parent.mkdir(parents=True)
        dm = TrajectoryDataModule(dataset_file=outside_file, seed=1, hf_dataset_repo="org/demos")
        # Since the fetch moved onto the shared helper, this now names both paths instead of being a
        # bare relative_to() failure.
        with pytest.raises(ValueError, match="outside") as error:
            dm.prepare_data()
        assert str(outside_file) in str(error.value)
