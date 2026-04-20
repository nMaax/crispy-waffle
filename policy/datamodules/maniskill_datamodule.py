import json
import os
from pathlib import Path
from typing import Any

import h5py
import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from policy.utils import extract_h5_shapes, load_h5_data, to_tensor


class DummyDataset(Dataset):
    """A minimal dataset to trigger Lightning loops for simulation-only phases."""

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {}


class ManiSkillTrajectoryDataset(Dataset):
    def __init__(
        self,
        dataset_file: str | Path,
        obs_horizon: int,
        pred_horizon: int,
        load_count: int = -1,
        success_only: bool = False,
    ) -> None:
        super().__init__()
        self.dataset_file = Path(dataset_file)
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon

        # Load JSON metadata
        json_path = self.dataset_file.with_suffix(".json")
        with open(json_path) as f:
            self.json_data = json.load(f)

        self.episodes = self.json_data["episodes"]
        self.trajectories = []

        if load_count == -1:
            load_count = len(self.episodes)

        # Load data into RAM (WARNING: Only do this for state-based observations!
        # For visual observations, you should lazily load from disk inside __getitem__)
        with h5py.File(self.dataset_file, "r") as data:
            print(f"Loading {load_count} episodes into memory...")
            for eps_id in tqdm(range(load_count), desc="Loading HDF5"):
                eps = self.episodes[eps_id]
                if success_only and not eps.get("success", False):
                    continue

                traj_data = data[f"traj_{eps['episode_id']}"]
                trajectory = load_h5_data(traj_data)

                self.trajectories.append(
                    {
                        "obs": trajectory["obs"],
                        "env_states": trajectory["env_states"],
                        "actions": trajectory["actions"],
                    }
                )

        # Peak into the tensors and extract dimensions for later use
        self.action_dim = self.trajectories[0]["actions"].shape[-1]
        self.env_state_dim = extract_h5_shapes(self.trajectories[0]["env_states"])
        self.obs_dim = extract_h5_shapes(self.trajectories[0]["obs"])

        # Pre-compute sliding windows centered around time step `t`
        self.slices = []
        for traj_idx, traj in enumerate(self.trajectories):
            L = len(traj["actions"])

            # Loop over every timestep. `t` represents the CURRENT frame.
            for t in range(L):
                # Observations: history leading up to and including `t`
                obs_start = t - self.obs_horizon + 1
                obs_end = t + 1

                # Actions: future execution starting exactly at `t`
                act_start = t
                act_end = t + self.pred_horizon

                self.slices.append((traj_idx, obs_start, obs_end, act_start, act_end, L))

        print(f"Dataset initialized: {len(self.slices)} temporal windows extracted.")

    def _slice_and_pad(self, data, start, end, L):
        if isinstance(data, dict):
            return {k: self._slice_and_pad(v, start, end, L) for k, v in data.items()}

        pad_before = max(0, -start)
        pad_after = max(0, end - L)

        # Prevents negative indices from acting as "end-of-array" slices
        valid_start = max(0, min(L, start))
        valid_end = max(0, min(L, end))

        seq = data[valid_start:valid_end]

        if pad_before > 0 or pad_after > 0:
            pad_width = [(pad_before, pad_after)] + [(0, 0)] * (seq.ndim - 1)
            seq = np.pad(seq, pad_width, mode="edge")

        return seq

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        traj_idx, obs_start, obs_end, act_start, act_end, L = self.slices[idx]
        traj = self.trajectories[traj_idx]

        obs_seq = self._slice_and_pad(traj["obs"], obs_start, obs_end, L)
        env_seq = self._slice_and_pad(traj["env_states"], obs_start, obs_end, L)
        action_seq = self._slice_and_pad(traj["actions"], act_start, act_end, L)

        return {
            "obs_seq": to_tensor(obs_seq),
            "env_seq": to_tensor(env_seq),
            "action_seq": to_tensor(action_seq),
        }


class ManiSkillDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_file: str | Path,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        batch_size: int = 256,
        num_workers: int = 4,
        val_split: float = 0.1,
    ):
        super().__init__()
        self.dataset_file = Path(dataset_file)
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

        # Prepare train and val split sets
        self.train_set: Dataset | None = None
        self.val_set: Dataset | None = None

        # Fetch dimensions instantly without loading the full dataset into RAM
        # Otherwise these may had to be set to None and wait for setup() to be called
        self.action_dim: int
        self.env_state_dim: int | dict[str, Any]
        self.obs_dim: int | dict[str, Any]
        self.action_dim, self.env_state_dim, self.obs_dim = self._peek_dimensions()

    def _peek_dimensions(self):
        """Reads the JSON and HDF5 headers to extract shapes without loading the full data."""
        json_path = self.dataset_file.with_suffix(".json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        with open(json_path) as f:
            json_data = json.load(f)

        episodes = json_data.get("episodes")
        if not episodes or "episode_id" not in episodes[0]:
            raise ValueError("No valid episode_id found in JSON.")

        first_episode_id = json_data["episodes"][0]["episode_id"]

        with h5py.File(self.dataset_file, "r") as data:
            traj_data = data[f"traj_{first_episode_id}"]

            # h5py can return .shape without loading into memory
            act_dim = traj_data["actions"].shape[-1]  # type: ignore
            env_state_dim = extract_h5_shapes(traj_data["env_states"])  # type: ignore
            obs_dim = extract_h5_shapes(traj_data["obs"])  # type: ignore

        if act_dim is None or env_state_dim is None or obs_dim is None:
            raise ValueError(
                "The h5 dataset was not found, thus dimensionalities could not be fetched."
            )

        return act_dim, env_state_dim, obs_dim

    def setup(self, stage=None):
        if self.train_set is None:
            # Load the dataset
            full_dataset = ManiSkillTrajectoryDataset(
                self.dataset_file, self.obs_horizon, self.pred_horizon
            )

            # Compute the sizes for splitting
            val_size = int(len(full_dataset) * self.val_split)
            train_size = len(full_dataset) - val_size

            # Use a fixed seed for reproducibility
            # TODO: NONONONONONO do not seed this way!!!!!
            self.train_set, self.val_set = random_split(
                full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
            )

    def train_dataloader(self):
        if self.train_set is None:
            raise TypeError(
                "It appears you asked for a dataloader without setting up a Dataset first. Call setup() first."
            )
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,  # Speeds up GPU transfer
        )

    def val_dataloader(self):
        if self.val_set is None:
            raise TypeError(
                "It appears you asked for a dataloader without setting up a Dataset first. Call setup() first."
            )
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        # Testing is done strictly via simulation in the Callback.
        # We return a dummy batch just to trigger the Callback logic.
        return DataLoader(DummyDataset(), batch_size=1)


if __name__ == "__main__":
    from pathlib import Path

    h5_path = (
        Path.home() / ".maniskill" / "demos" / "StackCube-v1" / "motionplanning" / "trajectory.h5"
    )
    obs_horizon = 8
    pred_horizon = 4
    # Load a datamodule instead of a dataset and get the action, obs and env shapes
    datamodule = ManiSkillDataModule(h5_path, obs_horizon, pred_horizon)
    print(datamodule.action_dim)
    print(datamodule.env_state_dim)
    print(datamodule.obs_dim)
