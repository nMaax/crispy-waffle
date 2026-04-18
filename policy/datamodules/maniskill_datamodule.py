import json
from pathlib import Path

import h5py
import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def load_h5_data(data):
    """Recursively loads h5py data into memory as numpy arrays."""
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out


def to_tensor(data, device=None):
    """Recursively converts numpy arrays to PyTorch tensors."""
    if isinstance(data, dict):
        return {k: to_tensor(v, device) for k, v in data.items()}
    tensor = torch.from_numpy(data).float()
    if device is not None:
        tensor = tensor.to(device)
    return tensor


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
        dataset_file: str,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        batch_size: int = 256,
        num_workers: int = 4,
    ):
        super().__init__()
        self.dataset_file = dataset_file
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = None

    def setup(self, stage=None):
        if self.dataset is None:
            # TODO: You'd split this into train/val datasets
            self.dataset = ManiSkillTrajectoryDataset(
                self.dataset_file, self.obs_horizon, self.pred_horizon
            )

    def train_dataloader(self):
        if self.dataset is None:
            raise TypeError(
                "It appears you asked for a dataloader without setting up a Dataset first. Call setup()."
            )
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,  # Speeds up GPU transfer
        )
    
    def val_dataloader(self):
        if self.dataset is None:
            raise TypeError("It appears you asked for a dataloader without setting up a Dataset first. Call setup().")
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        if self.dataset is None:
            raise TypeError("It appears you asked for a dataloader without setting up a Dataset first. Call setup().")
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

