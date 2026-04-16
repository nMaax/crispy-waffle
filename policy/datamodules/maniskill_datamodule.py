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
        self.dataset_file = dataset_file
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon

        # Load JSON metadata
        json_path = Path(dataset_file).with_suffix(".json")
        with open(json_path) as f:
            self.json_data = json.load(f)

        self.episodes = self.json_data["episodes"]
        self.trajectories = []

        if load_count == -1:
            load_count = len(self.episodes)

        # Load data into RAM (WARNING: Only do this for state-based observations!
        # For visual observations, you should lazily load from disk inside __getitem__)
        with h5py.File(dataset_file, "r") as data:
            print(f"Loading {load_count} episodes into memory...")
            for eps_id in tqdm(range(load_count), desc="Loading HDF5"):
                eps = self.episodes[eps_id]
                if success_only and not eps.get("success", False):
                    continue

                traj_data = data[f"traj_{eps['episode_id']}"]
                trajectory = load_h5_data(traj_data)

                # Use all available frames
                self.trajectories.append(
                    {
                        "obs": trajectory["obs"],
                        "env_states": trajectory["env_states"],
                        "actions": trajectory["actions"],
                    }
                )

        # Pre-compute sliding windows
        self.slices = []
        pad_before = self.obs_horizon - 1
        pad_after = self.pred_horizon - self.obs_horizon

        for traj_idx, traj in enumerate(self.trajectories):
            L = len(traj["actions"])
            for start in range(-pad_before, L - self.pred_horizon + pad_after):
                end = start + self.pred_horizon
                self.slices.append((traj_idx, start, end, L))

        print(f"Dataset initialized: {len(self.slices)} temporal windows extracted.")

    def _slice_and_pad(self, data, start, end, L):
        """Recursively slices and pads dictionaries or numpy arrays.

        - If start < 0, it repeats the first frame.
        - If end > L, it repeats the last frame.
        """
        if isinstance(data, dict):
            # If it's a nested dictionary (like priv_states), recurse!
            return {k: self._slice_and_pad(v, start, end, L) for k, v in data.items()}

        # Base case: It's a numpy array
        pad_before = max(0, -start)
        pad_after = max(0, end - L)

        valid_start = max(0, start)
        valid_end = min(L, end)

        # Grab the valid segment
        seq = data[valid_start:valid_end]

        # Pad by repeating the first frame
        if pad_before > 0:
            padding = np.repeat(seq[0:1], pad_before, axis=0)
            seq = np.concatenate([padding, seq], axis=0)

        # Pad by repeating the last frame
        if pad_after > 0:
            padding = np.repeat(seq[-1:], pad_after, axis=0)
            seq = np.concatenate([seq, padding], axis=0)

        return seq

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        traj_idx, start, end, L = self.slices[idx]
        traj = self.trajectories[traj_idx]

        obs_seq = self._slice_and_pad(traj["obs"], start, start + self.obs_horizon, L)
        env_seq = self._slice_and_pad(traj["env_states"], start, start + self.obs_horizon, L)
        action_seq = self._slice_and_pad(traj["actions"], start, end, L)

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


if __name__ == "__main__":
    from pathlib import Path

    from policy.utils import print_dict_tree

    h5_path = (
        Path.home() / ".maniskill" / "demos" / "StackCube-v1" / "motionplanning" / "trajectory.h5"
    )
    obs_horizon = 8
    pred_horizon = 16
    dataset = ManiSkillTrajectoryDataset(h5_path, obs_horizon, pred_horizon)
    print_dict_tree(dataset[0])
