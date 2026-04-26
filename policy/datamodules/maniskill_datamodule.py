import json
import os
import random
import warnings
from pathlib import Path
from typing import Any, Literal

import h5py
import lightning as L
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from policy.utils import extract_h5_shapes, load_h5_data, print_dict_tree, to_tensor

# NOTE: The use of DummyDataset is a bit smelly, ngl
# but it allows us to trigger the Lightning loops for validation and testing without needing
# actual data loading logic in those phases. The RolloutEvaluationCallback will handle the real evaluation logic in simulation,
# so we just need a placeholder here to satisfy the DataLoader interface.


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
        cond_source: Literal["env_states", "obs"],
        cond_horizon: int,
        pred_horizon: int,
        episodes: list[dict] | None = None,
        load_count: int = -1,
        success_only: bool = False,
    ) -> None:
        """
        Dataset for loading ManiSkill trajectories from HDF5 files.

        parameters:
            - dataset_file: Path to the HDF5 file containing trajectory data. The corresponding JSON metadata file should be in the same directory with the same name but .json extension.
            - cond_source: Whether to condition the policy on "env_states" (raw states of the physical engine), "obs" (observations, e.g. "state", "rgbd"), or "both".
            - cond_horizon: Number of past time steps to include in the conditioning sequence.
            - pred_horizon: Number of future time steps to include in the action sequence.
            - episodes: Optional list of episode metadata dicts to use. If None, the dataset oads all episodes from the JSON file.
            - load_count: If > 0, limits the number of episodes to load into memory (for faster debugging). If -1, loads all episodes.
            - success_only: If True, only loads episodes marked as successful in the JSON metadata
        """
        super().__init__()
        self.dataset_file = Path(dataset_file)
        self.cond_source = cond_source
        self.cond_horizon = cond_horizon
        self.pred_horizon = pred_horizon

        # Load JSON metadata if episodes aren't provided explicitly
        if episodes is not None:
            self.episodes = episodes
        else:
            json_path = self.dataset_file.with_suffix(".json")
            with open(json_path) as f:
                self.json_data = json.load(f)
            self.episodes = self.json_data["episodes"]

        self.trajectories = []

        if load_count == -1:
            load_count = len(self.episodes)

        # TODO: Address the warning
        # Load data into RAM: WARNING: Only do this for state-based observations!
        # For visual observations, you should lazily load from disk inside __getitem__
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
                cond_start = t - self.cond_horizon + 1
                cond_end = t + 1

                # Actions: future execution starting exactly at `t`
                act_start = t
                act_end = t + self.pred_horizon

                self.slices.append((traj_idx, cond_start, cond_end, act_start, act_end, L))

        print(
            f"Dataset initialized: {len(self.slices)} temporal windows \
            from {len(self.trajectories)} episodes "
        )
        print_dict_tree(self.trajectories[0])

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
        traj_idx, cond_start, cond_end, act_start, act_end, L = self.slices[idx]
        traj = self.trajectories[traj_idx]

        env_seq = self._slice_and_pad(traj["env_states"], cond_start, cond_end, L)
        obs_seq = self._slice_and_pad(traj["obs"], cond_start, cond_end, L)
        action_seq = self._slice_and_pad(traj["actions"], act_start, act_end, L)

        if self.cond_source == "env_states":
            cond_seq = env_seq
        elif self.cond_source == "obs":
            cond_seq = obs_seq
        else:
            raise ValueError(f"Invalid cond_source: {self.cond_source}")

        return {
            "cond_seq": to_tensor(cond_seq),
            "action_seq": to_tensor(action_seq),
        }


class ManiSkillDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_file: str | Path,
        cond_horizon: int = 2,
        pred_horizon: int = 16,
        batch_size: int = 256,
        num_workers: int = 4,
        val_split: float = 0.1,
        seed: int | None = None,
    ):
        """
        DataModule for loading ManiSkill trajectories from HDF5 files.

        parameters:
            - dataset_file: Path to the HDF5 file containing trajectory data. The corresponding JSON metadata file should be in the same directory with the same name but .json extension.
            - cond_horizon: Number of past time steps to include in the conditioning sequence.
            - pred_horizon: Number of future time steps to include in the action sequence.
            - batch_size: Number of samples per batch for training and validation.
            - num_workers: Number of subprocesses to use for data loading.
            - val_split: Fraction of episodes to reserve for validation (e.g. 0.1 for 10% validation).
            - seed: An optional main seed to ensure reproducible train/val splits. If None, a random seed will be generated.
        """
        super().__init__()
        self.dataset_file = Path(dataset_file)

        if not self.dataset_file.exists():
            raise FileNotFoundError(f"The dataset file was not found at: {self.dataset_file}")

        if self.dataset_file.suffix not in [".h5", ".hdf5"]:
            raise ValueError(
                f"Invalid file extension '{self.dataset_file.suffix}'. "
                "ManiSkill datasets must be HDF5 files (.h5 or .hdf5)."
            )

        json_path = self.dataset_file.with_suffix(".json")
        if not json_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {json_path}. "
                "ManiSkill requires a .json file alongside the .h5 file to index trajectories."
            )

        self.cond_horizon = cond_horizon
        self.pred_horizon = pred_horizon
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

        # Fetch data nature from filename
        self.obs_mode, self.cond_source, self.control_mode, self.physx_backend = (
            self._parse_dataset_filename()
        )

        # Fetch dimensions instantly without loading the full dataset into RAM
        self.action_dim, self.env_state_dim, self.obs_dim = self._peek_dimensions()

        # Prepare seeds for spliting
        main_seed = seed if seed is not None else random.randint(0, int(1e5))
        self.seed = main_seed

        # Debug
        print(f"Seeds for episodes datasplit fetched from main seed: {main_seed}")

        # Prepare train and val split sets
        self.train_set: Dataset | None = None
        self.val_set: Dataset | None = None

    @property
    def cond_dim(self) -> int | dict[str, Any]:
        """The dimensionality of the conditioning signal exposed to the policy.

        Returns the shape information for the source selected by cond_source:
        - `"env_states"` → `env_state_dim`
        - `"obs"` → `obs_dim`
        """
        if self.cond_source == "env_states":
            return self.env_state_dim
        elif self.cond_source == "obs":
            return self.obs_dim
        else:
            raise ValueError(f"Invalid cond_source: {self.cond_source}")

    def _parse_dataset_filename(self):
        """
        Parse the dataset filename to extract cond_source, obs_mode, control_mode, and physx_backend
        """
        stem = self.dataset_file.stem.lower()
        parts = stem.split(".")
        if len(parts) >= 4:
            obs_mode = parts[1]
            control_mode = parts[2]
            physx_backend = parts[3]
        else:
            warnings.warn(
                f"Dataset filename '{self.dataset_file.name}' does not follow the standard ManiSkill "
                "format: '[env_id].[obs_mode].[control_mode].[backend]'. "
                "Falling back to default parsing logic."
            )
            obs_mode = "state"
            control_mode = "pd_joint_pos"
            physx_backend = "physx_cpu"

        cond_source = "env_states" if obs_mode == "none" else "obs"

        return cond_source, obs_mode, control_mode, physx_backend

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

            # NOTE: h5py can return .shape without loading into memory
            act_dim = traj_data["actions"].shape[-1]
            env_state_dim = extract_h5_shapes(traj_data["env_states"])
            obs_dim = extract_h5_shapes(traj_data["obs"])  # type: ignore

        if act_dim is None or env_state_dim is None or obs_dim is None:
            raise ValueError(
                "The h5 dataset was not found, thus dimensionalities could not be fetched."
            )

        return act_dim, env_state_dim, obs_dim

    def setup(self, stage=None):
        if self.train_set is None:
            json_path = self.dataset_file.with_suffix(".json")
            with open(json_path) as f:
                all_episodes = json.load(f)["episodes"]

            rng = random.Random(self.seed)
            rng.shuffle(all_episodes)

            val_size = int(len(all_episodes) * self.val_split)
            train_size = len(all_episodes) - val_size

            train_episodes = all_episodes[:train_size]
            val_episodes = all_episodes[train_size:]

            print(
                f"Splitting dataset: {train_size} training episodes, {val_size} validation episodes."
            )

            if self.cond_source != "env_states" and self.cond_source != "obs":
                raise ValueError(f"Invalid cond_source: {self.cond_source}")

            self.train_set = ManiSkillTrajectoryDataset(
                dataset_file=self.dataset_file,
                cond_source=self.cond_source,
                cond_horizon=self.cond_horizon,
                pred_horizon=self.pred_horizon,
                episodes=train_episodes,
            )

            self.val_set = ManiSkillTrajectoryDataset(
                dataset_file=self.dataset_file,
                cond_source=self.cond_source,
                cond_horizon=self.cond_horizon,
                pred_horizon=self.pred_horizon,
                episodes=val_episodes,
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
            pin_memory=True,
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
        return DataLoader(DummyDataset(), batch_size=1)
