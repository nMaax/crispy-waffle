import json
import random
from pathlib import Path
from typing import Any, cast

import h5py
import lightning as L
import numpy as np
from lightning.fabric.utilities.rank_zero import rank_zero_warn
from torch.utils.data import DataLoader, Dataset

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
        use_phsyx_env_states: bool,
        cond_horizon: int,
        pred_horizon: int,
        episodes: list[dict] | None = None,
        load_count: int = -1,
        success_only: bool = False,
        lazy: bool = False,
    ) -> None:
        """
        Dataset for loading ManiSkill trajectories from HDF5 files.

        parameters:
            - dataset_file: Path to the HDF5 file containing trajectory data. The corresponding JSON metadata file should be in the same directory with the same name but .json extension.
            - use_phsyx_env_states: Whether to condition the policy on raw states of the physical engine (ignoring observaions).
            - cond_horizon: Number of past time steps to include in the conditioning sequence.
            - pred_horizon: Number of future time steps to include in the action sequence.
            - episodes: Optional list of episode metadata dicts to use. If None, the dataset loads all episodes from the JSON file.
            - load_count: If > 0, limits the number of episodes to load (for faster debugging). If -1, loads all episodes.
            - success_only: If True, only loads episodes marked as successful in the JSON metadata
            - lazy_load: If True, do not load trajectories into RAM. Only keep episode_id and read slices from disk in __getitem__.
        """
        super().__init__()
        self.dataset_file = Path(dataset_file)
        self.use_phsyx_env_states = use_phsyx_env_states
        self.cond_horizon = cond_horizon
        self.pred_horizon = pred_horizon
        self.lazy_load = lazy

        # Worker-specific HDF5 file handle for DataLoader multiprocessing (used in lazy mode)
        self._h5_file = None

        # Load JSON metadata if episodes aren't provided explicitly
        if episodes is not None:
            self.episodes = episodes
        else:
            json_path = self.dataset_file.with_suffix(".json")
            with open(json_path) as f:
                self.json_data = json.load(f)
            self.episodes = self.json_data["episodes"]

        if load_count == -1:
            load_count = len(self.episodes)
        else:
            load_count = min(load_count, len(self.episodes))

        self.trajectories: list[dict[str, Any]] = []
        self.slices: list[tuple[int, int, int, int, int, int]] = []

        def append_windows_for_episode(traj_idx: int, L: int) -> None:
            """Append all temporal windows for one episode of length L."""
            for t in range(L):
                cond_start = t - self.cond_horizon + 1
                cond_end = t + 1
                act_start = t
                act_end = t + self.pred_horizon
                self.slices.append((traj_idx, cond_start, cond_end, act_start, act_end, L))

        # Find one valid episode id for shape peeking
        first_valid_episode_id: int | None = None

        # Open the H5 only once during init:
        # - not lazy: load selected episodes into RAM
        # - lazy: peek dims from first valid episode
        with h5py.File(self.dataset_file, "r") as data:
            for i in range(load_count):
                eps = self.episodes[i]

                if success_only and not eps.get("success", False):
                    continue

                episode_id = int(eps["episode_id"])
                L = int(eps["elapsed_steps"])  # single source of truth for indexing/windows

                if first_valid_episode_id is None:
                    first_valid_episode_id = episode_id

                if self.lazy_load:
                    # Store only what's needed to locate the episode on disk
                    self.trajectories.append({"episode_id": episode_id, "length": L})
                else:
                    # Load episode tensors into RAM
                    traj_group = data[f"traj_{episode_id}"]
                    trajectory = load_h5_data(traj_group)

                    # Consistency check
                    h5_L = len(trajectory["actions"])
                    if h5_L != L:
                        raise ValueError(
                            f"Length mismatch for episode {episode_id}: "
                            f"JSON elapsed_steps={L} but H5 len(actions)={h5_L}."
                        )

                    self.trajectories.append(
                        {
                            "episode_id": episode_id,
                            "obs": trajectory["obs"],
                            "env_states": trajectory["env_states"],
                            "actions": trajectory["actions"],
                        }
                    )

                traj_idx = len(self.trajectories) - 1
                append_windows_for_episode(traj_idx, L)

            if first_valid_episode_id is None:
                raise ValueError(
                    f"No valid episodes found (success_only={success_only}) in metadata for dataset {self.dataset_file}."
                )

            # Peek into the tensors and extract dimensions for later use
            first_traj = cast(h5py.Group, data[f"traj_{first_valid_episode_id}"])
            actions_ds = cast(h5py.Dataset, first_traj["actions"])

            self.action_dim = actions_ds.shape[-1]
            self.env_state_dim = extract_h5_shapes(first_traj["env_states"])
            self.obs_dim = extract_h5_shapes(first_traj["obs"])

        print(
            f"Dataset initialized: {len(self.slices)} temporal windows "
            f"from {len(self.trajectories)} episodes "
            f"(lazy_load={self.lazy_load})."
        )
        print_dict_tree(self.trajectories[0])

    @property
    def h5_file(self):
        """Lazy HDF5 file opener to prevent multiprocessing crashes in PyTorch DataLoaders"""
        if self._h5_file is None:
            self._h5_file = h5py.File(self.dataset_file, "r")
        return self._h5_file

    def _slice_and_pad(self, data, start, end, L):
        """Slices the data from start to end, and pads with edge values if the slice goes out of bounds."""
        # Treat HDF5 groups like dicts (lazy nested observations)
        if isinstance(data, dict) or isinstance(data, h5py.Group):
            return {k: self._slice_and_pad(data[k], start, end, L) for k in data.keys()}

        pad_before = max(0, -start)
        pad_after = max(0, end - L)

        # Prevents negative indices from acting as "end-of-array" slices
        valid_start = max(0, min(L, start))
        valid_end = max(0, min(L, end))

        seq = data[valid_start:valid_end]  # works for numpy arrays and h5py.Dataset

        if pad_before > 0 or pad_after > 0:
            pad_width = [(pad_before, pad_after)] + [(0, 0)] * (seq.ndim - 1)
            seq = np.pad(seq, pad_width, mode="edge")

        return seq

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx: int):
        traj_idx, cond_start, cond_end, act_start, act_end, L = self.slices[idx]
        traj = self.trajectories[traj_idx]

        if self.lazy_load:
            episode_id = traj["episode_id"]
            traj = self.h5_file[f"traj_{episode_id}"]  # Read data on the fly

        traj = cast(h5py.Group, traj)
        # Select conditioning source
        cond_src = traj["env_states"] if self.use_phsyx_env_states else traj["obs"]
        act_src = traj["actions"]

        # Slice and pad conditinion and action sequences
        cond_seq = self._slice_and_pad(cond_src, cond_start, cond_end, L)
        act_seq = self._slice_and_pad(act_src, act_start, act_end, L)

        return {
            "cond_seq": to_tensor(cond_seq),
            "action_seq": to_tensor(act_seq),
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
        lazy: bool = False,
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

        self.json_path = self.dataset_file.with_suffix(".json")
        if not self.json_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {self.json_path}. "
                "ManiSkill requires a .json file alongside the .h5 file to index trajectories."
            )

        self.cond_horizon = cond_horizon
        self.pred_horizon = pred_horizon
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.lazy = lazy

        (
            self.env_id,
            self.obs_mode,
            self.control_mode,
            self.physx_backend,
            self.use_phsyx_env_states,
        ) = self._load_metadata_from_json()

        # Fetch dimensions instantly without loading the full dataset into RAM
        self.action_dim, self.env_state_dim, self.obs_dim = self._peek_dimensions()

        # Prepare seeds for spliting
        if seed is None:
            raise ValueError("seed must be provided.")
        self.seed = seed

        # Debug
        print(f"Seeds for episodes datasplit fetched from main seed: {seed}")

        # Prepare train and val split sets
        self.train_set: Dataset | None = None
        self.val_set: Dataset | None = None

    @property
    def cond_dim(self) -> int | dict[str, Any]:
        """The dimensionality of the conditioning signal exposed to the policy. Selected by use_phsyx_env_states."""
        if self.use_phsyx_env_states:
            return self.env_state_dim
        else:
            return self.obs_dim

    def _load_metadata_from_json(self):
        """
        Parse the dataset metadata JSON to extract obs_mode, control_mode, and physx_backend.
        """

        with open(self.json_path) as f:
            meta = json.load(f)

        env_info = meta.get("env_info", {})
        env_kwargs = env_info.get("env_kwargs", {})

        env_id = env_info.get("env_id", "StackCube-v1")
        obs_mode = env_kwargs.get("obs_mode", "state")
        control_mode = env_kwargs.get("control_mode", "pd_joint_pos")
        physx_backend = env_kwargs.get("sim_backend", "physx_cpu")

        if physx_backend == "auto":
            rank_zero_warn("Dataset specifies 'auto' sim_backend. Defaulting to 'physx_cpu'.")
            physx_backend = "physx_cpu"

        use_phsyx_env_states = obs_mode == "none"

        return env_id, obs_mode, control_mode, physx_backend, use_phsyx_env_states

    def _peek_dimensions(self):
        """Reads the JSON and HDF5 headers to extract shapes without loading the full data."""
        with open(self.json_path) as f:
            json_data = json.load(f)

        episodes = json_data.get("episodes")
        if not episodes or "episode_id" not in episodes[0]:
            raise ValueError("No valid episode_id found in JSON.")

        first_episode_id = json_data["episodes"][0]["episode_id"]

        # NOTE: h5py can return .shape without loading into memory

        with h5py.File(self.dataset_file, "r") as data:
            traj_data = cast(h5py.Group, data[f"traj_{first_episode_id}"])
            actions_ds = cast(h5py.Dataset, traj_data["actions"])

            act_dim = actions_ds.shape[-1]
            env_state_dim = extract_h5_shapes(traj_data["env_states"])
            obs_dim = extract_h5_shapes(traj_data["obs"])

        if act_dim is None or env_state_dim is None or obs_dim is None:
            raise ValueError(
                "The h5 dataset was not found, thus dimensionalities could not be fetched."
            )

        return act_dim, env_state_dim, obs_dim

    def setup(self, stage=None):
        if self.train_set is None:
            with open(self.json_path) as f:
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

            self.train_set = ManiSkillTrajectoryDataset(
                dataset_file=self.dataset_file,
                use_phsyx_env_states=self.use_phsyx_env_states,
                cond_horizon=self.cond_horizon,
                pred_horizon=self.pred_horizon,
                episodes=train_episodes,
                lazy=self.lazy,
            )

            self.val_set = ManiSkillTrajectoryDataset(
                dataset_file=self.dataset_file,
                use_phsyx_env_states=self.use_phsyx_env_states,
                cond_horizon=self.cond_horizon,
                pred_horizon=self.pred_horizon,
                episodes=val_episodes,
                lazy=self.lazy,
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
