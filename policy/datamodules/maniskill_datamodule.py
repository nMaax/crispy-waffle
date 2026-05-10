import json
import random
from pathlib import Path

import lightning as L
import numpy as np
from lightning.fabric.utilities.rank_zero import rank_zero_warn
from lightning_utilities.core.rank_zero import rank_zero_info
from torch.utils.data import DataLoader, Dataset

from policy.datamodules.maniskill_dataset import ManiSkillDataset


class DummyDataset(Dataset):
    """A minimal dataset to trigger Lightning loops for simulation-only phases."""

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {}


class ManiSkillDataModule(L.LightningDataModule):
    """DataModule for loading ManiSkill trajectories from HDF5 files, with train/val splitting and
    lazy loading support."""

    def __init__(
        self,
        dataset_file: str | Path,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        obs_dim: int = 48,
        act_dim: int = 4,
        batch_size: int = 256,
        num_workers: int = 4,
        val_split: float = 0.1,
        action_left_pad_as_zero_mask: list[bool] | None = None,
        action_right_pad_as_zero_mask: list[bool] | None = None,
        load_count: int = -1,
        success_only: bool = False,
        lazy: bool = False,
        seed: int | None = None,
    ):
        super().__init__()

        if seed is None:
            raise ValueError("seed must be provided.")

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

        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

        self.action_left_pad_as_zero_mask = action_left_pad_as_zero_mask
        self.action_right_pad_as_zero_mask = action_right_pad_as_zero_mask

        self.load_count = load_count
        self.success_only = success_only
        self.lazy = lazy
        self.seed = seed

        (
            self.env_id,
            self.obs_mode,
            self.control_mode,
            self.physx_backend,
        ) = self._load_metadata_from_json()

        rank_zero_info(f"Seed for episodes datasplit fetched from main seed: {seed}")

        self.train_set: Dataset | None = None
        self.val_set: Dataset | None = None

    def setup(self, stage: str | None = None):
        if self.train_set is None:
            with open(self.json_path) as f:
                all_episodes = json.load(f)["episodes"]

            rng = random.Random(self.seed)
            rng.shuffle(all_episodes)

            val_size = int(len(all_episodes) * self.val_split)
            train_size = len(all_episodes) - val_size

            train_episodes = all_episodes[:train_size]
            val_episodes = all_episodes[train_size:]

            rank_zero_info(
                f"Splitting dataset: {train_size} training episodes, {val_size} validation episodes."
            )

            left_mask, right_mask = self._infer_padding_masks()

            # TODO: should initialize only what needed given stage string

            self.train_set = ManiSkillDataset(
                dataset_file=self.dataset_file,
                obs_horizon=self.obs_horizon,
                pred_horizon=self.pred_horizon,
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                obs_left_pad_as_zero_mask=None,  # Condition padding should always be edge
                obs_right_pad_as_zero_mask=None,  # Condition padding should always be edge
                action_left_pad_as_zero_mask=left_mask,
                action_right_pad_as_zero_mask=right_mask,
                episodes=train_episodes,
                load_count=self.load_count,
                success_only=self.success_only,
                lazy=self.lazy,
            )

            self.val_set = ManiSkillDataset(
                dataset_file=self.dataset_file,
                obs_horizon=self.obs_horizon,
                pred_horizon=self.pred_horizon,
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                obs_left_pad_as_zero_mask=None,  # Condition padding should always be edge
                obs_right_pad_as_zero_mask=None,  # Condition padding should always be edge
                action_left_pad_as_zero_mask=left_mask,
                action_right_pad_as_zero_mask=right_mask,
                episodes=val_episodes,
                load_count=self.load_count,
                success_only=self.success_only,
                lazy=self.lazy,
            )

            self.test_set = DummyDataset()

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
        return DataLoader(self.test_set, batch_size=1)

    def _load_metadata_from_json(self):
        """Parse the dataset metadata JSON to extract obs_mode, control_mode, and physx_backend."""

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

        return env_id, obs_mode, control_mode, physx_backend

    def _infer_padding_masks(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Infers the left and right action padding masks based on the control mode."""
        # When the underlying dataset will generate windows it will need to pad sequences at the start/end of the episode
        # Padding can be either zeros or edge values, it will be the mask to dictate which
        # Generally, not padding mask is passed for observations, as we will default to edge padding
        # For actions instead, we allow users to specify which dimensions to pad as zeros vs edges,
        # as we would like the robot to infer some specific behavior,given its action space nature.
        # e.g. in delta_* control modes we want the robot to stand still after the task is complete,
        # while keeping the gripper closed, thus we need to pad zeros on some entries, and edge on some others
        final_left_mask = None
        final_right_mask = None

        # Padding with edge values should be mainly done with action spaces consisting of absolute values (e.g. pd_ee_pose, pd_joint_pos)
        # Padding with zeros should be preferred for action spaces made by deltas (e.g. pd_ee_delta_pose, pd_joint_delta_pos)
        # However exceptions exist, for example the gripper's entries should alays be edge padded if we want the model to learn to keep the hand closed
        # We will try to handle such cases gracefully, assuming a Franka (or UR5) robot with the last action dimension corresponding to the gripper,
        # but we also allow users to explicitly specify their own masks if we are not working with Franka/UR5
        is_non_abs_mode = "delta" in self.control_mode or "vel" in self.control_mode

        if is_non_abs_mode:
            franka_mask = np.ones(self.act_dim, dtype=bool)
            franka_mask[-1] = False

            # Handle LEFT Mask
            if self.action_left_pad_as_zero_mask is not None:
                final_left_mask = np.array(self.action_left_pad_as_zero_mask, dtype=bool)
                rank_zero_info(
                    "Using explicitly provided action LEFT padding mask given from config."
                )
            else:
                final_left_mask = franka_mask.copy()
                rank_zero_info(
                    f"Inferred action LEFT padding mask for '{self.control_mode}'. "
                    "Edge padding the last dimension only (presumed to be gripper)."
                )

            # Handle RIGHT Mask
            if self.action_right_pad_as_zero_mask is not None:
                final_right_mask = np.array(self.action_right_pad_as_zero_mask, dtype=bool)
                rank_zero_info(
                    "Using explicitly provided action RIGHT padding mask given from config."
                )
            else:
                final_right_mask = franka_mask.copy()
                rank_zero_info(
                    f"Inferred action RIGHT padding mask for '{self.control_mode}'. "
                    "Edge padding the last dimension only (presumed to be gripper)."
                )
        else:
            # Absolute mode fallback
            if self.action_left_pad_as_zero_mask is not None:
                rank_zero_warn(
                    f"A left padding mask was provided, but the control_mode '{self.control_mode}' "
                    "is absolute. The mask will be ignored (using standard edge padding)!"
                )
            if self.action_right_pad_as_zero_mask is not None:
                rank_zero_warn(
                    f"A right padding mask was provided, but the control_mode '{self.control_mode}' "
                    "is absolute. The mask will be ignored (using standard edge padding)!"
                )

        return final_left_mask, final_right_mask
