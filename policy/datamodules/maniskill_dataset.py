import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from lightning.fabric.utilities.rank_zero import rank_zero_warn
from lightning_utilities.core.rank_zero import rank_zero_info
from torch.utils.data import Dataset

from policy.utils import (
    load_h5_data,
    peek_trajectory_dimensions,
    print_dict_tree,
    to_tensor,
)


class ManiSkillDataset(Dataset):
    """Dataset for loading ManiSkill trajectories from HDF5 files."""

    def __init__(
        self,
        dataset_file: str | Path,
        use_phsyx_env_states: bool,
        cond_horizon: int,
        pred_horizon: int,
        act_dim: int | None = None,
        env_state_dim: dict[str, Any] | None = None,
        obs_dim: dict[str, Any] | None = None,
        episodes: list[dict] | None = None,
        action_right_zero_pad_mask: list[bool] | np.ndarray | torch.Tensor | None = None,
        load_count: int = -1,
        success_only: bool = False,
        lazy: bool = False,
        validate_lengths: bool = True,
    ):
        super().__init__()

        self.dataset_file = Path(dataset_file)

        if not self.dataset_file.exists():
            raise FileNotFoundError(f"The dataset file was not found at: {self.dataset_file}")

        if self.dataset_file.suffix not in [".h5", ".hdf5"]:
            raise ValueError(
                f"Invalid file extension '{self.dataset_file.suffix}'. "
                "ManiSkill datasets must be HDF5 files (.h5 or .hdf5)."
            )

        self.use_phsyx_env_states = use_phsyx_env_states
        self.lazy = lazy
        self.validate_lengths = validate_lengths

        self.cond_horizon = cond_horizon
        self.pred_horizon = pred_horizon
        self.act_dim = act_dim
        self.env_state_dim = env_state_dim
        self.obs_dim = obs_dim

        # Load metadata from JSON if episodes aren't provided explicitly
        if episodes is not None:
            self.episodes = episodes
        else:
            json_path = self.dataset_file.with_suffix(".json")
            with open(json_path) as f:
                self.json_data = json.load(f)
            self.episodes = self.json_data["episodes"]

        if action_right_zero_pad_mask is not None:
            if isinstance(action_right_zero_pad_mask, torch.Tensor):
                action_right_zero_pad_mask = action_right_zero_pad_mask.cpu()

            self.action_right_zero_pad_mask = np.asarray(action_right_zero_pad_mask, dtype=bool)
        else:
            self.action_right_zero_pad_mask = None

        if load_count == -1:
            load_count = len(self.episodes)
        else:
            load_count = min(load_count, len(self.episodes))

        self.trajectories: list[dict[str, Any]] = []
        self.slices: list[tuple[int, int, int, int, int, int]] = []

        # Open the H5 file, with different behavior based on lazy flag
        # - lazy: peek dims from first valid episode
        # - not lazy: load full episodes into RAM
        first_valid_episode_id = None
        with h5py.File(self.dataset_file, "r") as data:
            for i in range(load_count):
                eps = self.episodes[i]

                if success_only and not eps.get("success", False):
                    continue

                episode_id = int(eps["episode_id"])
                L = int(eps["elapsed_steps"])  # single source of truth for indexing/windows

                if first_valid_episode_id is None:
                    first_valid_episode_id = episode_id

                if self.lazy:
                    self.trajectories.append({"episode_id": episode_id, "length": L})
                else:
                    traj_group = data[f"traj_{episode_id}"]
                    if not isinstance(traj_group, h5py.Group):
                        raise TypeError(
                            f"Expected HDF5 group traj_{episode_id}, got {type(traj_group)}"
                        )

                    traj_actions = traj_group["actions"]
                    if not isinstance(traj_actions, h5py.Dataset):
                        raise TypeError(
                            f'Expected HDF5 dataset traj_{episode_id}["actions"], got {type(traj_actions)}'
                        )

                    if self.validate_lengths:
                        h5_L = len(traj_actions)
                        if h5_L != L:
                            raise ValueError(
                                f"Length mismatch for episode {episode_id}: "
                                f"JSON elapsed_steps={L} but H5 len(actions)={h5_L}."
                            )

                    trajectory = load_h5_data(traj_group)
                    self.trajectories.append(
                        {
                            "episode_id": episode_id,
                            "obs": trajectory["obs"],
                            "env_states": trajectory["env_states"],
                            "actions": trajectory["actions"],
                        }
                    )

                traj_idx = len(self.trajectories) - 1
                slice = self._compute_trajectory_slices(traj_idx, L)
                self.slices.append(*slice)

            if first_valid_episode_id is None:
                raise ValueError(
                    f"No valid episodes found (success_only={success_only}) in metadata for dataset {self.dataset_file}."
                )

            # Peek dimension from the h5 file if they are not provided explicitly
            if self.act_dim is None or self.env_state_dim is None or self.obs_dim is None:
                act_dim, env_state_dim, obs_dim = peek_trajectory_dimensions(
                    self.dataset_file, first_valid_episode_id
                )

                if self.act_dim is None:
                    self.act_dim = act_dim
                else:
                    assert self.act_dim == act_dim, (
                        f"Provided env_state_dim {self.env_state_dim} does not match peeked dimension {env_state_dim} from the dataset. Please check your configuration."
                    )

                if self.env_state_dim is None:
                    self.env_state_dim = env_state_dim
                else:
                    assert self.env_state_dim == env_state_dim, (
                        f"Provided env_state_dim {self.env_state_dim} does not match peeked dimension {env_state_dim} from the dataset. Please check your configuration."
                    )

                if self.obs_dim is None:
                    self.obs_dim = obs_dim
                else:
                    assert self.obs_dim == obs_dim, (
                        f"Provided obs_dim {self.obs_dim} does not match peeked dimension {obs_dim} from the dataset. Please check your configuration."
                    )

        # Worker-specific HDF5 file handle for DataLoader multiprocessing (used in lazy mode)
        self._h5_file = None

        rank_zero_info(
            f"Dataset initialized: {len(self.slices)} temporal windows "
            f"from {len(self.trajectories)} episodes. "
            f"{self.lazy=}, {self.use_phsyx_env_states=} "
        )
        print_dict_tree(self.trajectories[0], use_rank_zero_info=True)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx: int):
        traj_idx, cond_start, cond_end, act_start, act_end, L = self.slices[idx]
        traj = self.trajectories[traj_idx]

        if self.lazy:
            meta = traj
            episode_id = meta["episode_id"]

            h5_traj = self.h5_file[f"traj_{episode_id}"]
            if not isinstance(h5_traj, h5py.Group):
                raise TypeError(f"Expected HDF5 group traj_{episode_id}, got {type(h5_traj)}")

            h5_actions = h5_traj["actions"]
            if not isinstance(h5_actions, h5py.Dataset):
                raise TypeError(
                    f'Expected HDF5 dataset traj_{episode_id}["actions"], got {type(h5_actions)}'
                )

            if self.validate_lengths:
                h5_L = len(h5_actions)
                if h5_L != meta["length"]:
                    raise ValueError(
                        f"Length mismatch for episode {episode_id}: "
                        f"JSON elapsed_steps={traj['length']} but H5 len(actions)={h5_L}."
                    )

            traj = h5_traj

        cond_src = traj["env_states"] if self.use_phsyx_env_states else traj["obs"]
        if not isinstance(cond_src, h5py.Group | h5py.Dataset | dict | np.ndarray):
            raise TypeError(
                f"Expected env_states or obs to be a dataset or group, got {type(cond_src)}"
            )

        act_src = traj["actions"]
        if not isinstance(act_src, h5py.Dataset | np.ndarray):
            raise TypeError(f"Expected actions to be a dataset, got {type(act_src)}")

        cond_seq = self._slice_and_pad(cond_src, cond_start, cond_end, L, right_pad_mask=None)
        act_seq = self._slice_and_pad(
            act_src, act_start, act_end, L, right_pad_mask=self.action_right_zero_pad_mask
        )

        return {
            "cond_seq": to_tensor(cond_seq),
            "act_seq": to_tensor(act_seq),
        }

    @property
    def h5_file(self) -> h5py.File:
        """Lazy HDF5 file opener to prevent multiprocessing crashes in PyTorch DataLoaders."""
        if self._h5_file is None:
            self._h5_file = h5py.File(self.dataset_file, "r")
        return self._h5_file

    def _compute_trajectory_slices(
        self, traj_idx: int, L: int
    ) -> list[tuple[int, int, int, int, int, int]]:
        """Compute all temporal windows for one episode of length L."""

        # NOTE: To ensure temporal smoothness and momentum, we align the action sequence
        # to start at the same timestamp as the observation sequence (act_start = cond_start).
        # By forcing the model to re-predict actions that occurred during the observation
        # window (the "past"), the result is that the network learns a continuous trajectory where future
        # plans are physically grounded in recent history.
        #
        # More specifically, keep in mind that:
        # - At Training we do NOT throw away the past predictions; they are included in
        #   the loss calculation to act as a temporal anchor for the model.
        # - At Inference instead we discard the past actions (via slicing in get_action) and only
        #   return the actions intended for the present and future.
        # - We do NOT need to increase pred_horizon or act_horizon. The past
        #   actions (cond_horizon - 1) simply occupy the first few slots of the existing
        #   pred_horizon, which is already large enough to contain both the
        #   "anchor" steps and the steps we actually execute.
        #   Limits related to the sizes of the horizons are done in the DiffusionPolicy class, where the act_horizon is available

        slices = []
        for t in range(L):
            cond_start = t - self.cond_horizon + 1
            cond_end = t + 1

            act_start = cond_start
            act_end = act_start + self.pred_horizon

            slices.append((traj_idx, cond_start, cond_end, act_start, act_end, L))

        return slices

    def _slice_and_pad(
        self,
        data: h5py.Group | h5py.Dataset | dict[str, Any] | np.ndarray,
        start: int,
        end: int,
        L: int,
        right_pad_mask: torch.Tensor | np.ndarray | None = None,
    ):
        """Slices the data from start to end, and pads with edge values if the slice goes out of
        bounds."""
        # Treat HDF5 groups like dicts (lazy nested observations)
        if isinstance(data, h5py.Group | dict):
            result = {}
            for k in data.keys():
                nested_data = data[k]
                if not isinstance(nested_data, h5py.Group | h5py.Dataset | dict | np.ndarray):
                    raise TypeError(
                        f"Expected nested HDF5 group or dataset at key '{k}', got {type(nested_data)}"
                    )
                result[k] = self._slice_and_pad(nested_data, start, end, L, right_pad_mask)
            return result

        pad_before = max(0, -start)
        pad_after = max(0, end - L)

        # Prevents negative indices
        valid_start = max(0, min(L, start))
        valid_end = max(0, min(L, end))

        seq = data[valid_start:valid_end]

        # We can either pad with zeros or edge values.

        # Left padding should be edge values
        if pad_before > 0:
            pad_width = [(pad_before, 0)] + [(0, 0)] * (seq.ndim - 1)
            seq = np.pad(seq, pad_width, mode="edge")

        # Right padding can be either zeros or edge values, it will be the mask to dictate which
        # Rght padding with edge values should be mainly done with action spaces consisting of absolute values (e.g. pd_ee_pose, pd_joint_pos)
        # Right Padding with zeros must be preferred for action spaces made by deltas (e.g. pd_ee_delta_pose, pd_joint_delta_pos)
        # However exceptions exist, for example the gripper's entries should alays be edge padded if we want the model to learn to keep the hand closed
        if pad_after > 0:
            # TODO: seems like this is not correct
            if right_pad_mask is not None:
                # pad_mask is True for zeros, False for edge
                pad_frames = np.zeros((pad_after, *seq.shape[1:]), dtype=seq.dtype)
                pad_frames[..., ~right_pad_mask] = seq[-1, ~right_pad_mask]
                seq = np.concatenate([seq, pad_frames], axis=0)
            else:
                # No mask provided, we fallback on edge padding
                pad_width = [(0, pad_after)] + [(0, 0)] * (seq.ndim - 1)
                seq = np.pad(seq, pad_width, mode="edge")

        return seq

    def __del__(self):
        try:
            if self._h5_file is not None:
                self._h5_file.close()
        except Exception as e:
            rank_zero_warn(f"Failed to close HDF5 file handle. Got {e}")
