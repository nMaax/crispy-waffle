import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from lightning.fabric.utilities.rank_zero import rank_zero_warn
from lightning_utilities.core.rank_zero import rank_zero_info
from torch.utils.data import Dataset

from policy.utils import print_dict_tree, to_tensor
from policy.utils.h5_utils import load_h5_data, peek_trajectory_dimension


class TrajectoryDataset(Dataset):
    """Loads ManiSkill demonstrations from HDF5 files and yields raw state-action temporal
    windows."""

    def __init__(
        self,
        dataset_file: str | Path,
        obs_horizon: int,
        pred_horizon: int,
        obs_dim: int | None = None,
        act_dim: int | None = None,
        obs_left_pad_as_zero_mask: list[bool] | np.ndarray | torch.Tensor | None = None,
        obs_right_pad_as_zero_mask: list[bool] | np.ndarray | torch.Tensor | None = None,
        action_left_pad_as_zero_mask: list[bool] | np.ndarray | torch.Tensor | None = None,
        action_right_pad_as_zero_mask: list[bool] | np.ndarray | torch.Tensor | None = None,
        episodes: list[dict] | None = None,
        load_count: int = -1,
        success_only: bool = False,
        lazy: bool = False,
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

        self.lazy = lazy

        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon

        # Convert padding masks to numpy boolean arrays
        self.obs_left_pad_as_zero_mask = self._ensure_numpy_mask(obs_left_pad_as_zero_mask)
        self.obs_right_pad_as_zero_mask = self._ensure_numpy_mask(obs_right_pad_as_zero_mask)
        self.action_left_pad_as_zero_mask = self._ensure_numpy_mask(action_left_pad_as_zero_mask)
        self.action_right_pad_as_zero_mask = self._ensure_numpy_mask(action_right_pad_as_zero_mask)

        # Worker-specific HDF5 file handle for DataLoader multiprocessing (used in lazy mode)
        self._h5_file = None

        self.trajectories: list[dict[str, Any]] = []
        self.slices: list[tuple[int, int, int, int, int, int]] = []

        self._load_metadata(episodes)
        self._build_trajectories_and_slices(load_count, success_only)

        # Peek dimensions if not provided
        if obs_dim is None:
            obs_dim = peek_trajectory_dimension(
                self.dataset_file, f"traj_{self.first_valid_episode_id}", "obs"
            )
        self.obs_dim = obs_dim

        if act_dim is None:
            act_dim = peek_trajectory_dimension(
                self.dataset_file, f"traj_{self.first_valid_episode_id}", "actions"
            )
        self.act_dim = act_dim

        rank_zero_info(
            f"Dataset initialized: {len(self.slices)} temporal windows "
            f"from {len(self.trajectories)} episodes. "
            f"{self.lazy=}"
        )
        print_dict_tree(self.trajectories[0], use_rank_zero_info=True)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx: int):
        """Extracts a temporal window of observations and actions from an episode.

        Shapes:
            returns dict with:
                "obs_seq": [obs_horizon, obs_dim]
                "act_seq": [pred_horizon, act_dim]
        """
        traj_idx, obs_start, obs_end, act_start, act_end, L = self.slices[idx]
        traj = self.trajectories[traj_idx]

        if self.lazy:
            # If in lazy mode, the trajectories actually store simple metadata (episode_id and length)
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

            h5_L = len(h5_actions)
            if h5_L != meta["length"]:
                raise ValueError(
                    f"Length mismatch for episode {episode_id}: "
                    f"JSON elapsed_steps={traj['length']} but H5 len(actions)={h5_L}."
                )

            # So we overwrite trah with the actual H5 object for the episode, on which _slice_and_pad will extract what needed
            traj = h5_traj

        # Now traj is a h5 group for sure
        obs_src = traj["obs"]
        if not isinstance(obs_src, h5py.Group | h5py.Dataset | dict | np.ndarray):
            raise TypeError(f"Expected obs to be a dataset or group, got {type(obs_src)}")

        act_src = traj["actions"]
        if not isinstance(act_src, h5py.Dataset | np.ndarray):
            raise TypeError(f"Expected actions to be a dataset, got {type(act_src)}")

        # So we access it and retrieve what needed, with padding
        obs_seq = self._slice_and_pad(
            obs_src,
            obs_start,
            obs_end,
            L,
            left_pad_as_zero_mask=self.obs_left_pad_as_zero_mask,
            right_pad_as_zero_mask=self.obs_right_pad_as_zero_mask,
        )
        act_seq = self._slice_and_pad(
            act_src,
            act_start,
            act_end,
            L,
            left_pad_as_zero_mask=self.action_left_pad_as_zero_mask,
            right_pad_as_zero_mask=self.action_right_pad_as_zero_mask,
        )

        obs_seq = to_tensor(obs_seq)
        act_seq = to_tensor(act_seq)

        return {
            "obs_seq": obs_seq,
            "act_seq": act_seq,
        }

    @property
    def h5_file(self) -> h5py.File:
        """Lazy HDF5 file opener to prevent multiprocessing crashes in PyTorch DataLoaders."""
        if self._h5_file is None:
            self._h5_file = h5py.File(self.dataset_file, "r")
        return self._h5_file

    def _ensure_numpy_mask(
        self, mask: list[bool] | np.ndarray | torch.Tensor | None
    ) -> np.ndarray | None:
        """Helper to unify padding masks into boolean numpy arrays."""
        if mask is None:
            return None
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu()

        numpy_mask = np.asarray(mask, dtype=bool)
        return numpy_mask

    def _load_metadata(self, episodes: list[dict] | None) -> None:
        """Loads episode metadata from provided list or JSON file."""
        if episodes is not None:
            self.episodes = episodes
        else:
            json_path = self.dataset_file.with_suffix(".json")
            if not json_path.exists():
                raise FileNotFoundError(f"Metadata JSON not found. Expected at {json_path}.")
            with open(json_path) as f:
                self.episodes = json.load(f)["episodes"]

    def _build_trajectories_and_slices(self, load_count: int, success_only: bool) -> None:
        """Reads HDF5 to build trajectory buffers and temporal windows."""
        if load_count == -1:
            count = len(self.episodes)
        else:
            count = min(load_count, len(self.episodes))
        self.first_valid_episode_id = None

        # Open the H5 file, with different behavior based on lazy flag
        # - lazy: peek dims from first valid episode
        # - not lazy: load full episodes into RAM
        with h5py.File(self.dataset_file, "r") as data:
            for i in range(count):
                eps = self.episodes[i]
                if success_only and not eps.get("success", False):
                    continue

                episode_id = int(eps["episode_id"])
                L = int(eps["elapsed_steps"])

                if self.first_valid_episode_id is None:
                    self.first_valid_episode_id = episode_id

                if self.lazy:
                    # Only load simple metadata
                    self.trajectories.append({"episode_id": episode_id, "length": L})
                else:
                    # Directly load the actual h5 group object
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

                    h5_L = len(traj_actions)
                    if h5_L != L:
                        raise ValueError(
                            f"Length mismatch for episode {episode_id}: "
                            f"JSON elapsed_steps={L} but H5 len(actions)={h5_L}."
                        )

                    # And convert the trajectory group to an actual dictionary of numpy arrays
                    trajectory = load_h5_data(traj_group)
                    self.trajectories.append(
                        {
                            "episode_id": episode_id,
                            "obs": trajectory["obs"],
                            "actions": trajectory["actions"],
                        }
                    )

                # Either lazy or non-lazy, still pre-compute the slices (i.e., windows) for each trajectory
                # Lazy mode contains everything necessary for the windoeing function to work
                traj_idx = len(self.trajectories) - 1
                slice = self._compute_trajectory_slices(traj_idx, L)
                self.slices.extend(slice)

        if self.first_valid_episode_id is None:
            raise ValueError(f"No valid episodes found (success_only={success_only}).")

    def _compute_trajectory_slices(
        self, traj_idx: int, L: int
    ) -> list[tuple[int, int, int, int, int, int]]:
        """Compute all temporal windows for one episode of length L.

        This is very specific to Diffusion Policy.
        """

        # NOTE: To ensure temporal smoothness and momentum, we align the action sequence
        # to start at the same timestamp as the observation sequence (act_start = obs_start).
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
        #   actions (obs_horizon - 1) simply occupy the first few slots of the existing
        #   pred_horizon, which is already large enough to contain both the
        #   "anchor" steps and the steps we actually execute.
        #   Limits related to the sizes of the horizons are done in the DiffusionPolicy class, where the act_horizon is available
        #
        # Example horizons:
        #   |t|
        # |o|o|                             conditioning: 2
        # | |a|a|a|a|a|a|a|a|               actions executed: 8
        # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
        #
        # Note that in the code below act_start and act_end actually refer to the prediction bounds (|p|, not |a|)
        # in fact, the |a| line is irrelevant in the dataset, it doesn't exist here. It only concerns the DiffusionPolicy class
        # so we reuse the terminology "action" in pladce of "prediction" for simplicity

        slices = []
        for t in range(L):
            obs_start = t - self.obs_horizon + 1
            obs_end = t + 1

            act_start = obs_start
            act_end = act_start + self.pred_horizon

            slices.append((traj_idx, obs_start, obs_end, act_start, act_end, L))

        return slices

    def _slice_and_pad(
        self,
        data: h5py.Group | h5py.Dataset | dict[str, Any] | np.ndarray,
        start: int,
        end: int,
        L: int,
        left_pad_as_zero_mask: torch.Tensor | np.ndarray | None = None,
        right_pad_as_zero_mask: torch.Tensor | np.ndarray | None = None,
    ):
        """Slices a sequence from start to end, padding out-of-bounds indices with zeros or edge
        values.

        Shapes:
            data: [episode_len, ...], e.g. [227, 48]
            returns: [end - start, ...], e.g. [2, 48]
        """
        # Treat HDF5 groups like dicts (lazy nested observations)
        if isinstance(data, h5py.Group | dict):
            result = {}
            for k in data.keys():
                nested_data = data[k]
                if not isinstance(nested_data, h5py.Group | h5py.Dataset | dict | np.ndarray):
                    raise TypeError(
                        f"Expected nested HDF5 group or dataset at key '{k}', got {type(nested_data)}"
                    )
                result[k] = self._slice_and_pad(
                    nested_data,
                    start,
                    end,
                    L,
                    left_pad_as_zero_mask=left_pad_as_zero_mask,
                    right_pad_as_zero_mask=right_pad_as_zero_mask,
                )
            return result

        pad_before = max(0, -start)
        pad_after = max(0, end - L)

        # Prevents negative indices
        valid_start = max(0, min(L, start))
        valid_end = max(0, min(L, end))

        seq = data[valid_start:valid_end]

        # We can either pad with zeros or edge values
        if pad_before > 0:
            if left_pad_as_zero_mask is not None:
                # action_left_pad_as_zero_mask is True for padding with zeros, False for padding with edge
                pad_frames = np.zeros((pad_before, *seq.shape[1:]), dtype=seq.dtype)

                columns_to_copy = ~left_pad_as_zero_mask
                first_frame = seq[0]
                values_to_repeat = first_frame[columns_to_copy]
                pad_frames[..., columns_to_copy] = values_to_repeat

                seq = np.concatenate([pad_frames, seq], axis=0)
            else:
                # No mask provided, we fallback on edge padding
                pad_width = [(pad_before, 0)] + [(0, 0)] * (seq.ndim - 1)
                seq = np.pad(seq, pad_width, mode="edge")

        if pad_after > 0:
            if right_pad_as_zero_mask is not None:
                # action_right_pad_as_zero_mask is True for padding with zeros, False for padding with edge
                pad_frames = np.zeros((pad_after, *seq.shape[1:]), dtype=seq.dtype)

                columns_to_copy = ~right_pad_as_zero_mask
                last_frame = seq[-1]
                values_to_repeat = last_frame[columns_to_copy]
                pad_frames[..., columns_to_copy] = values_to_repeat

                seq = np.concatenate([seq, pad_frames], axis=0)
            else:
                # No mask provided, we fallback on edge padding
                # generally we don't fallback on zeros since we can't know for sure if 0 is a meaningful value in the space we padding (e.g. the gripper accept only -1 and +1, 0 would break it),
                # edge padding instead is a more safe choice, as we know for sure the valuers at the edge of the sequence are valid values in the space. Tho in delta action spaces this can lead to a bad behaviour
                pad_width = [(0, pad_after)] + [(0, 0)] * (seq.ndim - 1)
                seq = np.pad(seq, pad_width, mode="edge")

        return seq

    def __del__(self):
        try:
            if self._h5_file is not None:
                self._h5_file.close()
        except Exception as e:
            rank_zero_warn(f"Failed to close HDF5 file handle. Got {e}")
