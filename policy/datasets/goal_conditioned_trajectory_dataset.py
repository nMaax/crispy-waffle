import h5py
import numpy as np
import torch

from policy.datasets.trajectory_dataset import TrajectoryDataset
from policy.utils import recursive_index, to_tensor


class GoalConditionedTrajectoryDataset(TrajectoryDataset):
    def __init__(self, *args, her_ratio: float = 0.8, **kwargs):
        super().__init__(*args, **kwargs)
        self.her_ratio = her_ratio

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        batch = super().__getitem__(idx)

        traj_idx, obs_start, obs_end, act_start, act_end, L = self.slices[idx]
        traj_meta = self.trajectories[traj_idx]

        # The current timestep is the last observation in the sequence window
        current_t = obs_end - 1

        action_chunk_size = act_end - act_start
        min_goal_t = min(current_t + action_chunk_size, L - 1)

        # HER: Randomly sample a future state as the goal with probability her_ratio
        if self.her_ratio > 0.0 and torch.rand(1).item() < self.her_ratio:
            if min_goal_t < L - 1:
                # Our goal cannot be within the action chunk we are learning from
                # so it can only be sampled from the end of it, to the end of the episode
                # torch.randint(low, high) samples in range [low, high - 1]
                goal_t = torch.randint(min_goal_t, L, (1,)).item()
            else:
                # If the episode ends before the action chunk completes,
                # the only logical goal is the final frame.
                # remind also that such action chunk will simply be a padding over the last effective action
                goal_t = L - 1
        else:
            goal_t = L - 1

        if self.lazy:
            episode_id = traj_meta["episode_id"]

            h5_traj = self.h5_file[f"traj_{episode_id}"]
            if not isinstance(h5_traj, h5py.Group):
                raise ValueError(
                    f"Expected a h5py.Group for trajectory {episode_id}, but got {type(h5_traj)!r}"
                )

            obs_dataset = h5_traj["obs"]
            if not isinstance(obs_dataset, h5py.Group | h5py.Dataset | dict | np.ndarray):
                raise ValueError(
                    f"Expected obs to be a h5py.Dataset or numpy array, or h5py.Group or dictionary, in trajectory {episode_id}, but got {type(obs_dataset)!r}"
                )

            future_obs = recursive_index(obs_dataset, goal_t)
        else:
            future_obs = recursive_index(traj_meta["obs"], goal_t)

        goal = to_tensor(future_obs, dtype=torch.float32)

        if self.obs_transform is not None:
            goal = self.obs_transform(goal)

        batch["goal"] = goal
        return batch
