from typing import cast

import h5py
import torch

from policy.datasets.trajectory_dataset import TrajectoryDataset
from policy.utils import to_tensor


class GoalConditionedTrajectoryDataset(TrajectoryDataset):
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        batch = super().__getitem__(idx)

        traj_idx, obs_start, obs_end, act_start, act_end, L = self.slices[idx]
        traj_meta = self.trajectories[traj_idx]

        if self.lazy:
            episode_id = traj_meta["episode_id"]
            h5_traj = cast(h5py.Group, self.h5_file[f"traj_{episode_id}"])

            obs_dataset = cast(h5py.Dataset, h5_traj["obs"])
            final_obs = obs_dataset[-1]
        else:
            final_obs = traj_meta["obs"][-1]

        goal_state = to_tensor(final_obs, dtype=torch.float32)

        if self.obs_transform is not None:
            goal_state = self.obs_transform(goal_state)

        batch["goal_state"] = goal_state
        return batch
