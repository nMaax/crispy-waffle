import h5py
import torch

from policy.datasets.trajectory_dataset import TrajectoryDataset
from policy.utils import to_tensor


class GoalConditionedTrajectoryDataset(TrajectoryDataset):
    def __init__(self, *args, abs_goal: bool = True, her_ratio: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.abs_goal = abs_goal
        self.her_ratio = her_ratio

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        batch = super().__getitem__(idx)

        traj_idx, obs_start, obs_end, act_start, act_end, L = self.slices[idx]
        traj_meta = self.trajectories[traj_idx]

        # The current timestep is the last observation in the sequence window
        current_t = obs_end - 1

        # HER: Randomly sample a future state as the goal with probability her_ratio
        if self.her_ratio > 0.0 and torch.rand(1).item() < self.her_ratio:
            # torch.randint(low, high) samples in range [low, high - 1]
            goal_t = torch.randint(current_t, L, (1,)).item()
        else:
            goal_t = L - 1

        if self.lazy:
            episode_id = traj_meta["episode_id"]

            h5_traj = self.h5_file[f"traj_{episode_id}"]
            if not isinstance(h5_traj, h5py.Group):
                raise ValueError(
                    f"Expected a group for trajectory {episode_id}, but got {type(h5_traj)}"
                )

            obs_dataset = h5_traj["obs"]
            if not isinstance(obs_dataset, h5py.Dataset):
                raise ValueError(
                    f"Expected a dataset for observations in trajectory {episode_id}, but got {type(obs_dataset)}"
                )

            final_obs = obs_dataset[goal_t]
        else:
            final_obs = traj_meta["obs"][goal_t]

        goal_state = to_tensor(final_obs, dtype=torch.float32)

        if self.obs_transform is not None:
            goal_state = self.obs_transform(goal_state)

        # TODO: kind of dirty, we are assuming the canonicalized tensor
        cubeA_pos = goal_state[25:28]
        cubeB_pos = goal_state[32:35]

        if self.abs_goal:
            goal_state = torch.cat([cubeA_pos, cubeB_pos])
        else:
            goal_state = cubeB_pos - cubeA_pos

        batch["goal_state"] = goal_state
        return batch
