from policy.datamodules.trajectory_datamodule import TrajectoryDataModule
from policy.datasets.goal_conditioned_trajectory_dataset import GoalConditionedTrajectoryDataset


class GoalConditionedTrajectoryDataModule(TrajectoryDataModule):
    def __init__(self, *args, her_ratio: float = 0.8, **kwargs):
        super().__init__(*args, **kwargs)
        self.her_ratio = her_ratio

    def _create_dataset(self, episodes, left_mask, right_mask, obs_transform):
        return GoalConditionedTrajectoryDataset(
            dataset_file=self.dataset_file,
            obs_horizon=self.obs_horizon,
            pred_horizon=self.pred_horizon,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            obs_left_pad_as_zero_mask=None,
            obs_right_pad_as_zero_mask=None,
            action_left_pad_as_zero_mask=left_mask,
            action_right_pad_as_zero_mask=right_mask,
            episodes=episodes,
            load_count=self.load_count,
            success_only=self.success_only,
            lazy=self.lazy,
            obs_transform=obs_transform,
            her_ratio=self.her_ratio,
        )
