from typing import Any

import hydra_zen
import lightning as L
from torch.utils.data import ConcatDataset, DataLoader

from .task_conditioned_dataset import TaskConditionedDataset


class MultiTaskDataModule(L.LightningDataModule):
    def __init__(
        self,
        task_configs: dict[str, dict[str, Any]],
        batch_size: int = 256,
        num_workers: int = 4,
    ):
        super().__init__()
        self.task_configs = task_configs
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Dynamically assign an integer ID to each environment (e.g. "PlaceSphere-v1" -> 0)
        self.env_to_idx = {env_id: idx for idx, env_id in enumerate(task_configs.keys())}

        # Instantiate the underlying DataModules
        self.task_dms = {}
        for env_id, cfg in task_configs.items():
            self.task_dms[env_id] = hydra_zen.instantiate(cfg)

    def setup(self, stage: str | None = None):
        train_datasets = []
        val_datasets = []

        # TODO: a little doubtful on this? Review it later, it should include also the stage='validate' and 'test' cases(?)
        for env_id, task_dm in self.task_dms.items():
            task_dm.setup(stage)

            if stage == "fit" or stage is None:
                task_idx = self.env_to_idx[env_id]
                train_set = task_dm.train_set
                val_set = task_dm.val_set

                train_datasets.append(TaskConditionedDataset(train_set, env_id, task_idx))
                val_datasets.append(TaskConditionedDataset(val_set, env_id, task_idx))

        if stage == "fit" or stage is None:
            self.train_set = ConcatDataset(train_datasets)
            self.val_set = ConcatDataset(val_datasets)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
