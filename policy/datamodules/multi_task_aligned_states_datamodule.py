from typing import Any

import hydra_zen
import lightning as L
from torch.utils.data import ConcatDataset, DataLoader

from policy.datasets import TaskConditionedAlignedStatesDataset


class MultiTaskAlignedStatesDataModule(L.LightningDataModule):
    """Provides task-conditioned (source, target) state batches from multiple environments to train
    multi-task state translators."""

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
        test_datasets = []

        for env_id, task_dm in self.task_dms.items():
            task_dm.setup(stage)
            task_idx = self.env_to_idx[env_id]

            if stage == "fit" or stage is None:
                train_datasets.append(
                    TaskConditionedAlignedStatesDataset(task_dm.train_set, env_id, task_idx)
                )

            if stage in ("fit", "validate") or stage is None:
                val_datasets.append(
                    TaskConditionedAlignedStatesDataset(task_dm.val_set, env_id, task_idx)
                )

            if stage == "test" or stage is None:
                test_datasets.append(
                    TaskConditionedAlignedStatesDataset(task_dm.test_dataset, env_id, task_idx)
                )

        if stage == "fit" or stage is None:
            self.train_set = ConcatDataset(train_datasets)

        if stage in ("fit", "validate") or stage is None:
            self.val_set = ConcatDataset(val_datasets)

        if stage == "test" or stage is None:
            self.test_set = ConcatDataset(test_datasets)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
