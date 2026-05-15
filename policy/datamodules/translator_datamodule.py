from collections.abc import Callable

import lightning.pytorch as L
import torch
from torch.utils.data import DataLoader

from policy.datasets import ManiSkillDataset, TranslatorDataset

from .maniskill_datamodule import ManiSkillDataModule


class TranslatorDataModule(L.LightningDataModule):
    def __init__(
        self,
        base_datamodule: ManiSkillDataModule,
        adapter: Callable[[torch.Tensor], torch.Tensor],
        batch_size: int = 256,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        """
        Args:
            base_datamodule: An instantiated ManiSkillDataModule.
            adapter: An instantiated deterministic adapter class.
        """
        super().__init__()
        self.base_datamodule = base_datamodule
        self.adapter = adapter
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def prepare_data(self):
        self.base_datamodule.prepare_data()

    def setup(self, stage: str | None = None):
        self.base_datamodule.setup(stage)

        if stage == "fit" or stage is None:
            train_set = self.base_datamodule.train_set
            if not isinstance(train_set, ManiSkillDataset):
                raise ValueError(
                    f"Expected base_datamodule.train_set to be a ManiSkillDataset, but got {type(train_set)}"
                )
            self.train_set = TranslatorDataset(base_dataset=train_set, adapter=self.adapter)

            val_set = self.base_datamodule.val_set
            if not isinstance(val_set, ManiSkillDataset):
                raise ValueError(
                    f"Expected base_datamodule.val_set to be a ManiSkillDataset, but got {type(val_set)}"
                )
            self.val_set = TranslatorDataset(base_dataset=val_set, adapter=self.adapter)

        if stage == "test" or stage is None:
            test_set = self.base_datamodule.test_set
            if not isinstance(test_set, ManiSkillDataset):
                raise ValueError(
                    f"Expected base_datamodule.test_set to be a ManiSkillDataset, but got {type(test_set)}"
                )
            self.test_dataset = TranslatorDataset(base_dataset=test_set, adapter=self.adapter)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
            shuffle=False,
        )
