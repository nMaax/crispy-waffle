import functools

import hydra_zen
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from policy.utils.typing_utils import HydraConfigFor


class MLPAdapter(L.LightningModule):
    def __init__(
        self,
        network: HydraConfigFor[nn.Module],
        optimizer: HydraConfigFor[functools.partial[Optimizer]],
        lr_scheduler: HydraConfigFor[functools.partial[LRScheduler]] | None = None,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.network_config = network
        self.network: torch.nn.Module | None = None

        self.optimizer_config = optimizer
        self.optimizer: Optimizer | None = None

        self.lr_scheduler_config = lr_scheduler
        self.lr_scheduler: LRScheduler | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.network is None:
            raise ValueError(
                "Network is not configured. Call configure_model() before using the model."
            )

        return self.network(x)

    def configure_model(self) -> None:
        if self.network is not None:
            return

        self.network = hydra_zen.instantiate(self.network_config)

    def configure_optimizers(self) -> Optimizer | dict:

        # Optimizers and schedulers could actually be made in one shot, without partial,
        # however I prefer to follow the template prescription, just for coherence

        optimizer_partial = hydra_zen.instantiate(self.optimizer_config)
        optimizer = optimizer_partial(self.parameters())

        if self.lr_scheduler_config is not None:
            lr_scheduler_partial = hydra_zen.instantiate(self.lr_scheduler_config)
            lr_scheduler = lr_scheduler_partial(optimizer)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        else:
            return optimizer

    def _compute_loss(self, batch) -> torch.Tensor:
        if self.network is None:
            raise ValueError("Network is not configured. Call configure_model() before training.")

        x, y = batch
        y_hat = self.network(x)
        return F.mse_loss(y_hat, y)

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss
