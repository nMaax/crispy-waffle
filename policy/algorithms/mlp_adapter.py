from typing import Any

import hydra
import lightning.pytorch as pl
import torch
import torch.nn.functional as F


class MLPAdapter(pl.LightningModule):
    def __init__(
        self,
        network: torch.nn.Module,
        optimizer: dict[str, Any],
        lr_scheduler: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.network = network

        self.optimizer_config = optimizer
        self.lr_scheduler_config = lr_scheduler

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass purely for inference."""
        return self.network(x)

    def _compute_loss(self, batch) -> torch.Tensor:
        x, y = batch
        y_hat = self.forward(x)
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

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.optimizer_config, params=self.network.parameters()
        )

        if self.lr_scheduler_config is not None:
            scheduler = hydra.utils.instantiate(self.lr_scheduler_config, optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
