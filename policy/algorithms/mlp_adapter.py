import functools

import hydra_zen
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from policy.utils.normalizer import TensorNormalizer
from policy.utils.typing_utils import HydraConfigFor


class MLPAdapter(L.LightningModule):
    def __init__(
        self,
        network: HydraConfigFor[nn.Module],
        optimizer: HydraConfigFor[functools.partial[Optimizer]],
        lr_scheduler: HydraConfigFor[functools.partial[LRScheduler]] | None = None,
        loss_mask_indices: list[int] | None = None,  # Added Loss Masking
    ):
        super().__init__()

        self.save_hyperparameters()

        self.network_config = network
        self.network: torch.nn.Module | None = None

        self.optimizer_config = optimizer
        self.optimizer: Optimizer | None = None

        self.lr_scheduler_config = lr_scheduler
        self.lr_scheduler: LRScheduler | None = None

        self.x_normalizer = TensorNormalizer(network.input_dim)
        self.y_normalizer = TensorNormalizer(network.output_dim)

        self.loss_mask_indices = loss_mask_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inference pass: Normalize -> Predict -> Unnormalize"""

        if self.network is None:
            raise ValueError("Network is not configured. Call configure_model() before inference.")

        x_norm = self.x_normalizer.normalize(x)
        y_norm_pred = self.network(x_norm)
        y_pred = self.y_normalizer.unnormalize(y_norm_pred)
        return y_pred

    def setup(self, stage: str):
        if stage == "fit" and not self.x_normalizer.is_fit:
            # WARN: is there a better way to do this without loading everything in memory?

            if not hasattr(self.trainer, "datamodule"):
                raise ValueError(
                    f"Trainer does not have a datamodule. Are you sure you are training? Here the stage is {stage}"
                )

            train_set = self.trainer.datamodule.train_set

            all_x = []
            all_y = []
            for i in range(len(train_set)):
                x, y = train_set[i]
                all_x.append(x)
                all_y.append(y)

            self.x_normalizer.fit(torch.stack(all_x))
            self.y_normalizer.fit(torch.stack(all_y))

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
            raise ValueError("Network is not configured.")

        x, y = batch

        x_norm = self.x_normalizer.normalize(x)
        y_norm = self.y_normalizer.normalize(y)

        y_norm_hat = self.network(x_norm)

        if self.loss_mask_indices is not None:
            y_norm_hat = y_norm_hat[..., self.loss_mask_indices]
            y_norm = y_norm[..., self.loss_mask_indices]

        return F.mse_loss(y_norm_hat, y_norm)

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
