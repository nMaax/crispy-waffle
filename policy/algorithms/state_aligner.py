import functools

import hydra_zen
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from policy.transforms import TensorNormalizer
from policy.utils.hydra_utils import parse_slice
from policy.utils.typing_utils import HydraConfigFor


class StateAligner(L.LightningModule):
    """Trains a neural network to map/align states from one domain to another.

    The loss can be masked to only consider a subset of the output dimensions.
    """

    def __init__(
        self,
        network: HydraConfigFor[nn.Module],
        optimizer: HydraConfigFor[functools.partial[Optimizer]],
        lr_scheduler: HydraConfigFor[functools.partial[LRScheduler]] | None = None,
        loss_mask_slices: list[str | int] | None = None,
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

        self.loss_mask_slices = loss_mask_slices

    def setup(self, stage: str) -> None:
        if stage == "fit" and not self.x_normalizer.is_fit:
            self._parse_loss_mask()
            self._configure_normalizers()

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.network is None:
            raise ValueError("Network is not configured. Call configure_model() before inference.")

        x_norm = self.x_normalizer.normalize(x)
        y_norm_pred = self.network(x_norm)
        y_pred = self.y_normalizer.unnormalize(y_norm_pred)
        return y_pred

    def _compute_loss(self, batch) -> torch.Tensor:
        if self.network is None:
            raise ValueError("Network is not configured.")

        x, y = batch

        x_norm = self.x_normalizer.normalize(x)
        y_norm = self.y_normalizer.normalize(y)

        y_norm_hat = self.network(x_norm)

        if self.loss_mask is not None:
            y_norm_hat = y_norm_hat[..., self.loss_mask]
            y_norm = y_norm[..., self.loss_mask]

        return F.mse_loss(y_norm_hat, y_norm)

    def _parse_loss_mask(self) -> None:
        mask = torch.zeros(self.network_config.output_dim, dtype=torch.bool)

        if self.loss_mask_slices is not None:
            for s in self.loss_mask_slices:
                mask[parse_slice(s)] = True
        else:
            mask[:] = True

        self.register_buffer("loss_mask", mask)

    def _configure_normalizers(self) -> None:
        dm = getattr(self.trainer, "datamodule", None)
        if dm is None:
            raise ValueError(
                "Datamodule is not available in the trainer. Make sure to set the datamodule before training."
            )
        base_dm = dm.base_datamodule

        train_set = base_dm.train_set

        all_x = []
        all_y = []

        for traj in train_set.trajectories:
            if train_set.lazy:
                ep_id = traj["episode_id"]
                h5_traj = train_set.h5_file[f"traj_{ep_id}"]
                x_ep = torch.from_numpy(h5_traj["obs"][:])
            else:
                x_ep = torch.from_numpy(traj["obs"])

            with torch.no_grad():
                y_ep = dm.adapter.apply(x_ep)

            all_x.append(x_ep)
            all_y.append(y_ep)

        self.x_normalizer.fit(torch.cat(all_x, dim=0))
        self.y_normalizer.fit(torch.cat(all_y, dim=0))
