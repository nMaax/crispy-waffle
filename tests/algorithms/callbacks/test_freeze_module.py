import lightning as L
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from policy.algorithms.callbacks.freeze_module import FreezeModuleCallback


class LazyModule(L.LightningModule):
    """Mirrors the algorithms in this repo: submodules are built in ``configure_model``, not
    ``__init__``, so anything freezing them must run after that hook."""

    def __init__(self):
        super().__init__()
        self.embedder: nn.Module | None = None
        self.network: nn.Module | None = None

    def configure_model(self) -> None:
        if self.network is not None:
            return
        self.embedder = nn.Linear(4, 4)
        self.network = nn.Linear(4, 2)

    def training_step(self, batch, _batch_idx):
        (x,) = batch
        assert self.embedder is not None and self.network is not None
        return self.network(self.embedder(x)).square().mean()

    def configure_optimizers(self):
        # Same requires_grad filter as BaseDiffusionAgent.configure_optimizers.
        return torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=1.0)


def _loader() -> DataLoader:
    return DataLoader(TensorDataset(torch.randn(4, 4)), batch_size=2)


def _fit(module: L.LightningModule, callbacks: list[L.Callback]) -> None:
    L.Trainer(
        accelerator="cpu",
        max_steps=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=callbacks,
    ).fit(module, train_dataloaders=_loader())


def test_freeze_module_freezes_lazily_built_submodule():
    """Regression test for hook ordering: ``setup()`` (what ``BaseFinetuning`` uses) fires before
    ``configure_model()``, when ``embedder`` is still None."""
    module = LazyModule()

    _fit(module, [FreezeModuleCallback(attr_name="embedder")])

    assert module.embedder is not None
    for p in module.embedder.parameters():
        assert not p.requires_grad


class SnapshotCallback(L.Callback):
    """Captures the weights once training starts, i.e. after ``configure_model``."""

    def __init__(self):
        self.before: dict[str, torch.Tensor] = {}

    def on_train_start(self, _trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.before = {n: p.detach().clone() for n, p in pl_module.named_parameters()}


def test_freeze_module_leaves_frozen_weights_untouched_while_others_train():
    module = LazyModule()
    snapshot = SnapshotCallback()

    _fit(module, [FreezeModuleCallback(attr_name="embedder"), snapshot])

    after = dict(module.named_parameters())
    assert snapshot.before, "snapshot callback never ran"

    frozen = {n: p for n, p in after.items() if n.startswith("embedder.")}
    trainable = {n: p for n, p in after.items() if n.startswith("network.")}
    assert frozen and trainable

    for name, param in frozen.items():
        assert torch.equal(param, snapshot.before[name]), f"{name} moved despite being frozen"
        assert param.grad is None, f"{name} received a gradient despite being frozen"

    assert any(not torch.equal(p, snapshot.before[n]) for n, p in trainable.items()), (
        "no unfrozen parameter moved, so the test proves nothing about freezing"
    )


def test_freeze_module_raises_when_attribute_is_none():
    module = LazyModule()
    callback = FreezeModuleCallback(attr_name="embedder")

    with pytest.raises(AttributeError, match="is None"):
        callback.on_fit_start(L.Trainer(), module)


def test_freeze_module_raises_when_attribute_is_missing():
    module = LazyModule()
    module.configure_model()
    callback = FreezeModuleCallback(attr_name="nonexistent")

    with pytest.raises(AttributeError):
        callback.on_fit_start(L.Trainer(), module)
