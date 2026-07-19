from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from policy.algorithms.callbacks.finetune_planner import FinetunePlannerCallback


class DummyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Linear(4, 4)
        self.planner = nn.Linear(4, 4)


class DummyCustomModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.adapter = nn.Linear(4, 4)


def test_finetune_planner_freezes_network_makes_planner_trainable():
    module = DummyModule()
    callback = FinetunePlannerCallback(unfreeze_step=100, backbone_lr=1e-5)

    callback.freeze_before_training(module)

    for p in module.network.parameters():
        assert not p.requires_grad
    for p in module.planner.parameters():
        assert p.requires_grad


def test_finetune_planner_unfreezes_at_step():
    module = DummyModule()
    callback = FinetunePlannerCallback(unfreeze_step=100, backbone_lr=1e-5)
    callback.freeze_before_training(module)

    optimizer = torch.optim.Adam(module.planner.parameters(), lr=1e-3)
    trainer = MagicMock()
    trainer.optimizers = [optimizer]

    # Before step: network is frozen
    trainer.global_step = 50
    callback.on_train_batch_start(trainer, module, {}, 0)
    for p in module.network.parameters():
        assert not p.requires_grad
    assert len(optimizer.param_groups) == 1

    # At unfreeze step: network is unfrozen and added to optimizer
    trainer.global_step = 100
    callback.on_train_batch_start(trainer, module, {}, 0)
    for p in module.network.parameters():
        assert p.requires_grad
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[1]["lr"] == 1e-5


def test_finetune_planner_legacy_kwargs():
    callback = FinetunePlannerCallback(unet_unfreeze_step=200, unet_lr=2e-4)
    assert callback.unfreeze_step == 200
    assert callback.backbone_lr == 2e-4


def test_finetune_planner_custom_attributes():
    module = DummyCustomModule()
    callback = FinetunePlannerCallback(
        unfreeze_step=50, backbone_lr=1e-4, backbone_attr="backbone", adapter_attr="adapter"
    )
    callback.freeze_before_training(module)

    for p in module.backbone.parameters():
        assert not p.requires_grad
    for p in module.adapter.parameters():
        assert p.requires_grad


def test_finetune_planner_missing_attr_raises():
    module = DummyModule()
    callback = FinetunePlannerCallback(backbone_attr="missing_network")
    with pytest.raises(AttributeError, match="does not have attribute 'missing_network'"):
        callback.freeze_before_training(module)
