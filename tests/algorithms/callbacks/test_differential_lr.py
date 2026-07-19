import pytest
import torch
import torch.nn as nn

from policy.algorithms.callbacks.differential_lr import DifferentialLRCallback


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


def test_differential_lr_freezes_before_training():
    module = DummyModule()
    callback = DifferentialLRCallback(backbone_lr=1e-5)

    callback.freeze_before_training(module)

    for p in module.network.parameters():
        assert not p.requires_grad
    for p in module.planner.parameters():
        assert p.requires_grad


def test_differential_lr_unfreezes_at_epoch_zero():
    module = DummyModule()
    callback = DifferentialLRCallback(backbone_lr=1e-5)
    callback.freeze_before_training(module)

    optimizer = torch.optim.Adam(module.planner.parameters(), lr=1e-3)

    # Epoch 0: unfreezes network and adds param group
    callback.finetune_function(module, current_epoch=0, optimizer=optimizer)

    for p in module.network.parameters():
        assert p.requires_grad
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[1]["lr"] == 1e-5


def test_differential_lr_legacy_kwarg():
    callback = DifferentialLRCallback(unet_lr=2e-4)
    assert callback.backbone_lr == 2e-4


def test_differential_lr_custom_attributes():
    module = DummyCustomModule()
    callback = DifferentialLRCallback(
        backbone_lr=1e-4, backbone_attr="backbone", adapter_attr="adapter"
    )
    callback.freeze_before_training(module)

    for p in module.backbone.parameters():
        assert not p.requires_grad
    for p in module.adapter.parameters():
        assert p.requires_grad


def test_differential_lr_missing_attr_raises():
    module = DummyModule()
    callback = DifferentialLRCallback(backbone_attr="nonexistent")
    with pytest.raises(AttributeError, match="does not have attribute 'nonexistent'"):
        callback.freeze_before_training(module)
