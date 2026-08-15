import pytest
import torch

from policy.algorithms.networks.utils import (
    as_task_only,
    derive_task_dim,
    validate_proprio_dim,
)


def test_validate_proprio_dim_mapping():
    validate_proprio_dim({"proprio": 18, "tcp": 8}, 18)

    with pytest.raises(ValueError, match="must contain 'proprio' key"):
        validate_proprio_dim({"tcp": 8}, 18)

    with pytest.raises(ValueError, match="does not match proprio_dim"):
        validate_proprio_dim({"proprio": 10, "tcp": 8}, 18)


def test_validate_proprio_dim_int():
    validate_proprio_dim(48, 18)

    with pytest.raises(ValueError, match="must be >= proprio_dim"):
        validate_proprio_dim(10, 18)


def test_validate_proprio_dim_invalid_type():
    with pytest.raises(ValueError, match="must be an integer or dict"):
        validate_proprio_dim("invalid", 18)


def test_derive_task_dim_mapping():
    assert derive_task_dim({"proprio": 18, "task_a": 10, "task_b": 20}, 18) == 30
    assert derive_task_dim({"proprio": 18, "task_a": 10, "task_b": 20}, 18, task_dim=30) == 30

    with pytest.raises(ValueError, match="does not match task_dim"):
        derive_task_dim({"proprio": 18, "task_a": 10, "task_b": 20}, 18, task_dim=31)


def test_derive_task_dim_int():
    assert derive_task_dim(48, 18) == 30
    assert derive_task_dim(48, 18, task_dim=30) == 30

    with pytest.raises(ValueError, match="does not match task_dim"):
        derive_task_dim(48, 18, task_dim=31)


def test_as_task_only_already_task_width():
    tensor = torch.randn(2, 5)
    resolved = as_task_only(tensor, proprio_dim=3, task_dim=5)
    assert torch.equal(resolved, tensor)


def test_as_task_only_strips_leading_proprio():
    tensor = torch.arange(16, dtype=torch.float32).view(2, 8)
    resolved = as_task_only(tensor, proprio_dim=3, task_dim=5)
    assert torch.equal(resolved, tensor[..., 3:])


def test_as_task_only_rejects_mismatched_width():
    tensor = torch.randn(2, 7)
    with pytest.raises(ValueError, match="Expected width 5"):
        as_task_only(tensor, proprio_dim=3, task_dim=5)
