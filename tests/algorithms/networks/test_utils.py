import pytest

from policy.algorithms.networks.utils import (
    derive_task_dim,
    resolve_proprio_dim,
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


def test_resolve_proprio_dim_derives_from_dict():
    assert resolve_proprio_dim({"proprio": 18, "tcp": 8}) == 18


def test_resolve_proprio_dim_honours_an_explicit_value():
    assert resolve_proprio_dim({"proprio": 18, "tcp": 8}, 18) == 18
    assert resolve_proprio_dim(48, 18) == 18


def test_resolve_proprio_dim_flat_spec_requires_an_explicit_value():
    """A flat obs_dim carries no field names, so proprio_dim cannot silently default to 0."""
    with pytest.raises(ValueError, match="must be provided explicitly"):
        resolve_proprio_dim(48)


def test_resolve_proprio_dim_dict_without_proprio_key():
    with pytest.raises(ValueError, match="must contain 'proprio' key"):
        resolve_proprio_dim({"tcp": 8})


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
