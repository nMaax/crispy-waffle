import os
from pathlib import Path

import pytest

import policy.utils.env_vars as env_vars
from policy.utils.env_vars import (
    DATA_DIR,
    NUM_WORKERS,
    REPO_ROOTDIR,
    get_constant,
)


def test_env_vars_constants():
    assert isinstance(REPO_ROOTDIR, Path)
    assert (REPO_ROOTDIR / "README.md").exists()
    assert isinstance(DATA_DIR, Path)
    assert isinstance(NUM_WORKERS, int)
    assert NUM_WORKERS > 0


def test_get_constant_globals():
    val = get_constant("REPO_ROOTDIR")
    assert val == REPO_ROOTDIR


def test_get_constant_module_path():
    val = get_constant("os.name")
    assert val == os.name


def test_get_constant_fallback(monkeypatch):
    # Set NETWORK_DATASETS_DIR to None
    monkeypatch.setattr(env_vars, "NETWORK_DATASETS_DIR", None)

    # Calling with (NETWORK_DATASETS_DIR, DATA_DIR) should skip None and return DATA_DIR
    res = get_constant("NETWORK_DATASETS_DIR", "DATA_DIR")
    assert res == DATA_DIR


def test_get_constant_not_found():
    with pytest.raises(RuntimeError, match="Could not find non-None value for name non_existent"):
        get_constant("non_existent_constant_xyz")

    with pytest.raises(
        RuntimeError, match="Could not find non-None value for names \\('var1', 'var2'\\)"
    ):
        get_constant("var1", "var2")
