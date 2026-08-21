import pytest
from omegaconf import OmegaConf

from policy.configs.config import Config
from policy.utils.hydra_utils import (
    get_checkpoint_branch,
    get_experiment_phase,
    parse_slice,
    resolve_dictconfig,
    slice_size,
)


def test_resolve_dictconfig_valid():
    cfg = Config(algorithm={"_target_": "dummy"})
    dict_cfg = OmegaConf.structured(cfg)
    resolved = resolve_dictconfig(dict_cfg)
    assert isinstance(resolved, Config)
    assert resolved.branch is not None


def test_resolve_dictconfig_invalid():
    dict_cfg = OmegaConf.create({"key": "value"})
    with pytest.raises(
        TypeError, match="Expected the resolved config to be an instance of `Config`"
    ):
        resolve_dictconfig(dict_cfg)


def test_get_experiment_phase():
    assert get_experiment_phase("Algo__Data__Trainer__train") == "train"
    assert get_experiment_phase("Algo__Data__Trainer__test") == "test"
    assert get_experiment_phase("Algo__Data__Trainer__test__ZeroShot") == "test"
    assert get_experiment_phase("default") is None
    assert get_experiment_phase("Algo__Data__Trainer") is None


def test_get_checkpoint_branch(tmp_path):
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()
    ckpt_path = checkpoints_dir / "last.ckpt"
    ckpt_path.touch()
    hydra_dir = tmp_path / ".hydra"
    hydra_dir.mkdir()
    (hydra_dir / "config.yaml").write_text("branch: my-feature-branch\n")

    assert get_checkpoint_branch(str(ckpt_path)) == "my-feature-branch"


def test_get_checkpoint_branch_missing_hydra_config(tmp_path):
    ckpt_path = tmp_path / "last.ckpt"
    assert get_checkpoint_branch(str(ckpt_path)) is None


def test_parse_slice():
    # Int inputs
    assert parse_slice(5) == 5
    assert parse_slice("5") == 5

    # Slice string inputs
    assert parse_slice("25:48") == slice(25, 48, None)
    assert parse_slice(":25") == slice(None, 25, None)
    assert parse_slice("48:") == slice(48, None, None)
    assert parse_slice("1:10:2") == slice(1, 10, 2)


def test_slice_size():
    assert slice_size(10) == 1
    assert slice_size(slice(10, 25)) == 15

    with pytest.raises(TypeError, match="Expected int or slice"):
        slice_size("invalid")
