import pytest
from omegaconf import OmegaConf

from policy.configs.config import Config
from policy.utils.hydra_utils import parse_slice, resolve_dictconfig, slice_size


def test_resolve_dictconfig_valid():
    cfg = Config(algorithm={"_target_": "dummy"})
    dict_cfg = OmegaConf.structured(cfg)
    resolved = resolve_dictconfig(dict_cfg)
    assert isinstance(resolved, Config)


def test_resolve_dictconfig_invalid():
    dict_cfg = OmegaConf.create({"key": "value"})
    with pytest.raises(TypeError, match="Expected the resolved config to be an instance of `Config`"):
        resolve_dictconfig(dict_cfg)


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
