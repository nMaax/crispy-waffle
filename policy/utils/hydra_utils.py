"""Utility functions related to working with [Hydra](https://hydra.cc)."""

from __future__ import annotations

from logging import getLogger as get_logger

from omegaconf import DictConfig, OmegaConf

from policy.configs.config import Config

logger = get_logger(__name__)


def resolve_dictconfig(dict_config: DictConfig) -> Config:
    """Resolve all interpolations in the `DictConfig`."""

    config = OmegaConf.to_object(dict_config)

    if not isinstance(config, Config):
        raise TypeError(
            f"Expected the resolved config to be an instance of `Config`, but got {type(config)} "
            "Please check your config files and ensure they are properly structured."
        )

    return config


def parse_slice(slice_def: str | int) -> slice | int:
    """Converts a string like '25:48', '48:', or ':25' into a Python slice object."""

    if isinstance(slice_def, int):
        return slice_def

    if ":" not in slice_def:
        return int(slice_def)

    parts = slice_def.split(":")
    start = int(parts[0]) if parts[0] else None
    end = int(parts[1]) if parts[1] else None
    step = int(parts[2]) if len(parts) > 2 and parts[2] else None

    return slice(start, end, step)


def slice_size(s):
    if isinstance(s, int):
        return 1
    elif isinstance(s, slice):
        return s.stop - s.start
    else:
        raise TypeError("Expected int or slice")
