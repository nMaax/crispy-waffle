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
