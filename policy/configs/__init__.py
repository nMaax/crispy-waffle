"""All the configuration classes for the policy."""

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from policy.configs.config import Config
from policy.utils.env_vars import get_constant

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)


OmegaConf.register_new_resolver("constant", get_constant)
OmegaConf.register_new_resolver("eval", eval)

__all__ = [
    "Config",
]
