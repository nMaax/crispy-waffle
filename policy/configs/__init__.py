"""All the configuration classes for the policy."""

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from policy.configs.config import Config
from policy.utils.env_vars import get_constant

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)


OmegaConf.register_new_resolver("constant", get_constant)
OmegaConf.register_new_resolver("eval", eval)


def add_configs_to_hydra_store():
    from policy.utils.remote_launcher_plugin import RemoteSlurmQueueConf

    """Adds all configs to the Hydra Config store."""
    ConfigStore.instance().store(
        group="hydra/launcher",
        name="remote_submitit_slurm",
        node=RemoteSlurmQueueConf,
        provider="Mila",
    )


__all__ = [
    "Config",
]
