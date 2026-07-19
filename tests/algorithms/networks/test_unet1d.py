import hydra_zen
import torch
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from policy.utils.typing_utils import DiffusionNetworkProtocol


def test_unet1d_instantiates_and_runs():
    with initialize_config_module(config_module="policy.configs", version_base="1.2"):
        cfg = compose(config_name="algorithm/network/unet1d")

    # The composed config is nested under algorithm.network (config-group structure)
    net_cfg = cfg.algorithm.network
    OmegaConf.set_struct(net_cfg, False)

    batch_size = 128
    horizon = 16
    act_dim = 8
    external_cond_dim = 67

    net_cfg.act_dim = act_dim
    net_cfg.external_cond_dim = external_cond_dim
    network = hydra_zen.instantiate(net_cfg)

    sample = torch.randn(batch_size, horizon, act_dim)
    timestep = torch.randint(0, 100, (batch_size,))
    flatten_obs = torch.randn(batch_size, external_cond_dim)

    output = network(sample, timestep, obs=flatten_obs)
    assert output.shape == sample.shape

    # Protocol conformance
    assert isinstance(network, DiffusionNetworkProtocol)

    # Backward pass: grads must be finite
    loss = output.sum()
    loss.backward()
    for p in network.parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()
