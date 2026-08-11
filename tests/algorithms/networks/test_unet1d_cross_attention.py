import hydra_zen
import pytest
import torch
from hydra import compose, initialize_config_module
from hydra.errors import InstantiationException
from omegaconf import OmegaConf

from policy.utils.typing_utils import DiffusionNetworkProtocol


def test_unet1d_cross_attention_instantiates_and_runs():
    with initialize_config_module(config_module="policy.configs", version_base="1.2"):
        cfg = compose(config_name="algorithm/network/unet1d_cross_attention")

    # The composed config is nested under algorithm.network (config-group structure)
    net_cfg = cfg.algorithm.network
    OmegaConf.set_struct(net_cfg, False)

    batch_size = 128
    horizon = 16
    act_dim = 8
    proprio_dim = 18
    context_dim = 8
    context_len = 6  # e.g. 2 obs timesteps x 3 per-object tokens

    net_cfg.act_dim = act_dim
    net_cfg.cond_dims = {"obs": {"proprio": proprio_dim}, "context": context_dim}
    net_cfg.obs_horizon = horizon
    network = hydra_zen.instantiate(net_cfg)

    sample = torch.randn(batch_size, horizon, act_dim)
    timestep = torch.randint(0, 100, (batch_size,))
    proprio = torch.randn(batch_size, horizon, proprio_dim)
    context = torch.randn(batch_size, context_len, context_dim)

    output = network(
        sample,
        timestep,
        external_cond={"obs": {"proprio": proprio}, "context": context},
    )
    assert output.shape == sample.shape

    # Protocol conformance
    assert isinstance(network, DiffusionNetworkProtocol)

    # Backward pass: grads must be finite, including through the new cross-attention blocks
    loss = output.sum()
    loss.backward()
    for name, p in network.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{name} got no gradient"
            assert torch.isfinite(p.grad).all(), f"{name} got a non-finite gradient"


def test_unet1d_cross_attention_requires_a_context_entry():
    with initialize_config_module(config_module="policy.configs", version_base="1.2"):
        cfg = compose(config_name="algorithm/network/unet1d_cross_attention")

    net_cfg = cfg.algorithm.network
    OmegaConf.set_struct(net_cfg, False)

    net_cfg.act_dim = 8
    net_cfg.cond_dims = {"obs": {"proprio": 18}}  # no "context" entry
    net_cfg.obs_horizon = 16

    with pytest.raises(InstantiationException, match="context"):
        hydra_zen.instantiate(net_cfg)
