import hydra_zen
import pytest
import torch
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from policy.algorithms.networks.decoder.unet1d import CrossAttentionDecoder1D, FiLMDecoder1D
from policy.algorithms.networks.encoder import ConditioningContract


def _compose_film_decoder(**overrides):
    with initialize_config_module(config_module="policy.configs", version_base="1.2"):
        cfg = compose(config_name="algorithm/decoder/film_decoder1d")
    net_cfg = cfg.algorithm.decoder
    OmegaConf.set_struct(net_cfg, False)
    for key, value in overrides.items():
        net_cfg[key] = value
    return net_cfg


def _compose_cross_attn_decoder(**overrides):
    with initialize_config_module(config_module="policy.configs", version_base="1.2"):
        cfg = compose(config_name="algorithm/decoder/cross_attention_decoder1d")
    net_cfg = cfg.algorithm.decoder
    OmegaConf.set_struct(net_cfg, False)
    for key, value in overrides.items():
        net_cfg[key] = value
    return net_cfg


def test_film_decoder1d_instantiates_and_runs():
    """FiLMDecoder1D built directly from a hand-built ConditioningContract, bypassing
    ConditioningEncoder entirely -- exercises the decoder in isolation."""
    batch_size, horizon, act_dim = 8, 4, 4
    obs_width = 67

    cond_dims = ConditioningContract(step_dim=obs_width)
    net_cfg = _compose_film_decoder(act_dim=act_dim, obs_horizon=horizon)
    decoder = hydra_zen.instantiate(net_cfg, cond_dims=cond_dims)
    assert isinstance(decoder, FiLMDecoder1D)

    sample = torch.randn(batch_size, horizon, act_dim)
    timestep = torch.randint(0, 100, (batch_size,))
    encoded_cond = {"obs": torch.randn(batch_size, horizon, obs_width)}

    output = decoder(sample, timestep, external_cond=encoded_cond)
    assert output.shape == sample.shape

    loss = output.sum()
    loss.backward()
    for p in decoder.parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()


def test_film_decoder1d_with_a_goal_entry_instantiates_and_runs():
    """A static goal alongside 'obs' -- exercises get_film_width's global_dim addition."""
    batch_size, horizon, act_dim = 4, 4, 4
    obs_width, goal_width = 15, 10

    cond_dims = ConditioningContract(step_dim=obs_width, global_dim=goal_width)
    net_cfg = _compose_film_decoder(act_dim=act_dim, obs_horizon=horizon)
    decoder = hydra_zen.instantiate(net_cfg, cond_dims=cond_dims)

    sample = torch.randn(batch_size, horizon, act_dim)
    timestep = torch.randint(0, 100, (batch_size,))
    encoded_cond = {
        "obs": torch.randn(batch_size, horizon, obs_width),
        "goal": torch.randn(batch_size, goal_width),
    }

    output = decoder(sample, timestep, external_cond=encoded_cond)
    assert output.shape == sample.shape


def test_cross_attention_decoder1d_instantiates_and_runs():
    """CrossAttentionDecoder1D built directly from a hand-built ConditioningContract with
    context_dim."""
    batch_size, pred_horizon, obs_horizon, act_dim = 8, 16, 2, 4
    proprio_dim, context_dim, seq_len = 18, 8, 3

    cond_dims = ConditioningContract(
        step_dim=proprio_dim,
        context_dim=context_dim,
    )
    net_cfg = _compose_cross_attn_decoder(act_dim=act_dim, obs_horizon=obs_horizon)
    decoder = hydra_zen.instantiate(net_cfg, cond_dims=cond_dims)
    assert isinstance(decoder, CrossAttentionDecoder1D)

    sample = torch.randn(batch_size, pred_horizon, act_dim)
    timestep = torch.randint(0, 100, (batch_size,))
    encoded_cond = {
        "obs": torch.randn(batch_size, obs_horizon, proprio_dim),
        "context": torch.randn(batch_size, seq_len, context_dim),
    }

    output = decoder(sample, timestep, external_cond=encoded_cond)
    assert output.shape == sample.shape

    loss = output.sum()
    loss.backward()
    for name, p in decoder.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{name} got no gradient"
            assert torch.isfinite(p.grad).all(), f"{name} got a non-finite gradient"


def test_cross_attention_decoder1d_requires_a_context_dim():
    """CrossAttentionDecoder1D construction rejects a contract with no context_dim."""
    cond_dims = ConditioningContract(step_dim=10)
    net_cfg = _compose_cross_attn_decoder(act_dim=4, obs_horizon=4)
    with pytest.raises(Exception, match="requires context_dim to be set"):
        hydra_zen.instantiate(net_cfg, cond_dims=cond_dims)


def test_cross_attention_decoder1d_expects_the_declared_context_key():
    """A well-formed decoder still rejects an external_cond payload missing its context key."""
    cond_dims = ConditioningContract(
        step_dim=10,
        context_dim=8,
    )
    net_cfg = _compose_cross_attn_decoder(act_dim=4, obs_horizon=2)
    decoder = hydra_zen.instantiate(net_cfg, cond_dims=cond_dims)

    sample = torch.randn(2, 8, 4)
    timestep = torch.randint(0, 100, (2,))
    with pytest.raises(ValueError, match="expected a 'context' entry"):
        decoder(sample, timestep, external_cond={"obs": torch.randn(2, 2, 10)})
