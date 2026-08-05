import hydra_zen
import pytest
import torch
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from policy.algorithms.networks.mlp import MLP
from policy.algorithms.networks.residual_mlp import ResidualMLP


def _load_embedder_cfg(config_name: str):
    with initialize_config_module(config_module="policy.configs", version_base="1.2"):
        cfg = compose(config_name=f"algorithm/embedder/{config_name}")
    embedder_cfg = cfg.algorithm.embedder
    OmegaConf.set_struct(embedder_cfg, False)
    return embedder_cfg


def _load_self_attention_cfg_with_pooling(pooling_config_name: str):
    with initialize_config_module(config_module="policy.configs", version_base="1.2"):
        cfg = compose(
            config_name="algorithm/embedder/self_attention",
            overrides=[f"algorithm/embedder/pooling={pooling_config_name}"],
        )
    embedder_cfg = cfg.algorithm.embedder
    OmegaConf.set_struct(embedder_cfg, False)
    return embedder_cfg


def test_self_attention_embedder_instantiates_and_runs():
    embedder_cfg = _load_embedder_cfg("self_attention")

    batch_size = 8
    obs_horizon = 3
    input_dim = 16
    output_dim = 12

    embedder_cfg.input_dim = input_dim
    embedder_cfg.output_dim = output_dim
    embedder_cfg.obs_horizon = obs_horizon
    embedder = hydra_zen.instantiate(embedder_cfg)

    sample = torch.randn(batch_size, obs_horizon, input_dim)
    output = embedder(sample)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, obs_horizon, output_dim)

    # Backward pass: grads must be finite.
    loss = output.sum()
    loss.backward()
    for p in embedder.parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()


def test_self_attention_embedder_rejects_a_longer_window_than_configured():
    embedder_cfg = _load_embedder_cfg("self_attention")
    embedder_cfg.input_dim = 16
    embedder_cfg.output_dim = 8
    embedder_cfg.obs_horizon = 2
    embedder = hydra_zen.instantiate(embedder_cfg)

    with pytest.raises(ValueError, match="obs_horizon"):
        embedder(torch.randn(4, 3, 16))


def test_self_attention_embedder_mixes_across_timesteps():
    """The whole point of this embedder over the per-token MLP/ResidualMLP ones: the embedding of
    one token can depend on other tokens in the window."""
    torch.manual_seed(0)
    input_dim, output_dim, obs_horizon = 16, 12, 2
    embedder_cfg = _load_embedder_cfg("self_attention")
    embedder_cfg.input_dim = input_dim
    embedder_cfg.output_dim = output_dim
    embedder_cfg.obs_horizon = obs_horizon
    embedder = hydra_zen.instantiate(embedder_cfg)
    embedder.eval()

    x = torch.randn(4, obs_horizon, input_dim)
    x_perturbed = x.clone()
    x_perturbed[:, 0, :] += torch.randn(input_dim)

    with torch.no_grad():
        out = embedder(x)
        out_perturbed = embedder(x_perturbed)

    # Perturbing token 0 changes token 1's output too: attention mixed information across t.
    assert not torch.allclose(out[:, 1, :], out_perturbed[:, 1, :])


@pytest.mark.parametrize("pooling_config_name", ["mlp", "attention"])
def test_self_attention_embedder_with_pooling_collapses_the_time_axis(pooling_config_name):
    embedder_cfg = _load_self_attention_cfg_with_pooling(pooling_config_name)

    batch_size = 8
    obs_horizon = 3
    input_dim = 16
    output_dim = 12

    embedder_cfg.input_dim = input_dim
    embedder_cfg.output_dim = output_dim
    embedder_cfg.obs_horizon = obs_horizon
    embedder = hydra_zen.instantiate(embedder_cfg)

    sample = torch.randn(batch_size, obs_horizon, input_dim)
    output = embedder(sample)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, output_dim)

    loss = output.sum()
    loss.backward()
    for p in embedder.parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()


def test_per_token_embedders_do_not_mix_across_timesteps():
    """Contrast with the per-token ("siamese") MLP/ResidualMLP embedders: perturbing one timestep
    never affects another's embedding, since each row is embedded independently."""
    input_dim, output_dim = 16, 12
    for embedder in (
        MLP(input_dim=input_dim, output_dim=output_dim, hidden_dims=[32]),
        ResidualMLP(input_dim=input_dim, output_dim=output_dim, hidden_dims=[32]),
    ):
        x = torch.randn(4, 2, input_dim)
        x_perturbed = x.clone()
        x_perturbed[:, 0, :] += torch.randn(input_dim)

        with torch.no_grad():
            out = embedder(x)
            out_perturbed = embedder(x_perturbed)

        assert torch.equal(out[:, 1, :], out_perturbed[:, 1, :])
