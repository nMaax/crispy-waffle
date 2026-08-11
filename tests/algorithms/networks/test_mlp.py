import hydra_zen
import pytest
import torch
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from policy.algorithms.networks.mlp import MLP
from policy.algorithms.networks.pooling import AttentionPooling
from policy.algorithms.networks.residual_mlp import ResidualMLP


def _load_net_cfg(config_name: str):
    with initialize_config_module(config_module="policy.configs", version_base="1.2"):
        cfg = compose(config_name=f"algorithm/network/{config_name}")
    net_cfg = cfg.algorithm.network
    OmegaConf.set_struct(net_cfg, False)
    return net_cfg


def _load_embedder_cfg_with_pooling(embedder_name: str, pooling_config_name: str):
    with initialize_config_module(config_module="policy.configs", version_base="1.2"):
        cfg = compose(
            config_name=f"algorithm/embedder/{embedder_name}",
            overrides=[f"algorithm/embedder/pooling={pooling_config_name}"],
        )
    embedder_cfg = cfg.algorithm.embedder
    OmegaConf.set_struct(embedder_cfg, False)
    return embedder_cfg


def test_mlp_instantiates_and_runs():
    net_cfg = _load_net_cfg("mlp")

    batch_size = 32
    input_dim = 16
    output_dim = 4
    hidden_dims = [64, 128]

    net_cfg.input_dim = input_dim
    net_cfg.output_dim = output_dim
    net_cfg.hidden_dims = hidden_dims
    network = hydra_zen.instantiate(net_cfg)

    sample = torch.randn(batch_size, input_dim)
    output = network(sample)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, output_dim), (
        f"Expected shape {(batch_size, output_dim)}, got {output.shape}"
    )

    expected_layer_count = (len(hidden_dims) * 2) + 1
    assert len(network.net) == expected_layer_count

    # Backward pass: grads must be finite
    loss = output.sum()
    loss.backward()
    for p in network.parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()


def test_linear_config_zeroes_weights():
    """The `linear` config (hidden_dims=[], bias=False) triggers the weight-zeroing branch in
    MLP.__init__."""
    net_cfg = _load_net_cfg("linear")

    input_dim = 16
    output_dim = 4

    net_cfg.input_dim = input_dim
    net_cfg.output_dim = output_dim
    network = hydra_zen.instantiate(net_cfg)

    assert network.hidden_dims == []
    assert network.bias is False
    # A single Linear layer with no hidden layers
    assert len(network.net) == 1
    # The weight-zeroing branch should have produced an all-zero weight matrix
    assert network.net[0].weight.abs().sum().item() == 0.0

    # Forward still produces a (batch, output_dim) tensor of zeros (zero weight, no bias)
    out = network(torch.randn(8, input_dim))
    assert out.shape == (8, output_dim)
    assert torch.all(out == 0.0)


def test_residual_mlp_instantiates_and_runs():
    with initialize_config_module(config_module="policy.configs", version_base="1.2"):
        cfg = compose(config_name="algorithm/embedder/residual_mlp")
    embedder_cfg = cfg.algorithm.embedder
    OmegaConf.set_struct(embedder_cfg, False)

    input_dim = 16
    embedder_cfg.input_dim = input_dim
    embedder = hydra_zen.instantiate(embedder_cfg)

    sample = torch.randn(32, input_dim)
    output = embedder(sample)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (32, 64)

    # Test auto output_dim matching input_dim when output_dim is None
    embedder_cfg.output_dim = None
    embedder_auto = hydra_zen.instantiate(embedder_cfg)
    output_auto = embedder_auto(sample)
    assert output_auto.shape == (32, input_dim)
    assert isinstance(embedder_auto.shortcut, torch.nn.Identity)

    # Test network/residual_mlp config
    net_cfg = _load_net_cfg("residual_mlp")
    net_cfg.input_dim = input_dim
    net_cfg.output_dim = 64
    network_from_cfg = hydra_zen.instantiate(net_cfg)
    assert network_from_cfg(sample).shape == (32, 64)


@pytest.mark.parametrize("embedder_name", ["mlp", "residual_mlp"])
def test_mlp_family_embedder_with_attention_pooling_collapses_the_time_axis(embedder_name):
    """The per-token ("siamese") MLP/ResidualMLP embedders can plug in the same pooling heads
    SelfAttention uses, via `algorithm/embedder/pooling: attention`."""
    embedder_cfg = _load_embedder_cfg_with_pooling(embedder_name, "attention")

    batch_size, obs_horizon, input_dim, output_dim = 8, 3, 16, 12
    embedder_cfg.input_dim = input_dim
    embedder_cfg.output_dim = output_dim
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


@pytest.mark.parametrize("embedder_cls", [MLP, ResidualMLP])
def test_mlp_family_pooling_flattens_a_multi_token_per_step_axis_before_pooling(embedder_cls):
    """K tokens per timestep (e.g. per-object tokenization) must fold into the sequence axis a
    pooling head expects, exactly like SelfAttention does before pooling (see `pool_tokens` in
    policy/algorithms/networks/pooling.py)."""
    batch_size, obs_horizon, tokens_per_step, input_dim, output_dim = 4, 3, 2, 8, 12
    embedder = embedder_cls(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=[16],
        pooling=AttentionPooling(dim=output_dim),
    )

    sample = torch.randn(batch_size, obs_horizon, tokens_per_step, input_dim)
    output = embedder(sample)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, output_dim)

    loss = output.sum()
    loss.backward()
    for p in embedder.parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()
