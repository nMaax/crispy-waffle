import pytest
import torch

from policy.algorithms.networks.encoder.pooling import (
    AttentionPooling,
    MLPPooling,
)


@pytest.mark.parametrize(
    "pooling",
    [
        MLPPooling(dim=12, obs_horizon=3),
        AttentionPooling(dim=12),
    ],
    ids=["mlp", "attention"],
)
def test_pooling_reduces_the_time_axis(pooling):
    batch_size, obs_horizon, dim = 8, 3, 12
    x = torch.randn(batch_size, obs_horizon, dim)

    output = pooling(x)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, dim)

    loss = output.sum()
    loss.backward()
    for p in pooling.parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()


def test_attention_pooling_is_sensitive_to_every_timestep():
    """The whole point of attention pooling over flatten+MLP: it summarizes by content, not
    position, but every timestep must still be able to influence the pooled output."""
    torch.manual_seed(0)
    dim, obs_horizon = 12, 4
    pooling = AttentionPooling(dim=dim)
    pooling.eval()

    x = torch.randn(4, obs_horizon, dim)
    with torch.no_grad():
        out = pooling(x)
        for t in range(obs_horizon):
            x_perturbed = x.clone()
            x_perturbed[:, t, :] += torch.randn(dim)
            out_perturbed = pooling(x_perturbed)
            assert not torch.allclose(out, out_perturbed), (
                f"Perturbing timestep {t} should change the pooled output."
            )


def test_mlp_pooling_rejects_a_window_length_other_than_configured():
    pooling = MLPPooling(dim=12, obs_horizon=3)

    with pytest.raises(RuntimeError):
        pooling(torch.randn(4, 2, 12))


def test_mlp_pooling_accepts_tokens_per_step():
    """A K > 1 tokenizer (e.g. one token per object) folds K tokens per timestep into the token
    axis this pooling head flattens, so its width must account for K too."""
    batch_size, obs_horizon, tokens_per_step, dim = 8, 3, 3, 12
    pooling = MLPPooling(dim=dim, obs_horizon=obs_horizon, tokens_per_step=tokens_per_step)
    x = torch.randn(batch_size, obs_horizon * tokens_per_step, dim)

    output = pooling(x)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, dim)

    loss = output.sum()
    loss.backward()
    for p in pooling.parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()


def test_mlp_pooling_tokens_per_step_defaults_to_one():
    """tokens_per_step=1 (the default) must reproduce today's exact width."""
    pooling_default = MLPPooling(dim=12, obs_horizon=3)
    pooling_explicit = MLPPooling(dim=12, obs_horizon=3, tokens_per_step=1)

    mlp_default = pooling_default.net[1]
    mlp_explicit = pooling_explicit.net[1]
    assert mlp_default.in_features == mlp_explicit.in_features == 3 * 12


def test_mlp_pooling_accepts_hidden_dims_and_output_dim():
    pooling = MLPPooling(dim=12, obs_horizon=3, output_dim=16, hidden_dims=[64, 32])
    x = torch.randn(4, 3, 12)
    out = pooling(x)
    assert out.shape == (4, 16)


@pytest.mark.parametrize(
    "mode, expected_shape",
    [
        ("all", (8, 12)),
        ("objects", (8, 4, 12)),
        ("time", (8, 3, 12)),
    ],
)
def test_attention_pooling_modes_4d(mode, expected_shape):
    batch_size, obs_horizon, tokens_per_step, dim = 8, 4, 3, 12
    x = torch.randn(batch_size, obs_horizon, tokens_per_step, dim)

    pooling = AttentionPooling(dim=dim, mode=mode)
    out = pooling(x)
    assert out.shape == expected_shape

    loss = out.sum()
    loss.backward()
    for p in pooling.parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()


@pytest.mark.parametrize(
    "mode, expected_shape",
    [
        ("all", (8, 12)),
        ("objects", (8, 4, 12)),
        ("time", (8, 3, 12)),
    ],
)
def test_mlp_pooling_modes_4d(mode, expected_shape):
    batch_size, obs_horizon, tokens_per_step, dim = 8, 4, 3, 12
    x = torch.randn(batch_size, obs_horizon, tokens_per_step, dim)

    pooling = MLPPooling(
        dim=dim,
        obs_horizon=obs_horizon,
        tokens_per_step=tokens_per_step,
        mode=mode,
    )
    out = pooling(x)
    assert out.shape == expected_shape

    loss = out.sum()
    loss.backward()
    for p in pooling.parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()


def test_base_pooling_properties():
    all_pool = AttentionPooling(dim=12, mode="all")
    assert all_pool.pools_time is True
    assert all_pool.pools_objects is True

    obj_pool = AttentionPooling(dim=12, mode="objects")
    assert obj_pool.pools_time is False
    assert obj_pool.pools_objects is True

    time_pool = AttentionPooling(dim=12, mode="time")
    assert time_pool.pools_time is True
    assert time_pool.pools_objects is False


def test_pooling_invalid_mode():
    with pytest.raises(ValueError, match="Unknown pooling mode"):
        AttentionPooling(dim=12, mode="invalid")  # type: ignore[arg-type]
