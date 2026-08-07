import pytest
import torch

from policy.algorithms.networks.pooling import AttentionPooling, MLPPooling


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

    linear_default = pooling_default.net[1]
    linear_explicit = pooling_explicit.net[1]
    assert linear_default.in_features == linear_explicit.in_features == 3 * 12
