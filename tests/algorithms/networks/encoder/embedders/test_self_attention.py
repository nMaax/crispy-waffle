import hydra_zen
import pytest
import torch
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from policy.algorithms.networks.encoder.embedders import MLP, ResidualMLP


def _load_embedder_cfg(config_name: str):
    with initialize_config_module(config_module="policy.configs", version_base="1.2"):
        cfg = compose(config_name=f"algorithm/embedder/{config_name}")
    embedder_cfg = cfg.algorithm.embedder
    OmegaConf.set_struct(embedder_cfg, False)
    return embedder_cfg



def test_self_attention_instantiates_and_runs():
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


def test_self_attention_has_a_feed_forward_sublayer():
    """Regression guard: this is a full (post-norm) transformer block -- attention then FFN --
    not attention-only. Guards against the FFN sublayer silently disappearing later."""
    embedder_cfg = _load_embedder_cfg("self_attention")
    output_dim = 12
    embedder_cfg.input_dim = 16
    embedder_cfg.output_dim = output_dim
    embedder_cfg.obs_horizon = 3
    embedder = hydra_zen.instantiate(embedder_cfg)

    assert isinstance(embedder.mlp[0], torch.nn.Linear)
    assert embedder.mlp[0].out_features == 4 * output_dim
    assert isinstance(embedder.norm2, torch.nn.LayerNorm)


def test_self_attention_rejects_a_longer_window_than_configured():
    embedder_cfg = _load_embedder_cfg("self_attention")
    embedder_cfg.input_dim = 16
    embedder_cfg.output_dim = 8
    embedder_cfg.obs_horizon = 2
    embedder = hydra_zen.instantiate(embedder_cfg)

    with pytest.raises(ValueError, match="obs_horizon"):
        embedder(torch.randn(4, 3, 16))


def test_self_attention_mixes_across_timesteps():
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


def test_self_attention_accepts_multiple_tokens_per_step():
    """A 4D input (K tokens per timestep, e.g. one per object) is embedded to the same leading
    shape, output_dim swapped in for input_dim."""
    embedder_cfg = _load_embedder_cfg("self_attention")
    batch_size, obs_horizon, tokens_per_step, input_dim, output_dim = 8, 3, 3, 16, 12
    embedder_cfg.input_dim = input_dim
    embedder_cfg.output_dim = output_dim
    embedder_cfg.obs_horizon = obs_horizon
    embedder = hydra_zen.instantiate(embedder_cfg)

    sample = torch.randn(batch_size, obs_horizon, tokens_per_step, input_dim)
    output = embedder(sample)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, obs_horizon, tokens_per_step, output_dim)

    loss = output.sum()
    loss.backward()
    for p in embedder.parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()


def test_self_attention_with_tokens_per_step_one_matches_the_3d_call():
    """tokens_per_step=1, expressed as a 4D input with K=1, must reproduce the plain 3D call
    bit-for-bit -- the core backward-compatibility guarantee for this generalization."""
    embedder_cfg = _load_embedder_cfg("self_attention")
    batch_size, obs_horizon, input_dim, output_dim = 8, 3, 16, 12
    embedder_cfg.input_dim = input_dim
    embedder_cfg.output_dim = output_dim
    embedder_cfg.obs_horizon = obs_horizon
    embedder = hydra_zen.instantiate(embedder_cfg)
    embedder.eval()

    x_3d = torch.randn(batch_size, obs_horizon, input_dim)
    x_4d = x_3d.unsqueeze(2)

    with torch.no_grad():
        out_3d = embedder(x_3d)
        out_4d = embedder(x_4d)

    assert torch.equal(out_3d, out_4d.squeeze(2))


def test_self_attention_mixes_across_objects_at_the_same_timestep():
    """Tokens belonging to different objects at the same timestep attend to each other too, not
    just tokens across timesteps."""
    torch.manual_seed(0)
    embedder_cfg = _load_embedder_cfg("self_attention")
    batch_size, obs_horizon, tokens_per_step, input_dim, output_dim = 4, 2, 3, 16, 12
    embedder_cfg.input_dim = input_dim
    embedder_cfg.output_dim = output_dim
    embedder_cfg.obs_horizon = obs_horizon
    embedder = hydra_zen.instantiate(embedder_cfg)
    embedder.eval()

    x = torch.randn(batch_size, obs_horizon, tokens_per_step, input_dim)
    x_perturbed = x.clone()
    x_perturbed[:, 0, 0, :] += torch.randn(input_dim)  # perturb only (t=0, k=0)

    with torch.no_grad():
        out = embedder(x)
        out_perturbed = embedder(x_perturbed)

    # (t=0, k=1) is a different object at the *same* timestep as the perturbed token.
    assert not torch.allclose(out[:, 0, 1, :], out_perturbed[:, 0, 1, :])


def test_self_attention_shares_the_positional_embedding_across_tokens_per_step():
    """All K tokens at a given timestep must be nudged by the identical positional term."""
    embedder_cfg = _load_embedder_cfg("self_attention")
    obs_horizon, input_dim, output_dim = 2, 16, 12
    embedder_cfg.input_dim = input_dim
    embedder_cfg.output_dim = output_dim
    embedder_cfg.obs_horizon = obs_horizon
    embedder = hydra_zen.instantiate(embedder_cfg)

    with torch.no_grad():
        # Two otherwise-identical tokens at the same timestep t=0 (rows k=0 and k=1) must receive
        # the exact same input_proj(x) + pos_emb[t] term before attention ever mixes them.
        same_input = torch.randn(1, 1, input_dim)
        x = torch.stack([same_input.squeeze(0), same_input.squeeze(0)], dim=1).unsqueeze(0)
        assert x.shape == (1, 1, 2, input_dim)

        pre_attn = embedder.input_proj(x) + embedder.pos_emb[:, :1, :].unsqueeze(2)
        assert torch.equal(pre_attn[:, 0, 0, :], pre_attn[:, 0, 1, :])


def test_self_attention_rejects_a_rank_other_than_3_or_4():
    embedder_cfg = _load_embedder_cfg("self_attention")
    embedder_cfg.input_dim = 16
    embedder_cfg.output_dim = 8
    embedder_cfg.obs_horizon = 2
    embedder = hydra_zen.instantiate(embedder_cfg)

    with pytest.raises(ValueError, match="3D or 4D"):
        embedder(torch.randn(4, 2, 3, 4, 16))


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
