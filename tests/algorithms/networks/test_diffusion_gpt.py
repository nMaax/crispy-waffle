import pytest
import torch

from policy.algorithms.networks.diffusion_gpt import DiffusionGPT
from policy.utils.typing_utils import (
    DiffusionNetworkProtocol,
    GoalConditionedDiffusionNetworkProtocol,
)

# Small architecture dims for fast unit tests
ACT_DIM = 4
COND_DIM = 8
EMBED_DIM = 32
N_LAYERS = 2
N_HEADS = 4
HORIZON = 4


def _make_network(goal_seq_len: int = 0, **overrides) -> DiffusionGPT:
    kw = dict(
        act_dim=ACT_DIM,
        external_cond_dim=COND_DIM,
        embed_dim=EMBED_DIM,
        external_cond_horizon=HORIZON,
        pred_horizon=HORIZON,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        goal_seq_len=goal_seq_len,
    )
    kw.update(overrides)
    return DiffusionGPT(**kw)


def _sample_inputs(batch_size=4, horizon=HORIZON):
    sample = torch.randn(batch_size, horizon, ACT_DIM)
    timestep = torch.rand(batch_size)
    obs = torch.randn(batch_size, horizon, COND_DIM)
    return sample, timestep, obs


class TestDiffusionGPT:
    def test_forward_non_goal_conditioned(self):
        net = _make_network(goal_seq_len=0)
        sample, timestep, obs = _sample_inputs()
        out = net(sample, timestep, obs)
        assert out.shape == sample.shape

    def test_forward_goal_conditioned(self):
        goal_seq_len = 2
        net = _make_network(goal_seq_len=goal_seq_len)
        sample, timestep, obs = _sample_inputs()
        goal = torch.randn(4, goal_seq_len, COND_DIM)
        out = net(sample, timestep, obs, goal=goal)
        assert out.shape == sample.shape

    def test_forward_obs_flattened_2d(self):
        """Obs may be a 2D flattened tensor [B, horizon * cond_dim] instead of 3D."""
        net = _make_network(goal_seq_len=0)
        sample, timestep, _ = _sample_inputs()
        obs_flat = torch.randn(4, HORIZON * COND_DIM)
        out = net(sample, timestep, obs_flat)
        assert out.shape == sample.shape

    def test_backward_grads_finite(self):
        net = _make_network(goal_seq_len=0)
        sample, timestep, obs = _sample_inputs()
        out = net(sample, timestep, obs)
        loss = out.sum()
        loss.backward()
        for p in net.parameters():
            if p.requires_grad:
                assert p.grad is not None
                assert torch.isfinite(p.grad).all()

    # ------------------------------------------------------------------ #
    # Protocol conformance
    # ------------------------------------------------------------------ #
    def test_satisfies_diffusion_network_protocol(self):
        net = _make_network()
        assert isinstance(net, DiffusionNetworkProtocol)

    def test_satisfies_goal_conditioned_diffusion_network_protocol(self):
        net = _make_network(goal_seq_len=2)
        assert isinstance(net, GoalConditionedDiffusionNetworkProtocol)

    # ------------------------------------------------------------------ #
    # ValueError paths
    # ------------------------------------------------------------------ #
    def test_init_horizon_mismatch_raises(self):
        """external_cond_horizon != pred_horizon must raise at construction."""
        with pytest.raises(ValueError, match="Observation horizon and act horizon must be equal"):
            DiffusionGPT(
                act_dim=ACT_DIM,
                external_cond_dim=COND_DIM,
                embed_dim=EMBED_DIM,
                n_layers=N_LAYERS,
                n_heads=N_HEADS,
                external_cond_horizon=4,
                pred_horizon=8,
            )

    def test_forward_obs_action_horizon_mismatch_raises(self):
        """At forward time, obs sequence length must equal sample (action) sequence length."""
        net = _make_network(goal_seq_len=0)
        # sample has horizon=4 but obs has horizon=8
        sample = torch.randn(4, HORIZON, ACT_DIM)
        timestep = torch.rand(4)
        obs = torch.randn(4, HORIZON * 2, COND_DIM)  # 8 != 4
        with pytest.raises(ValueError, match="Observation sequence length .* and action sequence length"):
            net(sample, timestep, obs)

    def test_goal_required_when_goal_conditioned(self):
        """If goal_seq_len > 0, passing goal=None must raise."""
        net = _make_network(goal_seq_len=2)
        sample, timestep, obs = _sample_inputs()
        with pytest.raises(ValueError, match="goal must be provided"):
            net(sample, timestep, obs, goal=None)

    def test_goal_length_mismatch_raises(self):
        """The goal sequence length must match goal_seq_len."""
        net = _make_network(goal_seq_len=2)
        sample, timestep, obs = _sample_inputs()
        goal = torch.randn(4, 3, COND_DIM)  # 3 != 2
        with pytest.raises(ValueError, match="Expected goal sequence length 2, but got 3"):
            net(sample, timestep, obs, goal=goal)
