import pytest
import torch

from policy.algorithms.networks.decoder.diffusion_gpt import DiffusionGPT
from policy.utils.typing_utils import DiffusionNetworkProtocol

# Small architecture dims for fast unit tests
ACT_DIM = 4
COND_DIM = 8
EMBED_DIM = 32
N_LAYERS = 2
N_HEADS = 4
HORIZON = 4


def _make_network(goal_horizon: int = 0, **overrides) -> DiffusionGPT:
    # cond_dims["obs"] is the *per-timestep* width (COND_DIM); DiffusionGPT tokenizes per
    # timestep, so it uses this width directly.
    kw = dict(
        act_dim=ACT_DIM,
        cond_dims={"obs": COND_DIM},
        embed_dim=EMBED_DIM,
        obs_horizon=HORIZON,
        pred_horizon=HORIZON,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        goal_horizon=goal_horizon,
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
        net = _make_network(goal_horizon=0)
        sample, timestep, obs = _sample_inputs()
        out = net(sample, timestep, external_cond={"obs": obs})
        assert out.shape == sample.shape

    def test_forward_goal_conditioned(self):
        goal_horizon = 2
        net = _make_network(goal_horizon=goal_horizon)
        sample, timestep, obs = _sample_inputs()
        goal = torch.randn(4, goal_horizon, COND_DIM)
        out = net(sample, timestep, external_cond={"obs": obs, "goal": goal})
        assert out.shape == sample.shape

    def test_forward_obs_flattened_2d(self):
        """Obs may be a 2D flattened tensor [B, horizon * cond_dim] instead of 3D."""
        net = _make_network(goal_horizon=0)
        sample, timestep, _ = _sample_inputs()
        obs_flat = torch.randn(4, HORIZON * COND_DIM)
        out = net(sample, timestep, external_cond={"obs": obs_flat})
        assert out.shape == sample.shape

    def test_forward_obs_nested_mapping(self):
        """A nested obs mapping is merged on the feature axis before embedding."""
        net = _make_network(
            goal_horizon=0,
            cond_dims={"obs": {"a": COND_DIM // 2, "b": COND_DIM // 2}},
        )
        sample, timestep, _ = _sample_inputs()
        obs = {
            "a": torch.randn(4, HORIZON, COND_DIM // 2),
            "b": torch.randn(4, HORIZON, COND_DIM // 2),
        }
        out = net(sample, timestep, external_cond={"obs": obs})
        assert out.shape == sample.shape

    def test_backward_grads_finite(self):
        net = _make_network(goal_horizon=0)
        sample, timestep, obs = _sample_inputs()
        out = net(sample, timestep, external_cond={"obs": obs})
        loss = out.sum()
        loss.backward()
        for p in net.parameters():
            if p.requires_grad:
                assert p.grad is not None
                assert torch.isfinite(p.grad).all()

    def test_attention_is_causal(self):
        """Perturbing a later action token must not change earlier action outputs."""
        net = _make_network(goal_horizon=0)
        net.eval()
        sample, timestep, obs = _sample_inputs()
        perturbed_t = HORIZON - 2
        with torch.no_grad():
            out = net(sample, timestep, external_cond={"obs": obs})

            sample_perturbed = sample.clone()
            sample_perturbed[:, perturbed_t, :] += torch.randn(ACT_DIM)
            out_perturbed = net(sample_perturbed, timestep, external_cond={"obs": obs})

        # Earlier timesteps must be exactly unaffected by a later action token (causal mask).
        assert torch.allclose(out[:, :perturbed_t], out_perturbed[:, :perturbed_t], atol=1e-6)
        # The perturbed timestep itself must change (it attends to its own action token).
        assert not torch.allclose(out[:, perturbed_t], out_perturbed[:, perturbed_t])

    def test_init_weights_covers_the_attention_projections(self):
        """Regression guard: nn.MultiheadAttention's packed in-projection must go through the
        same _init_weights std=0.02 statistics as the rest of the network, not its own default
        (xavier) init."""
        net = _make_network(goal_horizon=0)
        attn = net.blocks[0].attn.attn  # Block -> CausalSelfAttention -> nn.MultiheadAttention
        assert torch.equal(attn.in_proj_bias, torch.zeros_like(attn.in_proj_bias))
        assert torch.equal(attn.out_proj.bias, torch.zeros_like(attn.out_proj.bias))
        std = attn.in_proj_weight.std().item()
        assert 0.015 < std < 0.026

    # ------------------------------------------------------------------ #
    # Protocol conformance
    # ------------------------------------------------------------------ #
    def test_satisfies_diffusion_network_protocol(self):
        net = _make_network()
        assert isinstance(net, DiffusionNetworkProtocol)

    # ------------------------------------------------------------------ #
    # ValueError paths
    # ------------------------------------------------------------------ #
    def test_init_horizon_mismatch_raises(self):
        """obs_horizon != pred_horizon must raise at construction."""
        with pytest.raises(ValueError, match="Observation horizon and act horizon must be equal"):
            DiffusionGPT(
                act_dim=ACT_DIM,
                cond_dims={"obs": COND_DIM},
                embed_dim=EMBED_DIM,
                n_layers=N_LAYERS,
                n_heads=N_HEADS,
                obs_horizon=4,
                pred_horizon=8,
            )

    def test_forward_obs_action_horizon_mismatch_raises(self):
        """At forward time, obs sequence length must equal sample (action) sequence length."""
        net = _make_network(goal_horizon=0)
        # sample has horizon=4 but obs has horizon=8
        sample = torch.randn(4, HORIZON, ACT_DIM)
        timestep = torch.rand(4)
        obs = torch.randn(4, HORIZON * 2, COND_DIM)  # 8 != 4
        with pytest.raises(ValueError, match="Observation sequence length .* and action sequence length"):
            net(sample, timestep, external_cond={"obs": obs})

    def test_goal_required_when_goal_conditioned(self):
        """If goal_horizon > 0, passing goal=None must raise."""
        net = _make_network(goal_horizon=2)
        sample, timestep, obs = _sample_inputs()
        with pytest.raises(ValueError, match="goal must be provided"):
            net(sample, timestep, external_cond={"obs": obs})

    def test_goal_length_mismatch_raises(self):
        """The goal sequence length must match goal_horizon."""
        net = _make_network(goal_horizon=2)
        sample, timestep, obs = _sample_inputs()
        goal = torch.randn(4, 3, COND_DIM)  # 3 != 2
        with pytest.raises(ValueError, match="Expected goal sequence length 2, but got 3"):
            net(sample, timestep, external_cond={"obs": obs, "goal": goal})


PROPRIO_DIM = 3
TASK_DIM = COND_DIM - PROPRIO_DIM  # 5


class TestDiffusionGPTMappingCond:
    """A `{"proprio", "task"}` conditioning tree is concatenated into one state token per step."""

    def test_forward_runs_with_dict_obs_and_goal(self):
        goal_horizon = 2
        net = _make_network(
            goal_horizon=goal_horizon,
            cond_dims={
                "obs": {"proprio": PROPRIO_DIM, "task": TASK_DIM},
                "goal": {"proprio": PROPRIO_DIM, "task": TASK_DIM},
            },
        )
        assert net.obs_dim == COND_DIM
        assert net.obs_emb.in_features == COND_DIM

        sample = torch.randn(4, HORIZON, ACT_DIM)
        timestep = torch.rand(4)
        obs = {
            "proprio": torch.randn(4, HORIZON, PROPRIO_DIM),
            "task": torch.randn(4, HORIZON, TASK_DIM),
        }
        goal = {
            "proprio": torch.randn(4, goal_horizon, PROPRIO_DIM),
            "task": torch.randn(4, goal_horizon, TASK_DIM),
        }
        out = net(sample, timestep, external_cond={"obs": obs, "goal": goal})
        assert out.shape == sample.shape

    def test_goal_cond_dim_mismatch_raises(self):
        """Goal tokens share obs_emb with obs tokens, so both sides must be the same width."""
        with pytest.raises(ValueError, match="must match the per-timestep obs width"):
            DiffusionGPT(
                act_dim=ACT_DIM,
                cond_dims={"obs": COND_DIM, "goal": TASK_DIM},
                embed_dim=EMBED_DIM,
                n_layers=N_LAYERS,
                n_heads=N_HEADS,
                obs_horizon=HORIZON,
                pred_horizon=HORIZON,
            )

    def test_goal_wrong_width_raises(self):
        goal_horizon = 1
        net = _make_network(goal_horizon=goal_horizon)
        sample, timestep, obs = _sample_inputs()
        goal = torch.randn(4, goal_horizon, COND_DIM + 1)
        with pytest.raises(ValueError, match="Expected goal width"):
            net(sample, timestep, external_cond={"obs": obs, "goal": goal})
