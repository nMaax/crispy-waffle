import pytest
import torch

from policy.algorithms.networks.diffusion_gpt import DiffusionGPT
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


class TestDiffusionGPTProprioDim:
    """Tests for the opt-in `use_proprio_token` ("robot-agnostic BESO") token layout."""

    def test_default_mode_has_no_proprio_emb(self):
        net = _make_network(goal_horizon=0)
        assert net.use_proprio_token is False
        assert net.proprio_emb is None
        assert net.proprio_dim is None
        assert net.task_dim == COND_DIM
        assert "proprio_emb" not in dict(net.named_parameters())

    def test_proprio_dim_none_state_dict_unaffected(self):
        """Regression guard: an unset (default) proprio_dim must not register any new
        parameters."""
        net = _make_network(goal_horizon=0)
        expected_top_level = {"obs_emb", "act_emb", "sigma_emb", "pos_emb", "blocks", "ln_f", "action_pred"}
        top_level_names = {name.split(".")[0] for name, _ in net.named_parameters()}
        assert top_level_names == expected_top_level

    def test_proprio_emb_and_obs_emb_shapes(self):
        net = _make_network(goal_horizon=0, proprio_dim=PROPRIO_DIM, use_proprio_token=True)
        assert net.proprio_emb is not None
        assert net.proprio_emb.in_features == PROPRIO_DIM
        assert net.obs_emb.in_features == TASK_DIM
        assert net.task_dim == TASK_DIM

    def test_proprio_dim_set_but_unused_when_token_disabled(self):
        """proprio_dim is inert unless use_proprio_token is also set -- true BESO stays untouched
        even if a (now-meaningless) width is passed alongside it."""
        net = _make_network(goal_horizon=0, proprio_dim=PROPRIO_DIM, use_proprio_token=False)
        assert net.proprio_emb is None
        assert net.task_dim == COND_DIM
        assert net.obs_emb.in_features == COND_DIM
        obs = torch.randn(4, HORIZON, COND_DIM)
        sample = torch.randn(4, HORIZON, ACT_DIM)
        timestep = torch.rand(4)
        out = net(sample, timestep, external_cond={"obs": obs})
        assert out.shape == sample.shape

    def test_forward_runs_with_proprio_dict_obs(self):
        net = _make_network(goal_horizon=0, proprio_dim=PROPRIO_DIM, use_proprio_token=True)
        sample = torch.randn(4, HORIZON, ACT_DIM)
        timestep = torch.rand(4)
        obs = {
            "proprio": torch.randn(4, HORIZON, PROPRIO_DIM),
            "task": torch.randn(4, HORIZON, TASK_DIM),
        }
        out = net(sample, timestep, external_cond={"obs": obs})
        assert out.shape == sample.shape

    def test_forward_runs_with_proprio_and_goal(self):
        goal_horizon = 2
        net = _make_network(goal_horizon=goal_horizon, proprio_dim=PROPRIO_DIM, use_proprio_token=True)
        sample = torch.randn(4, HORIZON, ACT_DIM)
        timestep = torch.rand(4)
        obs = {
            "proprio": torch.randn(4, HORIZON, PROPRIO_DIM),
            "task": torch.randn(4, HORIZON, TASK_DIM),
        }
        # Goal carries a "proprio" key too (e.g. zeroed by an upstream goal-crafter); it must be
        # discarded rather than embedded, so only the task-width remainder needs to line up.
        goal = {
            "proprio": torch.zeros(4, goal_horizon, PROPRIO_DIM),
            "task": torch.randn(4, goal_horizon, TASK_DIM),
        }
        out = net(sample, timestep, external_cond={"obs": obs, "goal": goal})
        assert out.shape == sample.shape

    def test_forward_runs_with_goal_lacking_proprio_key(self):
        """Goal may also structurally lack a 'proprio' key entirely."""
        goal_horizon = 1
        net = _make_network(goal_horizon=goal_horizon, proprio_dim=PROPRIO_DIM, use_proprio_token=True)
        sample = torch.randn(4, HORIZON, ACT_DIM)
        timestep = torch.rand(4)
        obs = {
            "proprio": torch.randn(4, HORIZON, PROPRIO_DIM),
            "task": torch.randn(4, HORIZON, TASK_DIM),
        }
        goal = torch.randn(4, goal_horizon, TASK_DIM)
        out = net(sample, timestep, external_cond={"obs": obs, "goal": goal})
        assert out.shape == sample.shape

    def test_forward_runs_with_flat_goal_carrying_proprio(self):
        """A flat/3D Tensor goal at full obs-width (proprio included, e.g. zeroed) is disambiguated
        by width and has its leading proprio slice stripped."""
        goal_horizon = 1
        net = _make_network(goal_horizon=goal_horizon, proprio_dim=PROPRIO_DIM, use_proprio_token=True)
        sample = torch.randn(4, HORIZON, ACT_DIM)
        timestep = torch.rand(4)
        obs = {
            "proprio": torch.randn(4, HORIZON, PROPRIO_DIM),
            "task": torch.randn(4, HORIZON, TASK_DIM),
        }
        goal = torch.zeros(4, goal_horizon, COND_DIM)  # full obs-width, proprio slice zeroed
        goal[..., PROPRIO_DIM:] = torch.randn(4, goal_horizon, TASK_DIM)
        out = net(sample, timestep, external_cond={"obs": obs, "goal": goal})
        assert out.shape == sample.shape

    def test_goal_wrong_width_raises(self):
        goal_horizon = 1
        net = _make_network(goal_horizon=goal_horizon, proprio_dim=PROPRIO_DIM, use_proprio_token=True)
        sample = torch.randn(4, HORIZON, ACT_DIM)
        timestep = torch.rand(4)
        obs = {
            "proprio": torch.randn(4, HORIZON, PROPRIO_DIM),
            "task": torch.randn(4, HORIZON, TASK_DIM),
        }
        goal = torch.randn(4, goal_horizon, TASK_DIM + 1)  # neither task_dim nor obs_dim
        with pytest.raises(ValueError, match="Expected goal width"):
            net(sample, timestep, external_cond={"obs": obs, "goal": goal})

    def test_obs_missing_proprio_key_raises(self):
        net = _make_network(goal_horizon=0, proprio_dim=PROPRIO_DIM, use_proprio_token=True)
        sample, timestep, _ = _sample_inputs()
        obs = {"task": torch.randn(4, HORIZON, TASK_DIM)}
        with pytest.raises(ValueError, match="to carry a 'proprio' key"):
            net(sample, timestep, external_cond={"obs": obs})

    def test_obs_flat_2d_raises_when_proprio_dim_set(self):
        """A pre-flattened 2D obs tensor's proprio/task boundary is ambiguous once multiple
        timesteps are packed into it -- must be 3D (or a Mapping of 3D tensors) instead."""
        net = _make_network(goal_horizon=0, proprio_dim=PROPRIO_DIM, use_proprio_token=True)
        sample, timestep, _ = _sample_inputs()
        obs_flat = torch.randn(4, HORIZON * COND_DIM)
        with pytest.raises(ValueError, match="to be 3D"):
            net(sample, timestep, external_cond={"obs": obs_flat})

    def test_goal_cond_dim_mismatch_raises_against_task_dim(self):
        """cond_dims['goal'] must match the task-only width, not the full obs width."""
        with pytest.raises(ValueError, match="must match the per-timestep task width"):
            DiffusionGPT(
                act_dim=ACT_DIM,
                cond_dims={"obs": COND_DIM, "goal": COND_DIM},  # should be TASK_DIM, not COND_DIM
                embed_dim=EMBED_DIM,
                n_layers=N_LAYERS,
                n_heads=N_HEADS,
                obs_horizon=HORIZON,
                pred_horizon=HORIZON,
                proprio_dim=PROPRIO_DIM,
                use_proprio_token=True,
            )

    def test_proprio_dim_too_large_raises(self):
        with pytest.raises(ValueError, match="must be smaller than the per-timestep obs width"):
            DiffusionGPT(
                act_dim=ACT_DIM,
                cond_dims={"obs": COND_DIM},
                embed_dim=EMBED_DIM,
                n_layers=N_LAYERS,
                n_heads=N_HEADS,
                obs_horizon=HORIZON,
                pred_horizon=HORIZON,
                proprio_dim=COND_DIM,
                use_proprio_token=True,
            )

    def test_use_proprio_token_requires_proprio_dim_raises(self):
        with pytest.raises(ValueError, match="use_proprio_token=True requires proprio_dim to be set"):
            DiffusionGPT(
                act_dim=ACT_DIM,
                cond_dims={"obs": COND_DIM},
                embed_dim=EMBED_DIM,
                n_layers=N_LAYERS,
                n_heads=N_HEADS,
                obs_horizon=HORIZON,
                pred_horizon=HORIZON,
                use_proprio_token=True,
            )

    def test_backward_grads_finite_with_proprio(self):
        net = _make_network(goal_horizon=0, proprio_dim=PROPRIO_DIM, use_proprio_token=True)
        sample = torch.randn(4, HORIZON, ACT_DIM)
        timestep = torch.rand(4)
        obs = {
            "proprio": torch.randn(4, HORIZON, PROPRIO_DIM),
            "task": torch.randn(4, HORIZON, TASK_DIM),
        }
        out = net(sample, timestep, external_cond={"obs": obs})
        loss = out.sum()
        loss.backward()
        for p in net.parameters():
            if p.requires_grad:
                assert p.grad is not None
                assert torch.isfinite(p.grad).all()
