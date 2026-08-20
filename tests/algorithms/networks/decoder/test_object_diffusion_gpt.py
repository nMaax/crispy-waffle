import pytest
import torch

from policy.algorithms.networks.decoder.diffusion_gpt import ObjectDiffusionGPT
from policy.utils.typing_utils import DiffusionNetworkProtocol

# Small architecture dims for fast unit tests
ACT_DIM = 4
PROPRIO_DIM = 3
TOKEN_DIM = 5
EMBED_DIM = 32
N_LAYERS = 2
N_HEADS = 4
HORIZON = 4


def _make_network(
    goal_horizon: int = 0, tokens_per_step: int = 3, **overrides
) -> ObjectDiffusionGPT:
    side = {"proprio": PROPRIO_DIM, "task": (tokens_per_step, TOKEN_DIM)}
    cond_dims = {"obs": side} if goal_horizon == 0 else {"obs": side, "goal": side}

    kw = dict(
        act_dim=ACT_DIM,
        cond_dims=cond_dims,
        embed_dim=EMBED_DIM,
        obs_horizon=HORIZON,
        pred_horizon=HORIZON,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        goal_horizon=goal_horizon,
    )
    kw.update(overrides)
    return ObjectDiffusionGPT(**kw)


def _obs(batch_size=4, tokens_per_step=3, horizon=HORIZON):
    task = (
        torch.randn(batch_size, horizon, TOKEN_DIM)
        if tokens_per_step == 1
        else torch.randn(batch_size, horizon, tokens_per_step, TOKEN_DIM)
    )
    return {"proprio": torch.randn(batch_size, horizon, PROPRIO_DIM), "task": task}


def _goal(goal_horizon, batch_size=4, tokens_per_step=3):
    return _obs(batch_size=batch_size, tokens_per_step=tokens_per_step, horizon=goal_horizon)


class TestObjectDiffusionGPT:
    @pytest.mark.parametrize("tokens_per_step", [1, 3])
    def test_forward_non_goal_conditioned(self, tokens_per_step):
        net = _make_network(goal_horizon=0, tokens_per_step=tokens_per_step)
        sample = torch.randn(4, HORIZON, ACT_DIM)
        out = net(sample, torch.rand(4), external_cond={"obs": _obs(tokens_per_step=tokens_per_step)})
        assert out.shape == sample.shape

    @pytest.mark.parametrize("tokens_per_step", [1, 3])
    def test_forward_goal_conditioned(self, tokens_per_step):
        goal_horizon = 2
        net = _make_network(goal_horizon=goal_horizon, tokens_per_step=tokens_per_step)
        sample = torch.randn(4, HORIZON, ACT_DIM)
        cond = {
            "obs": _obs(tokens_per_step=tokens_per_step),
            "goal": _goal(goal_horizon, tokens_per_step=tokens_per_step),
        }
        out = net(sample, torch.rand(4), external_cond=cond)
        assert out.shape == sample.shape

    def test_implements_diffusion_network_protocol(self):
        assert isinstance(_make_network(), DiffusionNetworkProtocol)

    @pytest.mark.parametrize(
        ("tokens_per_step", "goal_horizon"), [(1, 0), (1, 2), (3, 0), (3, 2)]
    )
    def test_sequence_lengths(self, tokens_per_step, goal_horizon):
        """Each frame contributes K object tokens plus a proprio and an action one; positions are
        counted per frame, not per token."""
        net = _make_network(goal_horizon=goal_horizon, tokens_per_step=tokens_per_step)
        assert net.block_size == 1 + goal_horizon * tokens_per_step + (tokens_per_step + 2) * HORIZON
        assert net.seq_len == goal_horizon + HORIZON + 1

    def test_single_token_layout_matches_proprio_token_beso(self):
        """K=1 reproduces the [sigma, goals, (p_t, s_t, a_t), ...] layout of BESO++."""
        net = _make_network(goal_horizon=1, tokens_per_step=1)
        assert net.block_size == 1 + 1 + 3 * HORIZON
        assert net.obj_emb.in_features == TOKEN_DIM
        assert net.proprio_emb.in_features == PROPRIO_DIM

    def test_goal_proprio_is_ignored(self):
        """Goal tokens are task-only: proprioception has its own stream, on the obs side alone."""
        goal_horizon = 1
        net = _make_network(goal_horizon=goal_horizon).eval()
        sample = torch.randn(4, HORIZON, ACT_DIM)
        timestep = torch.rand(4)
        obs = _obs()
        goal_task = torch.randn(4, goal_horizon, 3, TOKEN_DIM)

        def predict(goal_proprio):
            return net(
                sample,
                timestep,
                external_cond={"obs": obs, "goal": {"proprio": goal_proprio, "task": goal_task}},
            )

        zeroed = predict(torch.zeros(4, goal_horizon, PROPRIO_DIM))
        assert torch.equal(zeroed, predict(torch.randn(4, goal_horizon, PROPRIO_DIM)))

    def test_goal_sequence_length_mismatch_raises(self):
        net = _make_network(goal_horizon=2)
        sample = torch.randn(4, HORIZON, ACT_DIM)
        with pytest.raises(ValueError, match="Expected goal sequence length 2, but got 3"):
            net(sample, torch.rand(4), external_cond={"obs": _obs(), "goal": _goal(3)})

    def test_missing_goal_raises(self):
        net = _make_network(goal_horizon=1)
        sample = torch.randn(4, HORIZON, ACT_DIM)
        with pytest.raises(ValueError, match="goal must be provided"):
            net(sample, torch.rand(4), external_cond={"obs": _obs()})

    def test_obs_action_length_mismatch_raises(self):
        net = _make_network(goal_horizon=0)
        sample = torch.randn(4, HORIZON - 1, ACT_DIM)
        with pytest.raises(ValueError, match="must be equal"):
            net(sample, torch.rand(4), external_cond={"obs": _obs()})

    def test_goal_token_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="must match the observation's"):
            ObjectDiffusionGPT(
                act_dim=ACT_DIM,
                cond_dims={
                    "obs": {"proprio": PROPRIO_DIM, "task": (3, TOKEN_DIM)},
                    "goal": {"proprio": PROPRIO_DIM, "task": (3, TOKEN_DIM + 1)},
                },
                embed_dim=EMBED_DIM,
                n_layers=N_LAYERS,
                n_heads=N_HEADS,
                obs_horizon=HORIZON,
                pred_horizon=HORIZON,
                goal_horizon=1,
            )

    def test_flat_task_spec_raises(self):
        """The task spec must declare (tokens_per_step, token_dim), not a bare width."""
        with pytest.raises(ValueError, match=r"\(tokens_per_step, token_dim\)"):
            ObjectDiffusionGPT(
                act_dim=ACT_DIM,
                cond_dims={"obs": {"proprio": PROPRIO_DIM, "task": TOKEN_DIM}},
                embed_dim=EMBED_DIM,
                n_layers=N_LAYERS,
                n_heads=N_HEADS,
                obs_horizon=HORIZON,
                pred_horizon=HORIZON,
            )

    def test_backward_grads_finite(self):
        net = _make_network(goal_horizon=1)
        sample = torch.randn(4, HORIZON, ACT_DIM)
        out = net(sample, torch.rand(4), external_cond={"obs": _obs(), "goal": _goal(1)})
        out.sum().backward()

        for name, param in net.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradients"
