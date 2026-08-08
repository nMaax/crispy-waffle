from collections import deque
from unittest.mock import MagicMock

import pytest
import torch

from policy.algorithms.beso_policy import BesoPolicy
from tests.algorithms._beso_test_utils import mock_loop_internals


def _basic_kwargs(**overrides):
    """Mock kwargs that construct a BesoPolicy without invoking hydra_zen.instantiate."""
    kw = dict(
        network={"_target_": "policy.algorithms.networks.diffusion_gpt.DiffusionGPT"},
        ema={},
        optimizer={},
        act_dim=4,
        obs_dim=3,
        pred_horizon=16,
        obs_horizon=2,
    )
    kw.update(overrides)
    return kw


class TestBesoPolicyLogic:
    """Isolated unit tests for vanilla BesoPolicy behavior.

    Shared infra is covered in ``TestBaseDiffusionAgentLogic``; this suite only covers what is
    unique to BESO: the continuous-DDIM loop, goal handling, Karras scalings, CFG, and action
    history. Proprio-token-splitting and goal_delta ("BESO++"/"BESO++DeltaInput") are additions
    owned by `BesoPlusPlusPolicy` -- see ``test_beso_pp_policy.py`` for those.
    """

    @pytest.fixture
    def basic_kwargs(self):
        return _basic_kwargs()

    # ------------------------------------------------------------------ #
    # _get_cond_dims
    # ------------------------------------------------------------------ #
    def test_get_cond_dims_no_goal(self, basic_kwargs):
        """Without goal-conditioning, cond_dims carries no "goal" key (today's behavior)."""
        policy = BesoPolicy(**basic_kwargs)
        assert policy._get_cond_dims() == {"obs": 3}

    def test_get_cond_dims_goal_matches_obs(self, basic_kwargs):
        """Vanilla BESO ("true BESO"): goal width equals obs width."""
        policy = BesoPolicy(**_basic_kwargs(goal_horizon=1))
        assert policy._get_cond_dims() == {"obs": 3, "goal": 3}

    # ------------------------------------------------------------------ #
    # output_clip_range (post-unnormalize physical-space clamping)
    # ------------------------------------------------------------------ #
    def test_output_clip_range_clamps_without_normalizer(self, basic_kwargs):
        policy = BesoPolicy(**basic_kwargs)
        mock_loop_internals(policy)
        obs_cond = torch.zeros((1, 2, 3))
        out = policy._run_diffusion_loop(
            external_cond={"obs": obs_cond}, num_inference_steps=2, output_clip_range=(3.0, 6.0)
        )
        assert out.min() >= 3.0
        assert out.max() <= 6.0

    def test_output_clip_range_none_no_clamp(self, basic_kwargs):
        policy = BesoPolicy(**basic_kwargs)
        mock_loop_internals(policy)
        obs_cond = torch.zeros((1, 2, 3))
        out = policy._run_diffusion_loop(
            external_cond={"obs": obs_cond}, num_inference_steps=2, output_clip_range=None
        )
        # No clipping -> output is whatever the loop produces (finite, correct shape)
        assert out.shape == (1, 1, policy.act_dim)
        assert torch.isfinite(out).all()

    def test_output_clip_range_clamps_with_normalizer(self, basic_kwargs):
        """With an action normalizer, clamping happens in physical space (post-unnormalize)."""
        policy = BesoPolicy(**basic_kwargs, act_normalizer=True)
        # Fit the MinMax normalizer to a known range so unnormalize is well-defined.
        policy.act_normalizer.fit(torch.linspace(-5, 5, 40).view(10, 4))
        mock_loop_internals(policy)
        obs_cond = torch.zeros((1, 2, 3))
        out = policy._run_diffusion_loop(
            external_cond={"obs": obs_cond}, num_inference_steps=2, output_clip_range=(3.0, 6.0)
        )
        assert out.min() >= 3.0
        assert out.max() <= 6.0

    # ------------------------------------------------------------------ #
    # action_history + reset
    # ------------------------------------------------------------------ #
    def test_action_history_appended_after_loop(self, basic_kwargs):
        policy = BesoPolicy(**basic_kwargs)
        mock_loop_internals(policy)
        obs_cond = torch.zeros((1, 2, 3))
        assert len(policy.action_history) == 0
        policy._run_diffusion_loop(external_cond={"obs": obs_cond}, num_inference_steps=2)
        assert len(policy.action_history) == 1

    def test_reset_clears_action_history(self, basic_kwargs):
        policy = BesoPolicy(**basic_kwargs)
        mock_loop_internals(policy)
        obs_cond = torch.zeros((1, 2, 3))
        policy._run_diffusion_loop(external_cond={"obs": obs_cond}, num_inference_steps=2)
        assert len(policy.action_history) == 1
        policy.reset()
        assert len(policy.action_history) == 0
        assert isinstance(policy.action_history, deque)

    # ------------------------------------------------------------------ #
    # num_inference_steps required
    # ------------------------------------------------------------------ #
    def test_num_inference_steps_required(self, basic_kwargs):
        policy = BesoPolicy(**basic_kwargs)
        mock_loop_internals(policy)
        obs_cond = torch.zeros((1, 2, 3))
        with pytest.raises(ValueError, match="must be manually provided"):
            policy._run_diffusion_loop(external_cond={"obs": obs_cond}, num_inference_steps=None)

    # ------------------------------------------------------------------ #
    # Karras scalings + sigmas (pure functions, no mocks)
    # ------------------------------------------------------------------ #
    def test_karras_scalings_shapes(self, basic_kwargs):
        policy = BesoPolicy(**basic_kwargs)
        sigma = torch.tensor([0.5, 1.0, 2.0]).view(3, 1, 1)
        c_skip, c_out, c_in = policy._get_karras_scalings(sigma)
        assert c_skip.shape == (3, 1, 1)
        assert c_out.shape == (3, 1, 1)
        assert c_in.shape == (3, 1, 1)
        assert torch.isfinite(c_skip).all()
        assert torch.isfinite(c_out).all()
        assert torch.isfinite(c_in).all()

    def test_sigmas_exponential_shape_and_trailing_zero(self, basic_kwargs):
        policy = BesoPolicy(**basic_kwargs)
        sigmas = policy._get_sigmas_exponential(5, 0.005, 1.0)
        assert sigmas.shape == (6,)  # n + trailing 0
        assert sigmas[-1].item() == 0.0
        # Monotonically decreasing (except the appended zero)
        assert torch.all(sigmas[:-1] >= sigmas[1:])

    def test_t_fn_sigma_fn_are_inverses(self, basic_kwargs):
        policy = BesoPolicy(**basic_kwargs)
        sigma = torch.tensor([0.005, 0.1, 0.5, 1.0])
        # _sigma_fn(_t_fn(sigma)) == sigma
        round_trip = policy._sigma_fn(policy._t_fn(sigma))
        assert torch.allclose(round_trip, sigma)

    # ------------------------------------------------------------------ #
    # Goal dropout (CFG training)
    # ------------------------------------------------------------------ #
    def test_goal_dropout_zeros_goal_when_training(self, basic_kwargs):
        policy = BesoPolicy(**_basic_kwargs(goal_drop_prob=1.0, goal_horizon=1))
        policy.train()  # enable training mode
        policy.network = MagicMock(return_value=torch.zeros(1, 16, 4))
        obs_seq = torch.randn(1, 2, 3)
        act_seq = torch.randn(1, 16, 4)
        goal = torch.randn(1, 3)
        external_cond = {"obs": obs_seq, "goal": goal}
        policy._compute_loss(external_cond, act_seq)
        # With goal_drop_prob=1.0, the goal passed to the network must be all zeros.
        call_kwargs = policy.network.call_args.kwargs
        assert "goal" in call_kwargs["external_cond"]
        assert torch.all(call_kwargs["external_cond"]["goal"] == 0.0)

    def test_goal_dropout_zeros_goal_when_training_dict_goal(self, basic_kwargs):
        """Same as above, but with a genuine multi-key goal tree (not a bare Tensor)."""
        policy = BesoPolicy(**_basic_kwargs(goal_drop_prob=1.0, goal_horizon=1))
        policy.train()
        policy.network = MagicMock(return_value=torch.zeros(1, 16, 4))
        obs_seq = torch.randn(1, 2, 3)
        act_seq = torch.randn(1, 16, 4)
        goal = {"a": torch.randn(1, 3), "b": torch.randn(1, 5)}
        external_cond = {"obs": obs_seq, "goal": goal}
        policy._compute_loss(external_cond, act_seq)
        call_kwargs = policy.network.call_args.kwargs
        goal_out = call_kwargs["external_cond"]["goal"]
        assert torch.all(goal_out["a"] == 0.0)
        assert torch.all(goal_out["b"] == 0.0)

    # ------------------------------------------------------------------ #
    # CFG inference
    # ------------------------------------------------------------------ #
    def test_cfg_inference_two_network_calls(self, basic_kwargs):
        policy = BesoPolicy(**_basic_kwargs(cfg_lambda=1.0, goal_horizon=1))
        mock_loop_internals(policy)
        obs_cond = torch.zeros((1, 2, 3))
        goal_cond = torch.randn(1, 3)
        policy._run_diffusion_loop(
            external_cond={"obs": obs_cond, "goal": goal_cond}, num_inference_steps=2
        )
        # cond + uncond = 2 network calls per iteration.
        assert policy.network.call_count == 2

    def test_no_cfg_single_network_call(self, basic_kwargs):
        policy = BesoPolicy(**_basic_kwargs(cfg_lambda=None, goal_horizon=1))
        mock_loop_internals(policy)
        obs_cond = torch.zeros((1, 2, 3))
        goal_cond = torch.randn(1, 3)
        policy._run_diffusion_loop(
            external_cond={"obs": obs_cond, "goal": goal_cond}, num_inference_steps=2
        )
        assert policy.network.call_count == 1

    # ------------------------------------------------------------------ #
    # num_parallel_samples
    # ------------------------------------------------------------------ #
    def test_num_parallel_samples_averaged(self, basic_kwargs):
        policy = BesoPolicy(**_basic_kwargs(num_parallel_samples=2))
        mock_loop_internals(policy, network_return=torch.ones((2, 1, 4)))
        obs_cond = torch.zeros((1, 2, 3))
        out = policy._run_diffusion_loop(external_cond={"obs": obs_cond}, num_inference_steps=2)
        # B=1 averaged over 2 parallel samples -> (1, 1, act_dim)
        assert out.shape == (1, 1, policy.act_dim)
        assert torch.isfinite(out).all()
