from collections import deque
from unittest.mock import MagicMock

import pytest
import torch

from policy.algorithms.beso_policy import BesoPolicy


def _basic_kwargs(**overrides):
    """Mock kwargs that construct a BesoPolicy without invoking hydra_zen.instantiate.

    ``proprio_dim=0`` here means "this synthetic obs_dim=3 fixture has no proprioception at all"
    -- a real, explicit claim (not a placeholder), required because a flat obs_dim can't derive
    it. Tests that specifically exercise derivation/omission override it back to ``None``.
    """
    kw = dict(
        network={"_target_": "policy.algorithms.networks.diffusion_gpt.DiffusionGPT"},
        ema={},
        optimizer={},
        act_dim=4,
        obs_dim=3,
        proprio_dim=0,
        pred_horizon=16,
        obs_horizon=2,
    )
    kw.update(overrides)
    return kw


def _mock_loop_internals(policy, network_return=None):
    """Mocks the Karras/sigma helpers so _run_diffusion_loop runs deterministically."""
    if network_return is None:
        B_expanded = getattr(policy, "num_parallel_samples", 1)
        network_return = torch.ones((B_expanded, 1, policy.act_dim))
    policy.network = MagicMock(return_value=network_return)
    policy.ema = MagicMock()
    policy._get_karras_scalings = MagicMock(
        return_value=(torch.tensor(0.0), torch.tensor(1.0), torch.tensor(1.0))
    )
    policy._get_sigmas_exponential = MagicMock(return_value=torch.tensor([1.0, 0.0]))
    # Shape-aware mocks so .view(B_expanded, 1, 1) works for any batch size.
    policy._t_fn = MagicMock(side_effect=lambda sigma: torch.zeros_like(sigma))
    policy._sigma_fn = MagicMock(side_effect=lambda t: torch.ones_like(t))


class TestBesoPolicyLogic:
    """Isolated unit tests for BesoPolicy-specific behavior.

    Shared infra is covered in ``TestBaseDiffusionAgentLogic``; this suite only covers
    what is unique to BESO: the continuous-DDIM loop, goal handling, Karras scalings,
    CFG, and action history.
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

    def test_get_cond_dims_goal_matches_obs_when_use_proprio_token_off(self, basic_kwargs):
        """use_proprio_token=False ("true BESO"): goal width equals obs width, today's implicit
        invariant now made explicit -- regardless of whether proprio_dim is also set."""
        policy = BesoPolicy(**_basic_kwargs(goal_horizon=1))
        assert policy._get_cond_dims() == {"obs": 3, "goal": 3}

        policy_with_inert_proprio_dim = BesoPolicy(
            **_basic_kwargs(goal_horizon=1, proprio_dim=1, obs_dim=3)
        )
        assert policy_with_inert_proprio_dim._get_cond_dims() == {"obs": 3, "goal": 3}

    def test_get_cond_dims_goal_is_task_only_when_use_proprio_token_set(self, basic_kwargs):
        """use_proprio_token=True ("robot-agnostic BESO"): goal width is task-only
        (obs_total - proprio_dim)."""
        policy = BesoPolicy(
            **_basic_kwargs(goal_horizon=1, proprio_dim=1, use_proprio_token=True, obs_dim=3)
        )
        assert policy._get_cond_dims() == {"obs": 3, "goal": 2}

    def test_proprio_dim_validated_against_obs_dim(self, basic_kwargs):
        with pytest.raises(ValueError, match="must be >= proprio_dim"):
            BesoPolicy(**_basic_kwargs(proprio_dim=10, obs_dim=3))

    def test_task_dim_validated_against_obs_dim_and_proprio_dim(self, basic_kwargs):
        policy = BesoPolicy(**_basic_kwargs(proprio_dim=1, task_dim=2, obs_dim=3))
        assert policy.task_dim == 2

        with pytest.raises(ValueError, match="does not match task_dim"):
            BesoPolicy(**_basic_kwargs(proprio_dim=1, task_dim=5, obs_dim=3))

    def test_flat_obs_requires_explicit_proprio_dim(self, basic_kwargs):
        """A flat obs_dim can't auto-derive proprio_dim, so it must always be given explicitly --
        regardless of use_proprio_token -- rather than silently defaulting to 0."""
        kwargs = _basic_kwargs(proprio_dim=None)
        with pytest.raises(ValueError, match="proprio_dim must be provided explicitly"):
            BesoPolicy(**kwargs)

    def test_use_proprio_token_requires_proprio_dim(self, basic_kwargs):
        """With a flat obs_dim, proprio_dim can't be auto-derived and must be given explicitly."""
        with pytest.raises(ValueError, match="proprio_dim must be provided explicitly"):
            BesoPolicy(**_basic_kwargs(use_proprio_token=True, proprio_dim=None))

    def test_use_proprio_token_requires_proprio_dim_dict_obs(self, basic_kwargs):
        """With a dict obs_dim lacking a 'proprio' key, the same misconfiguration is still caught,
        since there's nothing to derive from."""
        kwargs = _basic_kwargs(use_proprio_token=True, proprio_dim=None)
        kwargs["obs_dim"] = {"task": 3}
        with pytest.raises(ValueError, match="must contain 'proprio' key"):
            BesoPolicy(**kwargs)

    def test_use_proprio_token_derives_proprio_dim_from_dict_obs(self, basic_kwargs):
        """With a dict obs_dim, proprio_dim is auto-derived from obs_dim['proprio'] when
        use_proprio_token=True and proprio_dim is omitted."""
        kwargs = _basic_kwargs(use_proprio_token=True, proprio_dim=None)
        kwargs["obs_dim"] = {"proprio": 2, "task": 1}
        policy = BesoPolicy(**kwargs)
        assert policy.proprio_dim == 2
        assert policy.task_dim == 1

    def test_derives_proprio_dim_from_dict_obs_even_without_proprio_token(self, basic_kwargs):
        """A dict obs_dim always records its own proprio width, so proprio_dim/task_dim are
        resolved from it regardless of use_proprio_token, instead of staying at an inert
        placeholder."""
        kwargs = _basic_kwargs(use_proprio_token=False, proprio_dim=None)
        kwargs["obs_dim"] = {"proprio": 2, "task": 1}
        policy = BesoPolicy(**kwargs)
        assert policy.proprio_dim == 2
        assert policy.task_dim == 1

    def test_flat_obs_without_proprio_token_still_resolves_task_dim(self, basic_kwargs):
        """A flat obs_dim with an explicit proprio_dim=0 ("no proprioception") still derives a real
        task_dim (the full obs width), never None -- even though use_proprio_token=False means
        nothing is actually split."""
        policy = BesoPolicy(**basic_kwargs)  # obs_dim=3 (flat), proprio_dim=0, no token
        assert policy.proprio_dim == 0
        assert policy.task_dim == 3

    # ------------------------------------------------------------------ #
    # output_clip_range (post-unnormalize physical-space clamping)
    # ------------------------------------------------------------------ #
    def test_output_clip_range_clamps_without_normalizer(self, basic_kwargs):
        policy = BesoPolicy(**basic_kwargs)
        _mock_loop_internals(policy)
        obs_cond = torch.zeros((1, 2, 3))
        out = policy._run_diffusion_loop(
            external_cond={"obs": obs_cond}, num_inference_steps=2, output_clip_range=(3.0, 6.0)
        )
        assert out.min() >= 3.0
        assert out.max() <= 6.0

    def test_output_clip_range_none_no_clamp(self, basic_kwargs):
        policy = BesoPolicy(**basic_kwargs)
        _mock_loop_internals(policy)
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
        _mock_loop_internals(policy)
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
        _mock_loop_internals(policy)
        obs_cond = torch.zeros((1, 2, 3))
        assert len(policy.action_history) == 0
        policy._run_diffusion_loop(external_cond={"obs": obs_cond}, num_inference_steps=2)
        assert len(policy.action_history) == 1

    def test_reset_clears_action_history(self, basic_kwargs):
        policy = BesoPolicy(**basic_kwargs)
        _mock_loop_internals(policy)
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
        _mock_loop_internals(policy)
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
        _mock_loop_internals(policy)
        obs_cond = torch.zeros((1, 2, 3))
        goal_cond = torch.randn(1, 3)
        policy._run_diffusion_loop(
            external_cond={"obs": obs_cond, "goal": goal_cond}, num_inference_steps=2
        )
        # cond + uncond = 2 network calls per iteration.
        assert policy.network.call_count == 2

    def test_no_cfg_single_network_call(self, basic_kwargs):
        policy = BesoPolicy(**_basic_kwargs(cfg_lambda=0.0, goal_horizon=1))
        _mock_loop_internals(policy)
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
        _mock_loop_internals(policy, network_return=torch.ones((2, 1, 4)))
        obs_cond = torch.zeros((1, 2, 3))
        out = policy._run_diffusion_loop(external_cond={"obs": obs_cond}, num_inference_steps=2)
        # B=1 averaged over 2 parallel samples -> (1, 1, act_dim)
        assert out.shape == (1, 1, policy.act_dim)
        assert torch.isfinite(out).all()
