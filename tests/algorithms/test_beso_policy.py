from collections import deque
from unittest.mock import MagicMock

import pytest
import torch

from policy.algorithms.beso_policy import BesoPolicy
from policy.transforms import MinMaxNormalizer
from tests.algorithms._beso_test_utils import mock_loop_internals


def _basic_kwargs(**overrides):
    """Mock kwargs that construct a BesoPolicy without invoking hydra_zen.instantiate."""
    kw = dict(
        decoder={"_target_": "policy.algorithms.networks.decoder.diffusion_gpt.DiffusionGPT"},
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
    """Isolated unit tests for vanilla BesoPolicy behavior (``relative_goal=False``,
    ``use_proprio_token=False`` -- both off, matching the upstream reference).

    Shared infra is covered in ``TestBaseDiffusionAgentLogic``; this suite only covers what is
    unique to BESO: the continuous-DDIM loop, goal handling, Karras scalings, CFG, and action
    history. Proprio-token-splitting and ``relative_goal=True`` ("BESO++DeltaInput") are
    repo-local additions, both opt-in flags on this same class -- see
    ``TestBesoPolicyDeltaInputLogic`` below for those.
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
        policy = BesoPolicy(**basic_kwargs)
        policy.act_normalizer = MinMaxNormalizer(policy.act_dim)
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
        policy.decoder = MagicMock(return_value=torch.zeros(1, 16, 4))
        obs_seq = torch.randn(1, 2, 3)
        act_seq = torch.randn(1, 16, 4)
        goal = torch.randn(1, 3)
        external_cond = {"obs": obs_seq, "goal": goal}
        policy._compute_loss(external_cond, act_seq)
        # With goal_drop_prob=1.0, the goal passed to the network must be all zeros.
        call_kwargs = policy.decoder.call_args.kwargs
        assert "goal" in call_kwargs["external_cond"]
        assert torch.all(call_kwargs["external_cond"]["goal"] == 0.0)

    def test_goal_dropout_zeros_goal_when_training_dict_goal(self, basic_kwargs):
        """Same as above, but with a genuine multi-key goal tree (not a bare Tensor)."""
        policy = BesoPolicy(**_basic_kwargs(goal_drop_prob=1.0, goal_horizon=1))
        policy.train()
        policy.decoder = MagicMock(return_value=torch.zeros(1, 16, 4))
        obs_seq = torch.randn(1, 2, 3)
        act_seq = torch.randn(1, 16, 4)
        goal = {"a": torch.randn(1, 3), "b": torch.randn(1, 5)}
        external_cond = {"obs": obs_seq, "goal": goal}
        policy._compute_loss(external_cond, act_seq)
        call_kwargs = policy.decoder.call_args.kwargs
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
        assert policy.decoder.call_count == 2

    def test_no_cfg_single_network_call(self, basic_kwargs):
        policy = BesoPolicy(**_basic_kwargs(cfg_lambda=None, goal_horizon=1))
        mock_loop_internals(policy)
        obs_cond = torch.zeros((1, 2, 3))
        goal_cond = torch.randn(1, 3)
        policy._run_diffusion_loop(
            external_cond={"obs": obs_cond, "goal": goal_cond}, num_inference_steps=2
        )
        assert policy.decoder.call_count == 1

    # ------------------------------------------------------------------ #
    # configure_optimizers (real DiffusionGPT network, incl. nn.MultiheadAttention)
    # ------------------------------------------------------------------ #
    def test_configure_optimizers_categorizes_every_parameter(self):
        """Regression guard: nn.MultiheadAttention's packed in-projection must land in a
        decay/no_decay group instead of silently tripping the "some parameters were not
        categorized" assertion (it isn't an nn.Linear, so it needs its own whitelist entry)."""
        kwargs = _basic_kwargs(
            obs_horizon=2,
            pred_horizon=2,
            act_horizon=1,
            decoder={
                "_target_": "policy.algorithms.networks.decoder.diffusion_gpt.DiffusionGPT",
                "act_dim": 4,
                "obs_horizon": 2,
                "pred_horizon": 2,
                "embed_dim": 8,
                "n_layers": 1,
                "n_heads": 2,
            },
            optimizer={"_target_": "torch.optim.AdamW", "_partial_": True, "weight_decay": 1e-4},
        )
        policy = BesoPolicy(**kwargs)
        policy.configure_model()

        result = policy.configure_optimizers()
        optimizer = result["optimizer"] if isinstance(result, dict) else result
        categorized = sum(len(g["params"]) for g in optimizer.param_groups)
        assert categorized == len(list(policy.decoder.parameters()))

    # ------------------------------------------------------------------ #
    # num_parallel_samples
    # ------------------------------------------------------------------ #
    def test_num_parallel_samples_averaged(self, basic_kwargs):
        policy = BesoPolicy(**_basic_kwargs(num_parallel_samples=2))
        mock_loop_internals(policy, decoder_return=torch.ones((2, 1, 4)))
        obs_cond = torch.zeros((1, 2, 3))
        out = policy._run_diffusion_loop(external_cond={"obs": obs_cond}, num_inference_steps=2)
        # B=1 averaged over 2 parallel samples -> (1, 1, act_dim)
        assert out.shape == (1, 1, policy.act_dim)
        assert torch.isfinite(out).all()


def _delta_kwargs(**overrides):
    """Mock kwargs that construct a valid ``BesoPolicy(relative_goal=True,
    use_proprio_token=True)`` ("BESO++DeltaInput") without invoking hydra_zen.instantiate.

    ``goal_horizon=1``/``goal_drop_prob=0.0``/``cfg_lambda=None`` are baked in here since every
    valid construction requires them -- ``relative_goal=True`` always folds the goal into a
    delta, which needs exactly one goal frame and has no meaning under classifier-free guidance.

    ``proprio_dim=0`` here means "this synthetic obs_dim=3 fixture has no proprioception at all"
    -- a real, explicit claim (not a placeholder), required because a flat obs_dim can't derive
    it. Tests that specifically exercise derivation/omission override it back to ``None``.
    """
    kw = dict(
        decoder={"_target_": "policy.algorithms.networks.decoder.diffusion_gpt.DiffusionGPT"},
        ema={},
        optimizer={},
        act_dim=4,
        obs_dim=3,
        proprio_dim=0,
        pred_horizon=16,
        obs_horizon=2,
        goal_horizon=1,
        relative_goal=True,
        use_proprio_token=True,
        goal_drop_prob=0.0,
        cfg_lambda=None,
    )
    kw.update(overrides)
    return kw


class TestBesoPolicyDeltaInputLogic:
    """Isolated unit tests for ``BesoPolicy(relative_goal=True, use_proprio_token=True)``
    ("BESO++DeltaInput"): both flags always active together here, always splitting.

    proprioception into its own network token and conditioning on the per-timestep goal-state
    delta (g - s_t) -- see ``TestBesoPolicyLogic`` above for the shared/inherited diffusion-loop,
    CFG, and Karras-scaling behavior common to every ``BesoPolicy`` configuration.
    """

    @pytest.fixture
    def basic_kwargs(self):
        return _delta_kwargs()

    # ------------------------------------------------------------------ #
    # proprio_dim / task_dim resolution
    #
    # These are lazily resolved BaseDiffusionAgent properties, so a misconfigured obs_dim raises
    # on first use rather than at construction; the validation itself belongs to (and is covered
    # with) `resolve_proprio_dim`/`derive_task_dim` in tests/algorithms/networks/test_utils.py.
    # ------------------------------------------------------------------ #
    def test_explicit_task_dim_is_honoured(self, basic_kwargs):
        policy = BesoPolicy(**_delta_kwargs(proprio_dim=1, task_dim=2, obs_dim=3))
        assert policy.task_dim == 2

    def test_derives_proprio_dim_from_dict_obs(self, basic_kwargs):
        """With a dict obs_dim, proprio_dim/task_dim are auto-derived from obs_dim['proprio'] when
        proprio_dim is omitted."""
        kwargs = _delta_kwargs(proprio_dim=None)
        kwargs["obs_dim"] = {"proprio": 2, "task": 1}
        policy = BesoPolicy(**kwargs)
        assert policy.proprio_dim == 2
        assert policy.task_dim == 1

    def test_resolves_task_dim_from_flat_obs_dim(self, basic_kwargs):
        """A flat obs_dim with an explicit proprio_dim=0 ("no proprioception") still derives a real
        task_dim (the full obs width), never None."""
        policy = BesoPolicy(**basic_kwargs)  # obs_dim=3 (flat), proprio_dim=0
        assert policy.proprio_dim == 0
        assert policy.task_dim == 3

    # ------------------------------------------------------------------ #
    # Fixed-configuration requirements (goal_horizon=1, no CFG)
    # ------------------------------------------------------------------ #
    def test_requires_goal_horizon_one(self, basic_kwargs):
        with pytest.raises(ValueError, match="requires goal_horizon=1"):
            BesoPolicy(**_delta_kwargs(goal_horizon=2))

    def test_rejects_goal_drop_prob(self, basic_kwargs):
        with pytest.raises(ValueError, match="mutually exclusive with classifier-free guidance"):
            BesoPolicy(**_delta_kwargs(goal_drop_prob=0.1))

    def test_rejects_cfg_lambda(self, basic_kwargs):
        with pytest.raises(ValueError, match="mutually exclusive with classifier-free guidance"):
            BesoPolicy(**_delta_kwargs(cfg_lambda=1.25))

    # ------------------------------------------------------------------ #
    # Network wiring (proprio token always on, no standalone goal token)
    # ------------------------------------------------------------------ #
    def test_cond_dims_omits_goal_key(self, basic_kwargs):
        policy = BesoPolicy(**_delta_kwargs(proprio_dim=1, obs_dim=3))
        assert policy._get_cond_dims() == {"obs": 3}

    def test_decoder_goal_horizon_forced_to_zero(self, basic_kwargs):
        policy = BesoPolicy(**_delta_kwargs(proprio_dim=1, obs_dim=3))
        # Includes "cond_dims" too: _decoder_extra_kwargs extends the base default (which is
        # just {"cond_dims": ...}) rather than replacing it, since DiffusionGPT still wants it.
        assert policy._decoder_extra_kwargs() == {
            "cond_dims": {"obs": 3},
            "proprio_dim": 1,
            "use_proprio_token": True,
            "goal_horizon": 0,
        }

    # ------------------------------------------------------------------ #
    # Delta conditioning (g - s_t)
    # ------------------------------------------------------------------ #
    def test_folds_goal_into_obs_flat_tensor(self, basic_kwargs):
        policy = BesoPolicy(**_delta_kwargs(proprio_dim=1, obs_dim=3))
        obs = torch.cat(
            [torch.randn(2, 4, 1), torch.randn(2, 4, 2)], dim=-1
        )  # [B=2, T=4, proprio(1)+task(2)]
        goal = torch.randn(2, 2)  # task-only, no time axis
        external_cond = policy._build_external_cond(obs, goal)
        assert set(external_cond.keys()) == {"obs"}
        assert set(external_cond["obs"].keys()) == {"proprio", "task"}
        assert torch.equal(external_cond["obs"]["proprio"], obs[..., :1])
        expected_delta = goal.unsqueeze(1) - obs[..., 1:]
        assert torch.allclose(external_cond["obs"]["task"], expected_delta)

    def test_folds_goal_into_obs_dict_obs_and_goal(self, basic_kwargs):
        policy = BesoPolicy(**_delta_kwargs(proprio_dim=1, obs_dim=3))
        obs = {"proprio": torch.randn(2, 4, 1), "task": torch.randn(2, 4, 2)}
        goal = {"proprio": torch.randn(2, 1), "task": torch.randn(2, 2)}
        external_cond = policy._build_external_cond(obs, goal)
        assert set(external_cond.keys()) == {"obs"}
        assert torch.equal(external_cond["obs"]["proprio"], obs["proprio"])
        expected_delta = goal["task"].unsqueeze(1) - obs["task"]
        assert torch.allclose(external_cond["obs"]["task"], expected_delta)

    def test_accepts_full_width_flat_goal(self, basic_kwargs):
        """A flat goal at full obs-width (proprio + task) has its leading proprio slice stripped,
        mirroring DiffusionGPT.forward's own tolerance for a flat goal of full obs-width."""
        policy = BesoPolicy(**_delta_kwargs(proprio_dim=1, obs_dim=3))
        obs = torch.cat([torch.randn(2, 4, 1), torch.randn(2, 4, 2)], dim=-1)
        goal_full = torch.randn(2, 3)  # proprio(1) + task(2)
        external_cond = policy._build_external_cond(obs, goal_full)
        expected_delta = goal_full[..., 1:].unsqueeze(1) - obs[..., 1:]
        assert torch.allclose(external_cond["obs"]["task"], expected_delta)

    def test_rejects_mismatched_goal_width(self, basic_kwargs):
        policy = BesoPolicy(**_delta_kwargs(proprio_dim=1, obs_dim=3))
        obs = torch.cat([torch.randn(2, 4, 1), torch.randn(2, 4, 2)], dim=-1)
        goal_bad = torch.randn(2, 5)
        with pytest.raises(ValueError, match="Expected width"):
            policy._build_external_cond(obs, goal_bad)

    def test_requires_proprio_key_in_obs(self, basic_kwargs):
        """A Mapping obs missing 'proprio' is a misconfiguration, caught the same way
        DiffusionGPT.forward's own use_proprio_token check would."""
        policy = BesoPolicy(**_delta_kwargs(proprio_dim=1, obs_dim=3))
        obs = {"task": torch.randn(2, 4, 2)}
        goal = torch.randn(2, 2)
        with pytest.raises(ValueError, match=r"requires external_cond\['obs'\] to carry"):
            policy._build_external_cond(obs, goal)

    def test_requires_goal(self, basic_kwargs):
        policy = BesoPolicy(**_delta_kwargs(proprio_dim=1, obs_dim=3))
        obs = torch.cat([torch.randn(2, 4, 1), torch.randn(2, 4, 2)], dim=-1)
        with pytest.raises(ValueError, match="but received goal=None"):
            policy._build_external_cond(obs, None)

    def test_end_to_end_through_real_diffusion_gpt(self, basic_kwargs):
        """Non-mocked smoke test: configure_model() actually builds a DiffusionGPT with
        goal_horizon=0 (no separate goal-token block), and a full loss pass runs without error --
        confirming the decoder-side override takes effect and the whole path is wired correctly.
        """
        kwargs = _delta_kwargs(proprio_dim=1, obs_dim=3, obs_horizon=2, pred_horizon=2, act_horizon=1)
        kwargs["decoder"] = {
            "_target_": "policy.algorithms.networks.decoder.diffusion_gpt.DiffusionGPT",
            "act_dim": 4,
            "obs_horizon": 2,
            "pred_horizon": 2,
            "embed_dim": 8,
            "n_layers": 1,
            "n_heads": 2,
        }
        policy = BesoPolicy(**kwargs)
        policy.configure_model()
        assert policy.decoder.goal_horizon == 0

        obs_seq = torch.cat([torch.randn(2, 2, 1), torch.randn(2, 2, 2)], dim=-1)
        goal = torch.randn(2, 2)
        act_seq = torch.randn(2, 2, 4)
        external_cond = policy._build_external_cond(obs_seq, goal)
        loss = policy._compute_loss(external_cond, act_seq)
        assert torch.isfinite(loss)

    def test_run_diffusion_loop_reuses_folded_obs(self, basic_kwargs):
        """Regression test: `_run_diffusion_loop` must not try to re-fold an already-delta-folded
        obs on every denoising step (goal has degraded to None by then, since delta mode leaves
        no standalone goal entry) -- it previously raised via a stale `_build_external_cond`
        call; `_run_diffusion_loop` now just packs whatever obs/goal it currently holds instead of
        re-invoking `_build_external_cond`."""
        policy = BesoPolicy(**_delta_kwargs(proprio_dim=1, obs_dim=3))
        mock_loop_internals(policy)
        obs_seq = torch.cat([torch.randn(1, 2, 1), torch.randn(1, 2, 2)], dim=-1)
        goal = torch.randn(1, 2)
        folded_external_cond = policy._build_external_cond(obs_seq, goal)
        out = policy._run_diffusion_loop(external_cond=folded_external_cond, num_inference_steps=2)
        assert torch.isfinite(out).all()
