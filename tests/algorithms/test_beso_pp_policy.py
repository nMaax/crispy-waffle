import pytest
import torch

from policy.algorithms.beso_pp_policy import BesoPlusPlusPolicy
from tests.algorithms._beso_test_utils import mock_loop_internals


def _basic_kwargs(**overrides):
    """Mock kwargs that construct a valid BesoPlusPlusPolicy without invoking
    hydra_zen.instantiate.

    ``goal_horizon=1``/``goal_drop_prob=0.0``/``cfg_lambda=None`` are baked in here since every
    valid construction requires them -- BesoPlusPlusPolicy always folds the goal into a delta,
    which needs exactly one goal frame and has no meaning under classifier-free guidance.

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
        goal_horizon=1,
        goal_drop_prob=0.0,
        cfg_lambda=None,
    )
    kw.update(overrides)
    return kw


class TestBesoPlusPlusPolicyLogic:
    """Isolated unit tests for BesoPlusPlusPolicy-specific behavior: it always splits
    proprioception into its own network token and conditions on the per-timestep goal-state
    delta (g - s_t) -- see ``test_beso_policy.py`` for the shared/inherited diffusion-loop, CFG,
    and Karras-scaling behavior.
    """

    @pytest.fixture
    def basic_kwargs(self):
        return _basic_kwargs()

    # ------------------------------------------------------------------ #
    # proprio_dim / task_dim resolution and validation
    # ------------------------------------------------------------------ #
    def test_proprio_dim_validated_against_obs_dim(self, basic_kwargs):
        with pytest.raises(ValueError, match="must be >= proprio_dim"):
            BesoPlusPlusPolicy(**_basic_kwargs(proprio_dim=10, obs_dim=3))

    def test_task_dim_validated_against_obs_dim_and_proprio_dim(self, basic_kwargs):
        policy = BesoPlusPlusPolicy(**_basic_kwargs(proprio_dim=1, task_dim=2, obs_dim=3))
        assert policy.task_dim == 2

        with pytest.raises(ValueError, match="does not match task_dim"):
            BesoPlusPlusPolicy(**_basic_kwargs(proprio_dim=1, task_dim=5, obs_dim=3))

    def test_flat_obs_requires_explicit_proprio_dim(self, basic_kwargs):
        """A flat obs_dim can't auto-derive proprio_dim, so it must always be given explicitly
        rather than silently defaulting to 0."""
        with pytest.raises(ValueError, match="proprio_dim must be provided explicitly"):
            BesoPlusPlusPolicy(**_basic_kwargs(proprio_dim=None))

    def test_dict_obs_missing_proprio_key_requires_explicit_proprio_dim(self, basic_kwargs):
        """With a dict obs_dim lacking a 'proprio' key, the same misconfiguration is still caught,
        since there's nothing to derive from."""
        kwargs = _basic_kwargs(proprio_dim=None)
        kwargs["obs_dim"] = {"task": 3}
        with pytest.raises(ValueError, match="must contain 'proprio' key"):
            BesoPlusPlusPolicy(**kwargs)

    def test_derives_proprio_dim_from_dict_obs(self, basic_kwargs):
        """With a dict obs_dim, proprio_dim/task_dim are auto-derived from obs_dim['proprio'] when
        proprio_dim is omitted."""
        kwargs = _basic_kwargs(proprio_dim=None)
        kwargs["obs_dim"] = {"proprio": 2, "task": 1}
        policy = BesoPlusPlusPolicy(**kwargs)
        assert policy.proprio_dim == 2
        assert policy.task_dim == 1

    def test_resolves_task_dim_from_flat_obs_dim(self, basic_kwargs):
        """A flat obs_dim with an explicit proprio_dim=0 ("no proprioception") still derives a real
        task_dim (the full obs width), never None."""
        policy = BesoPlusPlusPolicy(**basic_kwargs)  # obs_dim=3 (flat), proprio_dim=0
        assert policy.proprio_dim == 0
        assert policy.task_dim == 3

    # ------------------------------------------------------------------ #
    # Fixed-configuration requirements (goal_horizon=1, no CFG)
    # ------------------------------------------------------------------ #
    def test_requires_goal_horizon_one(self, basic_kwargs):
        with pytest.raises(ValueError, match="requires goal_horizon=1"):
            BesoPlusPlusPolicy(**_basic_kwargs(goal_horizon=2))

    def test_rejects_goal_drop_prob(self, basic_kwargs):
        with pytest.raises(ValueError, match="mutually exclusive with classifier-free guidance"):
            BesoPlusPlusPolicy(**_basic_kwargs(goal_drop_prob=0.1))

    def test_rejects_cfg_lambda(self, basic_kwargs):
        with pytest.raises(ValueError, match="mutually exclusive with classifier-free guidance"):
            BesoPlusPlusPolicy(**_basic_kwargs(cfg_lambda=1.25))

    # ------------------------------------------------------------------ #
    # Network wiring (proprio token always on, no standalone goal token)
    # ------------------------------------------------------------------ #
    def test_cond_dims_omits_goal_key(self, basic_kwargs):
        policy = BesoPlusPlusPolicy(**_basic_kwargs(proprio_dim=1, obs_dim=3))
        assert policy._get_cond_dims() == {"obs": 3}

    def test_network_goal_horizon_forced_to_zero(self, basic_kwargs):
        policy = BesoPlusPlusPolicy(**_basic_kwargs(proprio_dim=1, obs_dim=3))
        assert policy._network_extra_kwargs() == {
            "proprio_dim": 1,
            "use_proprio_token": True,
            "goal_horizon": 0,
        }

    # ------------------------------------------------------------------ #
    # Delta conditioning (g - s_t), the only mode BesoPlusPlusPolicy supports
    # ------------------------------------------------------------------ #
    def test_folds_goal_into_obs_flat_tensor(self, basic_kwargs):
        policy = BesoPlusPlusPolicy(**_basic_kwargs(proprio_dim=1, obs_dim=3))
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
        policy = BesoPlusPlusPolicy(**_basic_kwargs(proprio_dim=1, obs_dim=3))
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
        policy = BesoPlusPlusPolicy(**_basic_kwargs(proprio_dim=1, obs_dim=3))
        obs = torch.cat([torch.randn(2, 4, 1), torch.randn(2, 4, 2)], dim=-1)
        goal_full = torch.randn(2, 3)  # proprio(1) + task(2)
        external_cond = policy._build_external_cond(obs, goal_full)
        expected_delta = goal_full[..., 1:].unsqueeze(1) - obs[..., 1:]
        assert torch.allclose(external_cond["obs"]["task"], expected_delta)

    def test_rejects_mismatched_goal_width(self, basic_kwargs):
        policy = BesoPlusPlusPolicy(**_basic_kwargs(proprio_dim=1, obs_dim=3))
        obs = torch.cat([torch.randn(2, 4, 1), torch.randn(2, 4, 2)], dim=-1)
        goal_bad = torch.randn(2, 5)
        with pytest.raises(ValueError, match="Expected goal width"):
            policy._build_external_cond(obs, goal_bad)

    def test_requires_proprio_key_in_obs(self, basic_kwargs):
        """A Mapping obs missing 'proprio' is a misconfiguration, caught the same way
        DiffusionGPT.forward's own use_proprio_token check would."""
        policy = BesoPlusPlusPolicy(**_basic_kwargs(proprio_dim=1, obs_dim=3))
        obs = {"task": torch.randn(2, 4, 2)}
        goal = torch.randn(2, 2)
        with pytest.raises(ValueError, match=r"requires external_cond\['obs'\] to carry"):
            policy._build_external_cond(obs, goal)

    def test_requires_goal(self, basic_kwargs):
        policy = BesoPlusPlusPolicy(**_basic_kwargs(proprio_dim=1, obs_dim=3))
        obs = torch.cat([torch.randn(2, 4, 1), torch.randn(2, 4, 2)], dim=-1)
        with pytest.raises(ValueError, match="but received goal=None"):
            policy._build_external_cond(obs, None)

    def test_end_to_end_through_real_diffusion_gpt(self, basic_kwargs):
        """Non-mocked smoke test: configure_model() actually builds a DiffusionGPT with
        goal_horizon=0 (no separate goal-token block), and a full loss pass runs without error --
        confirming the network-side override takes effect and the whole path is wired correctly.
        """
        kwargs = _basic_kwargs(
            proprio_dim=1, obs_dim=3, obs_horizon=2, pred_horizon=2, act_horizon=1
        )
        kwargs["network"] = {
            "_target_": "policy.algorithms.networks.diffusion_gpt.DiffusionGPT",
            "act_dim": 4,
            "obs_horizon": 2,
            "pred_horizon": 2,
            "embed_dim": 8,
            "n_layers": 1,
            "n_heads": 2,
        }
        policy = BesoPlusPlusPolicy(**kwargs)
        policy.configure_model()
        assert policy.network.goal_horizon == 0

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
        policy = BesoPlusPlusPolicy(**_basic_kwargs(proprio_dim=1, obs_dim=3))
        mock_loop_internals(policy)
        obs_seq = torch.cat([torch.randn(1, 2, 1), torch.randn(1, 2, 2)], dim=-1)
        goal = torch.randn(1, 2)
        folded_external_cond = policy._build_external_cond(obs_seq, goal)
        out = policy._run_diffusion_loop(external_cond=folded_external_cond, num_inference_steps=2)
        assert torch.isfinite(out).all()
