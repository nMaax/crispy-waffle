"""Tokenization and token-space normalization, which the algorithm owns.

The tokenizer is parameterless geometry and the encoder is learned, so normalization sits between
them: ``_tokenize -> _normalize_obs -> encoder``. Normalizing the raw tree instead would hand
non-unit quaternions to the tokenizer's SE(3) math and z-score the one-hot role indicators to zero.
"""

import pytest
import torch

from policy.algorithms.goal_conditioned_diffusion_policy import GoalConditionedDiffusionPolicy
from policy.algorithms.tokenizers import ObjectTokenizer, StateTokenizer
from policy.transforms import MinMaxNormalizer, ZScoreNormalizer
from policy.transforms.canonicalization.spec import ROLE_PICK, ROLE_TARGET, canonical_dim_spec

DECODER = "policy.algorithms.networks.decoder.unet1d.FiLMDecoder1D"
ENCODER = "policy.algorithms.networks.encoder.encoder.ConditioningEncoder"
OBJECT_TOKENIZER = "policy.algorithms.tokenizers.object.ObjectTokenizer"
STATE_TOKENIZER = "policy.algorithms.tokenizers.state.StateTokenizer"

OBS_HORIZON = 2
NUM_OBJECTS = 2
ROLES = (ROLE_PICK, ROLE_TARGET)


def _policy(tokenizer=STATE_TOKENIZER, relative_goal=True, encoder=True, **overrides):
    kwargs = dict(
        decoder={"_target_": DECODER, "act_dim": 4, "obs_horizon": OBS_HORIZON},
        optimizer={},
        ema={"_target_": "diffusers.training_utils.EMAModel"},
        noise_scheduler={"_target_": "diffusers.schedulers.scheduling_ddpm.DDPMScheduler"},
        obs_dim=canonical_dim_spec(NUM_OBJECTS),
        act_dim=4,
        obs_horizon=OBS_HORIZON,
        pred_horizon=16,
        act_horizon=8,
        relative_goal=relative_goal,
        goal_horizon=1,
    )
    if tokenizer is not None:
        kwargs["tokenizer"] = (
            {"_target_": tokenizer, "object_keys": ["obj_0_pose", "obj_1_pose"]}
            if tokenizer == OBJECT_TOKENIZER
            else {"_target_": tokenizer}
        )
    if encoder:
        kwargs["encoder"] = {"_target_": ENCODER, "_recursive_": False}
    kwargs.update(overrides)
    return GoalConditionedDiffusionPolicy(**kwargs)


def _canonical_tree(batch_size=4, time_axis=True):
    """A canonicalized obs/goal tree with genuine unit quaternions and constant role one-hots."""
    tree = {}
    for key, dim in canonical_dim_spec(NUM_OBJECTS).items():
        shape = (batch_size, OBS_HORIZON) if time_axis else (batch_size,)
        if key.endswith("_pose"):
            quat = torch.randn(*shape, 4)
            tree[key] = torch.cat([torch.randn(*shape, 3), quat / quat.norm(dim=-1, keepdim=True)], -1)
        elif key.endswith("_role"):
            tree[key] = torch.tensor(ROLES[int(key.split("_")[1])]).expand(*shape, 3).clone()
        else:
            tree[key] = torch.randn(*shape, dim)
    return tree


class TestTokenizeRouting:
    """Coverage that moved off ConditioningEncoder when the tokenizer went up to the algorithm."""

    def test_relative_goal_folds_the_goal_into_the_task_tokens(self):
        policy = _policy(relative_goal=True)
        policy._instantiate_tokenizer()

        obs, goal = _canonical_tree(), _canonical_tree(time_axis=False)
        tokens = policy._tokenize(obs, goal)

        assert set(tokens) == {"proprio", "task"}
        assert torch.equal(tokens["proprio"], obs["proprio"])
        assert tokens["task"].shape == (4, OBS_HORIZON, policy.tokenizer.output_dim)

    def test_absolute_mode_emits_standalone_goal_tokens(self):
        policy = _policy(relative_goal=False)
        policy._instantiate_tokenizer()

        obs, goal = _canonical_tree(), _canonical_tree(time_axis=False)
        tokens = policy._tokenize(obs, goal)

        assert set(tokens) == {"proprio", "task", "goal_task"}
        assert tokens["task"].shape == (4, OBS_HORIZON, policy.tokenizer.output_dim)
        assert tokens["goal_task"].shape == (4, policy.tokenizer.output_dim)

    def test_goal_proprio_never_enters_conditioning(self):
        policy = _policy(relative_goal=False)
        policy._instantiate_tokenizer()

        obs, goal = _canonical_tree(), _canonical_tree(time_axis=False)
        tokens = policy._tokenize(obs, goal)
        assert torch.equal(tokens["proprio"], obs["proprio"])

    def test_relative_goal_rejects_obs_without_time_axis(self):
        policy = _policy(relative_goal=True)
        policy._instantiate_tokenizer()

        with pytest.raises(ValueError, match=r"expects observations of shape \[B, T, F\]"):
            policy._tokenize(_canonical_tree(time_axis=False), _canonical_tree(time_axis=False))

    def test_tokenize_requires_a_goal(self):
        policy = _policy()
        policy._instantiate_tokenizer()
        with pytest.raises(ValueError, match="received goal=None"):
            policy._tokenize(_canonical_tree(), None)

    def test_encoder_without_a_tokenizer_is_rejected(self):
        with pytest.raises(ValueError, match="an encoder but no tokenizer"):
            _policy(tokenizer=None, encoder=True)._instantiate_tokenizer()

    def test_rejects_incompatible_relative_goal_for_tokenizer(self):
        """A delta-only tokenizer has nothing to emit for absolute conditioning."""
        policy = _policy(relative_goal=False, encoder=False)
        policy.tokenizer = _SingleSideUnsupported(task_dim={"task": 30}, relative_goal=False)
        with pytest.raises(ValueError, match="cannot be used with relative_goal=False"):
            policy._validate_tokenizer()


class _SingleSideUnsupported(StateTokenizer):
    supports_single_side = False


class TestNormalizationHappensOnTokens:
    """The two defects this ordering exists to fix."""

    def test_tokenizer_receives_unnormalized_unit_quaternions(self):
        """Normalizing before tokenization would break `relative_se3_pose`'s quaternion algebra."""
        policy = _policy(tokenizer=OBJECT_TOKENIZER, relative_goal=True, obs_normalizer=True)
        policy.configure_model()

        obs, goal = _canonical_tree(), _canonical_tree(time_axis=False)
        policy.obs_normalizer.fit(policy._tokenize(obs, goal))

        seen = []
        raw_tokenize = policy.tokenizer.tokenize
        policy.tokenizer.tokenize = lambda o, g: (seen.append((o, g)), raw_tokenize(o, g))[1]

        policy._encode({"obs": obs, "goal": goal})

        assert seen, "tokenizer was never called"
        for obs_task, goal_task in seen:
            for key, value in (obs_task or {}).items():
                if key.endswith("_pose"):
                    norms = value[..., 3:7].norm(dim=-1)
                    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), key
                    assert torch.equal(value, obs[key]), f"{key} was normalized before tokenizing"

    def test_role_one_hots_survive_normalization(self):
        """A constant one-hot z-scores to exactly zero unless the mask excludes it."""
        policy = _policy(tokenizer=OBJECT_TOKENIZER, relative_goal=True, obs_normalizer=True)
        policy.configure_model()

        obs, goal = _canonical_tree(), _canonical_tree(time_axis=False)
        tokens = policy._tokenize(obs, goal)
        policy.obs_normalizer.fit(tokens)
        normalized = policy.obs_normalizer.normalize(tokens)

        mask = policy.tokenizer.categorical_mask
        assert torch.equal(normalized["task"][..., ~mask], tokens["task"][..., ~mask])
        assert not torch.allclose(normalized["task"][..., mask], tokens["task"][..., mask])

        # Without the mask the one-hot is destroyed -- the behaviour being fixed.
        unmasked = type(policy.obs_normalizer)(policy._obs_normalizer_spec())
        unmasked.fit(tokens)
        mangled = unmasked.normalize(tokens)["task"][..., ~mask]
        assert not torch.equal(mangled, tokens["task"][..., ~mask])
        assert not torch.isin(mangled, torch.tensor([0.0, 1.0])).all(), "still a valid one-hot"

    def test_normalizer_is_fit_in_token_space_not_observation_space(self):
        policy = _policy(tokenizer=OBJECT_TOKENIZER, relative_goal=True, obs_normalizer=True)
        policy.configure_model()

        assert policy._obs_normalizer_spec() == {
            "proprio": policy.proprio_dim,
            "task": policy.tokenizer.output_dim,
        }
        assert policy.obs_normalizer.norms["task"].mean.shape == (policy.tokenizer.output_dim,)

    def test_configure_model_wires_zscore_to_obs_and_minmax_to_act(self):
        """The bare-``True`` defaults, asserted through the real construction path."""
        policy = _policy(
            tokenizer=OBJECT_TOKENIZER,
            relative_goal=True,
            obs_normalizer=True,
            act_normalizer=True,
        )
        policy.configure_model()

        assert isinstance(policy.obs_normalizer, ZScoreNormalizer)
        assert isinstance(policy.act_normalizer, MinMaxNormalizer)
        assert policy.act_normalizer.min.shape == (policy.act_dim,)

    def test_normalizer_mask_falls_back_to_the_canonical_schema_without_a_tokenizer(self):
        """BesoPolicy has no tokenizer, so its role keys are masked at the canonical-tree level."""
        policy = _policy(tokenizer=None, encoder=False)

        mask = policy._obs_normalizer_mask()
        assert not mask["obj_0_role"].any()
        assert mask["obj_0_pose"].all()
        assert mask["proprio"].all()

    def test_masked_channels_are_untouched_by_the_round_trip(self):
        policy = _policy(tokenizer=None, encoder=False)
        policy.obs_normalizer = ZScoreNormalizer(
            policy._obs_normalizer_spec(), mask=policy._obs_normalizer_mask()
        )

        obs = _canonical_tree()
        policy.obs_normalizer.fit(obs)
        normalized = policy._normalize_obs(obs)

        assert torch.equal(normalized["obj_0_role"], obs["obj_0_role"])
        assert torch.equal(normalized["obj_1_role"], obs["obj_1_role"])
        assert not torch.allclose(normalized["proprio"], obs["proprio"])


class TestCategoricalMaskLayout:
    """The mask indexes real channel positions, which differ between the two tokenize paths."""

    SENTINEL = 7.0

    def _sentinel_tree(self, time_axis=True):
        tree = _canonical_tree(time_axis=time_axis)
        for key in tree:
            if key.endswith("_role"):
                tree[key] = torch.full_like(tree[key], self.SENTINEL)
        return tree

    @pytest.mark.parametrize("relative_goal", [True, False])
    def test_state_tokenizer_mask_matches_emitted_channels(self, relative_goal):
        task_dim = {k: v for k, v in canonical_dim_spec(NUM_OBJECTS).items() if k != "proprio"}
        tokenizer = StateTokenizer(task_dim=task_dim, relative_goal=relative_goal)

        obs = self._sentinel_tree()
        goal = self._sentinel_tree(time_axis=False)
        obs.pop("proprio"), goal.pop("proprio")
        goal = {k: v.unsqueeze(1) for k, v in goal.items()}

        tokens = tokenizer.tokenize(obs, goal) if relative_goal else tokenizer.tokenize(obs, None)
        mask = tokenizer.categorical_mask

        assert tokens.shape[-1] == tokenizer.output_dim == mask.shape[0]
        assert (tokens[..., ~mask] == self.SENTINEL).all(), "mask misses a role channel"
        assert (tokens[..., mask] != self.SENTINEL).all(), "mask covers a non-role channel"

    @pytest.mark.parametrize("relative_goal", [True, False])
    def test_object_tokenizer_mask_matches_emitted_channels(self, relative_goal):
        tokenizer = ObjectTokenizer(
            object_keys=["obj_0_pose", "obj_1_pose"], relative_goal=relative_goal
        )

        obs = self._sentinel_tree()
        goal = self._sentinel_tree(time_axis=False)
        obs.pop("proprio"), goal.pop("proprio")
        goal = {k: v.unsqueeze(1) for k, v in goal.items()}

        tokens = tokenizer.tokenize(obs, goal) if relative_goal else tokenizer.tokenize(obs, None)
        mask = tokenizer.categorical_mask

        assert tokens.shape[-1] == tokenizer.output_dim == mask.shape[0]
        assert (tokens[..., ~mask] == self.SENTINEL).all()
        assert (tokens[..., mask] != self.SENTINEL).all()


class TestNormalizerFitting:
    def test_fit_runs_over_tokenized_items(self):
        policy = _policy(tokenizer=OBJECT_TOKENIZER, relative_goal=True, obs_normalizer=True)
        policy.configure_model()

        items = [
            {
                "obs_seq": {k: v[i] for k, v in _canonical_tree().items()},
                "goal": {k: v[i] for k, v in _canonical_tree(time_axis=False).items()},
                "act_seq": torch.randn(16, 4),
            }
            for i in range(4)
        ]
        policy.trainer = _FakeTrainer(items)

        policy.on_fit_start()

        assert bool(policy.obs_normalizer.is_fit)
        assert policy.obs_normalizer.norms["task"].mean.shape == (policy.tokenizer.output_dim,)

    def test_fitting_one_normalizer_does_not_depend_on_the_other(self):
        """The old gate skipped fitting entirely unless BOTH normalizers were configured."""
        policy = _policy(tokenizer=OBJECT_TOKENIZER, relative_goal=True, obs_normalizer=True)
        policy.configure_model()
        assert policy.act_normalizer is None

        items = [
            {
                "obs_seq": {k: v[i] for k, v in _canonical_tree().items()},
                "goal": {k: v[i] for k, v in _canonical_tree(time_axis=False).items()},
                "act_seq": torch.randn(16, 4),
            }
            for i in range(4)
        ]
        policy.trainer = _FakeTrainer(items)
        policy.on_fit_start()

        assert bool(policy.obs_normalizer.is_fit)


class _FakeTrainSet(list):
    lazy = False


class _FakeDataModule:
    def __init__(self, items):
        self.train_set = _FakeTrainSet(items)


class _FakeTrainer:
    def __init__(self, items):
        self.datamodule = _FakeDataModule(items)
