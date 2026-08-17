from collections.abc import Mapping

import pytest
import torch

from policy.algorithms.networks.encoder import ConditioningContract, ConditioningEncoder
from policy.utils import flatten_and_concat_leaf_tensors

SELF_ATTENTION = "policy.algorithms.networks.encoder.embedders.self_attention.SelfAttention"
MLP = "policy.algorithms.networks.encoder.embedders.mlp.MLP"


def _tokens(batch_size=2, obs_horizon=2, proprio_dim=18, token_dim=30, tokens_per_step=None):
    """The ``{"proprio", "task"}`` tree the algorithm's ``_tokenize`` hands the encoder."""
    task = (
        torch.randn(batch_size, obs_horizon, token_dim)
        if tokens_per_step is None
        else torch.randn(batch_size, obs_horizon, tokens_per_step, token_dim)
    )
    return {"proprio": torch.randn(batch_size, obs_horizon, proprio_dim), "task": task}


def _goal_tokens(batch_size=2, token_dim=30, tokens_per_step=None):
    return (
        torch.randn(batch_size, token_dim)
        if tokens_per_step is None
        else torch.randn(batch_size, tokens_per_step, token_dim)
    )


class TestConditioningEncoderLogic:
    """Unit tests for ConditioningEncoder: embedder, pooling and payload packing.

    Tokenization moved up to the algorithm (see ``BaseDiffusionAgent._tokenize``) so that
    normalization can sit between the tokenizer's geometry and these learned layers; the
    tokenization/routing half of the old coverage now lives in
    ``tests/algorithms/test_tokenization.py``.
    """

    # ------------------------------------------------------------------ #
    # Packing and cond_dims
    # ------------------------------------------------------------------ #
    def test_default_goal_conditioned_is_false(self):
        encoder = ConditioningEncoder(proprio_dim=18, token_dim=30)
        assert encoder.goal_conditioned is False

    def test_no_embedder_and_no_pooling_passes_tokens_through(self):
        encoder = ConditioningEncoder(proprio_dim=18, token_dim=30)
        assert encoder.embedder is None
        assert encoder.pooling is None
        assert encoder.output_dim == 30

        tokens = _tokens()
        ext_cond = encoder(tokens)
        assert set(ext_cond) == {"obs"}
        assert torch.equal(ext_cond["obs"]["task"], tokens["task"])
        assert torch.equal(ext_cond["obs"]["proprio"], tokens["proprio"])

    def test_absolute_mode_reports_separate_goal_entry(self):
        encoder = ConditioningEncoder(proprio_dim=18, token_dim=30, goal_conditioned=True)
        assert encoder.cond_dims == ConditioningContract(step_dim=48, global_dim=30)
        assert encoder.cond_dims["obs"] == 48
        assert encoder.cond_dims["goal"] == 30

        tokens = _tokens() | {"goal_task": _goal_tokens()}
        ext_cond = encoder(tokens)

        assert set(ext_cond) == {"obs", "goal"}
        obs_cond = ext_cond["obs"]
        assert isinstance(obs_cond, Mapping)
        assert torch.equal(obs_cond["proprio"], tokens["proprio"])
        assert torch.equal(obs_cond["task"], tokens["task"])  # no-op embedder pass-through
        assert torch.equal(ext_cond["goal"], tokens["goal_task"])

    def test_unconditioned_has_no_goal_key(self):
        encoder = ConditioningEncoder(proprio_dim=18, token_dim=30, goal_conditioned=False)
        assert encoder.cond_dims == ConditioningContract(step_dim=48)

        ext_cond = encoder(_tokens())
        assert set(ext_cond) == {"obs"}

    def test_relative_goal_folds_the_goal_into_obs(self):
        """relative_goal tokens already carry the goal, so no standalone goal entry is reported."""
        encoder = ConditioningEncoder(
            proprio_dim=18, token_dim=30, goal_conditioned=True, relative_goal=True
        )
        assert encoder.cond_dims == ConditioningContract(step_dim=48)

        ext_cond = encoder(_tokens())
        assert set(ext_cond) == {"obs"}

    # ------------------------------------------------------------------ #
    # Configuration validation
    # ------------------------------------------------------------------ #
    def test_has_standalone_goal_tracks_the_configuration(self):
        """Whether a goal stream is embedded is decided by config, not by inspecting the
        payload."""
        assert not ConditioningEncoder(proprio_dim=18, token_dim=30).has_standalone_goal
        assert ConditioningEncoder(
            proprio_dim=18, token_dim=30, goal_conditioned=True
        ).has_standalone_goal
        assert not ConditioningEncoder(
            proprio_dim=18, token_dim=30, goal_conditioned=True, relative_goal=True
        ).has_standalone_goal

    def test_relative_goal_requires_goal_conditioning(self):
        with pytest.raises(ValueError, match="requires goal_conditioned=True"):
            ConditioningEncoder(
                proprio_dim=18, token_dim=30, goal_conditioned=False, relative_goal=True
            )

    # ------------------------------------------------------------------ #
    # Embedder
    # ------------------------------------------------------------------ #
    def test_embedder_is_applied_to_the_task_tokens(self):
        torch.manual_seed(0)
        encoder = ConditioningEncoder(
            proprio_dim=18,
            token_dim=30,
            goal_conditioned=True,
            relative_goal=True,
            embedder={"_target_": MLP, "output_dim": 8, "hidden_dims": [16]},
        )
        assert encoder.cond_dims == ConditioningContract(step_dim=26)

        tokens = _tokens()
        obs_cond = encoder(tokens)["obs"]
        assert isinstance(obs_cond, Mapping)

        with torch.no_grad():
            expected = encoder.embedder(tokens["task"])
        assert torch.allclose(obs_cond["task"], expected)

    # ------------------------------------------------------------------ #
    # Pooling collapses the time axis
    # ------------------------------------------------------------------ #
    def test_pooling_embedder_promotes_task_to_top_level(self):
        """A pooling embedder collapses the time axis, so its "task" entry must move out from under
        "obs" (which keeps a real per-timestep width) to a top-level key, mirroring "goal".

        Also regression-tests that the cond_dims a UNet would use to size FiLM (obs_horizon
        multiplies only "obs") stay consistent with the actual flattened conditioning width.
        """
        encoder = ConditioningEncoder(
            proprio_dim=18,
            token_dim=30,
            goal_conditioned=True,
            relative_goal=True,
            embedder={"_target_": SELF_ATTENTION, "output_dim": 8, "obs_horizon": 2, "num_heads": 2},
            pooling={
                "_target_": "policy.algorithms.networks.encoder.pooling.attention.AttentionPooling",
                "dim": 8,
                "num_heads": 2,
            },
        )
        cond_dims = encoder.cond_dims
        assert cond_dims == ConditioningContract(step_dim=18, global_dim=8)
        encoder.embedder.eval()

        ext_cond = encoder(_tokens())

        assert set(ext_cond) == {"obs", "task"}
        obs_cond = ext_cond["obs"]
        assert isinstance(obs_cond, Mapping)
        assert set(obs_cond) == {"proprio"}
        assert ext_cond["task"].shape == (2, 8)

        obs_horizon = 2
        actual_width = flatten_and_concat_leaf_tensors(ext_cond).shape[-1]
        expected_width = cond_dims.get_film_width(obs_horizon)
        assert actual_width == expected_width

    # ------------------------------------------------------------------ #
    # Multi-token conditioning (tokens_per_step > 1)
    # ------------------------------------------------------------------ #
    def _per_object_encoder(self, **overrides) -> ConditioningEncoder:
        kwargs = dict(
            proprio_dim=18,
            token_dim=15,
            tokens_per_step=3,
            goal_conditioned=True,
            relative_goal=True,
            embedder={"_target_": SELF_ATTENTION, "output_dim": 8, "obs_horizon": 2, "num_heads": 2},
        )
        kwargs.update(overrides)
        return ConditioningEncoder(**kwargs)

    def test_per_object_tokens_are_folded_without_cross_attention(self):
        encoder = self._per_object_encoder()
        assert encoder.cond_dims == ConditioningContract(step_dim=18 + 8 * 3)

        tokens = _tokens(token_dim=15, tokens_per_step=3)
        obs_cond = encoder(tokens)["obs"]
        assert isinstance(obs_cond, Mapping)
        assert torch.equal(obs_cond["proprio"], tokens["proprio"])
        assert obs_cond["task"].shape == (2, 2, 8 * 3)

    def test_per_object_tokens_with_cross_attention_keep_a_token_sequence(self):
        encoder = self._per_object_encoder(decoder_type="cross_attention")
        assert encoder.cond_dims == ConditioningContract(step_dim=18, context_dim=8)

        tokens = _tokens(token_dim=15, tokens_per_step=3)
        ext_cond = encoder(tokens)

        assert set(ext_cond) == {"obs", "context"}
        obs_cond = ext_cond["obs"]
        assert isinstance(obs_cond, Mapping)
        assert set(obs_cond) == {"proprio"}
        assert torch.equal(obs_cond["proprio"], tokens["proprio"])

        # 3 tokens per timestep x 2 timesteps, t-major k-minor, kept as a sequence (not folded
        # into a wider per-timestep vector).
        context = ext_cond["context"]
        assert isinstance(context, torch.Tensor)
        assert context.shape == (2, 2 * 3, 8)

    def test_cross_attention_rejects_a_single_token_stream(self):
        with pytest.raises(ValueError, match="tokens_per_step > 1"):
            ConditioningEncoder(
                proprio_dim=18,
                token_dim=30,
                tokens_per_step=1,
                goal_conditioned=True,
                relative_goal=True,
                decoder_type="cross_attention",
            )

    def test_film_rejects_a_dynamic_token_count_without_object_pooling(self):
        with pytest.raises(ValueError, match="requires\n?\\s*pooling across objects"):
            ConditioningEncoder(
                proprio_dim=18,
                token_dim=15,
                tokens_per_step=None,
                goal_conditioned=True,
                relative_goal=True,
            )

    # ------------------------------------------------------------------ #
    # Axis-Selective Pooling (mode="all", "objects", "time")
    # ------------------------------------------------------------------ #
    def _pooled_encoder(self, mode: str) -> ConditioningEncoder:
        from policy.algorithms.networks.encoder.pooling import AttentionPooling

        return ConditioningEncoder(
            proprio_dim=18,
            token_dim=15,
            tokens_per_step=3,
            goal_conditioned=True,
            relative_goal=True,
            pooling=AttentionPooling(dim=15, num_heads=3, mode=mode),
        )

    def test_encoder_pooling_mode_all(self):
        encoder = self._pooled_encoder("all")
        assert encoder.pools_time is True
        assert encoder.pools_objects is True
        assert encoder.cond_dims == ConditioningContract(step_dim=18, global_dim=15)

        ext_cond = encoder(_tokens(token_dim=15, tokens_per_step=3))
        assert set(ext_cond) == {"obs", "task"}
        assert ext_cond["task"].shape == (2, 15)

    def test_encoder_pooling_mode_objects(self):
        encoder = self._pooled_encoder("objects")
        assert encoder.pools_time is False
        assert encoder.pools_objects is True
        assert encoder.cond_dims == ConditioningContract(step_dim=18 + 15)

        ext_cond = encoder(_tokens(token_dim=15, tokens_per_step=3))
        assert set(ext_cond) == {"obs"}
        assert ext_cond["obs"]["task"].shape == (2, 2, 15)

    def test_encoder_pooling_mode_time(self):
        encoder = self._pooled_encoder("time")
        assert encoder.pools_time is True
        assert encoder.pools_objects is False
        assert encoder.cond_dims == ConditioningContract(step_dim=18, global_dim=15)

        ext_cond = encoder(_tokens(token_dim=15, tokens_per_step=3))
        assert set(ext_cond) == {"obs", "task"}
        assert ext_cond["task"].shape == (2, 3, 15)
