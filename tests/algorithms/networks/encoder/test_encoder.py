from collections.abc import Mapping

import pytest
import torch

from policy.algorithms.networks.encoder import (
    ConditioningContract,
    ConditioningEncoder,
    StateTokenizer,
)
from policy.transforms.canonicalization import Canonicalizer
from policy.utils import flatten_and_concat_leaf_tensors


class TestConditioningEncoderLogic:
    """Unit tests for ConditioningEncoder: the network-owned replacement for what
    GoalConditionedDiffusionPolicy used to own itself (tokenizer/embedder/relative-goal/packaging).

    Mirrors the equivalent coverage that lived in ``tests/algorithms/test_diffusion.py`` before
    the encoder moved to the network side; see that file's history for the pre-move behavior these
    tests are pinned against.
    """

    # ------------------------------------------------------------------ #
    # Default tokenizer, absolute conditioning (relative_goal=False)
    # ------------------------------------------------------------------ #
    def test_default_goal_conditioned_is_false(self):
        encoder = ConditioningEncoder(obs_dim=48, proprio_dim=18)
        assert encoder.goal_conditioned is False

    def test_default_tokenizer_is_state_tokenizer(self):
        encoder = ConditioningEncoder(obs_dim=48, goal_conditioned=True, proprio_dim=18)
        assert isinstance(encoder.tokenizer, StateTokenizer)
        assert encoder.tokenizer.relative_goal is False
        assert encoder.embedder is None
        assert encoder.pooling is None
        assert encoder.tokens_per_step == 1
        assert encoder.output_dim == 30

        encoder_rel = ConditioningEncoder(
            obs_dim=48, goal_conditioned=True, relative_goal=True, proprio_dim=18
        )
        assert isinstance(encoder_rel.tokenizer, StateTokenizer)
        assert encoder_rel.tokenizer.relative_goal is True

    def test_none_tokenizer_and_embedder_and_pooling_runs(self):
        encoder = ConditioningEncoder(
            obs_dim={"proprio": 18, "task": 30}, goal_conditioned=True, relative_goal=True
        )
        obs = {"proprio": torch.randn(2, 2, 18), "task": torch.randn(2, 2, 30)}
        goal = {"proprio": torch.randn(2, 18), "task": torch.randn(2, 30)}
        ext_cond = encoder(obs, goal)
        assert set(ext_cond) == {"obs"}
        assert ext_cond["obs"]["task"].shape == (2, 2, 30)

    def test_absolute_mode_reports_separate_goal_entry(self):
        encoder = ConditioningEncoder(obs_dim=48, goal_conditioned=True, proprio_dim=18)
        assert encoder.cond_dims == ConditioningContract(step_dim=48, global_dim=30)
        assert encoder.cond_dims["obs"] == 48
        assert encoder.cond_dims["goal"] == 30

        obs = torch.randn(2, 2, 48)
        goal = torch.randn(2, 48)
        ext_cond = encoder(obs, goal)

        assert set(ext_cond) == {"obs", "goal"}
        obs_cond = ext_cond["obs"]
        assert isinstance(obs_cond, Mapping)
        assert torch.equal(obs_cond["proprio"], obs[:, :, :18])
        assert torch.equal(obs_cond["task"], obs[:, :, 18:])  # no-op embedder pass-through
        assert torch.equal(ext_cond["goal"], goal[:, 18:])

    def test_unconditioned_has_no_goal_key(self):
        encoder = ConditioningEncoder(obs_dim=48, goal_conditioned=False, proprio_dim=18)
        assert encoder.cond_dims == ConditioningContract(step_dim=48)

        ext_cond = encoder(torch.randn(2, 2, 48))
        assert set(ext_cond) == {"obs"}

    def test_dict_obs_dim(self):
        encoder = ConditioningEncoder(
            obs_dim={"proprio": 18, "task_a": 10, "task_b": 20}, goal_conditioned=True
        )
        assert encoder.proprio_dim == 18
        assert encoder.task_dim == 30

        goal = {
            "proprio": torch.randn(2, 18),
            "task_a": torch.randn(2, 10),
            "task_b": torch.randn(2, 20),
        }
        obs = {
            "proprio": torch.randn(2, 2, 18),
            "task_a": torch.randn(2, 2, 10),
            "task_b": torch.randn(2, 2, 20),
        }
        ext_cond = encoder(obs, goal)
        assert ext_cond["goal"].shape == (2, 30)
        assert torch.equal(ext_cond["goal"][:, :10], goal["task_a"])
        assert torch.equal(ext_cond["goal"][:, 10:], goal["task_b"])

    # ------------------------------------------------------------------ #
    # relative_goal=True
    # ------------------------------------------------------------------ #
    def test_relative_goal_folds_goal_into_obs(self):
        encoder = ConditioningEncoder(
            obs_dim={"proprio": 18, "task": 30}, goal_conditioned=True, relative_goal=True
        )
        assert encoder.cond_dims == ConditioningContract(step_dim=48)

        obs = {"proprio": torch.randn(2, 2, 18), "task": torch.randn(2, 2, 30)}
        goal = {"proprio": torch.randn(2, 18), "task": torch.randn(2, 30)}
        ext_cond = encoder(obs, goal)

        assert set(ext_cond) == {"obs"}
        obs_cond = ext_cond["obs"]
        assert isinstance(obs_cond, Mapping)
        assert torch.equal(obs_cond["proprio"], obs["proprio"])
        expected = goal["task"].unsqueeze(1) - obs["task"]
        assert torch.equal(obs_cond["task"], expected)

    def test_relative_goal_taken_before_a_nonlinear_embedder(self):
        torch.manual_seed(0)
        encoder = ConditioningEncoder(
            obs_dim={"proprio": 18, "task": 30},
            goal_conditioned=True,
            relative_goal=True,
            embedder={
                "_target_": "policy.algorithms.networks.encoder.embedders.mlp.MLP",
                "output_dim": 8,
                "hidden_dims": [16],
            },
        )
        assert encoder.cond_dims == ConditioningContract(step_dim=26)

        obs = {"proprio": torch.randn(2, 2, 18), "task": torch.randn(2, 2, 30)}
        goal = {"proprio": torch.randn(2, 18), "task": torch.randn(2, 30)}
        obs_cond = encoder(obs, goal)["obs"]
        assert isinstance(obs_cond, Mapping)

        with torch.no_grad():
            expected = encoder.embedder(goal["task"].unsqueeze(1) - obs["task"])
        assert torch.allclose(obs_cond["task"], expected)

    def test_relative_goal_requires_goal_conditioning(self):
        with pytest.raises(ValueError, match="requires goal_conditioned=True"):
            ConditioningEncoder(
                obs_dim={"proprio": 18, "task": 30}, goal_conditioned=False, relative_goal=True
            )

    def test_relative_goal_rejects_obs_without_time_axis(self):
        encoder = ConditioningEncoder(
            obs_dim={"proprio": 18, "task": 30}, goal_conditioned=True, relative_goal=True
        )
        with pytest.raises(ValueError, match=r"expects observations of shape \[B, T, F\]"):
            encoder(
                {"proprio": torch.randn(2, 18), "task": torch.randn(2, 30)},
                {"proprio": torch.randn(2, 18), "task": torch.randn(2, 30)},
            )

    # ------------------------------------------------------------------ #
    # Pooling embedder collapses the time axis
    # ------------------------------------------------------------------ #
    def test_pooling_embedder_promotes_task_to_top_level(self):
        """A pooling embedder collapses the time axis, so its "task" entry must move out from under
        "obs" (which keeps a real per-timestep width) to a top-level key, mirroring "goal".

        Also regression-tests that the cond_dims a UNet would use to size FiLM (obs_horizon
        multiplies only "obs") stay consistent with the actual flattened conditioning width.
        """
        encoder = ConditioningEncoder(
            obs_dim={"proprio": 18, "task": 30},
            goal_conditioned=True,
            relative_goal=True,
            embedder={
                "_target_": "policy.algorithms.networks.encoder.embedders.self_attention.SelfAttention",
                "output_dim": 8,
                "obs_horizon": 2,
                "num_heads": 2,
            },
            pooling={
                "_target_": "policy.algorithms.networks.encoder.pooling.attention.AttentionPooling",
                "dim": 8,
                "num_heads": 2,
            },
        )
        cond_dims = encoder.cond_dims
        assert cond_dims == ConditioningContract(step_dim=18, global_dim=8)
        encoder.embedder.eval()

        obs = {"proprio": torch.randn(2, 2, 18), "task": torch.randn(2, 2, 30)}
        goal = {"proprio": torch.randn(2, 18), "task": torch.randn(2, 30)}
        ext_cond = encoder(obs, goal)

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
    # ObjectTokenizer (tokens_per_step > 1)
    # ------------------------------------------------------------------ #
    def _per_object_encoder(self, **overrides) -> ConditioningEncoder:
        kwargs = dict(
            obs_dim=Canonicalizer.dim_spec(3),
            goal_conditioned=True,
            relative_goal=True,
            tokenizer={
                "_target_": "policy.algorithms.networks.encoder.tokenizers.ObjectTokenizer",
                "object_keys": ("obj_0_pose", "obj_1_pose", "obj_2_pose"),
            },
            embedder={
                "_target_": "policy.algorithms.networks.encoder.embedders.self_attention.SelfAttention",
                "output_dim": 8,
                "obs_horizon": 2,
                "num_heads": 2,
            },
        )
        kwargs.update(overrides)
        return ConditioningEncoder(**kwargs)

    def _per_object_obs_goal(self, batch_size=2):
        obs = {
            "proprio": torch.randn(batch_size, 2, 18),
            "obj_0_pose": torch.randn(batch_size, 2, 7),
            "obj_0_role": torch.tensor([1.0, 0.0, 0.0]).expand(batch_size, 2, 3),
            "obj_1_pose": torch.randn(batch_size, 2, 7),
            "obj_1_role": torch.tensor([0.0, 1.0, 0.0]).expand(batch_size, 2, 3),
            "obj_2_pose": torch.randn(batch_size, 2, 7),
            "obj_2_role": torch.tensor([0.0, 0.0, 1.0]).expand(batch_size, 2, 3),
            "tcp_pose": torch.randn(batch_size, 2, 7),
        }
        goal = {
            "proprio": torch.randn(batch_size, 18),
            "obj_0_pose": torch.randn(batch_size, 7),
            "obj_0_role": torch.tensor([1.0, 0.0, 0.0]).expand(batch_size, 3),
            "obj_1_pose": torch.randn(batch_size, 7),
            "obj_1_role": torch.tensor([0.0, 1.0, 0.0]).expand(batch_size, 3),
            "obj_2_pose": torch.randn(batch_size, 7),
            "obj_2_role": torch.tensor([0.0, 0.0, 1.0]).expand(batch_size, 3),
            "tcp_pose": torch.randn(batch_size, 7),
        }
        return obs, goal

    def test_per_object_tokenizer_folds_tokens_without_cross_attention(self):
        encoder = self._per_object_encoder()
        assert encoder.cond_dims == ConditioningContract(step_dim=18 + 8 * 3)

        obs, goal = self._per_object_obs_goal()
        obs_cond = encoder(obs, goal)["obs"]
        assert isinstance(obs_cond, Mapping)
        assert torch.equal(obs_cond["proprio"], obs["proprio"])
        assert obs_cond["task"].shape == (2, 2, 8 * 3)

    def test_per_object_tokenizer_with_cross_attention_keeps_a_token_sequence(self):
        encoder = self._per_object_encoder(mode="cross_attention")
        assert encoder.cond_dims == ConditioningContract(
            step_dim=18,
            context_dim=8,
        )

        obs, goal = self._per_object_obs_goal()
        ext_cond = encoder(obs, goal)

        assert set(ext_cond) == {"obs", "context"}
        obs_cond = ext_cond["obs"]
        assert isinstance(obs_cond, Mapping)
        assert set(obs_cond) == {"proprio"}
        assert torch.equal(obs_cond["proprio"], obs["proprio"])

        # 3 tokens per timestep (obj_0, obj_1, obj_2) x 2 timesteps, t-major k-minor, kept as a
        # sequence (not folded into a wider per-timestep vector).
        context = ext_cond["context"]
        assert isinstance(context, torch.Tensor)
        assert context.shape == (2, 2 * 3, 8)

    def test_cross_attention_rejects_a_single_token_tokenizer(self):
        with pytest.raises(ValueError, match="tokens_per_step > 1"):
            ConditioningEncoder(
                obs_dim={"proprio": 18, "task": 30},
                goal_conditioned=True,
                relative_goal=True,
                mode="cross_attention",
            )

    # ------------------------------------------------------------------ #
    # Validation
    #
    # NOTE: ObjectTokenizer's compatible_relative_goal={True} means any relative_goal=False
    # trips that check before it could ever reach the cross-attention-specific "requires
    # relative_goal=True" branch, or the supports_single_side="cannot be used unconditioned"
    # branch (compatible_relative_goal not containing False already implies
    # supports_single_side=False is never reached unconditioned either) -- with today's two
    # tokenizers, those two branches are unreachable defensive code, same as before this class
    # existed (the tokenizer-level compatible_relative_goal check already covers every real case).
    # ------------------------------------------------------------------ #
    def test_rejects_incompatible_relative_goal_for_tokenizer(self):
        class SingleSideUnsupportedTokenizer(StateTokenizer):
            supports_single_side = False

        with pytest.raises(ValueError, match="cannot be used with relative_goal=False"):
            ConditioningEncoder(
                obs_dim=48,
                goal_conditioned=True,
                relative_goal=False,
                proprio_dim=18,
                tokenizer=SingleSideUnsupportedTokenizer(task_dim=30),
            )

    # ------------------------------------------------------------------ #
    # extract_embeddings
    # ------------------------------------------------------------------ #
    def test_extract_embeddings_reports_absolute_embeddings_under_delta_mode(self):
        """extract_embeddings stays absolute (not differenced), independently of relative_goal."""
        encoder = ConditioningEncoder(
            obs_dim={"proprio": 18, "task": 30}, goal_conditioned=True, relative_goal=True
        )
        obs = {"proprio": torch.randn(2, 2, 18), "task": torch.randn(2, 2, 30)}
        goal = {"proprio": torch.randn(2, 18), "task": torch.randn(2, 30)}
        embeddings = encoder.extract_embeddings(obs, goal=goal)
        assert torch.equal(embeddings["obs_embeddings"], obs["task"])
        assert torch.equal(embeddings["goal_embedding"], goal["task"])

    # ------------------------------------------------------------------ #
    # Axis-Selective Pooling (mode="all", "objects", "time")
    # ------------------------------------------------------------------ #
    def test_encoder_pooling_mode_all(self):
        from policy.algorithms.networks.encoder.pooling import AttentionPooling
        from policy.algorithms.networks.encoder.tokenizers.object import ObjectTokenizer

        tokenizer = ObjectTokenizer(object_keys=("obj_0_pose", "obj_1_pose", "obj_2_pose"))
        encoder = ConditioningEncoder(
            obs_dim=Canonicalizer.dim_spec(3),
            goal_conditioned=True,
            relative_goal=True,
            tokenizer=tokenizer,
            pooling=AttentionPooling(dim=15, num_heads=3, mode="all"),
        )
        assert encoder.pools_time is True
        assert encoder.pools_objects is True
        assert encoder.cond_dims == ConditioningContract(
            step_dim=18,
            global_dim=15,
        )

        obs, goal = self._per_object_obs_goal()
        ext_cond = encoder(obs, goal)
        assert set(ext_cond) == {"obs", "task"}
        assert ext_cond["task"].shape == (2, 15)

    def test_encoder_pooling_mode_objects(self):
        from policy.algorithms.networks.encoder.pooling import AttentionPooling
        from policy.algorithms.networks.encoder.tokenizers.object import ObjectTokenizer

        tokenizer = ObjectTokenizer(object_keys=("obj_0_pose", "obj_1_pose", "obj_2_pose"))
        encoder = ConditioningEncoder(
            obs_dim=Canonicalizer.dim_spec(3),
            goal_conditioned=True,
            relative_goal=True,
            tokenizer=tokenizer,
            pooling=AttentionPooling(dim=15, num_heads=3, mode="objects"),
        )
        assert encoder.pools_time is False
        assert encoder.pools_objects is True
        assert encoder.cond_dims == ConditioningContract(
            step_dim=18 + 15,
        )

        obs, goal = self._per_object_obs_goal()
        ext_cond = encoder(obs, goal)
        assert set(ext_cond) == {"obs"}
        assert ext_cond["obs"]["task"].shape == (2, 2, 15)

    def test_encoder_pooling_mode_time(self):
        from policy.algorithms.networks.encoder.pooling import AttentionPooling
        from policy.algorithms.networks.encoder.tokenizers.object import ObjectTokenizer

        tokenizer = ObjectTokenizer(object_keys=("obj_0_pose", "obj_1_pose", "obj_2_pose"))
        encoder = ConditioningEncoder(
            obs_dim=Canonicalizer.dim_spec(3),
            goal_conditioned=True,
            relative_goal=True,
            tokenizer=tokenizer,
            pooling=AttentionPooling(dim=15, num_heads=3, mode="time"),
        )
        assert encoder.pools_time is True
        assert encoder.pools_objects is False
        assert encoder.cond_dims == ConditioningContract(
            step_dim=18,
            global_dim=15,
        )

        obs, goal = self._per_object_obs_goal()
        ext_cond = encoder(obs, goal)
        assert set(ext_cond) == {"obs", "task"}
        assert ext_cond["task"].shape == (2, 3, 15)
