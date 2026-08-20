from collections import deque
from unittest.mock import MagicMock

import pytest
import torch

from policy.algorithms.beso_policy import BesoPolicy
from policy.transforms import MinMaxNormalizer
from policy.transforms.canonicalization.spec import (
    ROLE_PICK,
    ROLE_TARGET,
    ROLE_TCP,
    canonical_dim_spec,
    dim_shape,
)
from tests.algorithms._beso_test_utils import mock_loop_internals

DIFFUSION_GPT = "policy.algorithms.networks.decoder.diffusion_gpt.DiffusionGPT"
OBJECT_DIFFUSION_GPT = "policy.algorithms.networks.decoder.diffusion_gpt.ObjectDiffusionGPT"
STATE_TOKENIZER = "policy.algorithms.tokenizers.state.StateTokenizer"
OBJECT_TOKENIZER = "policy.algorithms.tokenizers.object.ObjectTokenizer"

OBS_HORIZON = 2
NUM_OBJECTS = 2
# Pool slot order: the TCP first, then the pick and target objects.
ROLES = (ROLE_TCP, ROLE_PICK, ROLE_TARGET)


def _canonical_tree(batch_size=1, horizon=OBS_HORIZON, time_axis=True):
    """A canonicalized obs/goal tree with genuine unit quaternions and constant role one-hots."""
    lead = (batch_size, horizon) if time_axis else (batch_size,)
    tree = {}
    for key, dim in canonical_dim_spec(NUM_OBJECTS).items():
        shape = lead + dim_shape(dim)
        if key.endswith("_pose"):
            quat = torch.randn(*shape[:-1], 4)
            tree[key] = torch.cat(
                [torch.randn(*shape[:-1], 3), quat / quat.norm(dim=-1, keepdim=True)], -1
            )
        elif key.endswith("_role"):
            tree[key] = torch.tensor(ROLES).expand(*shape).clone()
        elif key == "obj_valid":
            tree[key] = torch.ones(*shape)
        else:
            tree[key] = torch.randn(*shape)
    return tree


def _basic_kwargs(**overrides):
    """Mock kwargs that construct a BesoPolicy without invoking hydra_zen.instantiate."""
    kw = dict(
        decoder={"_target_": DIFFUSION_GPT},
        tokenizer={"_target_": STATE_TOKENIZER},
        ema={},
        optimizer={},
        act_dim=4,
        obs_dim=canonical_dim_spec(NUM_OBJECTS),
        pred_horizon=16,
        obs_horizon=OBS_HORIZON,
    )
    kw.update(overrides)
    return kw


def _policy(**overrides):
    """A BesoPolicy with its tokenizer built, but no decoder (the loop tests mock that)."""
    policy = BesoPolicy(**_basic_kwargs(**overrides))
    policy._instantiate_tokenizer()
    return policy


class TestBesoPolicyLogic:
    """Isolated unit tests for vanilla BesoPolicy behavior (``relative_goal=False``, a standalone
    goal token and a flat state token per timestep -- matching the upstream reference).

    Shared infra is covered in ``TestBaseDiffusionAgentLogic``; this suite only covers what is
    unique to BESO: the continuous-DDIM loop, goal handling, Karras scalings, CFG, and action
    history. ``relative_goal=True`` ("BESO++DeltaInput") and per-object tokens are repo-local
    additions -- see ``TestBesoPolicyObjectTokenLogic`` below for those.
    """

    # ------------------------------------------------------------------ #
    # _get_cond_dims
    # ------------------------------------------------------------------ #
    def test_get_cond_dims_no_goal(self):
        """Without goal-conditioning, cond_dims carries no "goal" key."""
        policy = _policy()
        assert policy._get_cond_dims() == {
            "obs": {"proprio": 18, "task": (1, policy.tokenizer.output_dim)}
        }

    def test_get_cond_dims_goal_matches_obs(self):
        """Vanilla BESO ("true BESO"): goal tokens share obs_emb, so both sides are alike."""
        policy = _policy(goal_horizon=1)
        cond_dims = policy._get_cond_dims()
        assert set(cond_dims) == {"obs", "goal"}
        assert cond_dims["goal"] == cond_dims["obs"]

    def test_get_cond_dims_requires_a_tokenizer(self):
        policy = BesoPolicy(**_basic_kwargs(tokenizer=None))
        policy._instantiate_tokenizer()
        with pytest.raises(ValueError, match="requires a tokenizer"):
            policy._get_cond_dims()

    # ------------------------------------------------------------------ #
    # output_clip_range (post-unnormalize physical-space clamping)
    # ------------------------------------------------------------------ #
    def test_output_clip_range_clamps_without_normalizer(self):
        policy = _policy()
        mock_loop_internals(policy)
        out = policy._run_diffusion_loop(
            external_cond=policy._build_external_cond(_canonical_tree(), None),
            num_inference_steps=2,
            output_clip_range=(3.0, 6.0),
        )
        assert out.min() >= 3.0
        assert out.max() <= 6.0

    def test_output_clip_range_none_no_clamp(self):
        policy = _policy()
        mock_loop_internals(policy)
        out = policy._run_diffusion_loop(
            external_cond=policy._build_external_cond(_canonical_tree(), None),
            num_inference_steps=2,
            output_clip_range=None,
        )
        # No clipping -> output is whatever the loop produces (finite, correct shape)
        assert out.shape == (1, 1, policy.act_dim)
        assert torch.isfinite(out).all()

    def test_output_clip_range_clamps_with_normalizer(self):
        """With an action normalizer, clamping happens in physical space (post-unnormalize)."""
        policy = _policy()
        policy.act_normalizer = MinMaxNormalizer(policy.act_dim)
        # Fit the MinMax normalizer to a known range so unnormalize is well-defined.
        policy.act_normalizer.fit(torch.linspace(-5, 5, 40).view(10, 4))
        mock_loop_internals(policy)
        out = policy._run_diffusion_loop(
            external_cond=policy._build_external_cond(_canonical_tree(), None),
            num_inference_steps=2,
            output_clip_range=(3.0, 6.0),
        )
        assert out.min() >= 3.0
        assert out.max() <= 6.0

    # ------------------------------------------------------------------ #
    # action_history + reset
    # ------------------------------------------------------------------ #
    def test_action_history_appended_after_loop(self):
        policy = _policy()
        mock_loop_internals(policy)
        assert len(policy.action_history) == 0
        policy._run_diffusion_loop(
            external_cond=policy._build_external_cond(_canonical_tree(), None),
            num_inference_steps=2,
        )
        assert len(policy.action_history) == 1

    def test_reset_clears_action_history(self):
        policy = _policy()
        mock_loop_internals(policy)
        policy._run_diffusion_loop(
            external_cond=policy._build_external_cond(_canonical_tree(), None),
            num_inference_steps=2,
        )
        assert len(policy.action_history) == 1
        policy.reset()
        assert len(policy.action_history) == 0
        assert isinstance(policy.action_history, deque)

    # ------------------------------------------------------------------ #
    # num_inference_steps required
    # ------------------------------------------------------------------ #
    def test_num_inference_steps_required(self):
        policy = _policy()
        mock_loop_internals(policy)
        with pytest.raises(ValueError, match="must be manually provided"):
            policy._run_diffusion_loop(
                external_cond=policy._build_external_cond(_canonical_tree(), None),
                num_inference_steps=None,
            )

    # ------------------------------------------------------------------ #
    # Karras scalings + sigmas (pure functions, no mocks)
    # ------------------------------------------------------------------ #
    def test_karras_scalings_shapes(self):
        policy = _policy()
        sigma = torch.tensor([0.5, 1.0, 2.0]).view(3, 1, 1)
        c_skip, c_out, c_in = policy._get_karras_scalings(sigma)
        assert c_skip.shape == (3, 1, 1)
        assert c_out.shape == (3, 1, 1)
        assert c_in.shape == (3, 1, 1)
        assert torch.isfinite(c_skip).all()
        assert torch.isfinite(c_out).all()
        assert torch.isfinite(c_in).all()

    def test_sigmas_exponential_shape_and_trailing_zero(self):
        policy = _policy()
        sigmas = policy._get_sigmas_exponential(5, 0.005, 1.0)
        assert sigmas.shape == (6,)  # n + trailing 0
        assert sigmas[-1].item() == 0.0
        # Monotonically decreasing (except the appended zero)
        assert torch.all(sigmas[:-1] >= sigmas[1:])

    def test_t_fn_sigma_fn_are_inverses(self):
        policy = _policy()
        sigma = torch.tensor([0.005, 0.1, 0.5, 1.0])
        # _sigma_fn(_t_fn(sigma)) == sigma
        round_trip = policy._sigma_fn(policy._t_fn(sigma))
        assert torch.allclose(round_trip, sigma)

    # ------------------------------------------------------------------ #
    # Goal dropout (CFG training)
    # ------------------------------------------------------------------ #
    def test_goal_dropout_zeros_goal_when_training(self):
        """The goal is zeroed after normalization, as the reference zeroes its scaled goal."""
        policy = _policy(goal_drop_prob=1.0, goal_horizon=1)
        policy.train()
        policy.decoder = MagicMock(return_value=torch.zeros(1, 16, 4))

        external_cond = policy._build_external_cond(
            _canonical_tree(), _canonical_tree(time_axis=False)
        )
        policy._compute_loss(external_cond, torch.randn(1, 16, 4))

        goal_out = policy.decoder.call_args.kwargs["external_cond"]["goal"]
        assert torch.all(goal_out["proprio"] == 0.0)
        assert torch.all(goal_out["task"] == 0.0)

    # ------------------------------------------------------------------ #
    # CFG inference
    # ------------------------------------------------------------------ #
    def test_cfg_inference_two_network_calls(self):
        policy = _policy(cfg_lambda=1.0, goal_horizon=1)
        mock_loop_internals(policy)
        policy._run_diffusion_loop(
            external_cond=policy._build_external_cond(
                _canonical_tree(), _canonical_tree(time_axis=False)
            ),
            num_inference_steps=2,
        )
        # cond + uncond = 2 network calls per iteration.
        assert policy.decoder.call_count == 2

    def test_no_cfg_single_network_call(self):
        policy = _policy(cfg_lambda=None, goal_horizon=1)
        mock_loop_internals(policy)
        policy._run_diffusion_loop(
            external_cond=policy._build_external_cond(
                _canonical_tree(), _canonical_tree(time_axis=False)
            ),
            num_inference_steps=2,
        )
        assert policy.decoder.call_count == 1

    def test_conditioning_is_encoded_once_per_loop(self):
        """Obs/goal don't change across denoising steps, so re-encoding them would be waste."""
        policy = _policy(goal_horizon=1)
        mock_loop_internals(policy)
        external_cond = policy._build_external_cond(
            _canonical_tree(), _canonical_tree(time_axis=False)
        )
        policy._encode = MagicMock(wraps=policy._encode)

        policy._run_diffusion_loop(external_cond=external_cond, num_inference_steps=4)

        assert policy._encode.call_count == 1

    # ------------------------------------------------------------------ #
    # configure_optimizers (real DiffusionGPT network, incl. nn.MultiheadAttention)
    # ------------------------------------------------------------------ #
    def test_configure_optimizers_categorizes_every_parameter(self):
        """Regression guard: nn.MultiheadAttention's packed in-projection must land in a
        decay/no_decay group instead of silently tripping the "some parameters were not
        categorized" assertion (it isn't an nn.Linear, so it needs its own whitelist entry)."""
        policy = BesoPolicy(
            **_basic_kwargs(
                obs_horizon=2,
                pred_horizon=2,
                act_horizon=1,
                decoder={
                    "_target_": DIFFUSION_GPT,
                    "act_dim": 4,
                    "obs_horizon": 2,
                    "pred_horizon": 2,
                    "embed_dim": 8,
                    "n_layers": 1,
                    "n_heads": 2,
                },
                optimizer={"_target_": "torch.optim.AdamW", "_partial_": True, "weight_decay": 1e-4},
            )
        )
        policy.configure_model()

        result = policy.configure_optimizers()
        optimizer = result["optimizer"] if isinstance(result, dict) else result
        categorized = sum(len(g["params"]) for g in optimizer.param_groups)
        assert categorized == len(list(policy.decoder.parameters()))

    # ------------------------------------------------------------------ #
    # num_parallel_samples
    # ------------------------------------------------------------------ #
    def test_num_parallel_samples_averaged(self):
        policy = _policy(num_parallel_samples=2)
        mock_loop_internals(policy, decoder_return=torch.ones((2, 1, 4)))
        out = policy._run_diffusion_loop(
            external_cond=policy._build_external_cond(_canonical_tree(), None),
            num_inference_steps=2,
        )
        # B=1 averaged over 2 parallel samples -> (1, 1, act_dim)
        assert out.shape == (1, 1, policy.act_dim)
        assert torch.isfinite(out).all()

    # ------------------------------------------------------------------ #
    # Goal presence validation
    # ------------------------------------------------------------------ #
    def test_requires_goal_when_goal_conditioned(self):
        policy = _policy(goal_horizon=1)
        with pytest.raises(ValueError, match="but received goal=None"):
            policy._build_external_cond(_canonical_tree(), None)

    def test_rejects_goal_when_not_goal_conditioned(self):
        policy = _policy()
        with pytest.raises(ValueError, match="not goal-conditioned"):
            policy._build_external_cond(_canonical_tree(), _canonical_tree(time_axis=False))


def _delta_kwargs(**overrides):
    """Kwargs for ``BesoPolicy(relative_goal=True)`` ("BESO++DeltaInput").

    ``goal_horizon=1``/``goal_drop_prob=0.0``/``cfg_lambda=None`` are baked in here since every
    valid construction requires them -- ``relative_goal=True`` always folds the goal into a
    delta, which needs exactly one goal frame and has no meaning under classifier-free guidance.
    """
    kw = _basic_kwargs(
        decoder={"_target_": OBJECT_DIFFUSION_GPT},
        goal_horizon=1,
        relative_goal=True,
        goal_drop_prob=0.0,
        cfg_lambda=None,
    )
    kw.update(overrides)
    return kw


class TestBesoPolicyObjectTokenLogic:
    """``BesoPolicy`` driving ``ObjectDiffusionGPT``: proprioception gets its own network token,
    and (with ``tokenizer: object``) so does every scene object.

    See ``TestBesoPolicyLogic`` above for the diffusion-loop, CFG and Karras behaviour shared by
    every configuration.
    """

    # ------------------------------------------------------------------ #
    # proprio_dim / task_dim resolution
    #
    # These are lazily resolved BaseDiffusionAgent properties, so a misconfigured obs_dim raises
    # on first use rather than at construction; the validation itself belongs to (and is covered
    # with) `resolve_proprio_dim`/`derive_task_dim` in tests/algorithms/networks/test_utils.py.
    # ------------------------------------------------------------------ #
    def test_derives_proprio_dim_from_dict_obs(self):
        """proprio_dim/task_dim are auto-derived from obs_dim['proprio'] when omitted."""
        policy = BesoPolicy(**_delta_kwargs(obs_dim={"proprio": 2, "task": 1}))
        assert policy.proprio_dim == 2
        assert policy.task_dim == 1

    def test_explicit_task_dim_is_honoured(self):
        policy = BesoPolicy(**_delta_kwargs(obs_dim=3, proprio_dim=1, task_dim=2))
        assert policy.task_dim == 2

    # ------------------------------------------------------------------ #
    # Fixed-configuration requirements (goal_horizon=1, no CFG)
    # ------------------------------------------------------------------ #
    def test_requires_goal_horizon_one(self):
        with pytest.raises(ValueError, match="requires goal_horizon=1"):
            BesoPolicy(**_delta_kwargs(goal_horizon=2))

    def test_rejects_goal_drop_prob(self):
        with pytest.raises(ValueError, match="mutually exclusive with classifier-free guidance"):
            BesoPolicy(**_delta_kwargs(goal_drop_prob=0.1))

    def test_rejects_cfg_lambda(self):
        with pytest.raises(ValueError, match="mutually exclusive with classifier-free guidance"):
            BesoPolicy(**_delta_kwargs(cfg_lambda=1.25))

    # ------------------------------------------------------------------ #
    # Network wiring (no standalone goal token)
    # ------------------------------------------------------------------ #
    def test_cond_dims_omits_goal_key(self):
        policy = BesoPolicy(**_delta_kwargs())
        policy._instantiate_tokenizer()
        assert set(policy._get_cond_dims()) == {"obs"}

    def test_decoder_goal_horizon_forced_to_zero(self):
        policy = BesoPolicy(**_delta_kwargs())
        policy._instantiate_tokenizer()
        # Extends the base default ({"cond_dims": ...}) rather than replacing it.
        assert policy._decoder_extra_kwargs() == {
            "cond_dims": policy._get_cond_dims(),
            "goal_horizon": 0,
        }

    @pytest.mark.parametrize(
        ("tokenizer", "tokens_per_step"), [(STATE_TOKENIZER, 1), (OBJECT_TOKENIZER, 3)]
    )
    def test_cond_dims_carry_the_tokenizers_slot_count(self, tokenizer, tokens_per_step):
        policy = BesoPolicy(**_delta_kwargs(tokenizer={"_target_": tokenizer}))
        policy._instantiate_tokenizer()
        assert policy._get_cond_dims()["obs"]["task"] == (
            tokens_per_step,
            policy.tokenizer.output_dim,
        )

    # ------------------------------------------------------------------ #
    # Delta conditioning (g - s_t)
    # ------------------------------------------------------------------ #
    def test_encode_folds_the_goal_into_the_obs_tokens(self):
        """No standalone goal entry survives encoding: it lives on inside the task tokens."""
        policy = BesoPolicy(**_delta_kwargs())
        policy._instantiate_tokenizer()

        obs = _canonical_tree()
        encoded = policy._encode(
            policy._build_external_cond(obs, _canonical_tree(time_axis=False))
        )

        assert set(encoded) == {"obs"}
        assert set(encoded["obs"]) == {"proprio", "task"}
        assert torch.equal(encoded["obs"]["proprio"], obs["proprio"])


class TestBesoPolicyOnCanonicalObservations:
    """End-to-end coverage on the real canonical schema, whose ``obj_pose`` leaf carries a slot
    axis (``[B, T, K, 7]``).

    Every variant used to die inside the decoder's ``concat_leaf_tensors`` on that 4D leaf; the
    tokenizer is what flattens it, so these run the whole path rather than a synthetic flat obs.
    """

    CASES = [
        pytest.param(DIFFUSION_GPT, STATE_TOKENIZER, False, id="vanilla-beso"),
        pytest.param(OBJECT_DIFFUSION_GPT, STATE_TOKENIZER, True, id="beso++-delta"),
        pytest.param(OBJECT_DIFFUSION_GPT, OBJECT_TOKENIZER, True, id="beso++-object-delta"),
        pytest.param(OBJECT_DIFFUSION_GPT, OBJECT_TOKENIZER, False, id="beso++-object-absolute"),
    ]

    def _policy(self, decoder, tokenizer, relative_goal):
        horizon = 2
        policy = BesoPolicy(
            **_basic_kwargs(
                decoder={
                    "_target_": decoder,
                    "act_dim": 4,
                    "obs_horizon": horizon,
                    "pred_horizon": horizon,
                    "embed_dim": 8,
                    "n_layers": 1,
                    "n_heads": 2,
                },
                tokenizer={"_target_": tokenizer},
                ema={"_target_": "diffusers.training_utils.EMAModel"},
                obs_horizon=horizon,
                pred_horizon=horizon,
                act_horizon=1,
                goal_horizon=1,
                relative_goal=relative_goal,
                goal_drop_prob=0.0 if relative_goal else 0.1,
                cfg_lambda=None if relative_goal else 1.25,
                obs_normalizer=True,
                act_normalizer=True,
            )
        )
        policy.configure_model()
        policy.obs_normalizer.fit(
            policy._tokenize(_canonical_tree(batch_size=4, horizon=horizon),
                             _canonical_tree(batch_size=4, time_axis=False))["obs"]
        )
        policy.act_normalizer.fit(torch.randn(4, horizon, policy.act_dim))
        return policy, horizon

    @pytest.mark.parametrize(("decoder", "tokenizer", "relative_goal"), CASES)
    def test_training_step_runs_and_backpropagates(self, decoder, tokenizer, relative_goal):
        policy, horizon = self._policy(decoder, tokenizer, relative_goal)
        policy.train()

        loss = policy._compute_loss(
            policy._build_external_cond(
                _canonical_tree(batch_size=2, horizon=horizon),
                _canonical_tree(batch_size=2, time_axis=False),
            ),
            torch.randn(2, horizon, 4),
        )
        assert torch.isfinite(loss)

        loss.backward()
        assert any(p.grad is not None for p in policy.decoder.parameters())

    @pytest.mark.parametrize(("decoder", "tokenizer", "relative_goal"), CASES)
    def test_get_action_runs(self, decoder, tokenizer, relative_goal):
        policy, horizon = self._policy(decoder, tokenizer, relative_goal)
        policy.eval()

        action = policy.get_action(
            _canonical_tree(batch_size=2, horizon=horizon),
            _canonical_tree(batch_size=2, time_axis=False),
            num_inference_steps=2,
        )
        assert action.shape == (2, 1, policy.act_dim)
        assert torch.isfinite(action).all()
