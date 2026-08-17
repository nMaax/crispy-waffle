from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf

from policy.algorithms.base_diffusion_agent import BaseDiffusionAgent
from policy.transforms import MinMaxNormalizer, ZScoreNormalizer
from policy.utils import get_batch_size
from policy.utils.typing_utils import TensorTree

DECODER_TARGET = "policy.algorithms.networks.decoder.unet1d.FiLMDecoder1D"


class _MinimalDiffusionAgent(BaseDiffusionAgent):
    """Trivial concrete double so the shared BaseDiffusionAgent infra can be exercised without
    committing to DP's diffusers scheduler or BESO's Karras math."""

    def _compute_loss(self, external_cond: TensorTree, act_seq: torch.Tensor) -> torch.Tensor:
        return external_cond["obs"].sum() * 0.0

    def _run_diffusion_loop(
        self,
        external_cond: TensorTree,
        num_inference_steps: int | None = None,
        output_clip_range: tuple | None = None,
    ):
        B = get_batch_size(external_cond)
        return torch.zeros((B, self.act_horizon, self.act_dim), device=self.device)


def _basic_kwargs(**overrides):
    kw = dict(
        decoder={"_target_": DECODER_TARGET},
        optimizer={},
        obs_dim=3,
        act_dim=4,
        pred_horizon=16,
        obs_horizon=2,
        act_horizon=8,
    )
    kw.update(overrides)
    return kw


class TestBaseDiffusionAgentLogic:
    """Shared infra tested once on the base via a minimal concrete stub.

    DP- and BESO-specific logic suites only need to cover their unique math.
    """

    # ------------------------------------------------------------------ #
    # Horizon validation
    # ------------------------------------------------------------------ #
    def test_horizon_act_gt_pred_raises(self):
        with pytest.raises(ValueError, match="cannot be greater than"):
            _MinimalDiffusionAgent(**_basic_kwargs(act_horizon=20))

    def test_horizon_window_too_long_raises(self):
        with pytest.raises(ValueError, match="is too short"):
            _MinimalDiffusionAgent(**_basic_kwargs(obs_horizon=4, pred_horizon=8, act_horizon=6))

    # ------------------------------------------------------------------ #
    # Normalizer construction (_instantiate_normalizer), run by configure_model()
    # ------------------------------------------------------------------ #
    def test_normalizer_bare_true_uses_the_default_class(self):
        agent = _MinimalDiffusionAgent(**_basic_kwargs())
        assert isinstance(
            agent._instantiate_normalizer(config=True, spec=3, default_cls=ZScoreNormalizer),
            ZScoreNormalizer,
        )
        assert isinstance(
            agent._instantiate_normalizer(config=True, spec=4, default_cls=MinMaxNormalizer),
            MinMaxNormalizer,
        )

    def test_normalizer_none_yields_no_normalizer(self):
        agent = _MinimalDiffusionAgent(**_basic_kwargs())
        assert agent._instantiate_normalizer(None, 3, ZScoreNormalizer) is None
        assert agent._instantiate_normalizer(False, 3, ZScoreNormalizer) is None

    def test_normalizer_dict_target_instantiated(self):
        agent = _MinimalDiffusionAgent(**_basic_kwargs())
        normalizer = agent._instantiate_normalizer(
            config={"_target_": "policy.transforms.ZScoreNormalizer"},
            spec=3,
            default_cls=MinMaxNormalizer,
        )
        assert isinstance(normalizer, ZScoreNormalizer)

    def test_normalizer_dict_target_instantiated_from_hydra_dictconfig(self):
        """A `DictConfig` reaches `_instantiate_normalizer` under `_recursive_: false`, not a plain
        dict."""
        agent = _MinimalDiffusionAgent(**_basic_kwargs())
        normalizer = agent._instantiate_normalizer(
            config=OmegaConf.create({"_target_": "policy.transforms.ZScoreNormalizer"}),
            spec=3,
            default_cls=MinMaxNormalizer,
        )
        assert isinstance(normalizer, ZScoreNormalizer)

    def test_normalizer_mask_is_threaded_to_the_normalizer(self):
        agent = _MinimalDiffusionAgent(**_basic_kwargs())
        mask = torch.tensor([True, False, True])
        normalizer = agent._instantiate_normalizer(
            config=True, spec=3, default_cls=ZScoreNormalizer, mask=mask
        )
        assert torch.equal(normalizer.mask, mask)

    def test_normalizer_not_built_before_configure_model(self):
        """The obs normalizer is sized from the tokenizer, so it cannot exist at __init__ time."""
        agent = _MinimalDiffusionAgent(**_basic_kwargs(obs_normalizer=True))
        assert agent.obs_normalizer is None

    # ------------------------------------------------------------------ #
    # _get_cond_dims / _build_external_cond / Normalization helpers
    # ------------------------------------------------------------------ #
    def test_get_cond_dims_wraps_obs_dim(self):
        agent = _MinimalDiffusionAgent(**_basic_kwargs())
        assert agent._get_cond_dims() == {"obs": agent.obs_dim}

    def test_build_external_cond_wraps_obs_unflattened(self):
        agent = _MinimalDiffusionAgent(**_basic_kwargs())
        obs = torch.randn(2, 2, 3)
        external_cond = agent._build_external_cond(obs)
        assert external_cond == {"obs": obs}

    def test_normalize_helpers_delegate_when_normalizers_present(self):
        agent = _MinimalDiffusionAgent(**_basic_kwargs())
        agent.obs_normalizer = MagicMock()
        agent.obs_normalizer.normalize.side_effect = lambda x: x + 1.0
        agent.act_normalizer = MagicMock()
        agent.act_normalizer.normalize.side_effect = lambda x: x * 2.0

        obs = torch.tensor([1.0, 2.0])
        act = torch.tensor([3.0, 4.0])
        assert torch.equal(agent._normalize_obs(obs), torch.tensor([2.0, 3.0]))
        assert torch.equal(agent._normalize_act(act), torch.tensor([6.0, 8.0]))

        # _build_external_cond hands the raw tree on; normalization happens later, on the tokens.
        assert torch.equal(agent._build_external_cond(obs)["obs"], obs)

    # ------------------------------------------------------------------ #
    # EMA optionality
    # ------------------------------------------------------------------ #
    def test_base_constructs_without_ema(self):
        agent = _MinimalDiffusionAgent(**_basic_kwargs())
        assert agent.ema_config is None
        assert agent.ema is None

    def test_on_train_batch_end_skips_when_ema_none(self):
        agent = _MinimalDiffusionAgent(**_basic_kwargs())
        agent.decoder = MagicMock()
        agent.ema = None
        # Should not raise even though EMA is absent.
        agent.on_train_batch_end(torch.tensor(0.0), {}, 0)

    def test_on_train_batch_end_raises_when_decoder_none(self):
        agent = _MinimalDiffusionAgent(**_basic_kwargs())
        with pytest.raises(ValueError, match="Decoder not initialized"):
            agent.on_train_batch_end(torch.tensor(0.0), {}, 0)

    def test_on_train_batch_end_steps_ema_when_present(self):
        agent = _MinimalDiffusionAgent(**_basic_kwargs())
        agent.decoder = MagicMock()
        agent.ema = MagicMock()
        agent.on_train_batch_end(torch.tensor(0.0), {}, 0)
        agent.ema.step.assert_called_once()

    # ------------------------------------------------------------------ #
    # Obs-only template methods
    # ------------------------------------------------------------------ #
    def test_shared_step_template(self):
        agent = _MinimalDiffusionAgent(**_basic_kwargs())
        agent.log = MagicMock()
        batch = {"obs_seq": torch.randn(2, 2, 3), "act_seq": torch.randn(2, 16, 4)}
        loss = agent._shared_step(batch, 0, "train")
        assert torch.isfinite(loss)
        agent.log.assert_called_once()
        assert agent.log.call_args[0][0] == "train/loss"

    def test_get_action_template(self):
        agent = _MinimalDiffusionAgent(**_basic_kwargs())
        obs_seq = torch.randn(2, 2, 3)
        out = agent.get_action(obs_seq)
        assert out.shape == (2, agent.act_horizon, agent.act_dim)
        assert torch.isfinite(out).all()

    # ------------------------------------------------------------------ #
    # Encoder scaffolding: _encode / _ema_parameters / _get_cond_dims
    # ------------------------------------------------------------------ #
    def test_encode_passes_through_unchanged_when_no_encoder(self):
        agent = _MinimalDiffusionAgent(**_basic_kwargs())
        assert agent.encoder is None
        external_cond = {"obs": torch.randn(2, 2, 3)}
        assert agent._encode(external_cond) is external_cond

    def test_encode_requires_an_obs_entry_when_encoder_present(self):
        agent = _MinimalDiffusionAgent(**_basic_kwargs())
        agent.encoder = MagicMock()
        with pytest.raises(ValueError, match="must contain an 'obs' entry"):
            agent._encode({})

    def test_encode_hands_normalized_tokens_to_the_encoder(self):
        agent = _MinimalDiffusionAgent(**_basic_kwargs())
        agent.encoder = MagicMock()
        tokens = {"proprio": torch.randn(2, 2, 3), "task": torch.randn(2, 2, 5)}
        agent._tokenize = MagicMock(return_value=tokens)
        agent.obs_normalizer = MagicMock()
        agent.obs_normalizer.normalize.side_effect = lambda x: x

        obs = torch.randn(2, 2, 3)
        agent._encode({"obs": obs})

        agent._tokenize.assert_called_once_with(obs=obs)
        agent.obs_normalizer.normalize.assert_called_once_with(tokens)
        agent.encoder.assert_called_once_with(tokens)

    def test_get_cond_dims_uses_encoder_cond_dims_when_present(self):
        agent = _MinimalDiffusionAgent(**_basic_kwargs())
        agent.encoder = MagicMock()
        agent.encoder.cond_dims = {"obs": 99}
        assert agent._get_cond_dims() == {"obs": 99}

    def test_ema_parameters_combines_encoder_and_decoder_when_encoder_present(self):
        agent = _MinimalDiffusionAgent(**_basic_kwargs())
        agent.decoder = torch.nn.Linear(3, 3)
        agent.encoder = torch.nn.Linear(2, 2)
        ema_params = agent._ema_parameters()
        expected = list(agent.encoder.parameters()) + list(agent.decoder.parameters())
        assert ema_params == expected

    def test_ema_parameters_is_decoder_only_when_no_encoder(self):
        agent = _MinimalDiffusionAgent(**_basic_kwargs())
        agent.decoder = torch.nn.Linear(3, 3)
        assert agent.encoder is None
        assert agent._ema_parameters() == list(agent.decoder.parameters())


class TestGoalConditionedDiffusionPolicyLogic:
    def test_init_with_goal_horizon_zero_raises(self):
        from policy.algorithms.goal_conditioned_diffusion_policy import (
            GoalConditionedDiffusionPolicy,
        )

        with pytest.raises(ValueError, match="requires goal_horizon >= 1"):
            GoalConditionedDiffusionPolicy(
                decoder={"_target_": DECODER_TARGET},
                optimizer={},
                ema={"_target_": "diffusers.training_utils.EMAModel"},
                noise_scheduler={"_target_": "diffusers.schedulers.scheduling_ddpm.DDPMScheduler"},
                goal_horizon=0,
            )

    def test_encoder_extra_kwargs_threads_goal_conditioned_true(self):
        from policy.algorithms.goal_conditioned_diffusion_policy import (
            GoalConditionedDiffusionPolicy,
        )

        policy = GoalConditionedDiffusionPolicy(
            decoder={"_target_": DECODER_TARGET},
            optimizer={},
            ema={"_target_": "diffusers.training_utils.EMAModel"},
            noise_scheduler={"_target_": "diffusers.schedulers.scheduling_ddpm.DDPMScheduler"},
            goal_horizon=1,
            obs_dim=48,
            proprio_dim=18,
            tokenizer={
                "_target_": "policy.algorithms.tokenizers.state.StateTokenizer"
            },
        )
        policy._instantiate_tokenizer()
        kwargs = policy._encoder_extra_kwargs()
        assert kwargs["goal_conditioned"] is True
        assert kwargs["proprio_dim"] == 18
        assert kwargs["token_dim"] == policy.tokenizer.output_dim == 30

    def test_build_external_cond_from_batch_missing_goal_raises(self):
        from policy.algorithms.goal_conditioned_diffusion_policy import (
            GoalConditionedDiffusionPolicy,
        )

        policy = GoalConditionedDiffusionPolicy(
            decoder={"_target_": DECODER_TARGET},
            optimizer={},
            ema={"_target_": "diffusers.training_utils.EMAModel"},
            noise_scheduler={"_target_": "diffusers.schedulers.scheduling_ddpm.DDPMScheduler"},
            goal_horizon=1,
        )
        with pytest.raises(ValueError, match="Expected batch\\['goal'\\]"):
            policy._build_external_cond_from_batch({"obs_seq": torch.randn(2, 2, 3)})

    def test_build_external_cond_packs_obs_and_goal(self):
        from policy.algorithms.goal_conditioned_diffusion_policy import (
            GoalConditionedDiffusionPolicy,
        )

        policy = GoalConditionedDiffusionPolicy(
            decoder={"_target_": DECODER_TARGET},
            optimizer={},
            ema={"_target_": "diffusers.training_utils.EMAModel"},
            noise_scheduler={"_target_": "diffusers.schedulers.scheduling_ddpm.DDPMScheduler"},
            goal_horizon=1,
        )
        obs = torch.randn(2, 2, 3)
        goal = torch.randn(2, 3)
        assert policy._build_external_cond(obs, goal) == {"obs": obs, "goal": goal}

    def test_shared_step_missing_goal_raises(self):
        from policy.algorithms.goal_conditioned_diffusion_policy import (
            GoalConditionedDiffusionPolicy,
        )

        policy = GoalConditionedDiffusionPolicy(
            decoder={"_target_": DECODER_TARGET},
            optimizer={},
            ema={"_target_": "diffusers.training_utils.EMAModel"},
            noise_scheduler={"_target_": "diffusers.schedulers.scheduling_ddpm.DDPMScheduler"},
            goal_horizon=1,
        )
        batch = {"obs_seq": torch.randn(2, 2, 3), "act_seq": torch.randn(2, 16, 4)}
        with pytest.raises(ValueError, match="Expected batch\\['goal'\\]"):
            policy._shared_step(batch, 0, "train")
