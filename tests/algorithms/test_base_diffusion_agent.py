from unittest.mock import MagicMock

import pytest
import torch

from policy.algorithms.base_diffusion_agent import BaseDiffusionAgent
from policy.transforms import MinMaxNormalizer, ZScoreNormalizer
from policy.utils import get_batch_size, get_total_dim

UNET_TARGET = "policy.algorithms.networks.unet1d.UNet1D"
GPT_TARGET = "policy.algorithms.networks.diffusion_gpt.DiffusionGPT"


class _MinimalDiffusionAgent(BaseDiffusionAgent):
    """Trivial concrete double so the shared BaseDiffusionAgent infra can be exercised without
    committing to DP's diffusers scheduler or BESO's Karras math."""

    def _compute_loss(self, obs_seq: torch.Tensor, act_seq: torch.Tensor) -> torch.Tensor:
        return obs_seq.sum() * 0.0

    def _run_diffusion_loop(
        self,
        obs_cond: torch.Tensor,
        goal_cond: torch.Tensor | None = None,
        num_inference_steps: int | None = None,
        output_clip_range: tuple | None = None,
    ):
        B = get_batch_size(obs_cond)
        return torch.zeros((B, self.act_horizon, self.act_dim), device=self.device)


def _basic_kwargs(network_target: str = UNET_TARGET, **overrides):
    kw = dict(
        network={"_target_": network_target},
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

    def test_horizon_valid_constructs(self):
        agent = _MinimalDiffusionAgent(**_basic_kwargs())
        assert agent.act_horizon == 8

    # ------------------------------------------------------------------ #
    # Normalizer building (_build_normalizer)
    # ------------------------------------------------------------------ #
    def test_normalizer_bare_true_obs_zscore_action_minmax(self):
        agent = _MinimalDiffusionAgent(**_basic_kwargs(obs_normalizer=True, act_normalizer=True))
        assert isinstance(agent.obs_normalizer, ZScoreNormalizer)
        assert isinstance(agent.act_normalizer, MinMaxNormalizer)

    def test_normalizer_none_default(self):
        agent = _MinimalDiffusionAgent(**_basic_kwargs())
        assert agent.obs_normalizer is None
        assert agent.act_normalizer is None

    def test_normalizer_dict_target_instantiated(self):
        agent = _MinimalDiffusionAgent(
            **_basic_kwargs(obs_normalizer={"_target_": "policy.transforms.ZScoreNormalizer"})
        )
        assert isinstance(agent.obs_normalizer, ZScoreNormalizer)

    # ------------------------------------------------------------------ #
    # flatten_obs auto-detection + network_cond_dim
    # ------------------------------------------------------------------ #
    def test_flatten_obs_auto_detect_unet(self):
        agent = _MinimalDiffusionAgent(**_basic_kwargs(network_target=UNET_TARGET))
        assert agent.flatten_obs is True
        assert agent.network_cond_dim == agent.obs_horizon * get_total_dim(agent.obs_dim)

    def test_flatten_obs_auto_detect_gpt(self):
        agent = _MinimalDiffusionAgent(**_basic_kwargs(network_target=GPT_TARGET))
        assert agent.flatten_obs is False
        assert agent.network_cond_dim == get_total_dim(agent.obs_dim)

    def test_flatten_obs_auto_detect_unknown_raises(self):
        with pytest.raises(ValueError, match="Cannot auto-detect"):
            _MinimalDiffusionAgent(
                **_basic_kwargs(network_target="policy.algorithms.networks.MLP")
            )

    def test_flatten_obs_explicit_respected(self):
        agent = _MinimalDiffusionAgent(**_basic_kwargs(flatten_obs=False))
        assert agent.flatten_obs is False

    # ------------------------------------------------------------------ #
    # EMA optionality
    # ------------------------------------------------------------------ #
    def test_base_constructs_without_ema(self):
        agent = _MinimalDiffusionAgent(**_basic_kwargs())
        assert agent.ema_config is None
        assert agent.ema is None

    def test_on_train_batch_end_skips_when_ema_none(self):
        agent = _MinimalDiffusionAgent(**_basic_kwargs())
        agent.network = MagicMock()
        agent.ema = None
        # Should not raise even though EMA is absent.
        agent.on_train_batch_end(torch.tensor(0.0), {}, 0)

    def test_on_train_batch_end_raises_when_network_none(self):
        agent = _MinimalDiffusionAgent(**_basic_kwargs())
        with pytest.raises(ValueError, match="Network not initialized"):
            agent.on_train_batch_end(torch.tensor(0.0), {}, 0)

    def test_on_train_batch_end_steps_ema_when_present(self):
        agent = _MinimalDiffusionAgent(**_basic_kwargs())
        agent.network = MagicMock()
        agent.ema = MagicMock()
        agent.on_train_batch_end(torch.tensor(0.0), {}, 0)
        agent.ema.step.assert_called_once()

    # ------------------------------------------------------------------ #
    # Abstract methods
    # ------------------------------------------------------------------ #
    def test_abstract_compute_loss_raises(self):
        with pytest.raises(NotImplementedError):
            BaseDiffusionAgent._compute_loss(None, torch.zeros(1), torch.zeros(1))

    def test_abstract_run_diffusion_loop_raises(self):
        with pytest.raises(NotImplementedError):
            BaseDiffusionAgent._run_diffusion_loop(None, torch.zeros(1))

    # ------------------------------------------------------------------ #
    # _prepare_obs
    # ------------------------------------------------------------------ #
    def test_prepare_network_cond_flatten(self):
        agent = _MinimalDiffusionAgent(**_basic_kwargs(flatten_obs=True))
        obs = {"a": torch.randn(2, 2, 3)}
        obs_cond = agent._prepare_obs(obs)
        assert obs_cond.shape == (2, 6)

    def test_prepare_network_cond_concat(self):
        agent = _MinimalDiffusionAgent(**_basic_kwargs(flatten_obs=False))
        obs = {"a": torch.randn(2, 2, 3)}
        obs_cond = agent._prepare_obs(obs)
        assert obs_cond.shape == (2, 2, 3)

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
        agent = _MinimalDiffusionAgent(**_basic_kwargs(flatten_obs=True))
        obs_seq = torch.randn(2, 2, 3)
        out = agent.get_action(obs_seq)
        assert out.shape == (2, agent.act_horizon, agent.act_dim)
        assert torch.isfinite(out).all()
