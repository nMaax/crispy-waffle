from unittest.mock import MagicMock

import pytest
import torch

from policy.algorithms.base_diffusion_agent import BaseDiffusionAgent
from policy.transforms import MinMaxNormalizer, ZScoreNormalizer
from policy.utils import get_batch_size
from policy.utils.typing_utils import TensorTree

UNET_TARGET = "policy.algorithms.networks.unet1d.UNet1D"


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
        network={"_target_": UNET_TARGET},
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
    # _get_cond_dims / _build_external_cond
    # ------------------------------------------------------------------ #
    def test_get_cond_dims_wraps_obs_dim(self):
        agent = _MinimalDiffusionAgent(**_basic_kwargs())
        assert agent._get_cond_dims() == {"obs": agent.obs_dim}

    def test_build_external_cond_wraps_obs_unflattened(self):
        agent = _MinimalDiffusionAgent(**_basic_kwargs())
        obs = torch.randn(2, 2, 3)
        external_cond = agent._build_external_cond(obs)
        assert external_cond == {"obs": obs}

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
            BaseDiffusionAgent._compute_loss(None, {"obs": torch.zeros(1)}, torch.zeros(1))

    def test_abstract_run_diffusion_loop_raises(self):
        with pytest.raises(NotImplementedError):
            BaseDiffusionAgent._run_diffusion_loop(None, {"obs": torch.zeros(1)})

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
