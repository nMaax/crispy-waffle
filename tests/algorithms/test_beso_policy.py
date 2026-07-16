import warnings
from unittest.mock import MagicMock, patch

import pytest
import torch

from policy.algorithms.beso_policy import BesoPolicy
from policy.transforms import MinMaxNormalizer, ZScoreNormalizer


class TestBesoPolicy:
    @pytest.fixture
    def basic_kwargs(self):
        return {
            "network": {"_target_": "policy.algorithms.networks.diffusion_gpt.DiffusionGPT"},
            "ema": {},
            "optimizer": {},
            "act_dim": 4,
            "obs_dim": 3,
            "pred_horizon": 16,
            "obs_horizon": 2,
        }

    @patch("policy.algorithms.diffusion_policy.hydra_zen.instantiate")
    def test_beso_policy_warning_on_zscore(self, mock_instantiate, basic_kwargs):
        # Setup mock normalizer to return a ZScoreNormalizer
        def mock_inst(config, *args, **kwargs):
            if config is None:
                return None
            spec = kwargs.get("spec")
            return ZScoreNormalizer(spec)

        mock_instantiate.side_effect = mock_inst

        policy = BesoPolicy(
            **basic_kwargs,
            action_normalizer=True,  # This triggers ZScoreNormalizer creation in DiffusionPolicy
        )
        policy.network = MagicMock()
        policy.network.return_value = torch.ones((1, 1, 4))
        policy.ema = MagicMock()
        policy._get_karras_scalings = MagicMock(
            return_value=(torch.tensor(0.0), torch.tensor(1.0), torch.tensor(1.0))
        )
        policy._get_sigmas_exponential = MagicMock(return_value=torch.tensor([1.0, 0.0]))
        policy._t_fn = MagicMock(return_value=torch.tensor(0.5))
        policy._sigma_fn = MagicMock(return_value=torch.tensor(0.5))

        obs_seq = torch.randn(1, 2, 3)

        # 1. Test clip_denoised=1.0 with zscore normalizer: should trigger a warning
        with pytest.warns(
            UserWarning, match="In-loop clipping is enabled.*with a ZScoreNormalizer"
        ):
            policy.get_action(
                obs_seq=obs_seq,
                num_inference_timesteps=2,
                clip_denoised=1.0,
            )

        # 2. Test clip_denoised=None with zscore normalizer: should NOT trigger a warning
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            policy.get_action(
                obs_seq=obs_seq,
                num_inference_timesteps=2,
                clip_denoised=None,
            )

        # 3. Test duplicate calls with clip_denoised=1.0 do not warn again
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            policy.get_action(
                obs_seq=obs_seq,
                num_inference_timesteps=2,
                clip_denoised=1.0,
            )

    @patch("policy.algorithms.diffusion_policy.hydra_zen.instantiate")
    def test_beso_policy_no_warning_on_minmax(self, mock_instantiate, basic_kwargs):
        # Setup mock normalizer to return a MinMaxNormalizer
        def mock_inst(config, *args, **kwargs):
            if config is None:
                return None
            spec = kwargs.get("spec")
            return MinMaxNormalizer(spec)

        mock_instantiate.side_effect = mock_inst

        policy = BesoPolicy(
            **basic_kwargs,
            action_normalizer={"_target_": "policy.transforms.MinMaxNormalizer"},
        )
        policy.network = MagicMock()
        policy.network.return_value = torch.ones((1, 1, 4))
        policy.ema = MagicMock()
        policy._get_karras_scalings = MagicMock(
            return_value=(torch.tensor(0.0), torch.tensor(1.0), torch.tensor(1.0))
        )
        policy._get_sigmas_exponential = MagicMock(return_value=torch.tensor([1.0, 0.0]))
        policy._t_fn = MagicMock(return_value=torch.tensor(0.5))
        policy._sigma_fn = MagicMock(return_value=torch.tensor(0.5))

        obs_seq = torch.randn(1, 2, 3)

        # Test clip_denoised=1.0 with minmax normalizer: should NOT trigger a warning
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            policy.get_action(
                obs_seq=obs_seq,
                num_inference_timesteps=2,
                clip_denoised=1.0,
            )

    @patch("policy.algorithms.diffusion_policy.hydra_zen.instantiate")
    def test_beso_in_loop_clamping(self, mock_instantiate, basic_kwargs):
        mock_instantiate.return_value = MagicMock()

        policy = BesoPolicy(
            **basic_kwargs,
        )

        # Mock network and ema to allow _run_diffusion_loop to run
        policy.network = MagicMock()
        policy.ema = MagicMock()

        # We need model_pred to return a tensor that would exceed the clamping bounds
        # shape of model_pred: [B, pred_horizon, act_dim] or similar.
        # Let's say we have B_expanded = 1.
        # model_pred will return a tensor of all 2.0.
        policy.network.return_value = torch.ones((1, 1, 4)) * 2.0

        # We also need to mock or set relevant attributes for Karras scalings
        # Let's mock _get_karras_scalings to return 0.0, 1.0, 1.0.
        # This way, scaled_pred = current_pred * 1.0 + current_noisy * 0.0 = current_pred.
        # So scaled_pred before clamping would be 2.0.
        policy._get_karras_scalings = MagicMock(
            return_value=(torch.tensor(0.0), torch.tensor(1.0), torch.tensor(1.0))
        )

        # We need to mock _t_fn, _sigma_fn, _get_sigmas_exponential.
        # Let's make _get_sigmas_exponential return a tensor with 2 steps: [1.0, 0.0].
        policy._get_sigmas_exponential = MagicMock(return_value=torch.tensor([1.0, 0.0]))
        policy._t_fn = MagicMock(return_value=torch.tensor(0.5))
        policy._sigma_fn = MagicMock(return_value=torch.tensor(0.5))

        # Now let's call _run_diffusion_loop!
        network_cond = torch.zeros((1, 2, 3))  # B=1, obs_horizon=2, obs_dim=3

        # We want to check that scaled_pred inside the loop is clamped.
        # Let's patch torch.clamp in policy.algorithms.beso_policy to see if it clamps to -0.5, 0.5.
        with patch("policy.algorithms.beso_policy.torch.clamp", wraps=torch.clamp) as mock_clamp:
            policy._run_diffusion_loop(
                network_cond=network_cond,
                num_inference_timesteps=2,
                clip_denoised=0.5,
            )
            # Find calls to torch.clamp where the min/max are -0.5 and 0.5
            clamp_calls = [
                call
                for call in mock_clamp.call_args_list
                if len(call[0]) >= 3 and call[0][1] == -0.5 and call[0][2] == 0.5
            ]
            assert len(clamp_calls) > 0, (
                "torch.clamp was not called with the clipping limits inside the loop"
            )
