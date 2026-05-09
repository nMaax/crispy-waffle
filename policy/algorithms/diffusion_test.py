from unittest.mock import MagicMock, patch

import pytest
import torch

from policy.algorithms.diffusion import DiffusionPolicy
from policy.algorithms.lightning_module_tests import LightningModuleTests
from policy.utils import flatten_tensor_from_mapping
from policy.utils.utils import sum_shapes


@pytest.mark.parametrize("algorithm_config", ["diffusion"], indirect=True)
@pytest.mark.parametrize("datamodule_config", ["maniskill_datamodule"], indirect=True)
class TestDiffusionPolicy(LightningModuleTests[DiffusionPolicy]):
    """Test suite for the DiffusionPolicy.

    By inheriting from LightningModuleTests, this class automatically runs:
    - test_initialization_is_reproducible
    - test_forward_pass_is_reproducible (SKIPPED)
    - test_backward_pass_is_reproducible
    - test_update_is_reproducible
    """

    def test_forward_pass_is_reproducible(
        self,
        algorithm,
        training_step_content,
        tensor_regression,
    ):
        pytest.skip(
            "The built-in test would have required to define forward() in the model. "
            "However, diffusion do not use a standard single-step `forward` pass "
            "during the training step. Inference is handled iteratively via multiple calls (see `get_action`)."
            "Thus, we skip this test as it is not applicable to the DiffusionPolicy architecture."
        )

    def test_get_action_runs(
        self,
        algorithm,
        training_step_content,
    ):
        """Check that get_action produces a finite tensor of the expected shape and device on a
        sample batch."""
        # Prepare model for inference and sync devices
        algorithm.eval()
        batch_device = training_step_content.batch["act_seq"].device
        algorithm.to(batch_device)

        # Execute inference
        obs_seq = training_step_content.batch["obs_seq"]
        obs_seq = flatten_tensor_from_mapping(obs_seq, device=algorithm.device)
        with torch.no_grad():
            out = algorithm.get_action(obs_seq)

        # Assert instance, finiteness, device, and shape of the output
        assert isinstance(out, torch.Tensor)
        assert torch.isfinite(out).all(), "Output contains NaN or Inf"
        assert out.device == algorithm.device
        assert out.shape == (
            obs_seq.shape[0],
            algorithm.act_horizon,
            algorithm.act_dim,
        )

    def test_get_action_is_reproducible(
        self,
        algorithm,
        training_step_content,
        tensor_regression,
    ):
        """Check that get_action produces the same action tensor given a fixed random seed."""
        #  Prepare model for inference and sync devices
        algorithm.eval()
        batch_device = training_step_content.batch["act_seq"].device
        algorithm.to(batch_device)

        obs_seq = training_step_content.batch["obs_seq"]
        obs_seq = flatten_tensor_from_mapping(obs_seq, device=algorithm.device)
        with torch.no_grad():
            out = algorithm.get_action(obs_seq)

        # Regression check
        tensor_regression.check(
            {"action": out},
            default_tolerance={"rtol": 1e-5, "atol": 1e-6},
            include_gpu_name_in_stats=False,
        )


class TestDiffusionPolicyLogic:
    """Isolated unit tests for DiffusionPolicy boundary conditions, assertions, and logic.

    These run instantly using mocks, independent of the heavy regression suite.
    """

    @pytest.fixture(autouse=True)
    def patch_instantiate(self):
        """Intercepts hydra_zen.instantiate to prevent Hydra from crashing on mock configs.

        Returns a fresh MagicMock every time it is called.
        """
        with patch(
            "policy.algorithms.diffusion.hydra_zen.instantiate",
            side_effect=lambda *args, **kwargs: MagicMock(),
        ) as mock:
            yield mock

    @pytest.fixture
    def basic_kwargs(self):
        # We pass empty dictionaries to satisfy Hydra config type hints for modules,
        # but our patch above handles the actual return value.
        # We now pass explicit dimensions instead of a datamodule!
        return {
            "network": {},
            "ema": {},
            "noise_scheduler": {},
            "optimizer": {},
            "act_dim": 4,
            "obs_dim": 3,
            "pred_horizon": 16,
            "obs_horizon": 2,
        }

    def test_horizon_validations(self, basic_kwargs):
        """Ensures the init method strictly catches invalid horizon configurations."""

        # act_horizon > pred_horizon
        with pytest.raises(ValueError, match="cannot be greater than prediction horizon"):
            DiffusionPolicy(**basic_kwargs, act_horizon=20)

        # act_horizon < obs_horizon
        with pytest.raises(ValueError, match="cannot be less than observation horizon"):
            kwargs = basic_kwargs.copy()
            kwargs["obs_horizon"] = 4
            DiffusionPolicy(**kwargs, act_horizon=2)

        # obs_horizon + act_horizon - 1 > pred_horizon
        with pytest.raises(ValueError, match="Prediction horizon .* is too short"):
            kwargs = basic_kwargs.copy()
            kwargs["obs_horizon"] = 4
            kwargs["pred_horizon"] = 8
            # 4 + 6 - 1 = 9. 9 > 8, so this should fail.
            DiffusionPolicy(**kwargs, act_horizon=6)

    def test_obs_dim_parsing(self, basic_kwargs):
        """Ensures obs_dim handles both flat integers and nested dictionaries."""

        # Test flat int
        kwargs = basic_kwargs.copy()
        kwargs["obs_dim"] = 5
        policy_flat = DiffusionPolicy(**kwargs)
        # Note: Depending on your exact implementation, this attribute might be
        # called .obs_dim or .obs_dim internally. Using obs_dim based on your __init__.
        assert policy_flat.obs_dim == 5

        # Test nested dict (should trigger sum_shapes)
        kwargs["obs_dim"] = {"camera1": {"shape": (3, 64, 64)}, "proprio": {"shape": (12,)}}
        policy_dict = DiffusionPolicy(**kwargs)
        # 64 + 12 = 76
        assert sum_shapes(policy_dict.obs_dim) == 76

    def test_uninitialized_errors(self, basic_kwargs):
        """Ensures methods fail gracefully if configure_model is not called."""
        policy = DiffusionPolicy(**basic_kwargs)
        # We DO NOT call policy.configure_model() here, so network remains None

        with pytest.raises(ValueError, match="Network not initialized"):
            policy.get_action(torch.randn(1, 2, 3))

        with pytest.raises(ValueError, match="Network not initialized"):
            policy.on_train_batch_end(None, {}, 0)

        with pytest.raises(ValueError, match="Network not initialized"):
            policy._compute_loss(torch.randn(1, 2, 3), torch.randn(1, 4, 4))

    def test_compute_loss_prediction_types(self, basic_kwargs):
        """Ensures _compute_loss correctly maps prediction_type to the right target tensor."""
        policy = DiffusionPolicy(**basic_kwargs)
        policy.configure_model()  # Populates self.network and self.ema with our mock patches

        # Mock the initialized components behavior
        policy.network = MagicMock(return_value=torch.ones(1, 16, 4))  # Dummy prediction
        policy.noise_scheduler.add_noise.return_value = torch.zeros(1, 16, 4)
        policy.noise_scheduler.config = {"num_train_timesteps": 100}

        B = 2
        obs_seq = torch.randn(B, 2, policy.obs_dim)
        act_seq = torch.ones(B, 16, policy.act_dim)  # Act sequence is 1s

        # Test "sample" (target should be act_seq)
        policy.noise_scheduler.config["prediction_type"] = "sample"
        loss_sample = policy._compute_loss(obs_seq, act_seq)
        assert isinstance(loss_sample, torch.Tensor)

        # Test "v_prediction"
        policy.noise_scheduler.config["prediction_type"] = "v_prediction"
        policy.noise_scheduler.get_velocity.return_value = (
            torch.ones(B, 16, policy.act_dim) * 2
        )  # Vel is 2s
        loss_v = policy._compute_loss(obs_seq, act_seq)
        assert isinstance(loss_v, torch.Tensor)

        # Test Invalid prediction type
        policy.noise_scheduler.config["prediction_type"] = "invalid_type"
        with pytest.raises(ValueError, match="Unsupported prediction_type: invalid_type"):
            policy._compute_loss(obs_seq, act_seq)

    def test_get_action_slicing_and_clamping(self, basic_kwargs):
        """Ensures get_action slices the predicted sequence correctly based on horizons and clamps
        bounds."""
        policy = DiffusionPolicy(**basic_kwargs, act_horizon=8)
        policy.configure_model()  # Populates self.network and self.ema with our mock patches

        # Set up mocks
        policy.noise_scheduler.config = {"num_train_timesteps": 10}
        policy.noise_scheduler.timesteps = torch.tensor([0, 1])

        # Create a deterministic "noisy" output sequence to track slicing
        # Shape: (B, pred_horizon, act_dim) -> (1, 16, 4)
        pred_output = torch.arange(16).view(1, 16, 1).expand(1, 16, 4).float()

        # Step returns a tuple where output[0] is the denoised sequence
        policy.noise_scheduler.step.return_value = (pred_output,)

        obs_seq = torch.randn(1, 2, 3)  # obs_horizon = 2

        # obs_horizon = 2 -> start = 1. act_horizon = 8 -> end = 9.
        # Sliced indices should be 1 through 8.
        out = policy.get_action(obs_seq, clamp_range=None)

        assert out.shape == (1, 8, 4)
        assert torch.allclose(out[0, 0, 0], torch.tensor(1.0))  # Index 1
        assert torch.allclose(out[0, -1, 0], torch.tensor(8.0))  # Index 8

        # Test Clamping
        out_clamped = policy.get_action(obs_seq, clamp_range=(3.0, 6.0))
        assert out_clamped.min() >= 3.0
        assert out_clamped.max() <= 6.0
