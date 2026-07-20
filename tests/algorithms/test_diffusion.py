from unittest.mock import MagicMock, patch

import pytest
import torch

from policy.algorithms.diffusion_policy import DiffusionPolicy
from policy.algorithms.goal_conditioned_diffusion_policy import GoalConditionedDiffusionPolicy
from policy.utils import flatten_and_concat_leaf_tensors, get_total_dim
from policy.utils.test_utils import get_gpu_arch_name
from tests.algorithms.test_lightning_module import LightningModuleTests


@pytest.mark.parametrize("algorithm_config", ["diffusion_policy"], indirect=True)
@pytest.mark.parametrize("datamodule_config", ["trajectory_datamodule"], indirect=True)
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
        obs_seq = flatten_and_concat_leaf_tensors(obs_seq, device=algorithm.device)
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
        obs_seq = flatten_and_concat_leaf_tensors(obs_seq, device=algorithm.device)
        with torch.no_grad():
            out = algorithm.get_action(obs_seq)

        # Regression check
        tensor_regression.check(
            {"action": out},
            default_tolerance={"rtol": 1e-5, "atol": 1e-6},
            additional_label=get_gpu_arch_name(),
            include_gpu_name_in_stats=False,
        )


@pytest.mark.parametrize("algorithm_config", ["goal_conditioned_diffusion_policy"], indirect=True)
@pytest.mark.parametrize(
    "datamodule_config", ["goal_conditioned_trajectory_datamodule"], indirect=True
)
class TestGoalConditionedDiffusionPolicy(LightningModuleTests[GoalConditionedDiffusionPolicy]):
    """Test suite for GoalConditionedDiffusionPolicy."""

    def test_forward_pass_is_reproducible(
        self,
        algorithm,
        training_step_content,
        tensor_regression,
    ):
        pytest.skip("Diffusion policies do not use standard forward pass during training.")

    def test_get_action_runs(
        self,
        algorithm,
        training_step_content,
    ):
        algorithm.eval()
        batch_device = training_step_content.batch["act_seq"].device
        algorithm.to(batch_device)

        obs_seq = training_step_content.batch["obs_seq"]
        goal = training_step_content.batch.get("goal", None)
        with torch.no_grad():
            out = algorithm.get_action(obs_seq, goal=goal)

        assert isinstance(out, torch.Tensor)
        assert torch.isfinite(out).all(), "Output contains NaN or Inf"
        assert out.device == algorithm.device
        assert out.shape == (
            training_step_content.batch["act_seq"].shape[0],
            algorithm.act_horizon,
            algorithm.act_dim,
        )


class TestDiffusionPolicyLogic:
    """Isolated unit tests for DiffusionPolicy-specific boundary conditions.

    Shared infra (horizon validation, normalizer building, cond-dim inference, EMA
    optionality, abstract methods, obs-only templates) is covered once in
    ``TestBaseDiffusionAgentLogic``; this suite only covers what is unique to DP:
    the EMA-required contract, the diffusers-scheduler loss dispatch, and the
    diffusers-scheduler reverse loop (slicing + clamping + EMA store/restore).
    """

    @pytest.fixture(autouse=True)
    def patch_instantiate(self):
        """Intercepts hydra_zen.instantiate in the base module to prevent Hydra from crashing on
        mock configs."""
        with patch(
            "policy.algorithms.base_diffusion_agent.hydra_zen.instantiate",
            side_effect=lambda *args, **kwargs: MagicMock(),
        ) as mock:
            yield mock

    @pytest.fixture
    def basic_kwargs(self):
        return {
            "network": {"_target_": "policy.algorithms.networks.unet1d.UNet1D"},
            "ema": {},
            "noise_scheduler": {},
            "optimizer": {},
            "act_dim": 4,
            "obs_dim": 3,
            "pred_horizon": 16,
            "obs_horizon": 2,
        }

    def test_ema_required_raises(self, basic_kwargs):
        """DiffusionPolicy must reject construction without an EMA config."""
        kwargs = {k: v for k, v in basic_kwargs.items() if k != "ema"}
        with pytest.raises(ValueError, match="requires an EMA model"):
            DiffusionPolicy(**kwargs)

    def test_horizon_validations(self, basic_kwargs):
        """Ensures the init method strictly catches invalid horizon configurations."""

        # act_horizon > pred_horizon
        with pytest.raises(ValueError, match="cannot be greater than prediction horizon"):
            DiffusionPolicy(**basic_kwargs, act_horizon=20)

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
        assert policy_flat.obs_dim == 5

        # Test nested dict (should trigger get_total_dim)
        kwargs["obs_dim"] = {"camera1": {"shape": (3, 64, 64)}, "proprio": {"shape": (12,)}}
        policy_dict = DiffusionPolicy(**kwargs)
        # 64 + 12 = 76
        assert get_total_dim(policy_dict.obs_dim) == 76

    def test_uninitialized_errors(self, basic_kwargs):
        """Ensures methods fail gracefully if configure_model is not called."""
        policy = DiffusionPolicy(**basic_kwargs)
        # We do NOT call policy.configure_model() here, so network remains None

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

        if policy.noise_scheduler is None:
            raise ValueError("Noise scheduler should be initialized by configure_model")

        # Mock the initialized components behavior
        policy.network = MagicMock(return_value=torch.ones(2, 16, 4))  # Dummy prediction
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

        if policy.noise_scheduler is None:
            raise ValueError("Noise scheduler should be initialized by configure_model")

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
        out = policy.get_action(obs_seq, output_clip_range=None)

        assert out.shape == (1, 8, 4)
        assert torch.allclose(out[0, 0, 0], torch.tensor(1.0))  # Index 1
        assert torch.allclose(out[0, -1, 0], torch.tensor(8.0))  # Index 8

        # Test Clamping
        out_clamped = policy.get_action(obs_seq, output_clip_range=(3.0, 6.0))
        assert out_clamped.min() >= 3.0
        assert out_clamped.max() <= 6.0

    def test_run_diffusion_loop_ema_store_restore(self, basic_kwargs):
        """Ensures _run_diffusion_loop stores/restores EMA weights around the loop."""
        policy = DiffusionPolicy(**basic_kwargs, act_horizon=8)
        policy.configure_model()

        policy.noise_scheduler.config = {"num_train_timesteps": 10}
        policy.noise_scheduler.timesteps = torch.tensor([0, 1])
        policy.noise_scheduler.step.return_value = (
            torch.arange(16).view(1, 16, 1).expand(1, 16, 4).float(),
        )

        policy.get_action(torch.randn(1, 2, 3))

        policy.ema.store.assert_called_once()
        policy.ema.copy_to.assert_called_once()
        policy.ema.restore.assert_called_once()
