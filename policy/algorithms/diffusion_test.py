import pytest
import torch

from policy.algorithms.diffusion import DiffusionPolicy
from policy.algorithms.lightning_module_tests import LightningModuleTests
from policy.utils import get_batch_size


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
        cond_seq = training_step_content.batch["cond_seq"]
        with torch.no_grad():
            out = algorithm.get_action(cond_seq)

        # Assert instance, finiteness, device, and shape of the output
        assert isinstance(out, torch.Tensor)
        assert torch.isfinite(out).all(), "Output contains NaN or Inf"
        assert out.device == algorithm.device
        assert out.shape == (
            get_batch_size(cond_seq),
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

        cond_seq = training_step_content.batch["cond_seq"]
        with torch.no_grad():
            out = algorithm.get_action(cond_seq)

        # Regression check
        tensor_regression.check(
            {"action": out},
            default_tolerance={"rtol": 1e-5, "atol": 1e-6},
            include_gpu_name_in_stats=False,
        )
