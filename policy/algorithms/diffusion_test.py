import pytest

from policy.algorithms.diffusion import DiffusionPolicy
from policy.algorithms.lightning_module_tests import LightningModuleTests


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
        # Use the cond_seq from the training batch as input
        cond_seq = training_step_content.batch["cond_seq"]
        out = algorithm.get_action(cond_seq)
        # Check output is a torch.Tensor
        assert isinstance(out, type(cond_seq))
        # Check output shape: (batch, act_horizon, act_dim)
        assert out.shape[0] == cond_seq.shape[0]
        assert out.shape[1] == algorithm.act_horizon
        assert out.shape[2] == algorithm.act_dim
