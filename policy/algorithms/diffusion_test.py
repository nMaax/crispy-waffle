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
            "Diffusion policies do not use a standard single-step `forward` pass "
            "during the training step. Inference is handled iteratively via `get_action`."
            "Thus, we skip this test as it is not applicable to the DiffusionPolicy architecture."
        )
