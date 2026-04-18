import pytest
from policy.datamodules.datamodule_tests import DataModuleTests
from policy.datamodules.maniskill_datamodule import ManiSkillDataModule

@pytest.mark.parametrize("datamodule_config", ["maniskill_datamodule"], indirect=True)
class TestManiSkillDataModule(DataModuleTests[ManiSkillDataModule]):
    """Test suite for the ManiSkillDataModule."""
    pass