import abc
import sys
from typing import Generic, TypeVar

import hydra
import lightning
import omegaconf
import pytest
from lightning import LightningDataModule
from lightning.fabric.utilities.exceptions import MisconfigurationException
from lightning.pytorch.trainer.states import RunningStage
from tensor_regression.fixture import TensorRegressionFixture
from torch.utils.data import DataLoader

from policy.utils.test_utils import IN_GITHUB_CLOUD_CI, get_gpu_arch_name
from tests.algorithms.test_lightning_module import convert_list_and_tuples_to_dicts
from tests.conftest import algorithm_config

DataModuleType = TypeVar("DataModuleType", bound=LightningDataModule)

# Use a dummy, empty algorithm, to keep the datamodule tests independent of the algorithms.
# This is a unit test for the datamodule, so we don't want to involve the algorithm here.


@pytest.mark.skipif(
    IN_GITHUB_CLOUD_CI and sys.platform == "darwin",
    reason="Getting weird bugs with MacOS on GitHub CI.",
)
@pytest.mark.parametrize(algorithm_config.__name__, ["no_op"], indirect=True, ids=[""])
class DataModuleTests(Generic[DataModuleType], abc.ABC):
    @pytest.fixture(
        scope="class",
        params=[
            RunningStage.TRAINING,
            RunningStage.VALIDATING,
            RunningStage.TESTING,
            pytest.param(
                RunningStage.PREDICTING,
                marks=pytest.mark.xfail(
                    reason="Might not be implemented by the datamodule.",
                    raises=MisconfigurationException,
                ),
            ),
        ],
    )
    def stage(self, request: pytest.FixtureRequest):
        return getattr(request, "param", RunningStage.TRAINING)

    @pytest.fixture(scope="class")
    def datamodule(self, dict_config: omegaconf.DictConfig) -> DataModuleType:
        """Fixture that creates the datamodule instance, given the current Hydra config."""
        datamodule = hydra.utils.instantiate(dict_config["datamodule"])
        return datamodule

    @pytest.fixture(scope="class")
    def dataloader(self, datamodule: DataModuleType, stage: RunningStage) -> DataLoader:
        # NOTE: This fixture is class-scoped, so it executes BEFORE the function-scoped
        # 'seed' fixture in conftest.py. We must manually pin the seed here to ensure
        # that randomness inside setup() (like random_split) is deterministic across
        # different test runs.

        lightning.seed_everything(42, workers=True)
        datamodule.prepare_data()
        if stage == RunningStage.TRAINING:
            datamodule.setup("fit")
            dataloader = datamodule.train_dataloader()
        elif stage in [RunningStage.VALIDATING, RunningStage.SANITY_CHECKING]:
            datamodule.setup("validate")
            dataloader = datamodule.val_dataloader()
        elif stage == RunningStage.TESTING:
            datamodule.setup("test")
            dataloader = datamodule.test_dataloader()
        else:
            assert stage == RunningStage.PREDICTING
            datamodule.setup("predict")
            dataloader = datamodule.predict_dataloader()
        return dataloader

    @pytest.fixture(scope="class")
    def batch(self, dataloader: DataLoader):
        # NOTE: This fixture is class-scoped, so it executes BEFORE the function-scoped
        # 'seed' fixture in conftest.py. We must manually pin the seed here to ensure
        # that randomness inside setup() (like random_split) is deterministic across
        # different test runs.

        lightning.seed_everything(42, workers=True)
        iterator = iter(dataloader)
        batch = next(iterator)
        return batch

    def test_first_batch(
        self,
        batch,
        tensor_regression: TensorRegressionFixture,
    ):
        batch = convert_list_and_tuples_to_dicts(batch)
        tensor_regression.check(
            batch, additional_label=get_gpu_arch_name(), include_gpu_name_in_stats=False
        )
