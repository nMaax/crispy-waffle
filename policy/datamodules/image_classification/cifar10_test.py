from policy.conftest import setup_with_overrides
from policy.datamodules.image_classification.cifar10 import CIFAR10DataModule
from policy.datamodules.image_classification.image_classification_test import (
    ImageClassificationDataModuleTests,
)


@setup_with_overrides("datamodule=cifar10")
class TestCIFAR10DataModule(ImageClassificationDataModuleTests[CIFAR10DataModule]): ...
