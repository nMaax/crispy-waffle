"""The training and evaluation functions.

NOTE: This has to be in a different file than `main` because we want to allow registering different
variants of the `train_and_validate` function for different algorithms with
`functools.singledispatch`. If we have everything in `main.py`, the registration doesn't happen
correctly.
"""

import dataclasses
import functools
import os
from logging import getLogger
from typing import Any

import hydra
import lightning
import rich
from omegaconf import DictConfig

from policy.configs.config import Config

logger = getLogger(__name__)


@functools.singledispatch
def train_and_validate(
    algorithm,
    /,
    *,
    datamodule: lightning.LightningDataModule | None = None,
    config: Config,
):
    """Generic function that trains and validates a learning algorithm.

    This by default assumes that the algorithm is a LightningModule, but can be extended to
    implement specific training / validation procedures for different algorithms.

    The default implementation here does roughly the same thing as
    https://github.com/ashleve/lightning-hydra-template/blob/main/src/train.py

    1. Instantiates the experiment components from the Hydra configuration:
        - algorithm (already instantiated)
        - trainer
        - datamodule (optional)
    2. Calls `trainer.fit` to train the algorithm
    3. Calls `trainer.validate` to validate the model
    4. Returns the metrics.

    ## Extending to other algorithms or training procedures


    For example, if your algorithm has to be trained in two distinct phases, or if you want to use
    a different kind of Trainer that does something other than just call `.fit` and `.validate`,
    you could do something like this:

    ```python
    @train_and_evaluate.register(MyAlgorithm)
    def train_and_evaluate_my_algo(algorithm: MyAlgorithm, /, *, trainer, datamodule)
        # making this up, this isn't doable with any of the datamodules at the moment.
        datamodule.set_task(1)
        trainer.fit(algorithm, datamodule)

        datamodule.set_task(2)
        trainer.fit(algorithm, datamodule)
    ```
    """

    trainer = instantiate_trainer(config.trainer)

    for lightning_logger in trainer.loggers:
        lightning_logger.log_hyperparams(dataclasses.asdict(config))
        lightning_logger.log_hyperparams(
            {k: v for k, v in os.environ.items() if k.startswith("SLURM")}
        )

    if not (
        isinstance(algorithm, lightning.LightningModule) and isinstance(trainer, lightning.Trainer)
    ):
        _this_fn_name = train_and_validate.__name__  # type: ignore
        raise NotImplementedError(
            f"The `{_this_fn_name}` function assumes that the algorithm is a "
            f"lightning.LightningModule and that the trainer is a lightning.Trainer, but got "
            f"algorithm {algorithm} and trainer {trainer}!\n"
            f"You can register a new handler for that algorithm type using "
            f"`@{_this_fn_name}.register`.\n"
            f"Registered handlers: "
            + "\n\t".join([f"- {k}: {v.__name__}" for k, v in train_and_validate.registry.items()])
        )

    algorithm = train_lightning(
        algorithm,
        trainer=trainer,
        config=config,
        datamodule=datamodule,
    )

    metric_name, error, _metrics = validate_lightning(
        algorithm,
        trainer=trainer,
        datamodule=datamodule,
    )

    return metric_name, error


def train_lightning(
    algorithm: lightning.LightningModule,
    /,
    *,
    trainer: lightning.Trainer,
    datamodule: lightning.LightningDataModule | None = None,
    config: Config,
):
    """Trains the algorithm using the trainer and datamodule."""

    if datamodule is None:
        if hasattr(algorithm, "datamodule"):
            datamodule = getattr(algorithm, "datamodule")
        elif isinstance(config.datamodule, lightning.LightningDataModule):
            datamodule = config.datamodule
        elif config.datamodule is not None:
            # TODO: what about using hydra_zen?
            datamodule = hydra.utils.instantiate(config.datamodule)

    trainer.fit(algorithm, datamodule=datamodule, ckpt_path=config.ckpt_path)
    return algorithm


def validate_lightning(
    algorithm: lightning.LightningModule,
    /,
    *,
    trainer: lightning.Trainer,
    datamodule: lightning.LightningDataModule | None = None,
) -> tuple[str, float | None, dict]:
    """Validates the algorithm and returns the metrics.

    By default, if validation is to be performed, returns the validation error. Returns the
    training error when `trainer.overfit_batches != 0` (e.g. when debugging).
    """

    datamodule = datamodule or getattr(algorithm, "datamodule", None)

    # When overfitting on a single batch or only training, we return the train error.
    if (trainer.limit_val_batches == trainer.limit_test_batches == 0) or (
        trainer.overfit_batches == 1  # type: ignore
    ):
        results_type = "train"
        results = [
            {
                **trainer.logged_metrics,
                **trainer.callback_metrics,
                **trainer.progress_bar_metrics,
            }
        ]
    else:
        results_type = "val"
        results = trainer.validate(model=algorithm, datamodule=datamodule)

    if results is None:
        rich.print("RUN FAILED!")
        return "fail", None, {}

    metrics = dict(results[0])
    for key, value in metrics.items():
        rich.print(f"{results_type} {key}: ", value)

    if (success_once_rate := metrics.get(f"{results_type}/success_once_rate")) is not None:
        # Added for Imitation Learning
        metric_name = "1-success_once_rate"
        error = 1.0 - success_once_rate
    elif (loss := metrics.get(f"{results_type}/loss")) is not None:
        logger.info("Assuming that the objective to minimize is the loss metric.")
        metric_name = "loss"
        error = loss
    else:
        raise RuntimeError(
            f"Don't know which metric to use to calculate the 'error' of this run.\n"
            f"Here are the available metric names:\n"
            f"{list(metrics.keys())}"
        )
    return metric_name, error, metrics


def instantiate_trainer(trainer_config: dict | DictConfig) -> lightning.Trainer | Any:
    """Instantiates the callbacks and loggers first, then creates the trainer from its config."""
    trainer_config = trainer_config.copy()  # Avoid mutating the config.
    callbacks: list | None = instantiate_values(trainer_config.pop("callbacks", None))
    logger: list | None = instantiate_values(trainer_config.pop("logger", None))
    trainer = hydra.utils.instantiate(trainer_config, callbacks=callbacks, logger=logger)
    return trainer


def instantiate_values(config_dict: DictConfig | None) -> list[Any] | None:
    """Returns the list of objects at the values in this dict of configs.

    This is used for the config of the `trainer/logger` and `trainer/callbacks` fields, where
    we can combine multiple config groups by adding entries in a dict.

    For example, using `trainer/logger=wandb` and `trainer/logger=tensorboard` would result in a
    dict with `wandb` and `tensorboard` as keys, and the corresponding config groups as values.

    This would then return a list with the instantiated WandbLogger and TensorBoardLogger objects.
    """
    if not config_dict:
        return None
    objects_dict = hydra.utils.instantiate(config_dict, _recursive_=True)
    if objects_dict is None:
        return None

    assert isinstance(objects_dict, dict | DictConfig)
    return [v for v in objects_dict.values() if v is not None]
