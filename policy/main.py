"""Training script using [Hydra](https://hydra.cc).

This does the following:
1. Parses the config using Hydra;
2. Instantiated the components (trainer / algorithm), optionally datamodule and network;
3. Trains the model;
4. Optionally runs an evaluation loop.

"""

from __future__ import annotations

import functools
import logging
from pathlib import Path

import hydra
import lightning
import rich
import rich.logging
import wandb
from omegaconf import DictConfig
from rich.panel import Panel

import policy
from policy.configs.config import Config
from policy.experiment import train_and_validate
from policy.utils.hydra_utils import resolve_dictconfig
from policy.utils.typing_utils import HydraConfigFor
from policy.utils.utils import print_config

PROJECT_NAME = policy.__name__
REPO_ROOTDIR = Path(__file__).parent.parent
logger = logging.getLogger(__name__)

# TODO: since I sometimes train on a RTX 4080 I should use `torch.set_float32_matmul_precision=high` instead of default 'highest'
# tho this is not supported on TITAN X and other architectures, I think Lightning provides a handy way to handle this anyway.


@hydra.main(
    config_path=f"pkg://{PROJECT_NAME}.configs",
    config_name="config",
    version_base="1.2",
)
def main(dict_config: DictConfig) -> dict:
    """Main entry point: trains & evaluates a learning algorithm."""

    print_config(dict_config, resolve=False)
    assert dict_config["algorithm"] is not None

    # Resolve all the interpolations in the configs.
    config: Config = resolve_dictconfig(dict_config)

    limit_train = config.trainer.get("limit_train_batches", None)
    limit_val = config.trainer.get("limit_val_batches", None)

    if limit_train == 0 and limit_val == 0:
        warning_msg = (
            "[bold red]TESTING VIA MAIN IS DEPRECATED[/bold red]\n\n"
            "It looks like you are trying to run an evaluation-only pipeline by setting "
            "[yellow]limit_train_batches=0[/yellow] and [yellow]limit_val_batches=0[/yellow].\n\n"
            "Using [bold cyan]main.py[/bold cyan] for pure evaluation is deprecated because it needlessly loads "
            "the training datamodule and skips the test hooks.\n\n"
            "Please use the dedicated evaluation script instead:\n"
            "[bold green]uv run python policy/eval.py experiment=YOUR_EXP ckpt_path=YOUR_CKPT[/bold green]\n\n"
            "[dim]The script will now proceed with validation only, no test rollouts will occur.[/dim]"
        )
        rich.print(Panel(warning_msg, title="Architecture Notice", border_style="red"))

    setup_logging(
        log_level=config.log_level,
        global_log_level="DEBUG" if config.debug else "INFO" if config.verbose else "WARNING",
    )

    # Seed the random number generators, so the weights that are
    # constructed are deterministic and reproducible.
    lightning.seed_everything(seed=config.seed, workers=True)

    if config.datamodule is None:
        datamodule = None
    elif isinstance(config.datamodule, lightning.LightningDataModule):
        # The datamodule was already instantiated for the `instance_attr` resolver to
        # get an attribute like `num_classes` when instantiating a network config.
        datamodule = config.datamodule
    else:
        datamodule = hydra.utils.instantiate(config.datamodule)

    # Create the algo.
    algorithm = instantiate_algorithm(config.algorithm)

    # Do the training and evaluation, returns the metric name and the overall 'error' to minimize.
    metric_name, error = train_and_validate(algorithm, config=config, datamodule=datamodule)

    if wandb.run:
        wandb.finish()

    assert error is not None
    # Results are returned like this so that the Orion sweeper can parse the results correctly.
    return dict(name=metric_name, type="objective", value=error)


def setup_logging(log_level: str, global_log_level: str = "WARNING") -> None:
    from policy.main import PROJECT_NAME

    logging.basicConfig(
        level=global_log_level.upper(),
        # format="%(asctime)s - %(levelname)s - %(message)s",
        format="%(message)s",
        datefmt="[%X]",
        force=True,
        handlers=[
            rich.logging.RichHandler(
                markup=True,
                rich_tracebacks=True,
                tracebacks_width=100,
                tracebacks_show_locals=False,
            )
        ],
    )

    policy_logger = logging.getLogger(PROJECT_NAME)
    policy_logger.setLevel(log_level.upper())


def instantiate_algorithm(
    algorithm_config: HydraConfigFor[lightning.LightningModule],
) -> lightning.LightningModule:
    """Function used to instantiate the algorithm."""
    # Create the algorithm

    # TODO: can I use hydra_zen?
    algo_or_algo_partial = hydra.utils.instantiate(algorithm_config)

    if isinstance(algo_or_algo_partial, functools.partial):
        return algo_or_algo_partial()
    else:
        return algo_or_algo_partial


if __name__ == "__main__":
    main()
