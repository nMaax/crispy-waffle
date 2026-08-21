"""Evaluation script using [Hydra](https://hydra.cc).

This does the following:
1. Parses the config using Hydra;
2. Instantiated the components (trainer / algorithm);
3. Evaluates the model;

"""

import dataclasses
import logging
from pathlib import Path

import hydra
import lightning
import rich
import torch
from omegaconf import DictConfig, OmegaConf
from rich.panel import Panel
from torch.utils.data import DataLoader

import wandb
from policy.algorithms.callbacks.rollout_evaluation import SUCCESS_METRICS
from policy.datamodules.trajectory_datamodule import DummyDataset
from policy.experiment import instantiate_trainer
from policy.utils.hf_hub_utils import ensure_checkpoint
from policy.utils.hydra_utils import (
    find_checkpoint_hydra_config,
    get_checkpoint_branch,
    get_experiment_phase,
    resolve_dictconfig,
)
from policy.utils.logging_utils import setup_logging

torch.set_float32_matmul_precision("high")

logger = logging.getLogger(__name__)


def _log_zero_shot_bar_charts(trainer: lightning.Trainer) -> None:
    """Logs one wandb bar chart per metric, across any env-namespaced `test/<env>/<metric>` entries
    produced by running multiple RolloutEvaluationCallback instances in one test run.

    A no-op for ordinary single-env test runs, whose metrics aren't env-namespaced.
    """
    per_metric: dict[str, dict[str, float]] = {}
    for key, value in trainer.callback_metrics.items():
        parts = key.split("/")
        if len(parts) == 3 and parts[0] == "test" and parts[2] in SUCCESS_METRICS:
            per_metric.setdefault(parts[2], {})[parts[1]] = float(value)

    for metric_name, env_values in per_metric.items():
        table = wandb.Table(columns=["env", metric_name], data=list(env_values.items()))
        wandb.log({f"test/{metric_name}_by_env": wandb.plot.bar(table, "env", metric_name)})


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(dict_config: DictConfig):
    config = resolve_dictconfig(dict_config)

    experiment_phase = get_experiment_phase(config.name)
    if experiment_phase == "train":
        warning_msg = (
            "[bold red]TRAINING VIA EVAL IS NOT SUPPORTED[/bold red]\n\n"
            f"It looks like you are trying to evaluate [yellow]{config.name}[/yellow], whose name "
            "marks it as a [yellow]train[/yellow]-phase experiment, through [bold cyan]eval.py[/bold cyan].\n\n"
            "eval.py loads a dummy dataloader and only runs the rollout evaluation callbacks — it "
            "never trains. Please use the dedicated training entrypoint instead:\n"
            "[bold green]uv run python policy/main.py experiment=YOUR_EXP[/bold green]"
        )
        rich.print(Panel(warning_msg, title="Notice", border_style="red"))
        raise ValueError(
            "Running a train-phase experiment via eval.py is not supported; use main.py instead."
        )

    setup_logging(
        log_level=config.log_level,
        global_log_level="DEBUG" if config.debug else "INFO" if config.verbose else "WARNING",
    )

    if not hasattr(config, "ckpt_path") or config.ckpt_path is None:
        raise ValueError("Checkpoint path must be specified in the config under 'ckpt_path'.")
    ckpt_path = Path(config.ckpt_path)
    ensure_checkpoint(ckpt_path, config.hf_checkpoint_repo_id)

    loaded_branch = get_checkpoint_branch(config.ckpt_path)
    if loaded_branch is not None and loaded_branch != config.branch:
        logger.warning(
            f"Branch mismatch! The checkpoint at '{config.ckpt_path}' was generated on branch "
            f"'{loaded_branch}', but you are currently on branch '{config.branch}'. Code "
            "differences between the two branches may affect evaluation results."
        )

    # Seed everything for reproducibility during evaluation (env seeding + model stochastic actions)
    lightning.seed_everything(seed=config.seed, workers=True)

    print(f"Loading policy from {ckpt_path}...")
    # Load the model class dynamically from the config
    model_class = hydra.utils.get_class(dict_config.algorithm._target_)
    model = model_class.load_from_checkpoint(ckpt_path, weights_only=False)

    hparams = dataclasses.asdict(config)
    checkpoint_hydra_config = find_checkpoint_hydra_config(config.ckpt_path)
    if checkpoint_hydra_config is not None and "datamodule" in checkpoint_hydra_config:
        hparams["datamodule"] = OmegaConf.to_container(
            checkpoint_hydra_config.datamodule, resolve=True
        )
    else:
        hparams["datamodule"] = None
        logger.warning(
            f"Could not find a '.hydra/config.yaml' snapshot for checkpoint '{ckpt_path}'; "
            "datamodule hyperparameters (e.g. her_ratio, load_count) will not be logged to wandb."
        )

    if "logger" in config.trainer and "wandb" in config.trainer["logger"]:
        config.trainer["logger"]["wandb"]["job_type"] = "test"

    trainer = instantiate_trainer(config.trainer)

    for lightning_logger in trainer.loggers:
        lightning_logger.log_hyperparams(hparams)

    dummy_loader = DataLoader(DummyDataset(), batch_size=1)

    trainer.test(model=model, dataloaders=dummy_loader)

    if wandb.run:
        _log_zero_shot_bar_charts(trainer)
        wandb.finish()


if __name__ == "__main__":
    main()
