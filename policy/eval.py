from pathlib import Path

import hydra
import lightning
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

import wandb
from policy.datamodules.trajectory_datamodule import DummyDataset
from policy.experiment import instantiate_trainer
from policy.utils.hydra_utils import resolve_dictconfig

torch.set_float32_matmul_precision("high")


def _log_zero_shot_bar_charts(trainer: lightning.Trainer) -> None:
    """Logs one wandb bar chart per metric, across any env-namespaced `test/<env>/<metric>` entries
    produced by running multiple RolloutEvaluationCallback instances in one test run.

    A no-op for ordinary single-env test runs, whose metrics aren't env-namespaced.
    """
    per_metric: dict[str, dict[str, float]] = {}
    for key, value in trainer.callback_metrics.items():
        parts = key.split("/")
        if len(parts) == 3 and parts[0] == "test" and parts[2] in (
            "success_once_rate",
            "success_at_end_rate",
        ):
            per_metric.setdefault(parts[2], {})[parts[1]] = float(value)

    for metric_name, env_values in per_metric.items():
        table = wandb.Table(columns=["env", metric_name], data=list(env_values.items()))
        wandb.log({f"test/{metric_name}_by_env": wandb.plot.bar(table, "env", metric_name)})


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(dict_config: DictConfig):
    config = resolve_dictconfig(dict_config)
    if not hasattr(config, "ckpt_path") or config.ckpt_path is None:
        raise ValueError("Checkpoint path must be specified in the config under 'ckpt_path'.")
    ckpt_path = Path(config.ckpt_path)

    # Seed everything for reproducibility during evaluation (env seeding + model stochastic actions)
    lightning.seed_everything(seed=config.seed, workers=True)

    print(f"Loading policy from {ckpt_path}...")
    # Load the model class dynamically from the config
    model_class = hydra.utils.get_class(dict_config.algorithm._target_)
    model = model_class.load_from_checkpoint(ckpt_path)

    trainer = instantiate_trainer(config.trainer)

    dummy_loader = DataLoader(DummyDataset(), batch_size=1)

    trainer.test(model=model, dataloaders=dummy_loader)

    if wandb.run:
        _log_zero_shot_bar_charts(trainer)
        wandb.finish()


if __name__ == "__main__":
    main()
