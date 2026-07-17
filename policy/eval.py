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
        wandb.finish()


if __name__ == "__main__":
    main()
