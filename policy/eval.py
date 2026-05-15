from pathlib import Path

import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from policy.algorithms.diffusion import DiffusionPolicy
from policy.datamodules.trajectory_datamodule import DummyDataset
from policy.experiment import instantiate_trainer
from policy.utils.hydra_utils import resolve_dictconfig


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(dict_config: DictConfig):
    config = resolve_dictconfig(dict_config)
    if not hasattr(config, "ckpt_path") or config.ckpt_path is None:
        raise ValueError("Checkpoint path must be specified in the config under 'ckpt_path'.")
    ckpt_path = Path(config.ckpt_path)

    print(f"Loading policy from {ckpt_path}...")
    model = DiffusionPolicy.load_from_checkpoint(ckpt_path)

    trainer = instantiate_trainer(config.trainer)

    dummy_loader = DataLoader(DummyDataset(), batch_size=1)

    trainer.test(model=model, dataloaders=dummy_loader)


if __name__ == "__main__":
    main()
