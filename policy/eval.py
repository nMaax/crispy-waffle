import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from policy.algorithms.diffusion import DiffusionPolicy
from policy.datamodules.maniskill_datamodule import DummyDataset
from policy.experiment import instantiate_trainer


@hydra.main(config_path="../policy/configs", config_name="config", version_base="1.2")
def main(config: DictConfig):
    # Load the model purely from the checkpoint, completely bypassing Hydra's algorithm instantiation
    # Because you used save_hyperparameters(), the shape integers are already in the ckpt!
    model = DiffusionPolicy.load_from_checkpoint(config.ckpt_path)

    # Instantiate the Trainer and Callbacks via Hydra
    # This automatically builds your RolloutEvaluationCallback!
    trainer = instantiate_trainer(config.trainer)

    # Create a raw Dummy DataLoader (No Heavy ManiSkillDataModule required!)
    dummy_loader = DataLoader(DummyDataset(), batch_size=1)

    # Run the Test Loop!
    # This sends 1 fake batch through the system, immediately triggering `on_test_epoch_end`,
    # which then runs the rollout simulation.
    trainer.test(model=model, dataloaders=dummy_loader)


if __name__ == "__main__":
    main()
