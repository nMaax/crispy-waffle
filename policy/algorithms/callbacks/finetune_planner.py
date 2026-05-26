from lightning.pytorch.callbacks import BaseFinetuning
from lightning.pytorch.utilities import rank_zero_info


class FinetunePlannerCallback(BaseFinetuning):
    def __init__(self, unet_unfreeze_step: int | None = 500, unet_lr: float | None = 1e-6):
        super().__init__()
        self.unet_unfreeze_step = unet_unfreeze_step
        self.unet_lr = unet_lr
        self._unet_is_frozen = True

    def freeze_before_training(self, pl_module):
        self.freeze(pl_module.network)
        self.make_trainable(pl_module.planner)

    def finetune_function(self, pl_module, current_epoch, optimizer):
        # We leave this empty because we are handling it per-step, not per-epoch
        pass

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if (
            self.unet_unfreeze_step is not None
            and trainer.global_step >= self.unet_unfreeze_step
            and self._unet_is_frozen
        ):
            for optimizer in trainer.optimizers:
                self.unfreeze_and_add_param_group(
                    modules=pl_module.network,
                    optimizer=optimizer,
                    train_bn=True,
                    lr=self.unet_lr,
                )

            self._unet_is_frozen = False
            rank_zero_info(
                f"Unfroze U-Net at global step {trainer.global_step} with LR {self.unet_lr}"
            )
