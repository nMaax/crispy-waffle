from lightning.pytorch.callbacks import BaseFinetuning
from lightning.pytorch.utilities import rank_zero_info


class FinetunePlannerCallback(BaseFinetuning):
    """Finetuning callback that freezes a backbone network and keeps an adapter/planner module
    trainable, then unfreezes the backbone network at a specified global step with a custom
    learning rate."""

    def __init__(
        self,
        unfreeze_step: int | None = 500,
        backbone_lr: float | None = 1e-6,
        backbone_attr: str = "network",
        adapter_attr: str = "planner",
        *,
        unet_unfreeze_step: int | None = None,
        unet_lr: float | None = None,
    ):
        super().__init__()
        self.unfreeze_step = unet_unfreeze_step if unet_unfreeze_step is not None else unfreeze_step
        self.backbone_lr = unet_lr if unet_lr is not None else backbone_lr
        self.backbone_attr = backbone_attr
        self.adapter_attr = adapter_attr
        self._backbone_is_frozen = True

    def _get_module_attr(self, pl_module, attr_name: str):
        if not hasattr(pl_module, attr_name):
            raise AttributeError(
                f"{type(pl_module).__name__} does not have attribute '{attr_name}' expected by {type(self).__name__}."
            )
        return getattr(pl_module, attr_name)

    def freeze_before_training(self, pl_module):
        backbone = self._get_module_attr(pl_module, self.backbone_attr)
        adapter = self._get_module_attr(pl_module, self.adapter_attr)
        self.freeze(backbone)
        self.make_trainable(adapter)

    def finetune_function(self, pl_module, current_epoch, optimizer):
        # We leave this empty because we are handling it per-step, not per-epoch
        pass

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if (
            self.unfreeze_step is not None
            and trainer.global_step >= self.unfreeze_step
            and self._backbone_is_frozen
        ):
            backbone = self._get_module_attr(pl_module, self.backbone_attr)
            for optimizer in trainer.optimizers:
                self.unfreeze_and_add_param_group(
                    modules=backbone,
                    optimizer=optimizer,
                    train_bn=True,
                    lr=self.backbone_lr,
                )

            self._backbone_is_frozen = False
            rank_zero_info(
                f"Unfroze backbone network '{self.backbone_attr}' at global step {trainer.global_step} with LR {self.backbone_lr}"
            )
