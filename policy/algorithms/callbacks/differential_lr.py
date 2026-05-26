from lightning.pytorch.callbacks import BaseFinetuning


class DifferentialLRCallback(BaseFinetuning):
    def __init__(self, unet_lr: float = 1e-6):
        super().__init__()
        self.unet_lr = unet_lr

    def freeze_before_training(self, pl_module):
        # Temporarily freeze U-Net so it isn't added to the default parameter group
        self.freeze(pl_module.network)
        self.make_trainable(pl_module.planner)

    def finetune_function(self, pl_module, current_epoch, optimizer):
        # Immediately unfreeze at epoch 0 and assign the custom learning rate
        if current_epoch == 0:
            self.unfreeze_and_add_param_group(
                modules=pl_module.network,
                optimizer=optimizer,
                train_bn=True,
                lr=self.unet_lr,
            )
