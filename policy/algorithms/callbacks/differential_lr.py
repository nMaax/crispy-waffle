from lightning.pytorch.callbacks import BaseFinetuning


class DifferentialLRCallback(BaseFinetuning):
    """Callback that assigns a different learning rate to a backbone network than to the rest of
    the model (e.g. planner/adapter)."""

    def __init__(
        self,
        backbone_lr: float = 1e-6,
        backbone_attr: str = "network",
        adapter_attr: str = "planner",
        *,
        unet_lr: float | None = None,
    ):
        super().__init__()
        self.backbone_lr = unet_lr if unet_lr is not None else backbone_lr
        self.backbone_attr = backbone_attr
        self.adapter_attr = adapter_attr

    def _get_module_attr(self, pl_module, attr_name: str):
        if not hasattr(pl_module, attr_name):
            raise AttributeError(
                f"{type(pl_module).__name__} does not have attribute '{attr_name}' expected by {type(self).__name__}."
            )
        return getattr(pl_module, attr_name)

    def freeze_before_training(self, pl_module):
        # Temporarily freeze backbone so it isn't added to the default parameter group
        backbone = self._get_module_attr(pl_module, self.backbone_attr)
        adapter = self._get_module_attr(pl_module, self.adapter_attr)
        self.freeze(backbone)
        self.make_trainable(adapter)

    def finetune_function(self, pl_module, current_epoch, optimizer):
        # Immediately unfreeze at epoch 0 and assign the custom learning rate
        if current_epoch == 0:
            backbone = self._get_module_attr(pl_module, self.backbone_attr)
            self.unfreeze_and_add_param_group(
                modules=backbone,
                optimizer=optimizer,
                train_bn=True,
                lr=self.backbone_lr,
            )
