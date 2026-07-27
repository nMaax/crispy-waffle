import lightning as L
from lightning.pytorch.callbacks import Callback


class FreezeModuleCallback(Callback):
    """Permanently freezes a named submodule of the LightningModule (sets requires_grad=False)."""

    def __init__(self, attr_name: str):
        super().__init__()
        self.attr_name = attr_name

    def on_fit_start(self, _trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # on_fit_start (not BaseFinetuning's setup()) because algorithms like GCDP build their
        # submodules lazily in configure_model(), which hasn't run yet when setup() fires.
        module = getattr(pl_module, self.attr_name)
        if module is None:
            raise AttributeError(
                f"{type(pl_module).__name__}.{self.attr_name} is None; nothing to freeze."
            )
        for param in module.parameters():
            param.requires_grad_(False)
