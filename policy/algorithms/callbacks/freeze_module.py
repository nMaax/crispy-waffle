import lightning as L
from lightning.pytorch.callbacks import Callback


class FreezeModuleCallback(Callback):
    """Permanently freezes a named submodule of the LightningModule (sets requires_grad=False).

    Deliberately a plain ``Callback`` rather than a ``BaseFinetuning`` subclass. ``BaseFinetuning``
    freezes from ``setup()``, which Lightning runs *before* ``configure_model()`` (see
    ``trainer.py``: ``_call_setup_hook`` → ``_call_configure_model`` → ``strategy.setup`` →
    ``on_fit_start``). Every algorithm here builds its submodules lazily in ``configure_model()``,
    so at ``setup()`` time the attribute is still ``None`` and ``BaseFinetuning.freeze`` raises
    ``AttributeError: 'NoneType' object has no attribute 'modules'``.

    Freezing at ``on_fit_start`` therefore happens *after* ``configure_optimizers``, so the frozen
    parameters do sit in a param group -- but they never receive a gradient, so the optimizer skips
    them and their weights stay bit-identical (asserted in ``tests/.../test_freeze_module.py``).
    """

    def __init__(self, attr_name: str):
        super().__init__()
        self.attr_name = attr_name

    def on_fit_start(self, _trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        module = getattr(pl_module, self.attr_name)
        if module is None:
            raise AttributeError(
                f"{type(pl_module).__name__}.{self.attr_name} is None; nothing to freeze."
            )
        for param in module.parameters():
            param.requires_grad_(False)
