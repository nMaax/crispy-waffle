import logging
import threading
from pathlib import Path

import lightning
from huggingface_hub import HfApi
from lightning.pytorch.callbacks import ModelCheckpoint
from typing_extensions import override

from policy.utils.env_vars import REPO_ROOTDIR

logger = logging.getLogger(__name__)


class HFSyncModelCheckpoint(ModelCheckpoint):
    """`ModelCheckpoint` that also uploads checkpoints to a HF Hub model repo.

    Identical to `ModelCheckpoint` when `hf_repo_id` is None

    Only `last.ckpt` (the resume-critical file) is synced on every save; the
    top-k best checkpoints are pushed once, whenever training stops -- normal
    completion, a Ctrl+C, or a crash.
    """

    def __init__(self, *args, hf_repo_id: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.hf_repo_id = hf_repo_id
        self._upload_lock = threading.Lock()
        self._hydra_config_uploaded = False

    @override
    def _save_checkpoint(self, trainer: lightning.Trainer, filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)

        if self.hf_repo_id is None:
            return
        if not trainer.is_global_zero:
            return

        # Only "last.ckpt" is resume-critical; top-k "best" checkpoints sync at train end instead.
        if Path(filepath).name != f"{self.CHECKPOINT_NAME_LAST}{self.FILE_EXTENSION}":
            return

        self._upload_last_checkpoint_async(filepath)

    def _upload_last_checkpoint_async(self, filepath: str) -> None:
        if not self._upload_lock.acquire(blocking=False):
            logger.warning("Skipping HF Hub checkpoint upload: previous upload still running.")
            return
        thread = threading.Thread(
            target=self._upload_last_checkpoint, args=(filepath,), daemon=True
        )
        thread.start()

    def _hf_path_prefix(self) -> str:
        """The path this run already uses locally under `logs/`, reused as the upload prefix so the
        HF Hub repo mirrors `logs/` exactly -- one folder tree, one repo, one branch."""
        assert self.dirpath is not None
        return Path(self.dirpath).parent.relative_to(REPO_ROOTDIR).as_posix()

    def _upload_last_checkpoint(self, filepath: str) -> None:
        assert self.hf_repo_id is not None
        api = HfApi()
        try:
            prefix = self._hf_path_prefix()
            self._upload_hydra_config_once(api, self.hf_repo_id, prefix)
            api.upload_file(
                path_or_fileobj=filepath,
                path_in_repo=f"{prefix}/checkpoints/last.ckpt",
                repo_id=self.hf_repo_id,
                repo_type="model",
            )
        except Exception:
            # If we fail we do not crash, we will wait for the next save call
            logger.exception(f"Failed to upload checkpoint to HF Hub repo '{self.hf_repo_id}'.")
        finally:
            self._upload_lock.release()

    def _upload_hydra_config_once(self, api: HfApi, hf_repo_id: str, prefix: str) -> None:
        if self._hydra_config_uploaded:
            return

        if self.dirpath is None:
            return

        config_path = Path(self.dirpath).parent / ".hydra" / "config.yaml"
        if not config_path.exists():
            logger.debug(f"No .hydra/config.yaml found next to {self.dirpath}, skipping upload.")
            return

        api.upload_file(
            path_or_fileobj=str(config_path),
            path_in_repo=f"{prefix}/.hydra/config.yaml",
            repo_id=hf_repo_id,
            repo_type="model",
        )
        self._hydra_config_uploaded = True

    @override
    def on_train_end(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        super().on_train_end(trainer, pl_module)
        self._sync_best_checkpoints(trainer)

    @override
    def on_exception(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        exception: BaseException,
    ) -> None:
        # Covers both a manual Ctrl+C and a genuine crash: either way, training is ending
        # without ever reaching on_train_end, so the top-k checkpoints would otherwise never sync.
        super().on_exception(trainer, pl_module, exception)
        self._sync_best_checkpoints(trainer)

    def _sync_best_checkpoints(self, trainer: lightning.Trainer) -> None:
        if self.hf_repo_id is None:
            return
        if not trainer.is_global_zero:
            return

        with self._upload_lock:
            self._upload_best_checkpoints(self.hf_repo_id)

    def _upload_best_checkpoints(self, hf_repo_id: str) -> None:
        api = HfApi()
        prefix = self._hf_path_prefix()
        for filepath in self.best_k_models:
            api.upload_file(
                path_or_fileobj=filepath,
                path_in_repo=f"{prefix}/checkpoints/{Path(filepath).name}",
                repo_id=hf_repo_id,
                repo_type="model",
            )
