import datetime
import json
import logging
import threading
from pathlib import Path

import lightning
from huggingface_hub import HfApi
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.trainer.states import TrainerFn
from typing_extensions import override

from policy.utils.env_vars import REPO_ROOTDIR
from policy.utils.hf_hub_utils import MODEL_REPO_TYPE, repo_relative_path

logger = logging.getLogger(__name__)

RUN_STATUS_FILENAMES = {
    "completed": "RUN_COMPLETED.json",
    "interrupted": "RUN_INTERRUPTED.json",
}


class HFSyncModelCheckpoint(ModelCheckpoint):
    """`ModelCheckpoint` that also uploads checkpoints to a HF Hub model repo.

    Identical to `ModelCheckpoint` when `hf_repo_id` is unset.

    Only `last.ckpt` (the resume-critical file) is synced on every save; the
    top-k best checkpoints are pushed once, whenever training stops
    It also uploads a `RUN_COMPLETED.json`/`RUN_INTERRUPTED.json` marking
    if a run properly completed the loop or if it has been interrupted.
    """

    def __init__(self, *args, hf_repo_id: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.hf_repo_id = hf_repo_id
        self._upload_lock = threading.Lock()
        self._hydra_config_uploaded = False
        self._marker_uploaded = False

    @override
    def _save_checkpoint(self, trainer: lightning.Trainer, filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)

        if not self.hf_repo_id:
            return
        if not trainer.is_global_zero:
            return
        if self._should_skip_hf_upload(trainer):
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
        HF Hub repo mirrors `logs/` exactly."""
        assert self.dirpath is not None
        return repo_relative_path(Path(self.dirpath).parent, anchor=REPO_ROOTDIR)

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
                repo_type=MODEL_REPO_TYPE,
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
            repo_type=MODEL_REPO_TYPE,
        )
        self._hydra_config_uploaded = True

    @override
    def on_train_end(
        self, trainer: lightning.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        super().on_train_end(trainer, pl_module)
        self._sync_best_checkpoints(trainer, status="completed")

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
        self._sync_best_checkpoints(trainer, status="interrupted")

    def _sync_best_checkpoints(self, trainer: lightning.Trainer, status: str) -> None:
        if not self.hf_repo_id:
            return
        if not trainer.is_global_zero:
            return
        if self._should_skip_hf_upload(trainer):
            return

        # Blocking, unlike the `last.ckpt` path: this waits out any in-flight async upload, which is
        # what lets the marker be written last and mean "everything else already landed".
        with self._upload_lock:
            try:
                api = HfApi()
                prefix = self._hf_path_prefix()
                uploaded = self._upload_best_checkpoints(api, self.hf_repo_id, prefix)
                self._upload_run_status(api, self.hf_repo_id, prefix, trainer, status, uploaded)
            except Exception:
                # A network blip here must not crash the end of an otherwise successful run, the
                # same way it doesn't on the `last.ckpt` path.
                logger.exception(
                    f"Failed to sync the final checkpoints to HF Hub repo '{self.hf_repo_id}'."
                )

    def _should_skip_hf_upload(self, trainer: lightning.Trainer) -> bool:
        """Whether this run is too throwaway to be worth putting on the Hub.

        Mirrors `ModelCheckpoint._should_skip_saving_checkpoint`.
        """
        return (
            bool(getattr(trainer, "fast_dev_run", False))
            or bool(getattr(trainer, "overfit_batches", 0))
            or trainer.sanity_checking
            or trainer.state.fn != TrainerFn.FITTING
        )

    def _upload_best_checkpoints(self, api: HfApi, hf_repo_id: str, prefix: str) -> list[str]:
        """Uploads the top-k checkpoints, returning their filenames."""
        names = []
        for filepath in self.best_k_models:
            name = Path(filepath).name
            api.upload_file(
                path_or_fileobj=filepath,
                path_in_repo=f"{prefix}/checkpoints/{name}",
                repo_id=hf_repo_id,
                repo_type=MODEL_REPO_TYPE,
            )
            names.append(name)
        return names

    def _upload_run_status(
        self,
        api: HfApi,
        hf_repo_id: str,
        prefix: str,
        trainer: lightning.Trainer,
        status: str,
        uploaded: list[str],
    ) -> None:
        """Records how this run ended, so an abandoned run on the Hub is recognisable as one."""
        if self._marker_uploaded:
            return

        filename = RUN_STATUS_FILENAMES[status]
        marker = json.dumps(self._run_details(trainer, uploaded), indent=2)
        self._write_run_status_locally(filename, marker)
        api.upload_file(
            path_or_fileobj=marker.encode(),
            path_in_repo=f"{prefix}/{filename}",
            repo_id=hf_repo_id,
            repo_type=MODEL_REPO_TYPE,
        )
        self._marker_uploaded = True

    def _run_details(self, trainer: lightning.Trainer, uploaded: list[str]) -> dict[str, object]:
        """The marker's contents.

        Containing some useful metadata.
        """
        score = self.best_model_score
        return {
            "global_step": int(trainer.global_step),
            "epoch": int(trainer.current_epoch),
            "monitor": str(self.monitor) if self.monitor is not None else None,
            "best_model_path": Path(self.best_model_path).name if self.best_model_path else None,
            "best_model_score": float(score) if score is not None else None,
            # Distinguishes "the scheduler asked us to stop" from "training genuinely finished".
            "received_sigterm": bool(getattr(trainer, "received_sigterm", False)),
            "uploaded_checkpoints": uploaded,
            "finished_at": datetime.datetime.now(datetime.UTC).isoformat(),
        }

    def _write_run_status_locally(self, filename: str, marker: str) -> None:
        """Mirrors the marker into `logs/`, keeping the local tree and the repo identical."""
        if self.dirpath is None:
            return
        try:
            (Path(self.dirpath).parent / filename).write_text(marker, encoding="utf-8")
        except OSError as error:
            logger.warning(f"Could not write {filename} next to the checkpoints: {error}")
