import json
from pathlib import Path, PurePosixPath
from unittest.mock import MagicMock, patch

import pytest
import torch
from lightning.pytorch.trainer.states import TrainerFn

import policy.algorithms.callbacks.hf_sync_model_checkpoint as hf_sync_model_checkpoint
from policy.algorithms.callbacks.hf_sync_model_checkpoint import (
    RUN_STATUS_FILENAMES,
    HFSyncModelCheckpoint,
)

COMPLETED = RUN_STATUS_FILENAMES["completed"]
INTERRUPTED = RUN_STATUS_FILENAMES["interrupted"]

PREFIX = "logs/my-experiment/runs/2026-01-01/12-00-00"


def _mock_trainer(is_global_zero=True, **overrides):
    """A trainer that looks like a normal fitting run.

    Every attribute `_should_skip_hf_upload` reads has to be set explicitly: a bare `MagicMock`
    attribute is truthy, which would make the guard skip every upload and fail every test here.
    """
    trainer = MagicMock(
        is_global_zero=is_global_zero,
        loggers=[],
        global_step=10,
        current_epoch=2,
        fast_dev_run=False,
        overfit_batches=0,
        sanity_checking=False,
        received_sigterm=False,
        state=MagicMock(fn=TrainerFn.FITTING),
    )
    for name, value in overrides.items():
        setattr(trainer, name, value)
    return trainer


def _make_run_dirpath(tmp_path: Path, monkeypatch) -> Path:
    """Builds a fake local run directory (mirroring logs/<name>/runs/<date>/<time>/checkpoints) and
    points REPO_ROOTDIR at tmp_path, so `_hf_path_prefix()` resolves to `PREFIX`."""
    monkeypatch.setattr(hf_sync_model_checkpoint, "REPO_ROOTDIR", tmp_path)
    dirpath = tmp_path / PREFIX / "checkpoints"
    dirpath.mkdir(parents=True)
    return dirpath


def test_save_checkpoint_noop_when_repo_id_none(tmp_path):
    cb = HFSyncModelCheckpoint(dirpath=str(tmp_path), hf_repo_id=None)
    trainer = _mock_trainer()

    with patch("policy.algorithms.callbacks.hf_sync_model_checkpoint.HfApi") as mock_api_cls:
        cb._save_checkpoint(trainer, str(tmp_path / "last.ckpt"))

    trainer.save_checkpoint.assert_called_once()
    mock_api_cls.assert_not_called()


def test_save_checkpoint_skips_non_rank_zero(tmp_path):
    cb = HFSyncModelCheckpoint(dirpath=str(tmp_path), hf_repo_id="org/repo")
    trainer = _mock_trainer(is_global_zero=False)

    with patch.object(cb, "_upload_last_checkpoint_async") as mock_upload:
        cb._save_checkpoint(trainer, str(tmp_path / "last.ckpt"))

    mock_upload.assert_not_called()


def test_save_checkpoint_skips_non_last_file(tmp_path):
    cb = HFSyncModelCheckpoint(dirpath=str(tmp_path), hf_repo_id="org/repo")
    trainer = _mock_trainer()

    with patch.object(cb, "_upload_last_checkpoint_async") as mock_upload:
        cb._save_checkpoint(trainer, str(tmp_path / "step_000010.ckpt"))

    mock_upload.assert_not_called()


def test_save_checkpoint_triggers_upload_for_last_ckpt(tmp_path):
    cb = HFSyncModelCheckpoint(dirpath=str(tmp_path), hf_repo_id="org/repo")
    trainer = _mock_trainer()
    last_ckpt_path = str(tmp_path / "last.ckpt")

    with patch.object(cb, "_upload_last_checkpoint_async") as mock_upload:
        cb._save_checkpoint(trainer, last_ckpt_path)

    mock_upload.assert_called_once_with(last_ckpt_path)


def test_upload_last_checkpoint_async_busy_flag_skips(tmp_path, caplog):
    cb = HFSyncModelCheckpoint(dirpath=str(tmp_path), hf_repo_id="org/repo")
    cb._upload_lock.acquire()
    try:
        with patch("threading.Thread") as mock_thread_cls:
            cb._upload_last_checkpoint_async(str(tmp_path / "last.ckpt"))
        mock_thread_cls.assert_not_called()
        assert "previous upload still running" in caplog.text
    finally:
        cb._upload_lock.release()


def test_upload_last_checkpoint_uploads_config_once(tmp_path, monkeypatch):
    dirpath = _make_run_dirpath(tmp_path, monkeypatch)
    hydra_dir = dirpath.parent / ".hydra"
    hydra_dir.mkdir()
    (hydra_dir / "config.yaml").write_text("seed: 1\n")

    cb = HFSyncModelCheckpoint(dirpath=str(dirpath), hf_repo_id="org/repo")
    ckpt_path = dirpath / "last.ckpt"
    ckpt_path.write_text("fake checkpoint")

    with patch("policy.algorithms.callbacks.hf_sync_model_checkpoint.HfApi") as mock_api_cls:
        mock_api = mock_api_cls.return_value
        # _upload_last_checkpoint releases the lock itself; caller must hold it first
        # (normally done by _upload_last_checkpoint_async's non-blocking acquire).
        cb._upload_lock.acquire()
        cb._upload_last_checkpoint(str(ckpt_path))
        cb._upload_lock.acquire()
        cb._upload_last_checkpoint(str(ckpt_path))

    # config.yaml only uploaded on the first call, the checkpoint on both.
    assert mock_api.upload_file.call_count == 3
    uploaded_repo_paths = [c.kwargs["path_in_repo"] for c in mock_api.upload_file.call_args_list]
    assert uploaded_repo_paths.count(f"{PREFIX}/.hydra/config.yaml") == 1
    assert uploaded_repo_paths.count(f"{PREFIX}/checkpoints/last.ckpt") == 2
    assert not cb._upload_lock.locked()


def test_upload_last_checkpoint_swallows_exceptions(tmp_path, monkeypatch, caplog):
    dirpath = _make_run_dirpath(tmp_path, monkeypatch)
    cb = HFSyncModelCheckpoint(dirpath=str(dirpath), hf_repo_id="org/repo")

    with patch("policy.algorithms.callbacks.hf_sync_model_checkpoint.HfApi") as mock_api_cls:
        mock_api_cls.return_value.upload_file.side_effect = RuntimeError("network down")
        cb._upload_lock.acquire()
        cb._upload_last_checkpoint(str(dirpath / "last.ckpt"))  # must not raise

    assert not cb._upload_lock.locked()
    assert "Failed to upload checkpoint" in caplog.text


def test_on_train_end_uploads_best_k_models(tmp_path, monkeypatch):
    dirpath = _make_run_dirpath(tmp_path, monkeypatch)
    cb = HFSyncModelCheckpoint(dirpath=str(dirpath), hf_repo_id="org/repo")
    cb.best_k_models = {
        str(dirpath / "step_000010.ckpt"): torch.tensor(0.1),
        str(dirpath / "step_000020.ckpt"): torch.tensor(0.2),
    }
    trainer = _mock_trainer()

    with patch("policy.algorithms.callbacks.hf_sync_model_checkpoint.HfApi") as mock_api_cls:
        mock_api = mock_api_cls.return_value
        cb.on_train_end(trainer, MagicMock())

    # The two checkpoints, plus the run-status marker.
    assert mock_api.upload_file.call_count == 3
    uploaded_repo_paths = {c.kwargs["path_in_repo"] for c in mock_api.upload_file.call_args_list}
    assert uploaded_repo_paths == {
        f"{PREFIX}/checkpoints/step_000010.ckpt",
        f"{PREFIX}/checkpoints/step_000020.ckpt",
        f"{PREFIX}/{COMPLETED}",
    }


def test_on_train_end_noop_when_repo_id_none(tmp_path):
    cb = HFSyncModelCheckpoint(dirpath=str(tmp_path), hf_repo_id=None)
    trainer = _mock_trainer()

    with patch("policy.algorithms.callbacks.hf_sync_model_checkpoint.HfApi") as mock_api_cls:
        cb.on_train_end(trainer, MagicMock())

    mock_api_cls.assert_not_called()


def test_on_train_end_skips_non_rank_zero(tmp_path):
    cb = HFSyncModelCheckpoint(dirpath=str(tmp_path), hf_repo_id="org/repo")
    cb.best_k_models = {str(tmp_path / "step_000010.ckpt"): torch.tensor(0.1)}
    trainer = _mock_trainer(is_global_zero=False)

    with patch("policy.algorithms.callbacks.hf_sync_model_checkpoint.HfApi") as mock_api_cls:
        cb.on_train_end(trainer, MagicMock())

    mock_api_cls.assert_not_called()


def test_on_exception_uploads_best_k_models(tmp_path, monkeypatch):
    dirpath = _make_run_dirpath(tmp_path, monkeypatch)
    cb = HFSyncModelCheckpoint(dirpath=str(dirpath), hf_repo_id="org/repo")
    cb.best_k_models = {str(dirpath / "step_000010.ckpt"): torch.tensor(0.1)}
    trainer = _mock_trainer()

    with patch("policy.algorithms.callbacks.hf_sync_model_checkpoint.HfApi") as mock_api_cls:
        mock_api = mock_api_cls.return_value
        cb.on_exception(trainer, MagicMock(), KeyboardInterrupt())

    assert mock_api.upload_file.call_args_list[0].kwargs == {
        "path_or_fileobj": str(dirpath / "step_000010.ckpt"),
        "path_in_repo": f"{PREFIX}/checkpoints/step_000010.ckpt",
        "repo_id": "org/repo",
        "repo_type": "model",
    }


def test_on_exception_noop_when_repo_id_none(tmp_path):
    cb = HFSyncModelCheckpoint(dirpath=str(tmp_path), hf_repo_id=None)
    trainer = _mock_trainer()

    with patch("policy.algorithms.callbacks.hf_sync_model_checkpoint.HfApi") as mock_api_cls:
        cb.on_exception(trainer, MagicMock(), KeyboardInterrupt())

    mock_api_cls.assert_not_called()


def test_on_exception_skips_non_rank_zero(tmp_path):
    cb = HFSyncModelCheckpoint(dirpath=str(tmp_path), hf_repo_id="org/repo")
    cb.best_k_models = {str(tmp_path / "step_000010.ckpt"): torch.tensor(0.1)}
    trainer = _mock_trainer(is_global_zero=False)

    with patch("policy.algorithms.callbacks.hf_sync_model_checkpoint.HfApi") as mock_api_cls:
        cb.on_exception(trainer, MagicMock(), KeyboardInterrupt())

    mock_api_cls.assert_not_called()


def test_hf_path_prefix_honours_patched_repo_rootdir(tmp_path, monkeypatch):
    """The path mapping lives in policy.utils.hf_hub_utils, but the anchor must stay this module's.

    Tests patch `REPO_ROOTDIR` in the callback's namespace, so `_hf_path_prefix` has to pass it
    explicitly rather than let the helper look it up.
    """
    dirpath = _make_run_dirpath(tmp_path, monkeypatch)
    cb = HFSyncModelCheckpoint(dirpath=str(dirpath), hf_repo_id="org/repo")

    assert cb._hf_path_prefix() == PREFIX


def test_falsy_repo_id_disables_upload(tmp_path):
    """An empty HF_CHECKPOINT_REPO_ID resolves to "" rather than None, and must still mean
    "off"."""
    cb = HFSyncModelCheckpoint(dirpath=str(tmp_path), hf_repo_id="")
    trainer = _mock_trainer()

    with patch("policy.algorithms.callbacks.hf_sync_model_checkpoint.HfApi") as mock_api_cls:
        cb._save_checkpoint(trainer, str(tmp_path / "last.ckpt"))
        cb.on_train_end(trainer, MagicMock())

    mock_api_cls.assert_not_called()


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"fast_dev_run": True}, id="fast_dev_run"),
        pytest.param({"overfit_batches": 1}, id="overfit_batches"),
        pytest.param({"sanity_checking": True}, id="sanity_check"),
        pytest.param({"state": MagicMock(fn=TrainerFn.TESTING)}, id="not_fitting"),
    ],
)
def test_no_upload_for_throwaway_runs(tmp_path, monkeypatch, overrides):
    """Debug runs must not reach the Hub.

    The `TESTING` case is the one that matters beyond hygiene: `*__test.yaml` inherits
    `callbacks: default`, so `eval.py` builds this callback too, and its dirpath is not a training
    run directory.
    """
    dirpath = _make_run_dirpath(tmp_path, monkeypatch)
    cb = HFSyncModelCheckpoint(dirpath=str(dirpath), hf_repo_id="org/repo")
    cb.best_k_models = {str(dirpath / "step_000010.ckpt"): torch.tensor(0.1)}
    trainer = _mock_trainer(**overrides)

    with patch("policy.algorithms.callbacks.hf_sync_model_checkpoint.HfApi") as mock_api_cls:
        cb._save_checkpoint(trainer, str(dirpath / "last.ckpt"))
        cb.on_train_end(trainer, MagicMock())
        cb.on_exception(trainer, MagicMock(), KeyboardInterrupt())

    mock_api_cls.assert_not_called()


def test_on_train_end_uploads_completion_marker_last(tmp_path, monkeypatch):
    """The marker goes up after everything else, so its presence means the run synced fully."""
    dirpath = _make_run_dirpath(tmp_path, monkeypatch)
    cb = HFSyncModelCheckpoint(dirpath=str(dirpath), hf_repo_id="org/repo")
    cb.best_k_models = {str(dirpath / "step_000010.ckpt"): torch.tensor(0.1)}
    trainer = _mock_trainer()

    with patch("policy.algorithms.callbacks.hf_sync_model_checkpoint.HfApi") as mock_api_cls:
        mock_api = mock_api_cls.return_value
        cb.on_train_end(trainer, MagicMock())

    calls = mock_api.upload_file.call_args_list
    assert calls[-1].kwargs["path_in_repo"] == f"{PREFIX}/{COMPLETED}"

    uploaded = json.loads(calls[-1].kwargs["path_or_fileobj"].decode())
    # The status is the filename, so it must not be duplicated in the body.
    assert "status" not in uploaded
    assert uploaded["global_step"] == 10
    assert uploaded["epoch"] == 2
    assert uploaded["uploaded_checkpoints"] == ["step_000010.ckpt"]

    # Mirrored into logs/ too, so the local tree and the repo stay identical.
    assert json.loads((dirpath.parent / COMPLETED).read_text()) == uploaded


def test_on_exception_uses_the_interrupted_marker(tmp_path, monkeypatch):
    dirpath = _make_run_dirpath(tmp_path, monkeypatch)
    cb = HFSyncModelCheckpoint(dirpath=str(dirpath), hf_repo_id="org/repo")
    trainer = _mock_trainer()

    with patch("policy.algorithms.callbacks.hf_sync_model_checkpoint.HfApi") as mock_api_cls:
        mock_api = mock_api_cls.return_value
        cb.on_exception(trainer, MagicMock(), KeyboardInterrupt())

    last_call = mock_api.upload_file.call_args_list[-1]
    assert last_call.kwargs["path_in_repo"] == f"{PREFIX}/{INTERRUPTED}"
    assert (dirpath.parent / INTERRUPTED).exists()
    assert not (dirpath.parent / COMPLETED).exists()


def test_marker_uploaded_only_once(tmp_path, monkeypatch):
    """`on_exception` can follow `on_train_end` if teardown throws; the verdict must not flip."""
    dirpath = _make_run_dirpath(tmp_path, monkeypatch)
    cb = HFSyncModelCheckpoint(dirpath=str(dirpath), hf_repo_id="org/repo")
    trainer = _mock_trainer()

    with patch("policy.algorithms.callbacks.hf_sync_model_checkpoint.HfApi") as mock_api_cls:
        mock_api = mock_api_cls.return_value
        cb.on_train_end(trainer, MagicMock())
        cb.on_exception(trainer, MagicMock(), RuntimeError("teardown blew up"))

    markers = [
        c
        for c in mock_api.upload_file.call_args_list
        if PurePosixPath(c.kwargs["path_in_repo"]).name in RUN_STATUS_FILENAMES.values()
    ]
    assert len(markers) == 1
    assert markers[0].kwargs["path_in_repo"] == f"{PREFIX}/{COMPLETED}"


def test_final_sync_failure_does_not_raise(tmp_path, monkeypatch, caplog):
    """A network blip at train end must not crash the end of an otherwise successful run."""
    dirpath = _make_run_dirpath(tmp_path, monkeypatch)
    cb = HFSyncModelCheckpoint(dirpath=str(dirpath), hf_repo_id="org/repo")
    cb.best_k_models = {str(dirpath / "step_000010.ckpt"): torch.tensor(0.1)}
    trainer = _mock_trainer()

    with patch("policy.algorithms.callbacks.hf_sync_model_checkpoint.HfApi") as mock_api_cls:
        mock_api_cls.return_value.upload_file.side_effect = RuntimeError("network down")
        cb.on_train_end(trainer, MagicMock())  # must not raise

    assert not cb._upload_lock.locked()
    assert "Failed to sync the final checkpoints" in caplog.text
