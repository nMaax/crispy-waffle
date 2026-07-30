from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

import policy.algorithms.callbacks.hf_sync_model_checkpoint as hf_sync_model_checkpoint
from policy.algorithms.callbacks.hf_sync_model_checkpoint import HFSyncModelCheckpoint

PREFIX = "logs/my-experiment/runs/2026-01-01/12-00-00"


def _mock_trainer(is_global_zero=True):
    return MagicMock(is_global_zero=is_global_zero, loggers=[], global_step=10)


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

    assert mock_api.upload_file.call_count == 2
    uploaded_repo_paths = {c.kwargs["path_in_repo"] for c in mock_api.upload_file.call_args_list}
    assert uploaded_repo_paths == {
        f"{PREFIX}/checkpoints/step_000010.ckpt",
        f"{PREFIX}/checkpoints/step_000020.ckpt",
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
