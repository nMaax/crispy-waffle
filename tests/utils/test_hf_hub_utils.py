"""Tests for the shared HF Hub fetch and the local<->repo path mapping.

Mirrors `TestManiSkillDataModuleHFFetch` in `tests/datamodules/test_trajectory_datamodule.py`:
`huggingface_hub.hf_hub_download` is patched at its source (which works because
`policy.utils.hf_hub_utils` imports it lazily), and the fake download materialises real files so the
relative-path round trip is actually proven rather than asserted on call arguments alone.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from policy.utils import env_vars, hf_hub_utils
from policy.utils.hf_hub_utils import (
    default_checkpoint_repo_id,
    ensure_checkpoint,
    fetch_missing,
    hydra_config_path_of,
    repo_relative_path,
    run_dir_of,
)

RUN_PREFIX = "logs/my-experiment/runs/2026-01-01/12-00-00"
CKPT_RELATIVE = f"{RUN_PREFIX}/checkpoints/last.ckpt"
CONFIG_RELATIVE = f"{RUN_PREFIX}/.hydra/config.yaml"


@pytest.fixture
def repo_root(tmp_path, monkeypatch) -> Path:
    """A fake repo root, so checkpoint paths resolve inside it without touching the real one."""
    monkeypatch.setattr(env_vars, "REPO_ROOTDIR", tmp_path)
    return tmp_path


def _fake_download(filename, local_dir, **_kwargs):
    """Stands in for `hf_hub_download`, materialising the file at the path it claims to write."""
    target = Path(local_dir) / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("downloaded", encoding="utf-8")
    return str(target)


class TestRepoRelativePath:
    def test_maps_path_under_anchor(self, tmp_path):
        assert repo_relative_path(tmp_path / CKPT_RELATIVE, anchor=tmp_path) == CKPT_RELATIVE

    def test_rejects_path_outside_anchor(self, tmp_path):
        outside = tmp_path.parent / "elsewhere" / "last.ckpt"
        with pytest.raises(ValueError, match="outside") as error:
            repo_relative_path(outside, anchor=tmp_path)
        # Both paths are named, so the mistake is obvious without re-deriving anything.
        assert str(outside) in str(error.value)
        assert str(tmp_path) in str(error.value)

    def test_accepts_relative_input(self, tmp_path, monkeypatch):
        """CLI values like `ckpt_path=logs/...` are relative; both sides must be resolved."""
        (tmp_path / RUN_PREFIX / "checkpoints").mkdir(parents=True)
        monkeypatch.chdir(tmp_path)
        assert repo_relative_path(CKPT_RELATIVE, anchor=tmp_path) == CKPT_RELATIVE

    def test_anchor_need_not_be_the_repo_root(self, tmp_path):
        """The mapping is anchor-agnostic, which is what lets datasets reuse it."""
        demos = tmp_path / "demos"
        trajectory = demos / "StackCube-v1" / "motionplanning" / "trajectory.h5"
        expected = "StackCube-v1/motionplanning/trajectory.h5"
        assert repo_relative_path(trajectory, anchor=demos) == expected


class TestFetchMissing:
    def test_skips_files_already_present(self, tmp_path):
        existing = tmp_path / "a.bin"
        existing.write_text("here")
        with patch("huggingface_hub.hf_hub_download") as mock_download:
            assert fetch_missing(
                [existing], repo_id="org/repo", repo_type="model", anchor=tmp_path
            ) == []
        mock_download.assert_not_called()

    def test_fetches_only_what_is_missing(self, tmp_path):
        present = tmp_path / "present.bin"
        present.write_text("here")
        absent = tmp_path / "nested" / "absent.bin"

        with patch(
            "huggingface_hub.hf_hub_download", side_effect=_fake_download
        ) as mock_download:
            fetched = fetch_missing(
                [present, absent], repo_id="org/repo", repo_type="model", anchor=tmp_path
            )

        assert fetched == [absent]
        assert mock_download.call_count == 1
        assert mock_download.call_args.kwargs["filename"] == "nested/absent.bin"

    def test_passes_through_the_repo_type(self, tmp_path):
        """A dataset fetch differs from a checkpoint fetch only in this argument."""
        with patch(
            "huggingface_hub.hf_hub_download", side_effect=_fake_download
        ) as mock_download:
            fetch_missing(
                [tmp_path / "x.h5"],
                repo_id="org/demos",
                repo_type=hf_hub_utils.DATASET_REPO_TYPE,
                anchor=tmp_path,
            )
        assert mock_download.call_args.kwargs["repo_type"] == "dataset"

    def test_raises_actionable_error_on_failure(self, tmp_path):
        with patch("huggingface_hub.hf_hub_download", side_effect=RuntimeError("404")):
            with pytest.raises(FileNotFoundError) as error:
                fetch_missing(
                    [tmp_path / "nested" / "x.bin"],
                    repo_id="org/repo",
                    repo_type="model",
                    anchor=tmp_path,
                )
        message = str(error.value)
        assert "org/repo" in message
        assert "nested/x.bin" in message
        assert "HF_TOKEN" in message


class TestRunLayoutHelpers:
    def test_run_dir_is_the_parent_of_checkpoints(self, tmp_path):
        assert run_dir_of(tmp_path / CKPT_RELATIVE) == tmp_path / RUN_PREFIX

    def test_run_dir_handles_the_multirun_job_level(self, tmp_path):
        """A multirun inserts a `<job>` level, which *is* that job's Hydra output dir.

        Verified against the real layout on the training machine: `<job>/` holds `checkpoints/` and
        `.hydra/` as siblings, exactly like a single run's `<time>/` does. So the extra level shifts
        nothing -- what matters is that `checkpoints/` is a direct child of the output dir.
        """
        job_dir = tmp_path / "logs/my-experiment/multiruns/2026-01-01/12-00-00/0"
        assert run_dir_of(job_dir / "checkpoints" / "last.ckpt") == job_dir
        assert (
            hydra_config_path_of(job_dir / "checkpoints" / "last.ckpt")
            == job_dir / ".hydra" / "config.yaml"
        )

    def test_hydra_config_sits_beside_checkpoints(self, tmp_path):
        assert hydra_config_path_of(tmp_path / CKPT_RELATIVE) == tmp_path / CONFIG_RELATIVE

    def test_rejects_checkpoint_outside_a_checkpoints_dir(self, tmp_path):
        """A hand-placed `logs/last.ckpt` has no run dir; say so instead of guessing one."""
        with pytest.raises(ValueError, match="checkpoints") as error:
            run_dir_of(tmp_path / "logs" / "last.ckpt")
        assert "'logs'" in str(error.value)


class TestDefaultCheckpointRepoId:
    def test_reads_env_at_call_time(self, monkeypatch):
        monkeypatch.setenv("HF_CHECKPOINT_REPO_ID", "org/checkpoints")
        assert default_checkpoint_repo_id() == "org/checkpoints"

    def test_empty_value_is_unset(self, monkeypatch):
        """`${oc.env:HF_CHECKPOINT_REPO_ID,null}` yields "" for an empty export, not None."""
        monkeypatch.setenv("HF_CHECKPOINT_REPO_ID", "")
        assert default_checkpoint_repo_id() is None

    def test_missing_value_is_none(self, monkeypatch):
        monkeypatch.delenv("HF_CHECKPOINT_REPO_ID", raising=False)
        assert default_checkpoint_repo_id() is None


class TestEnsureCheckpoint:
    def test_noop_without_repo_id(self, repo_root):
        """A missing checkpoint and no repo stays a plain local problem, with no network call."""
        missing = repo_root / CKPT_RELATIVE
        with patch("huggingface_hub.hf_hub_download") as mock_download:
            assert ensure_checkpoint(missing, None) == missing
        mock_download.assert_not_called()

    def test_noop_for_empty_repo_id(self, repo_root):
        missing = repo_root / CKPT_RELATIVE
        with patch("huggingface_hub.hf_hub_download") as mock_download:
            ensure_checkpoint(missing, "")
        mock_download.assert_not_called()

    def test_noop_when_file_exists(self, repo_root):
        """The local-existence check comes first, which is what keeps offline machines working."""
        existing = repo_root / CKPT_RELATIVE
        existing.parent.mkdir(parents=True)
        existing.write_text("already here")

        with patch("huggingface_hub.hf_hub_download") as mock_download:
            ensure_checkpoint(existing, "org/checkpoints")
        mock_download.assert_not_called()

    def test_downloads_checkpoint_and_run_config(self, repo_root):
        """The anchor comes from `env_vars.REPO_ROOTDIR`, looked up at call time."""
        ckpt = repo_root / CKPT_RELATIVE

        with patch(
            "huggingface_hub.hf_hub_download", side_effect=_fake_download
        ) as mock_download:
            assert ensure_checkpoint(ckpt, "org/checkpoints") == ckpt

        assert ckpt.exists()
        assert (repo_root / CONFIG_RELATIVE).exists()
        assert [c.kwargs["filename"] for c in mock_download.call_args_list] == [
            CKPT_RELATIVE,
            CONFIG_RELATIVE,
        ]
        for call in mock_download.call_args_list:
            assert call.kwargs["repo_id"] == "org/checkpoints"
            assert call.kwargs["repo_type"] == "model"
            assert Path(call.kwargs["local_dir"]) == repo_root

    def test_skips_run_config_when_already_local(self, repo_root):
        ckpt = repo_root / CKPT_RELATIVE
        config = repo_root / CONFIG_RELATIVE
        config.parent.mkdir(parents=True)
        config.write_text("seed: 1\n")

        with patch(
            "huggingface_hub.hf_hub_download", side_effect=_fake_download
        ) as mock_download:
            ensure_checkpoint(ckpt, "org/checkpoints")

        assert mock_download.call_count == 1
        assert config.read_text() == "seed: 1\n"

    def test_tolerates_missing_run_config(self, repo_root, caplog):
        """Runs synced before the config upload existed have no snapshot; that is not fatal."""
        ckpt = repo_root / CKPT_RELATIVE

        def only_the_checkpoint(filename, local_dir, **kwargs):
            if filename != CKPT_RELATIVE:
                raise RuntimeError("404 not found")
            return _fake_download(filename, local_dir, **kwargs)

        with patch("huggingface_hub.hf_hub_download", side_effect=only_the_checkpoint):
            ensure_checkpoint(ckpt, "org/checkpoints")  # must not raise

        assert ckpt.exists()
        assert not (repo_root / CONFIG_RELATIVE).exists()
        assert "not its '.hydra/config.yaml'" in caplog.text

    def test_raises_when_checkpoint_missing_remotely(self, repo_root):
        with patch("huggingface_hub.hf_hub_download", side_effect=RuntimeError("404")):
            with pytest.raises(FileNotFoundError, match="org/checkpoints") as error:
                ensure_checkpoint(repo_root / CKPT_RELATIVE, "org/checkpoints")
        assert CKPT_RELATIVE in str(error.value)

    def test_raises_when_download_leaves_file_missing(self, repo_root):
        """A download that silently writes nowhere must still be reported as a miss."""
        with patch("huggingface_hub.hf_hub_download"):  # no-op: materialises nothing
            with pytest.raises(FileNotFoundError, match="still does not exist"):
                ensure_checkpoint(repo_root / CKPT_RELATIVE, "org/checkpoints")

    def test_raises_for_checkpoint_outside_repo(self, repo_root):
        outside = repo_root.parent / "elsewhere" / "last.ckpt"
        with patch("huggingface_hub.hf_hub_download") as mock_download:
            with pytest.raises(ValueError, match="outside"):
                ensure_checkpoint(outside, "org/checkpoints")
        mock_download.assert_not_called()

    def test_repo_types(self):
        assert hf_hub_utils.MODEL_REPO_TYPE == "model"
        assert hf_hub_utils.DATASET_REPO_TYPE == "dataset"
