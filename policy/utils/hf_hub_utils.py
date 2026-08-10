from __future__ import annotations

import os
from collections.abc import Iterable
from logging import getLogger as get_logger
from pathlib import Path

logger = get_logger(__name__)

MODEL_REPO_TYPE = "model"
"""HF Hub repo type for checkpoints."""

DATASET_REPO_TYPE = "dataset"
"""HF Hub repo type for trajectory datasets."""

CHECKPOINTS_DIRNAME = "checkpoints"
"""Directory a run's checkpoints sit in, per `dirpath` in `trainer/callbacks/default.yaml`."""


def repo_relative_path(local_path: Path | str, *, anchor: Path | str) -> str:
    """The HF Hub path for a local path, i.e. its location relative to `anchor`."""
    resolved = Path(local_path).expanduser().resolve()
    resolved_anchor = Path(anchor).expanduser().resolve()
    try:
        return resolved.relative_to(resolved_anchor).as_posix()
    except ValueError as error:
        raise ValueError(
            f"Cannot map '{resolved}' to a HF Hub path: it is outside '{resolved_anchor}'. The "
            "repo layout mirrors that directory exactly, so files have to live under it."
        ) from error


def fetch_missing(
    paths: Iterable[Path | str],
    *,
    repo_id: str,
    repo_type: str,
    anchor: Path | str,
) -> list[Path]:
    """Downloads whichever of `paths` are not already on disk, returning the ones fetched."""
    from huggingface_hub import hf_hub_download

    fetched = []
    for candidate in paths:
        path = Path(candidate)
        if path.exists():
            continue

        relative = repo_relative_path(path, anchor=anchor)
        logger.info(f"'{path}' is missing locally; fetching '{relative}' from '{repo_id}'.")
        try:
            hf_hub_download(
                repo_id=repo_id,
                repo_type=repo_type,
                filename=relative,
                local_dir=str(Path(anchor)),
            )
        except Exception as error:
            raise FileNotFoundError(
                f"Could not fetch '{path}' from the HF Hub. Looked for '{relative}' in repo "
                f"'{repo_id}' ({repo_type}) and got: {error}. Check that the repo id is right, "
                "that HF_TOKEN grants access to it if it is private, and that the file was "
                "actually uploaded."
            ) from error
        fetched.append(path)
    return fetched


def default_checkpoint_repo_id() -> str | None:
    """`HF_CHECKPOINT_REPO_ID`, or None when unset or empty."""
    return os.environ.get("HF_CHECKPOINT_REPO_ID") or None


def default_dataset_repo_id() -> str | None:
    """`HF_DATASET_REPO`, or None when unset or empty."""
    return os.environ.get("HF_DATASET_REPO") or None


def _repo_root() -> Path:
    """The anchor checkpoint paths are relative to."""
    from policy.utils import env_vars

    return env_vars.REPO_ROOTDIR


def run_dir_of(ckpt_path: Path | str) -> Path:
    """The Hydra output directory a checkpoint belongs to."""
    path = Path(ckpt_path)
    if path.parent.name != CHECKPOINTS_DIRNAME:
        raise ValueError(
            f"Cannot tell which run '{path}' belongs to: a checkpoint is expected to sit in a "
            f"'{CHECKPOINTS_DIRNAME}/' directory inside its Hydra output directory, but its parent "
            f"is '{path.parent.name or path.parent}'. Move it under "
            f"<run-dir>/{CHECKPOINTS_DIRNAME}/, as a training run would."
        )
    return path.parent.parent


def hydra_config_path_of(ckpt_path: Path | str) -> Path:
    """The Hydra config snapshot for the run that produced a checkpoint."""
    return run_dir_of(ckpt_path) / ".hydra" / "config.yaml"


def ensure_checkpoint(ckpt_path: Path | str, repo_id: str | None) -> Path:
    """Downloads a checkpoint from the HF Hub checkpoint repo if it is missing locally."""
    path = Path(ckpt_path)
    if path.exists():
        return path
    if not repo_id:
        return path

    anchor = _repo_root()
    fetch_missing([path], repo_id=repo_id, repo_type=MODEL_REPO_TYPE, anchor=anchor)
    _fetch_hydra_config(path, repo_id, anchor)

    if not path.exists():
        raise FileNotFoundError(
            f"The HF Hub download reported success but '{path}' still does not exist. The repo "
            "layout may not mirror 'logs/'."
        )
    return path


def _fetch_hydra_config(ckpt_path: Path, repo_id: str, anchor: Path) -> None:
    """Best-effort download of the run's Hydra config snapshot."""
    try:
        fetch_missing(
            [hydra_config_path_of(ckpt_path)],
            repo_id=repo_id,
            repo_type=MODEL_REPO_TYPE,
            anchor=anchor,
        )
    except Exception as error:
        logger.warning(
            f"Fetched the checkpoint but not its '.hydra/config.yaml' ({error}). Anything that "
            "needs the run's training settings -- the resume seed check, the analysis scripts -- "
            "will behave as if the snapshot were absent."
        )
