import importlib
import os
from logging import getLogger as get_logger
from pathlib import Path

import torch

logger = get_logger(__name__)


NETWORK_DATASETS_DIR: Path | None = (
    Path(os.environ["NETWORK_DIR"]) if "NETWORK_DIR" in os.environ else None
)
"""The (read-only) network directory that contains pre-downloaded datasets.

Set the `NETWORK_DIR` environment variable to point to your local datasets directory.
"""

REPO_ROOTDIR = Path(__file__).parent
"""The root directory of this repository on this machine."""

for level in range(5):
    if "README.md" in list(p.name for p in REPO_ROOTDIR.iterdir()):
        break
    REPO_ROOTDIR = REPO_ROOTDIR.parent


DATA_DIR = Path(os.environ.get("DATA_DIR", REPO_ROOTDIR / "data"))
"""Local directory where datasets should be extracted on this machine."""


def get_constant(*names: str):
    """Resolver for Hydra to get the value of a constant in this file."""
    assert names
    for name in names:
        if name in globals():
            obj = globals()[name]
            if obj is None:
                logger.debug(f"Value of {name} is None, moving on to the next value.")
                continue
            return obj
        parts = name.split(".")
        obj = importlib.import_module(parts[0])
        for part in parts[1:]:
            obj = getattr(obj, part)
        if obj is not None:
            return obj
        logger.debug(f"Value of {name} is None, moving on to the next value.")

    if len(names) == 1:
        raise RuntimeError(f"Could not find non-None value for name {names[0]}")
    raise RuntimeError(f"Could not find non-None value for names {names}")


NUM_WORKERS = (
    len(os.sched_getaffinity(0))
    if hasattr(os, "sched_getaffinity")
    else torch.multiprocessing.cpu_count()
)
"""Default number of workers to be used by dataloaders, based on the number of available CPUs."""
