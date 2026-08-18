import random
import subprocess
from dataclasses import dataclass, field
from logging import getLogger as get_logger
from typing import Any

logger = get_logger(__name__)


def get_git_branch() -> str | None:
    """Returns the name of the current active git branch, or None."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        branch = result.stdout.strip()
        return branch if branch and branch != "HEAD" else None
    except Exception:
        return None


@dataclass
class Config:
    """The options required for a run. This dataclass acts as a structure for the Hydra configs.

    For more info, see https://hydra.cc/docs/tutorials/structured_config/schema/
    """

    algorithm: Any
    """Configuration for the algorithm (a
    [LightningModule][lightning.pytorch.core.module.LightningModule]).

    It is suggested for this class to accept a `datamodule` and `network` as arguments. The
    instantiated datamodule and network will be passed to the algorithm's constructor.

    For more info, see the [instantiate_algorithm][policy.main.instantiate_algorithm] function.
    """

    datamodule: Any | None = None
    """Configuration for the datamodule (dataset + transforms + dataloader creation).

    This should normally create a [LightningDataModule][lightning.pytorch.core.datamodule.LightningDataModule].
    See the [MNISTDataModule][policy.datamodules.image_classification.mnist.MNISTDataModule] for an example.
    """

    trainer: dict = field(default_factory=dict)
    """Configuration for the 'Trainer'."""

    log_level: str = "info"
    """Logging level."""

    seed: int = field(default_factory=lambda: random.randint(0, int(1e5)))
    """Random seed for reproducibility.

    If None, a random seed is generated.
    """

    name: str = "default"
    """Name for the experiment."""

    branch: str | None = field(default_factory=get_git_branch)
    """Git branch name for tracking ablations and experiment variations."""

    debug: bool = False
    """Debug mode flag."""

    verbose: bool = False
    """Verbose mode flag."""

    ckpt_path: str | None = None
    """Path to a checkpoint to load the training state and resume the training run.

    This is the same as the `ckpt_path` argument in the `lightning.Trainer.fit` method.
    """

    finetuning_ckpt_path: str | None = None
    """Path to a checkpoint to load the weights for fine-tuning.

    If provided, the weights will be loaded into the algorithm, and the trainer's `ckpt_path`
    will be set to None to avoid resuming the training state.
    """

    hf_checkpoint_repo_id: str | None = None
    """HF Hub model repo id to mirror checkpoints to, at the same path they use locally.

    None will ignore HFH and save locally only. Resuming from a synced checkpoint is manual:
    download the one you mean, then pass its local path as `ckpt_path`.
    """

    render: Any | None = None
    """Optional render mode."""

    validate_at_end: bool = False
    """Whether to run a final validation loop after training completes."""
