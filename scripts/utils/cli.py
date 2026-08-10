from __future__ import annotations

import argparse
from pathlib import Path

from scripts.utils.episodes import DEFAULT_FRAME_SPEC


def checkpoint_args(required: bool = True) -> argparse.ArgumentParser:
    """`--ckpt-path` and `--seed`."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--ckpt-path",
        type=Path,
        required=required,
        help="Checkpoint to analyse. Its training config is read from the nearest .hydra/ "
        "directory above it.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for rollouts and any sampling. Episode N uses seed + N. (default: 42)",
    )
    return parser


def output_args() -> argparse.ArgumentParser:
    """Where output goes and how it is rendered."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Root for saved figures and reports. (default: scripts/figures/)",
    )
    parser.add_argument(
        "--run-label",
        type=str,
        default=None,
        help="Overrides the auto-derived filename prefix identifying this checkpoint/config.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Figure resolution. (default: 180)")
    parser.add_argument(
        "--show", action="store_true", help="Open each figure in a window as it is produced."
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Print the analysis to the terminal but do not save the .txt alongside the figure.",
    )
    return parser


def rollout_args(default_num_episodes: int = 5) -> argparse.ArgumentParser:
    """Live-rollout settings, including the zero-shot env sweep."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--env-id",
        type=str,
        nargs="+",
        default=None,
        help="Environments to roll out in, one run each. Defaults to the whole LockedRotation "
        "family (the training task plus its three zero-shot targets).",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=default_num_episodes,
        help=f"Episodes per environment. (default: {default_num_episodes})",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="Step budget per episode. Defaults to the checkpoint's own training-time rollout "
        "budget, which is usually far longer than the env's registered default.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Denoising steps per action chunk. Defaults to the policy's own setting.",
    )
    parser.add_argument(
        "--no-clamp-action",
        dest="clamp_action",
        action="store_false",
        help="Do not clamp actions to the action space bounds.",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default=None,
        choices=["human", "rgb_array"],
        help="Render the rollout. 'human' opens a viewer; 'rgb_array' is needed to record video.",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Record rollout videos into this directory.",
    )
    return parser


def dataset_args(required: bool = False) -> argparse.ArgumentParser:
    """Offline demonstration-file settings."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=required,
        default=None,
        help="Demonstration .h5 to read. Defaults to the conventional demo path for the env.",
    )
    parser.add_argument(
        "--episode-idx",
        type=int,
        default=0,
        help="Episode to analyse. Negative counts from the end. (default: 0)",
    )
    parser.add_argument(
        "--frames",
        type=str,
        default=DEFAULT_FRAME_SPEC,
        # argparse runs help through %-formatting, so every literal % has to be doubled -- including
        # the one inside the default value.
        help="Frames to sample, as percentages or indices, e.g. '0%%,50%%,100%%' or '0,10,-1'. "
        f"(default: {DEFAULT_FRAME_SPEC.replace('%', '%%')})",
    )
    return parser


def source_args(default: str = "dataset") -> argparse.ArgumentParser:
    """Chooses between offline episodes and live rollouts."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--source",
        choices=["dataset", "rollout"],
        default=default,
        help=f"Analyse recorded demonstrations or a live policy rollout. (default: {default})",
    )
    return parser
