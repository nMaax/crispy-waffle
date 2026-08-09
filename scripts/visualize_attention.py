"""Visualizes attention weights from a GoalConditionedDiffusionPolicy checkpoint's embedder:
`SelfAttentionEmbedder`'s self-attention across obs tokens, and/or `AttentionPooling`'s
learned-query attention that collapses the token sequence into one vector. Both are optional and
independent -- whichever is present in the loaded checkpoint's embedder gets visualized.

Both modules discard their attention weights by default (`need_weights=False` hardcoded in both
`forward()`s), so this hooks the underlying `nn.MultiheadAttention` submodule directly to recover
them, without altering either module's forward pass at all.

Replays a single HDF5 demonstration episode (mirrors `scripts/visualize_embeddings.py`), sampling a
handful of frames (`--frames`).
"""

import argparse
import contextlib
import json
import warnings
from collections.abc import Callable, Generator
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from policy.algorithms.goal_conditioned_diffusion_policy import GoalConditionedDiffusionPolicy
from policy.algorithms.networks.pooling import AttentionPooling
from policy.algorithms.networks.self_attention_embedder import SelfAttentionEmbedder
from policy.algorithms.tokenizers import PerObjectStateTokenizer
from policy.transforms import observation_pipeline
from policy.utils import cat_dicts, get_batch_size, map_leaves, recursive_index, to_tensor
from policy.utils.checkpoint_utils import load_goal_conditioned_diffusion_policy
from policy.utils.h5_utils import load_h5_data, peek_trajectory_is_dataset
from policy.utils.typing_utils import RawTree, TensorTree

DARK_RCPARAMS = {
    "figure.facecolor": "#0f172a",
    "axes.facecolor": "#1e293b",
    "text.color": "#f8fafc",
    "axes.labelcolor": "#94a3b8",
    "xtick.color": "#64748b",
    "ytick.color": "#64748b",
    "grid.color": "#334155",
    "grid.alpha": 0.5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="logs/GoalConditionedDiffusionPolicyMLP__StackCube-v1__default__train/runs/2026-07-07/16-54-32/checkpoints/last.ckpt",
        help="Path to the GoalConditionedDiffusionPolicy checkpoint.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=str(
            Path.home()
            / ".maniskill/demos/StackCubeLockedRotation-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cuda.h5"
        ),
        help="Path to the HDF5 dataset file.",
    )
    parser.add_argument(
        "--episode_idx", type=int, default=0, help="Index of the target episode to visualize."
    )
    parser.add_argument(
        "--frames",
        type=str,
        default="0%,25%,50%,75%,100%",
        help="Comma-separated frame indices or percentages (e.g. 0%%,25%%,50%%,75%%,100%%).",
    )
    parser.add_argument(
        "--goal_frame_idx",
        type=int,
        default=-1,
        help="Frame index used as the (fixed) goal for every sampled frame. Default: last frame.",
    )
    parser.add_argument(
        "--head_agg",
        type=str,
        default="mean",
        choices=["mean", "max"],
        help="How to collapse attention heads in the default figures.",
    )
    parser.add_argument(
        "--show_heads",
        action="store_true",
        default=False,
        help="Also emit a per-head attention breakdown figure.",
    )
    parser.add_argument(
        "--save_path_prefix",
        type=str,
        default=None,
        help="Prefix for saved figures under scripts/figures/. Default: derived from ckpt_path.",
    )
    parser.add_argument(
        "--show", action="store_true", default=False, help="Display plots interactively too."
    )
    return parser.parse_args()


def parse_frames(frames_str: str, seq_len: int) -> list[tuple[int, str]]:
    """Parses a comma-separated list of percentages ("25%") or absolute (possibly negative)
    indices into `(frame_idx, display_label)` pairs, clamped to `[0, seq_len - 1]`."""
    frames: list[tuple[int, str]] = []
    for f_str in frames_str.split(","):
        f_str = f_str.strip()
        if not f_str:
            continue
        if f_str.endswith("%"):
            try:
                pct = float(f_str[:-1])
                idx = int(round((pct / 100.0) * (seq_len - 1)))
                frames.append((max(0, min(idx, seq_len - 1)), f_str))
            except ValueError:
                pass
        else:
            try:
                idx = int(f_str)
                if idx < 0:
                    idx = seq_len + idx
                frames.append((max(0, min(idx, seq_len - 1)), f"t={idx}"))
            except ValueError:
                pass
    if not frames:
        raise ValueError(f"--frames={frames_str!r} did not yield any valid frame indices.")
    return frames


def load_ckpt_config(ckpt_path: Path) -> DictConfig | None:
    """Loads the checkpoint's saved Hydra run config, if present.

    Returns None (with a printed warning) rather than raising: env_id/obs-transform flags all have
    sensible fallbacks here, unlike the live-rollout script where the env config is strictly
    required.
    """
    config_file = ckpt_path.parent.parent / ".hydra" / "config.yaml"
    if not config_file.exists():
        warnings.warn(
            f"No saved Hydra run config found at {config_file}; falling back to default "
            "env_id/obs-transform resolution.",
            stacklevel=2,
        )
        return None
    cfg = OmegaConf.load(config_file)
    if not isinstance(cfg, DictConfig):
        warnings.warn(f"Expected a DictConfig at {config_file}, got {type(cfg).__name__}.")
        return None
    return cfg


def resolve_env_id(cfg: DictConfig | None, dataset_path: Path) -> str:
    """Prefers the checkpoint's own training env_id (authoritative for which Canonicalizer
    parser/STATE_SCHEMA applies), falling back to the dataset's sidecar JSON, then the dataset
    path's directory name -- mirrors `scripts/visualize_embeddings.py`'s fallback chain."""
    if cfg is not None:
        env_id = cfg.get("datamodule", {}).get("env_id", None)
        if env_id:
            return str(env_id)

    json_path = dataset_path.with_suffix(".json")
    if json_path.exists():
        with open(json_path) as f:
            meta = json.load(f)
        env_id = meta.get("env_info", {}).get("env_id")
        if env_id:
            return str(env_id)

    return dataset_path.parent.parent.name


def resolve_obs_transform(
    cfg: DictConfig | None, dataset_path: Path, env_id: str
) -> Callable[[TensorTree], TensorTree]:
    """Rebuilds the exact obs transform this checkpoint was trained with (deflatten + canonicalize
    + optional no-proprio-vel), reading flags off the saved datamodule config and defaulting to
    `TrajectoryDataModule`'s own defaults if missing."""
    canonicalize, as_dict, no_proprio_vel = True, True, False
    if cfg is not None:
        dm_cfg = cfg.get("datamodule", {})
        canonicalize = bool(dm_cfg.get("canonicalize", canonicalize))
        as_dict = bool(dm_cfg.get("as_dict", as_dict))
        no_proprio_vel = bool(dm_cfg.get("no_proprio_vel", no_proprio_vel))

    is_flat = peek_trajectory_is_dataset(dataset_path, dimension_key="obs")
    return observation_pipeline(
        env_id=env_id,
        is_flat=is_flat,
        canonicalize=canonicalize,
        as_dict=as_dict,
        no_proprio_vel=no_proprio_vel,
    )


def load_episode_obs(dataset_path: Path, episode_idx: int) -> tuple[str, RawTree, int]:
    """Resolves `traj_<episode_idx>` (or the episode_idx-th trajectory key if that exact name is
    absent) and loads its raw `obs` -- an ndarray if flat, or a nested dict of ndarrays if the
    dataset stores structured `state_dict` observations.

    Mirrors `scripts/visualize_embeddings.py`'s traj_key resolution.
    """
    with h5py.File(dataset_path, "r") as f:
        traj_keys = sorted(
            (k for k in f.keys() if k.startswith("traj_")), key=lambda x: int(x.split("_")[1])
        )
        if not traj_keys:
            raise ValueError(f"No 'traj_*' episodes found in {dataset_path}.")

        target_key = f"traj_{episode_idx}"
        if target_key not in traj_keys:
            if episode_idx >= len(traj_keys) or episode_idx < -len(traj_keys):
                raise IndexError(f"episode_idx={episode_idx} is out of bounds.")
            target_key = traj_keys[episode_idx]

        target_grp = f[target_key]
        if not isinstance(target_grp, h5py.Group):
            raise TypeError(f"Expected an HDF5 group at {target_key}.")

        obs_node = target_grp["obs"]
        obs_tree: RawTree
        if isinstance(obs_node, h5py.Dataset):
            obs_array = np.asarray(obs_node)
            obs_tree = obs_array
            seq_len = obs_array.shape[0]
        elif isinstance(obs_node, h5py.Group):
            obs_dict = load_h5_data(obs_node)
            obs_tree = obs_dict
            seq_len = len(next(iter(obs_dict.values())))
        else:
            raise TypeError(f"Unexpected obs node type: {type(obs_node)}")

    return target_key, obs_tree, seq_len


def window_indices(end_idx: int, obs_horizon: int) -> list[int]:
    """`[end_idx - obs_horizon + 1, ..., end_idx]`, edge-padded by clamping the low end to 0 --
    mirrors `TrajectoryDataset._slice_and_pad`'s edge-pad fallback (obs never gets a
    zero-pad mask)."""
    start = end_idx - obs_horizon + 1
    return [max(0, i) for i in range(start, end_idx + 1)]


def _add_batch_axis(t: torch.Tensor) -> torch.Tensor:
    return t.unsqueeze(0)


def build_obs_batch(
    obs_tree: RawTree, frame_indices: list[int], obs_horizon: int, device: torch.device
) -> TensorTree:
    """Builds an `[obs_horizon, ...]` window ending at each sampled frame index and stacks them
    into a batch axis 0, giving `[B, obs_horizon, ...]`."""
    windows = []
    for end_idx in frame_indices:
        window = recursive_index(obs_tree, window_indices(end_idx, obs_horizon))
        window_t = to_tensor(window, device=device, dtype=torch.float32)
        windows.append(map_leaves(_add_batch_axis, window_t))
    return cat_dicts(windows)


def build_goal_batch(
    obs_tree: RawTree, goal_frame_idx: int, batch_size: int, device: torch.device
) -> TensorTree:
    """A single goal frame, broadcast (no copy, via `expand`) across the batch so every sampled
    frame is evaluated against the same fixed goal."""
    goal = recursive_index(obs_tree, goal_frame_idx)
    goal_t = to_tensor(goal, device=device, dtype=torch.float32)

    def _expand(t: torch.Tensor) -> torch.Tensor:
        return t.unsqueeze(0).expand(batch_size, *t.shape)

    return map_leaves(_expand, goal_t)


def detect_attention_modules(
    model: GoalConditionedDiffusionPolicy,
) -> dict[str, nn.MultiheadAttention]:
    """Auto-detects whichever attention module(s) the loaded embedder has -- both may be present (a
    SelfAttentionEmbedder with a nested AttentionPooling), one, or neither."""
    modules: dict[str, nn.MultiheadAttention] = {}
    if isinstance(model.embedder, SelfAttentionEmbedder):
        modules["self_attention"] = model.embedder.attn
    pooling = getattr(model.embedder, "pooling", None)
    if isinstance(pooling, AttentionPooling):
        modules["pooling"] = pooling.attn

    if not modules:
        raise RuntimeError(
            f"{type(model.embedder).__name__} has no SelfAttentionEmbedder or AttentionPooling "
            "attention module to visualize."
        )
    return modules


@contextlib.contextmanager
def capture_attention_weights(
    mha: nn.MultiheadAttention,
) -> Generator[list[torch.Tensor], None, None]:
    """Forces `need_weights=True, average_attn_weights=False` on every real call this
    `nn.MultiheadAttention` makes -- without touching the owning module's `forward()` at all -- and
    collects every resulting weights tensor (detached, CPU) into the yielded list."""
    captured: list[torch.Tensor] = []

    def pre_hook(_module: nn.Module, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        return args, {**kwargs, "need_weights": True, "average_attn_weights": False}

    def fwd_hook(_module: nn.Module, _args: tuple, output: tuple) -> None:
        weights = output[1]
        if weights is not None:
            captured.append(weights.detach().cpu())

    handle_pre = mha.register_forward_pre_hook(pre_hook, with_kwargs=True)
    handle_fwd = mha.register_forward_hook(fwd_hook)
    try:
        yield captured
    finally:
        handle_pre.remove()
        handle_fwd.remove()


def select_obs_capture(
    captured: list[torch.Tensor], expected_batch: int, expected_seq_len: int
) -> torch.Tensor:
    """Disambiguates the real observation-window capture from an incidental goal-branch capture on
    the same submodule (occurs for `goal_delta in (None, "embedding")`, which call the embedder a
    second time for the goal).

    Matches on `(batch, key_len)`; falls back to the widest capture with a warning if nothing
    matches exactly.
    """
    for weights in captured:
        if weights.shape[0] == expected_batch and weights.shape[-1] == expected_seq_len:
            return weights
    warnings.warn(
        f"No attention capture matched batch={expected_batch}, seq_len={expected_seq_len}; "
        "falling back to the widest capture (the goal-branch call may have been picked up "
        "instead).",
        stacklevel=2,
    )
    return max(captured, key=lambda w: w.shape[-1])


def run_and_capture(
    model: GoalConditionedDiffusionPolicy,
    obs_batch: TensorTree,
    goal_batch: TensorTree,
    target_modules: dict[str, nn.MultiheadAttention],
    obs_horizon: int,
    tokens_per_step: int,
) -> dict[str, torch.Tensor]:
    """Normalizes obs/goal, installs capture hooks on every target module, drives the real
    tokenizer+embedder pipeline via `_build_external_cond` (no diffusion loop needed), and returns
    each module's disambiguated observation-window attention capture."""
    if model.obs_normalizer is not None:
        obs_batch = model.obs_normalizer.normalize(obs_batch)
        goal_batch = model.obs_normalizer.normalize(goal_batch)

    expected_batch = get_batch_size(obs_batch)
    expected_seq_len = obs_horizon * tokens_per_step

    with contextlib.ExitStack() as stack:
        captures = {
            name: stack.enter_context(capture_attention_weights(module))
            for name, module in target_modules.items()
        }
        with torch.no_grad():
            model._build_external_cond(obs_batch, goal_batch)

    return {
        name: select_obs_capture(capture, expected_batch, expected_seq_len)
        for name, capture in captures.items()
        if capture
    }


def build_metadata_str(
    model: GoalConditionedDiffusionPolicy, cfg: DictConfig | None, env_id: str
) -> str:
    """A compact `key=value` summary of the config this checkpoint was trained with -- env,
    tokenizer/embedder/pooling architecture, goal-conditioning mode, HER ratio -- appended under
    every figure's title so a saved PNG identifies its own provenance without cross-referencing the
    checkpoint path."""
    pooling = getattr(model.embedder, "pooling", None)
    parts = [
        f"env={env_id}",
        f"tokenizer={type(model.tokenizer).__name__ if model.tokenizer is not None else 'none'}",
        f"embedder={type(model.embedder).__name__ if model.embedder is not None else 'none'}",
        f"pooling={type(pooling).__name__ if pooling is not None else 'none'}",
        f"goal_delta={model.goal_delta!r}",
    ]
    her_ratio = cfg.get("datamodule", {}).get("her_ratio", None) if cfg is not None else None
    if her_ratio is not None:
        parts.append(f"her_ratio={her_ratio}")
    return " | ".join(parts)


def build_token_labels(model: GoalConditionedDiffusionPolicy, obs_horizon: int) -> list[str]:
    """T-major, k-minor order (`index = t*K + k`), matching `SelfAttentionEmbedder`'s own
    documented token-flattening convention."""
    if isinstance(model.tokenizer, PerObjectStateTokenizer):
        return [f"t{t}/{key}" for t in range(obs_horizon) for key in model.tokenizer.object_keys]
    return [f"t{t}" for t in range(obs_horizon)]


def _agg_heads(weights: np.ndarray, head_agg: str) -> np.ndarray:
    """weights: [B, H, Lq, Lk] -> [B, Lq, Lk]."""
    return weights.mean(axis=1) if head_agg == "mean" else weights.max(axis=1)


def _apply_dark_theme() -> None:
    plt.rcParams.update(DARK_RCPARAMS)


def plot_self_attention(
    weights: np.ndarray,  # [B, H, S, S]
    token_labels: list[str],
    frame_labels: list[str],
    head_agg: str,
    metadata_str: str,
    save_path: Path,
    show: bool,
) -> None:
    agg = _agg_heads(weights, head_agg)  # [B, S, S]
    vmax = float(agg.max())
    n = agg.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5), squeeze=False)
    for i, ax in enumerate(axes[0]):
        im = ax.imshow(agg[i], cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
        ax.set_title(frame_labels[i], fontsize=11)
        ax.set_xticks(range(len(token_labels)))
        ax.set_xticklabels(token_labels, rotation=90, fontsize=7)
        if i == 0:
            ax.set_yticks(range(len(token_labels)))
            ax.set_yticklabels(token_labels, fontsize=7)
        else:
            ax.set_yticks([])

    fig.suptitle(
        f"Self-Attention ({head_agg}-over-heads)\n{metadata_str}",
        fontsize=14,
        fontweight="bold",
        y=1.05,
    )
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="attention weight")
    _save_and_maybe_show(fig, save_path, show)


def plot_self_attention_heads(
    weights: np.ndarray,
    token_labels: list[str],
    frame_labels: list[str],
    metadata_str: str,
    save_path: Path,
    show: bool,
) -> None:
    n_frames, n_heads = weights.shape[0], weights.shape[1]
    vmax = float(weights.max())
    fig, axes = plt.subplots(n_frames, n_heads, figsize=(3 * n_heads, 3 * n_frames), squeeze=False)
    for i in range(n_frames):
        for h in range(n_heads):
            ax = axes[i][h]
            ax.imshow(weights[i, h], cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(f"head {h}", fontsize=9)
            if h == 0:
                ax.set_ylabel(frame_labels[i], fontsize=9)
            if i == n_frames - 1:
                ax.set_xticks(range(len(token_labels)))
                ax.set_xticklabels(token_labels, rotation=90, fontsize=6)

    fig.suptitle(
        f"Self-Attention (per head)\n{metadata_str}", fontsize=14, fontweight="bold", y=1.03
    )
    _save_and_maybe_show(fig, save_path, show)


def plot_pooling_attention(
    weights: np.ndarray,  # [B, H, S]
    token_labels: list[str],
    frame_labels: list[str],
    head_agg: str,
    metadata_str: str,
    save_path: Path,
    show: bool,
) -> None:
    agg = weights.mean(axis=1) if head_agg == "mean" else weights.max(axis=1)  # [B, S]

    fig, ax = plt.subplots(
        figsize=(max(6, len(token_labels) * 0.6), max(4, len(frame_labels) * 0.6))
    )
    im = ax.imshow(agg, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(token_labels)))
    ax.set_xticklabels(token_labels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(frame_labels)))
    ax.set_yticklabels(frame_labels, fontsize=9)
    ax.set_xlabel("Token")
    ax.set_ylabel("Sampled frame")

    if agg.size <= 40:
        for i in range(agg.shape[0]):
            for j in range(agg.shape[1]):
                ax.text(
                    j, i, f"{agg[i, j]:.2f}", ha="center", va="center", fontsize=7, color="#0f172a"
                )

    fig.colorbar(im, ax=ax, shrink=0.8, label="attention weight")
    ax.set_title(
        f"Pooling Attention over the Episode ({head_agg}-over-heads)\n{metadata_str}",
        fontsize=13,
        fontweight="bold",
    )
    _save_and_maybe_show(fig, save_path, show)


def plot_pooling_attention_heads(
    weights: np.ndarray,
    token_labels: list[str],
    frame_labels: list[str],
    metadata_str: str,
    save_path: Path,
    show: bool,
) -> None:
    n_heads = weights.shape[1]
    vmax = float(weights.max())
    fig, axes = plt.subplots(1, n_heads, figsize=(4.5 * n_heads, 5), squeeze=False)
    for h, ax in enumerate(axes[0]):
        im = ax.imshow(weights[:, h], cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
        ax.set_title(f"head {h}", fontsize=11)
        ax.set_xticks(range(len(token_labels)))
        ax.set_xticklabels(token_labels, rotation=90, fontsize=7)
        if h == 0:
            ax.set_yticks(range(len(frame_labels)))
            ax.set_yticklabels(frame_labels, fontsize=8)
        else:
            ax.set_yticks([])

    fig.suptitle(
        f"Pooling Attention (per head)\n{metadata_str}", fontsize=14, fontweight="bold", y=1.05
    )
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="attention weight")
    _save_and_maybe_show(fig, save_path, show)


def _save_and_maybe_show(fig: plt.Figure, save_path: Path, show: bool) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        save_path, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight"
    )
    print(f"Saved: {save_path.resolve()}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.ckpt_path)
    dataset_path = Path(args.dataset_path)

    print(f"Loading checkpoint from: {ckpt_path}")
    model = load_goal_conditioned_diffusion_policy(ckpt_path)
    target_modules = detect_attention_modules(model)
    print(f"Detected attention module(s): {sorted(target_modules)}")

    cfg = load_ckpt_config(ckpt_path)
    env_id = resolve_env_id(cfg, dataset_path)
    obs_transform = resolve_obs_transform(cfg, dataset_path, env_id)

    target_key, obs_tree, seq_len = load_episode_obs(dataset_path, args.episode_idx)
    print(f"Episode: {target_key} (length {seq_len}, env_id={env_id})")

    frames = parse_frames(args.frames, seq_len)
    frame_indices = [idx for idx, _ in frames]
    frame_labels = [label for _, label in frames]

    goal_frame_idx = (
        args.goal_frame_idx if args.goal_frame_idx >= 0 else seq_len + args.goal_frame_idx
    )
    goal_frame_idx = max(0, min(goal_frame_idx, seq_len - 1))

    obs_horizon = model.obs_horizon
    assert model.tokenizer is not None, (
        "configure_model() must run before tokens_per_step is known."
    )
    tokens_per_step = model.tokenizer.tokens_per_step

    obs_batch = build_obs_batch(obs_tree, frame_indices, obs_horizon, model.device)
    goal_batch = build_goal_batch(obs_tree, goal_frame_idx, len(frame_indices), model.device)
    obs_batch = obs_transform(obs_batch)
    goal_batch = obs_transform(goal_batch)

    captures = run_and_capture(
        model, obs_batch, goal_batch, target_modules, obs_horizon, tokens_per_step
    )
    token_labels = build_token_labels(model, obs_horizon)
    metadata_str = build_metadata_str(model, cfg, env_id)

    _apply_dark_theme()
    prefix = args.save_path_prefix or f"{ckpt_path.parent.parent.parent.name}_ep{args.episode_idx}"
    save_dir = Path("scripts/figures")

    if "self_attention" in captures:
        weights = captures["self_attention"].numpy()
        plot_self_attention(
            weights,
            token_labels,
            frame_labels,
            args.head_agg,
            metadata_str,
            save_dir / f"{prefix}_self_attention.png",
            args.show,
        )
        if args.show_heads:
            plot_self_attention_heads(
                weights,
                token_labels,
                frame_labels,
                metadata_str,
                save_dir / f"{prefix}_self_attention_heads.png",
                args.show,
            )

    if "pooling" in captures:
        weights = captures["pooling"].numpy().squeeze(2)  # [B, H, 1, S] -> [B, H, S]
        plot_pooling_attention(
            weights,
            token_labels,
            frame_labels,
            args.head_agg,
            metadata_str,
            save_dir / f"{prefix}_pooling_attention.png",
            args.show,
        )
        if args.show_heads:
            plot_pooling_attention_heads(
                weights,
                token_labels,
                frame_labels,
                metadata_str,
                save_dir / f"{prefix}_pooling_attention_heads.png",
                args.show,
            )


if __name__ == "__main__":
    main()
