"""Shows what a policy's embedder attends to.

Two attention modules can be present in a `GoalConditionedDiffusionPolicy`'s embedder.
Both are visualised when found:

- `SelfAttentionEmbedder`'s self-attention across observation tokens
- `AttentionPooling`'s learned-query attention

Frames come either from a recorded episode (`--source dataset`) or from a live rollout
(`--source rollout`), the latter defaulting to a sweep of the whole LockedRotation family.
"""

from __future__ import annotations

import argparse
import contextlib
import warnings
from collections.abc import Generator
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.image import AxesImage

import policy.environments  # noqa: F401  (registers the project's envs as a side effect)
from policy.algorithms.goal_conditioned_diffusion_policy import GoalConditionedDiffusionPolicy
from policy.algorithms.networks.pooling import AttentionPooling
from policy.algorithms.networks.self_attention_embedder import SelfAttentionEmbedder
from policy.algorithms.tokenizers import PerObjectStateTokenizer
from policy.transforms import observation_pipeline
from policy.utils import (
    cat_dicts,
    get_batch_size,
    map_leaves,
    recursive_index,
    stack_dicts,
    to_tensor,
)
from policy.utils.h5_utils import load_h5_data, peek_trajectory_is_dataset
from policy.utils.typing_utils import RawTree, TensorTree
from scripts.utils import cli, theme
from scripts.utils.checkpoints import (
    describe_model_config,
    load_goal_conditioned_diffusion_policy,
    require_run_config,
    run_slug,
)
from scripts.utils.episodes import (
    default_demo_path,
    parse_frame_spec,
    resolve_env_id,
    resolve_trajectory_key,
)
from scripts.utils.figures import figure_path, save_figure
from scripts.utils.report import Report
from scripts.utils.rollouts import (
    build_obs_transform,
    build_rollout_env,
    iter_env_kwargs,
    resolve_max_episode_steps,
    run_episode,
)

SCRIPT_NAME = Path(__file__).stem

# Above this many cells, per-cell value labels stop being readable.
ANNOTATE_MAX_CELLS = 40


# --- Capturing attention ----------------------------------------------------------------------


def detect_attention_modules(
    model: GoalConditionedDiffusionPolicy,
) -> dict[str, nn.MultiheadAttention]:
    """Finds whichever attention modules the loaded embedder has; either, both, or neither."""
    modules: dict[str, nn.MultiheadAttention] = {}
    if isinstance(model.embedder, SelfAttentionEmbedder):
        modules["self_attention"] = model.embedder.attn
    pooling = getattr(model.embedder, "pooling", None)
    if isinstance(pooling, AttentionPooling):
        modules["pooling"] = pooling.attn

    if not modules:
        raise RuntimeError(
            f"{type(model.embedder).__name__} contains no SelfAttentionEmbedder or "
            "AttentionPooling, so there is no attention to visualise."
        )
    return modules


@contextlib.contextmanager
def capture_attention_weights(
    mha: nn.MultiheadAttention,
) -> Generator[list[torch.Tensor], None, None]:
    """Collects attention weights from an `nn.MultiheadAttention` for the duration of the block.

    Forces `need_weights=True, average_attn_weights=False` on every call through a pre-hook.
    """
    captured: list[torch.Tensor] = []

    def pre_hook(_module: nn.Module, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        return args, {**kwargs, "need_weights": True, "average_attn_weights": False}

    def forward_hook(_module: nn.Module, _args: tuple, output: tuple) -> None:
        weights = output[1]
        if weights is not None:
            captured.append(weights.detach().cpu())

    pre_handle = mha.register_forward_pre_hook(pre_hook, with_kwargs=True)
    forward_handle = mha.register_forward_hook(forward_hook)
    try:
        yield captured
    finally:
        pre_handle.remove()
        forward_handle.remove()


def select_obs_capture(
    captured: list[torch.Tensor], expected_batch: int, expected_seq_len: int
) -> torch.Tensor:
    """Picks the observation-window capture out of everything the hook saw."""
    for weights in captured:
        if weights.shape[0] == expected_batch and weights.shape[-1] == expected_seq_len:
            return weights
    warnings.warn(
        f"No attention capture matched batch={expected_batch}, seq_len={expected_seq_len}; "
        "falling back to the widest capture, which may be the goal-branch call.",
        stacklevel=2,
    )
    return max(captured, key=lambda weights: weights.shape[-1])


def run_and_capture(
    model: GoalConditionedDiffusionPolicy,
    obs_batch: TensorTree,
    goal_batch: TensorTree,
    target_modules: dict[str, nn.MultiheadAttention],
    obs_horizon: int,
    tokens_per_step: int,
) -> dict[str, torch.Tensor]:
    """Runs the tokenizer/embedder pipeline over a batch of frames and returns the attention.

    Goes only through `_build_external_cond`, so the diffusion loop is never invoked.
    """
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


# --- Assembling frames ------------------------------------------------------------------------


def build_token_labels(model: GoalConditionedDiffusionPolicy, obs_horizon: int) -> list[str]:
    """Token names in `t * K + k` order, matching `SelfAttentionEmbedder`'s flattening."""
    if isinstance(model.tokenizer, PerObjectStateTokenizer):
        return [f"t{t}/{key}" for t in range(obs_horizon) for key in model.tokenizer.object_keys]
    return [f"t{t}" for t in range(obs_horizon)]


def window_indices(end_index: int, obs_horizon: int) -> list[int]:
    """The `obs_horizon` indices ending at `end_index`, edge-padded at the start of the episode."""
    return [max(0, i) for i in range(end_index - obs_horizon + 1, end_index + 1)]


def build_obs_batch(
    obs_tree: RawTree, frame_indices: list[int], obs_horizon: int, device: torch.device
) -> TensorTree:
    """Stacks one observation window per sampled frame into a batch."""
    windows = []
    for end_index in frame_indices:
        window = recursive_index(obs_tree, window_indices(end_index, obs_horizon))
        tensor_window = to_tensor(window, device=device, dtype=torch.float32)
        windows.append(map_leaves(lambda t: t.unsqueeze(0), tensor_window))
    return cat_dicts(windows)


def broadcast_goal(goal: TensorTree, batch_size: int) -> TensorTree:
    """Expands one goal instance across the batch, so every frame is scored against the same
    goal."""
    return map_leaves(lambda t: t.unsqueeze(0).expand(batch_size, *t.shape), goal)


def build_goal_batch(
    obs_tree: RawTree, goal_frame_index: int, batch_size: int, device: torch.device
) -> TensorTree:
    """Takes one frame of a recorded episode as the goal and broadcasts it."""
    goal = to_tensor(
        recursive_index(obs_tree, goal_frame_index), device=device, dtype=torch.float32
    )
    return broadcast_goal(goal, batch_size)


def load_episode_obs(dataset_path: Path, episode_idx: int) -> tuple[str, RawTree, int]:
    """Loads one recorded episode's observations, flat or structured."""
    with h5py.File(dataset_path, "r") as handle:
        key = resolve_trajectory_key(handle, episode_idx)
        group = handle[key]
        if not isinstance(group, h5py.Group):
            raise TypeError(f"Expected an HDF5 group at {key}.")

        node = group["obs"]
        if isinstance(node, h5py.Dataset):
            array = np.asarray(node)
            return key, array, array.shape[0]
        if isinstance(node, h5py.Group):
            tree = load_h5_data(node)
            return key, tree, len(next(iter(tree.values())))
        raise TypeError(f"Unexpected obs node type: {type(node)}")


def latest_frame(obs: TensorTree) -> TensorTree:
    """Drops the batch and time axes from a windowed observation, leaving one bare frame."""
    return map_leaves(lambda t: t[:, -1].squeeze(0), obs)


def collect_live_episode(model, cfg, env_kwargs: dict, args: argparse.Namespace):
    """Drives one live episode.

    Returns `(obs_tree, goal_raw, seq_len, success, transform)`.
    """
    max_episode_steps = resolve_max_episode_steps(cfg, args.max_episode_steps)
    print(
        f"Env: {env_kwargs['env_id']}  obs_mode={env_kwargs['obs_mode']}  "
        f"control_mode={env_kwargs['control_mode']}  "
        f"physx_backend={env_kwargs['physx_backend']}  max_episode_steps={max_episode_steps}"
    )

    env, inner_env = build_rollout_env(
        env_kwargs, model.obs_horizon, max_episode_steps, args.render_mode, args.video_dir
    )
    transform = build_obs_transform(env, env_kwargs, cfg)
    goal_frames: list[TensorTree] = []

    def record(step) -> TensorTree:
        if not goal_frames:
            goal_frames.append(step.goal_raw)
        return latest_frame(step.obs_raw)

    try:
        rollout = run_episode(
            model,
            env,
            inner_env,
            transform,
            seed=args.seed,
            num_inference_steps=args.num_inference_steps,
            clamp_action=args.clamp_action,
            render_mode=args.render_mode,
            # The heuristic goal is anchored to a static target here, so one capture is enough.
            refresh_goal_each_step=False,
            on_step=record,
        )
    finally:
        env.close()

    obs_tree = stack_dicts(rollout.records)
    goal_raw = map_leaves(lambda t: t.squeeze(0), goal_frames[0])
    return obs_tree, goal_raw, len(rollout.records), rollout.success_once, transform


# --- Plotting ---------------------------------------------------------------------------------


def aggregate_heads(weights: np.ndarray, how: str) -> np.ndarray:
    """Collapses the head axis: `[B, H, ...] -> [B, ...]`."""
    return weights.mean(axis=1) if how == "mean" else weights.max(axis=1)


def plot_self_attention(
    weights: np.ndarray,
    token_labels: list[str],
    frame_labels: list[str],
    head_agg: str,
    fields,
    save_path: Path,
    *,
    show: bool,
    dpi: int,
) -> None:
    """One token-by-token attention map per sampled frame."""
    aggregated = aggregate_heads(weights, head_agg)
    vmax = float(aggregated.max())
    count = aggregated.shape[0]

    fig, axes = plt.subplots(1, count, figsize=(4.5 * count, 5), squeeze=False)
    image: AxesImage | None = None
    for index, ax in enumerate(axes[0]):
        image = ax.imshow(
            aggregated[index], cmap=theme.SEQUENTIAL_CMAP, vmin=0, vmax=vmax, aspect="auto"
        )
        ax.set_title(frame_labels[index], fontsize=10, color=theme.TEXT_SECONDARY)
        ax.set_xticks(range(len(token_labels)))
        ax.set_xticklabels(token_labels, rotation=90, fontsize=7)
        if index == 0:
            ax.set_yticks(range(len(token_labels)))
            ax.set_yticklabels(token_labels, fontsize=7)
        else:
            ax.set_yticks([])

    assert image is not None  # squeeze=False guarantees at least one panel was drawn
    fig.colorbar(image, ax=axes[0].tolist(), shrink=0.8, label="attention weight")
    theme.set_title(fig, f"Self-attention across tokens ({head_agg} over heads)", fields)
    save_figure(fig, save_path, show=show, dpi=dpi, reserve_title_space=False)
    print(f"Figure saved: {save_path}")


def plot_self_attention_heads(
    weights: np.ndarray,
    token_labels: list[str],
    frame_labels: list[str],
    fields,
    save_path: Path,
    *,
    show: bool,
    dpi: int,
) -> None:
    """The same maps split per head, to see whether heads specialise."""
    num_frames, num_heads = weights.shape[0], weights.shape[1]
    vmax = float(weights.max())

    fig, axes = plt.subplots(
        num_frames, num_heads, figsize=(3 * num_heads, 3 * num_frames), squeeze=False
    )
    for frame in range(num_frames):
        for head in range(num_heads):
            ax = axes[frame][head]
            ax.imshow(
                weights[frame, head], cmap=theme.SEQUENTIAL_CMAP, vmin=0, vmax=vmax, aspect="auto"
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if frame == 0:
                ax.set_title(f"head {head}", fontsize=9, color=theme.TEXT_SECONDARY)
            if head == 0:
                ax.set_ylabel(frame_labels[frame], fontsize=9)
            if frame == num_frames - 1:
                ax.set_xticks(range(len(token_labels)))
                ax.set_xticklabels(token_labels, rotation=90, fontsize=6)

    theme.set_title(fig, "Self-attention per head", fields)
    save_figure(fig, save_path, show=show, dpi=dpi, reserve_title_space=False)
    print(f"Figure saved: {save_path}")


def plot_pooling_attention(
    weights: np.ndarray,
    token_labels: list[str],
    frame_labels: list[str],
    head_agg: str,
    fields,
    save_path: Path,
    *,
    show: bool,
    dpi: int,
) -> None:
    """How much each token contributes to the pooled conditioning vector, frame by frame."""
    aggregated = aggregate_heads(weights, head_agg)

    fig, ax = plt.subplots(
        figsize=(max(6, len(token_labels) * 0.6), max(4, len(frame_labels) * 0.6))
    )
    image = ax.imshow(aggregated, cmap=theme.SEQUENTIAL_CMAP, aspect="auto")
    ax.set_xticks(range(len(token_labels)))
    ax.set_xticklabels(token_labels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(frame_labels)))
    ax.set_yticklabels(frame_labels, fontsize=9)
    ax.set_xlabel("token")
    ax.set_ylabel("sampled frame")

    if aggregated.size <= ANNOTATE_MAX_CELLS:
        # Dark ink, because the viridis cells these sit on are light at the high end.
        for row in range(aggregated.shape[0]):
            for col in range(aggregated.shape[1]):
                ax.text(
                    col,
                    row,
                    f"{aggregated[row, col]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=theme.BACKGROUND,
                )

    fig.colorbar(image, ax=ax, shrink=0.8, label="attention weight")
    theme.set_title(fig, f"Pooling attention over tokens ({head_agg} over heads)", fields)
    save_figure(fig, save_path, show=show, dpi=dpi, reserve_title_space=False)
    print(f"Figure saved: {save_path}")


def plot_pooling_attention_heads(
    weights: np.ndarray,
    token_labels: list[str],
    frame_labels: list[str],
    fields,
    save_path: Path,
    *,
    show: bool,
    dpi: int,
) -> None:
    """Pooling attention split per head."""
    num_heads = weights.shape[1]
    vmax = float(weights.max())

    fig, axes = plt.subplots(1, num_heads, figsize=(4.5 * num_heads, 5), squeeze=False)
    image: AxesImage | None = None
    for head, ax in enumerate(axes[0]):
        image = ax.imshow(
            weights[:, head], cmap=theme.SEQUENTIAL_CMAP, vmin=0, vmax=vmax, aspect="auto"
        )
        ax.set_title(f"head {head}", fontsize=10, color=theme.TEXT_SECONDARY)
        ax.set_xticks(range(len(token_labels)))
        ax.set_xticklabels(token_labels, rotation=90, fontsize=7)
        if head == 0:
            ax.set_yticks(range(len(frame_labels)))
            ax.set_yticklabels(frame_labels, fontsize=8)
        else:
            ax.set_yticks([])

    assert image is not None  # squeeze=False guarantees at least one panel was drawn
    fig.colorbar(image, ax=axes[0].tolist(), shrink=0.8, label="attention weight")
    theme.set_title(fig, "Pooling attention per head", fields)
    save_figure(fig, save_path, show=show, dpi=dpi, reserve_title_space=False)
    print(f"Figure saved: {save_path}")


def emit_plots(
    captures: dict[str, torch.Tensor],
    token_labels: list[str],
    frame_labels: list[str],
    fields,
    path_for,
    args: argparse.Namespace,
) -> tuple[Path | None, dict[str, np.ndarray]]:
    """Emits whichever figures the captured modules support."""
    anchor: Path | None = None
    numeric: dict[str, np.ndarray] = {}

    if "self_attention" in captures:
        weights = captures["self_attention"].numpy()
        numeric["self_attention"] = weights
        self_attention_path = path_for("self-attention")
        anchor = self_attention_path
        plot_self_attention(
            weights,
            token_labels,
            frame_labels,
            args.head_agg,
            fields,
            self_attention_path,
            show=args.show,
            dpi=args.dpi,
        )
        if args.show_heads:
            plot_self_attention_heads(
                weights,
                token_labels,
                frame_labels,
                fields,
                path_for("self-attention-heads"),
                show=args.show,
                dpi=args.dpi,
            )

    if "pooling" in captures:
        weights = captures["pooling"].numpy().squeeze(2)  # [B, H, 1, S] -> [B, H, S]
        numeric["pooling"] = weights
        pooling_path = path_for("pooling-attention")
        anchor = anchor or pooling_path
        plot_pooling_attention(
            weights,
            token_labels,
            frame_labels,
            args.head_agg,
            fields,
            pooling_path,
            show=args.show,
            dpi=args.dpi,
        )
        if args.show_heads:
            plot_pooling_attention_heads(
                weights,
                token_labels,
                frame_labels,
                fields,
                path_for("pooling-attention-heads"),
                show=args.show,
                dpi=args.dpi,
            )

    return anchor, numeric


def build_report(
    numeric: dict[str, np.ndarray],
    token_labels: list[str],
    frame_labels: list[str],
    fields,
    ckpt_path: Path,
    extra: list[tuple[str, str]],
) -> Report:
    """Turns the attention maps into numbers."""
    report = Report("Embedder attention", [("ckpt", str(ckpt_path)), *fields])

    if "pooling" in numeric:
        pooled = numeric["pooling"].mean(axis=1)  # average heads -> [frames, tokens]
        report.section("Pooling attention by token (mean over heads)")
        report.table(
            ["frame", *token_labels],
            [
                [frame_labels[row], *[f"{value:.3f}" for value in pooled[row]]]
                for row in range(pooled.shape[0])
            ],
        )
        mean_by_token = pooled.mean(axis=0)
        ranking = np.argsort(mean_by_token)[::-1]
        report.section("Most-attended tokens (averaged over frames)")
        report.table(
            ["rank", "token", "mean weight"],
            [
                [rank + 1, token_labels[index], f"{mean_by_token[index]:.4f}"]
                for rank, index in enumerate(ranking)
            ],
        )
        report.note(
            f"Uniform attention would give every token {1 / len(token_labels):.3f}. Values far "
            "above that mean the pooled conditioning vector is dominated by a few tokens."
        )

    if "self_attention" in numeric:
        weights = numeric["self_attention"].mean(axis=1)  # [frames, query, key]
        received = weights.mean(axis=(0, 1))
        ranking = np.argsort(received)[::-1]
        report.section("Most-attended-to tokens in self-attention")
        report.table(
            ["rank", "token", "mean attention received"],
            [
                [rank + 1, token_labels[index], f"{received[index]:.4f}"]
                for rank, index in enumerate(ranking)
            ],
        )

    if extra:
        report.section("Run")
        for key, value in extra:
            report.kv(key, value)
    return report


# --- Entry point ------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        parents=[
            cli.checkpoint_args(),
            cli.output_args(),
            cli.source_args(),
            cli.dataset_args(),
            cli.rollout_args(default_num_episodes=1),
        ],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--goal-frame-idx",
        type=int,
        default=-1,
        help="Frame of the recorded episode to use as the goal, in --source dataset. Negative "
        "counts from the end. (default: -1, the final frame)",
    )
    parser.add_argument(
        "--head-agg",
        choices=["mean", "max"],
        default="mean",
        help="How to combine attention heads in the summary figures. (default: mean)",
    )
    parser.add_argument(
        "--show-heads",
        action="store_true",
        help="Also emit the per-head breakdown figures.",
    )
    args = parser.parse_args()

    if args.source == "rollout" and args.dataset_path is not None:
        parser.error("--dataset-path only applies to --source dataset.")
    if args.num_episodes != 1:
        parser.error(
            "This script visualises a single episode; use --seed to look at a different one."
        )
    return args


def visualize(
    model,
    cfg,
    target_modules,
    token_labels: list[str],
    tokens_per_step: int,
    env_id: str,
    obs_tree: RawTree,
    goal_batch_source,
    seq_len: int,
    episode_label: str,
    goal_label: str,
    extra: list[tuple[str, str]],
    transform,
    args: argparse.Namespace,
    slug: str,
) -> None:
    """Sample frames, capture attention over them, plot and report."""
    frames = parse_frame_spec(args.frames, seq_len)
    frame_indices = [index for index, _ in frames]
    frame_labels = [label for _, label in frames]

    obs_batch = transform(
        build_obs_batch(obs_tree, frame_indices, model.obs_horizon, model.device)
    )
    goal_batch = transform(goal_batch_source(len(frame_indices)))

    captures = run_and_capture(
        model, obs_batch, goal_batch, target_modules, model.obs_horizon, tokens_per_step
    )

    fields = describe_model_config(
        model,
        cfg,
        extra=[("env", env_id), ("episode", episode_label), ("goal", goal_label)],
    )

    def path_for(name: str) -> Path:
        return figure_path(SCRIPT_NAME, name, env_id=env_id, run_slug=slug, out_dir=args.out_dir)

    anchor, numeric = emit_plots(captures, token_labels, frame_labels, fields, path_for, args)
    build_report(numeric, token_labels, frame_labels, fields, args.ckpt_path, extra).emit(
        anchor, save=not args.no_report
    )


def main() -> None:
    args = parse_args()
    theme.apply_theme()

    model = load_goal_conditioned_diffusion_policy(args.ckpt_path)
    target_modules = detect_attention_modules(model)
    print(f"Detected attention module(s): {sorted(target_modules)}")

    if model.tokenizer is None:
        raise RuntimeError("configure_model() must run before tokens_per_step is known.")
    tokens_per_step = model.tokenizer.tokens_per_step
    token_labels = build_token_labels(model, model.obs_horizon)

    if args.source == "rollout":
        cfg = require_run_config(args.ckpt_path)
        slug = args.run_label or run_slug(args.ckpt_path, model, cfg, args.seed)

        for _, _, env_kwargs in iter_env_kwargs(cfg, args.env_id):
            env_id = env_kwargs["env_id"]
            obs_tree, goal_raw, seq_len, success, transform = collect_live_episode(
                model, cfg, env_kwargs, args
            )
            print(f"Episode: live seed={args.seed} (length {seq_len}, success={success})")

            visualize(
                model,
                cfg,
                target_modules,
                token_labels,
                tokens_per_step,
                env_id,
                obs_tree,
                lambda batch_size: broadcast_goal(
                    to_tensor(goal_raw, device=model.device, dtype=torch.float32), batch_size
                ),
                seq_len,
                f"live seed={args.seed}",
                "heuristic",
                [("source", "rollout"), ("success_once", str(success))],
                transform,
                args,
                slug,
            )
        return

    cfg = require_run_config(args.ckpt_path)
    slug = args.run_label or run_slug(args.ckpt_path, model, cfg, args.seed)
    env_id = (args.env_id or [None])[0] or str(cfg.get("datamodule", {}).get("env_id", "") or "")
    dataset_path = args.dataset_path or default_demo_path(env_id or "StackCubeLockedRotation-v1")
    env_id = resolve_env_id(dataset_path, env_id or None)

    transform = observation_pipeline(
        env_id=env_id,
        is_flat=peek_trajectory_is_dataset(dataset_path, dimension_key="obs"),
        canonicalize=bool(cfg.get("datamodule", {}).get("canonicalize", True)),
        as_dict=bool(cfg.get("datamodule", {}).get("as_dict", True)),
        no_proprio_vel=bool(cfg.get("datamodule", {}).get("no_proprio_vel", False)),
    )

    episode_key, obs_tree, seq_len = load_episode_obs(dataset_path, args.episode_idx)
    print(f"Episode: {episode_key} (length {seq_len}, env_id={env_id})")

    goal_frame = args.goal_frame_idx if args.goal_frame_idx >= 0 else seq_len + args.goal_frame_idx
    goal_frame = max(0, min(goal_frame, seq_len - 1))

    visualize(
        model,
        cfg,
        target_modules,
        token_labels,
        tokens_per_step,
        env_id,
        obs_tree,
        lambda batch_size: build_goal_batch(obs_tree, goal_frame, batch_size, model.device),
        seq_len,
        episode_key,
        f"frame {goal_frame}/{seq_len - 1}",
        [("source", "dataset"), ("dataset", str(dataset_path))],
        transform,
        args,
        slug,
    )


if __name__ == "__main__":
    main()
