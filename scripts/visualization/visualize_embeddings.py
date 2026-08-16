"""Projects a policy's state embeddings into 2D to see how an episode moves through latent space.

The embedding of one episode is drawn as a path coloured by time, over a faint backdrop of other
episodes, with the goal state marked.

Frames come either from recorded demonstrations (`--source dataset`) or from a live rollout
(`--source rollout`), the latter defaulting to a sweep of the LockedRotation family.

`--highlight-stages` labels the grasp / mid-air / place moments instead of fixed percentages.
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Callable
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

import policy.environments  # noqa: F401  (registers the project's envs as a side effect)
from policy.transforms import observation_pipeline
from policy.utils import map_leaves, stack_dicts, to_tensor
from policy.utils.typing_utils import TensorTree
from scripts.utils import cli, theme
from scripts.utils.checkpoints import (
    describe_model_config,
    ensure_local_checkpoint,
    load_goal_conditioned_diffusion_policy,
    require_run_config,
    run_slug,
)
from scripts.utils.episodes import (
    broadcast_goal,
    build_obs_batch,
    default_demo_path,
    ensure_local_dataset,
    parse_frame_spec,
    resolve_env_id,
    resolve_trajectory_key,
    trajectory_keys,
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
from scripts.utils.taps import capture_pre_norm

SCRIPT_NAME = Path(__file__).stem

# Slices into the raw StackCube-family observation, matching `StackCubeEnv.STATE_SCHEMA`. Only used
# by the semantic-stage detector, which is specific to this task family.
TCP_TO_CUBE_A = slice(39, 42)
CUBE_A_POS = slice(25, 28)
CUBE_A_TO_CUBE_B = slice(45, 48)

# How far the cube must rise above its resting height to count as lifted.
LIFT_HEIGHT = 0.005
# Where a successfully placed cube A sits relative to cube B, and how close counts as placed.
PLACE_TARGET_OFFSET = np.array([0.0, 0.0, -0.04])
PLACE_TOLERANCE = 0.02


def detect_key_moments(obs: np.ndarray) -> dict[str, int]:
    """Finds the frames where the robot grasps, lifts, and places the cube."""
    seq_len = len(obs)
    cube_height = obs[:, CUBE_A_POS][:, 2]
    risen = np.where(cube_height - cube_height[0] > LIFT_HEIGHT)[0]
    lift = int(risen[0]) if risen.size else seq_len // 2

    approach_distance = np.linalg.norm(obs[:, TCP_TO_CUBE_A], axis=-1)
    grab = int(np.argmin(approach_distance[:lift])) if lift > 0 else 0

    placement_error = np.linalg.norm(obs[:, CUBE_A_TO_CUBE_B] - PLACE_TARGET_OFFSET, axis=-1)
    placed = np.where((placement_error < PLACE_TOLERANCE) & (np.arange(seq_len) > grab))[0]
    place = int(placed[0]) if placed.size else int(grab + np.argmin(placement_error[grab:]))

    if place > grab + 1:
        midair = int(grab + np.argmax(cube_height[grab:place]))
    else:
        midair = (grab + place) // 2

    return {"grab": grab, "midair": midair, "place": place}


def stage_markers(obs: np.ndarray) -> list[tuple[int, str, str]]:
    """`(frame, label, colour)` for each semantic stage, in time order."""
    moments = detect_key_moments(obs)
    return [
        (0, "start", theme.SERIES[0]),
        (moments["grab"], "grab", theme.SERIES[1]),
        (moments["midair"], "mid-air", theme.SERIES[2]),
        (moments["place"], "place", theme.SERIES[3]),
    ]


# --- Sources ----------------------------------------------------------------------------------


def load_from_dataset(
    dataset_path: Path, episode_idx: int, num_background: int, seed: int
) -> tuple[str, np.ndarray, list[np.ndarray]]:
    """Reads the target episode plus a random sample of others for context."""
    with h5py.File(dataset_path, "r") as handle:
        keys = trajectory_keys(handle)
        target_key = resolve_trajectory_key(handle, episode_idx)

        group = handle[target_key]
        if not isinstance(group, h5py.Group):
            raise TypeError(f"Expected an HDF5 group at {target_key}.")
        target = np.asarray(group["obs"])

        others = [key for key in keys if key != target_key]
        if len(others) > num_background:
            others = random.Random(seed).sample(others, num_background)

        background = []
        for key in others:
            group = handle[key]
            if isinstance(group, h5py.Group):
                background.append(np.asarray(group["obs"]))

    print(
        f"Loaded {target_key} ({len(target)} frames) plus {len(background)} background episodes."
    )
    return target_key, target, background


def load_from_rollout(
    model, cfg, env_kwargs: dict, args: argparse.Namespace
) -> tuple[str, np.ndarray, list[np.ndarray], list[bool], Callable, TensorTree]:
    """Runs live episodes in one environment.

    Returns the first episode as the analysis target and the rest as background context.
    """
    max_episode_steps = resolve_max_episode_steps(cfg, args.max_episode_steps)
    env, inner_env = build_rollout_env(
        env_kwargs, model.obs_horizon, max_episode_steps, args.render_mode, args.video_dir
    )
    transform = build_obs_transform(env, env_kwargs, cfg)

    episodes: list[np.ndarray] = []
    successes: list[bool] = []
    goal_frames: list = []

    def record(step):
        if not goal_frames:
            goal_frames.append(step.goal_raw)
        return map_leaves(lambda t: t[:, -1].squeeze(0), step.obs_raw)

    try:
        for index in range(args.num_episodes):
            rollout = run_episode(
                model,
                env,
                inner_env,
                transform,
                episode_idx=index,
                seed=args.seed + index,
                num_inference_steps=args.num_inference_steps,
                clamp_action=args.clamp_action,
                render_mode=args.render_mode,
                refresh_goal_each_step=False,
                on_step=record,
            )
            frames = stack_dicts(rollout.records)
            assert isinstance(frames, torch.Tensor)  # flat observations stack into a tensor
            episodes.append(np.asarray(frames.detach().cpu()))
            successes.append(rollout.success_once)
            print(
                f"  episode {index}: {rollout.num_steps} steps, "
                f"success_once={rollout.success_once}"
            )
    finally:
        env.close()

    goal_raw = map_leaves(lambda t: t.squeeze(0), goal_frames[0])
    return (
        f"live seed={args.seed}",
        episodes[0],
        episodes[1:],
        successes,
        transform,
        goal_raw,
    )


# --- Projection -------------------------------------------------------------------------------


def embed(
    model, obs_tree, goal, transform, seq_len: int
) -> tuple[np.ndarray, np.ndarray | None, str]:
    """Embeds every frame of an episode, and reports which kind of embedding it produced.

    Two architectures need different treatment:

    - Embedders that can encode a single state on their own give an *absolute* embedding.
    - Token-based embedders such as `ObjectTokenizer` only ever see goal-relative deltas
      (`supports_single_side=False`), so no standalone state embedding exists. There the
      comparable quantity is the conditioning vector the network actually receives.

    `model.extract_embeddings` reports which of the two it produced; see
    `ConditioningEncoder.extract_embeddings`.

    Also returns the pre-normalisation representation when the embedder has an output norm. That
    norm pins the magnitude of the returned embedding, so any distance computed from it is confined
    to a narrow band -- see `scripts/utils/taps.py`. None when there is no such norm.
    """
    frame_indices = list(range(seq_len))
    obs_batch = transform(
        build_obs_batch(obs_tree, frame_indices, model.obs_horizon, model.device)
    )
    goal_batch = transform(broadcast_goal(goal, seq_len))

    with torch.no_grad(), capture_pre_norm(model.encoder.embedder) as pre_norm:
        embeddings_dict, mode = model.extract_embeddings(obs_batch, goal=goal_batch)
        embeddings = embeddings_dict["obs_embeddings"]

        # The first capture is the observation branch; goal-delta modes that embed the goal
        # separately append a second one after it.
        pre = pre_norm[0].reshape(seq_len, -1).cpu().numpy() if pre_norm else None

    return embeddings.detach().cpu().numpy(), pre, mode


def project(embeddings: np.ndarray, reducer: str, seed: int) -> dict[str, np.ndarray]:
    """Reduces embeddings to 2D with whichever methods were asked for."""
    projections: dict[str, np.ndarray] = {}
    choice = reducer.lower()

    if choice in ("all", "pca"):
        try:
            from sklearn.decomposition import PCA

            projections["PCA"] = np.asarray(PCA(n_components=2).fit_transform(embeddings))
        except Exception as error:
            print(f"PCA failed: {error}")

    if choice in ("all", "tsne"):
        try:
            from sklearn.manifold import TSNE

            perplexity = min(30, max(1, len(embeddings) // 3))
            projections["t-SNE"] = np.asarray(
                TSNE(n_components=2, random_state=seed, perplexity=perplexity).fit_transform(
                    embeddings
                )
            )
        except Exception as error:
            print(f"t-SNE failed: {error}")

    if choice in ("all", "umap"):
        try:
            import umap

            neighbours = min(15, max(2, len(embeddings) // 2))
            projections["UMAP"] = np.asarray(
                umap.UMAP(n_components=2, random_state=seed, n_neighbors=neighbours).fit_transform(
                    embeddings
                )
            )
        except Exception as error:
            print(f"UMAP failed: {error}")

    if not projections:
        raise RuntimeError("No dimensionality reduction succeeded.")
    return projections


# --- Plotting ---------------------------------------------------------------------------------


def draw_panel(
    ax: plt.Axes,
    coords: np.ndarray,
    split_index: int,
    background_lengths: list[int],
    markers: list[tuple[int, str, str]],
    goal_index: int,
    name: str,
    equal_aspect: bool,
    goal_label: str,
) -> None:
    """Draws one projection: background paths, the time-coloured target path, markers, goal."""
    background = coords[:split_index]
    target = coords[split_index:]

    offset = 0
    for length in background_lengths:
        segment = background[offset : offset + length]
        offset += length
        # One muted colour for all background episodes: they are context, not identity, and
        # cycling hues here would imply distinctions that do not exist.
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            color=theme.TEXT_MUTED,
            alpha=0.10,
            linewidth=0.6,
            zorder=1,
        )

    points = target.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    line = LineCollection(
        segments.tolist(),
        cmap=plt.get_cmap(theme.SEQUENTIAL_CMAP),
        norm=plt.Normalize(0, len(target) - 1),
        linewidths=1.6,
        alpha=0.9,
        zorder=3,
    )
    line.set_array(np.arange(len(target) - 1))
    ax.add_collection(line)

    for index, label, color in markers:
        if index == goal_index or index >= len(target):
            continue
        ax.scatter(
            target[index, 0],
            target[index, 1],
            color=color,
            marker="o",
            s=40,
            edgecolors=theme.TEXT_PRIMARY,
            linewidths=1.0,
            zorder=6,
        )
        ax.annotate(
            label,
            (target[index, 0], target[index, 1]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7,
            color=color,
            zorder=7,
        )

    ax.scatter(
        target[goal_index, 0],
        target[goal_index, 1],
        color=theme.SERIES[4],
        marker="X",
        s=90,
        edgecolors=theme.TEXT_PRIMARY,
        linewidths=1.2,
        zorder=8,
    )
    ax.annotate(
        goal_label,
        (target[goal_index, 0], target[goal_index, 1]),
        textcoords="offset points",
        # Offset downward: the goal frequently coincides with the final stage marker, whose label
        # is offset upward.
        xytext=(6, -12),
        fontsize=7.5,
        color=theme.SERIES[4],
        zorder=9,
    )

    ax.set_title(f"{name} projection", fontsize=11, color=theme.TEXT_SECONDARY, pad=10)
    ax.set_xlabel("dimension 1", fontsize=9)
    ax.set_ylabel("dimension 2", fontsize=9)
    theme.style_axes(ax)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")


def plot_projections(
    projections: dict[str, np.ndarray],
    split_index: int,
    background_lengths: list[int],
    markers: list[tuple[int, str, str]],
    goal_index: int,
    fields,
    save_path: Path,
    *,
    goal_label: str,
    equal_aspect: bool,
    show: bool,
    dpi: int,
) -> None:
    """One panel per reduction method, sharing a legend."""
    fig, axes = plt.subplots(
        1, len(projections), figsize=(6 * len(projections), 6.5), squeeze=False
    )

    for index, (name, coords) in enumerate(projections.items()):
        draw_panel(
            axes[0][index],
            coords,
            split_index,
            background_lengths,
            markers,
            goal_index,
            name,
            equal_aspect,
            goal_label,
        )

    handles = [
        Line2D(
            [],
            [],
            color=plt.get_cmap(theme.SEQUENTIAL_CMAP)(0.5),
            lw=2.5,
            label="target episode (colour = time)",
        ),
        Line2D([], [], color=theme.TEXT_MUTED, lw=1.5, label="other episodes"),
        Line2D(
            [],
            [],
            color=theme.SERIES[4],
            marker="X",
            linestyle="none",
            markersize=9,
            label=goal_label,
        ),
        *[
            Line2D([], [], color=color, marker="o", linestyle="none", markersize=7, label=label)
            for _, label, color in markers
        ],
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=min(len(handles), 7),
        frameon=True,
        facecolor=theme.SURFACE,
        edgecolor=theme.GRID,
        fontsize=9,
    )

    theme.set_title(fig, "Latent embeddings of an episode", fields)
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    save_figure(fig, save_path, show=show, dpi=dpi, apply_tight_layout=False)
    print(f"Figure saved: {save_path}")


def stage_distance(embeddings: np.ndarray, index: int, goal_index: int, embed_mode: str) -> float:
    """Latent distance from one frame to the goal, measured the way the embedding allows."""
    if embed_mode == "goal-relative":
        return float(np.linalg.norm(embeddings[index]))
    return float(np.linalg.norm(embeddings[index] - embeddings[goal_index]))


def geometry_report(
    target_embeddings: np.ndarray,
    pre_norm_embeddings: np.ndarray | None,
    goal_index: int,
    markers: list[tuple[int, str, str]],
    embed_mode: str,
) -> list[list[str]]:
    """Distance in latent space from each stage to the goal."""
    rows = []
    for index, label, _ in markers:
        if index >= len(target_embeddings):
            continue
        row = [
            label,
            str(index),
            f"{stage_distance(target_embeddings, index, goal_index, embed_mode):.4f}",
        ]
        if pre_norm_embeddings is not None and index < len(pre_norm_embeddings):
            row.append(f"{stage_distance(pre_norm_embeddings, index, goal_index, embed_mode):.4f}")
        rows.append(row)
    return rows


# --- Entry point ------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        parents=[
            cli.checkpoint_args(),
            cli.output_args(),
            cli.source_args(),
            cli.dataset_args(),
            cli.rollout_args(default_num_episodes=6),
        ],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--reducer",
        choices=["all", "pca", "tsne", "umap"],
        default="pca",
        help="Dimensionality reduction to apply. (default: pca)",
    )
    parser.add_argument(
        "--goal-frame-idx",
        type=int,
        default=-1,
        help="Frame treated as the goal. Negative counts from the end. (default: -1)",
    )
    parser.add_argument(
        "--num-background-episodes",
        type=int,
        default=20,
        help="Other episodes drawn faintly for context, in --source dataset. (default: 20)",
    )
    parser.add_argument(
        "--highlight-stages",
        action="store_true",
        help="Mark the detected grasp / mid-air / place moments instead of fixed frame "
        "percentages. StackCube-family observations only.",
    )
    parser.add_argument(
        "--equal-aspect",
        action="store_true",
        help="Force equal axis scaling, so latent distances are visually comparable.",
    )
    return parser.parse_args()


def analyse(
    model,
    cfg,
    env_id: str,
    episode_label: str,
    target: np.ndarray,
    background: list[np.ndarray],
    extra: list[tuple[str, str]],
    args: argparse.Namespace,
    slug: str,
    transform,
    goal_for,
) -> None:
    """Embeds, projects, plots and reports one episode."""
    seq_len = len(target)
    goal_index = args.goal_frame_idx if args.goal_frame_idx >= 0 else seq_len + args.goal_frame_idx
    goal_index = max(0, min(goal_index, seq_len - 1))

    if args.highlight_stages:
        markers = stage_markers(target)
    else:
        markers = [
            (index, label, theme.SERIES[position % len(theme.SERIES)])
            for position, (index, label) in enumerate(parse_frame_spec(args.frames, seq_len))
        ]

    target_embeddings, target_pre_norm, embed_mode = embed(
        model, target, goal_for(target), transform, seq_len
    )
    background_embeddings = [
        embed(model, episode, goal_for(episode), transform, len(episode))[0]
        for episode in background
    ]
    background_lengths = [len(episode) for episode in background_embeddings]

    stacked = (
        np.concatenate([*background_embeddings, target_embeddings], axis=0)
        if background_embeddings
        else target_embeddings
    )
    split_index = sum(background_lengths)

    projections = project(stacked, args.reducer, args.seed)
    fields = describe_model_config(
        model,
        cfg,
        extra=[
            ("env", env_id),
            ("episode", episode_label),
            ("goal", str(goal_index)),
            ("embedding", embed_mode),
        ],
    )

    save_path = figure_path(
        SCRIPT_NAME,
        f"embeddings-{args.source}",
        env_id=env_id,
        run_slug=slug,
        out_dir=args.out_dir,
    )

    # In rollout mode the goal is the environment's heuristic goal, not a frame of the episode, so
    # marking a frame as "goal" would be wrong.
    goal_label = "goal frame" if args.source == "dataset" else "final frame"

    plot_projections(
        projections,
        split_index,
        background_lengths,
        markers,
        goal_index,
        fields,
        save_path,
        goal_label=goal_label,
        equal_aspect=args.equal_aspect,
        show=args.show,
        dpi=args.dpi,
    )

    report = Report("Latent embeddings", [("ckpt", str(args.ckpt_path)), *fields])
    report.section("Episode")
    report.kv("frames", seq_len)
    report.kv("embedding dim", target_embeddings.shape[1])
    report.kv("reductions", ", ".join(projections))
    report.kv("embedding kind", embed_mode)
    report.kv("background episodes", len(background))

    report.section("Latent distance to the goal state")
    headers = ["stage", "frame", "distance"]
    if target_pre_norm is not None:
        headers.append("distance (pre-norm)")
    report.table(
        headers,
        geometry_report(target_embeddings, target_pre_norm, goal_index, markers, embed_mode),
    )
    report.note(
        "If the embedding encodes progress toward the goal, this should decrease down the table."
    )
    if embed_mode == "goal-relative":
        report.note(
            "Measured as the magnitude of the goal-relative embedding, so the goal is the origin."
        )
    if target_pre_norm is not None:
        report.note(
            "The `distance` column is computed from the post-normalisation embedding, whose "
            "magnitude the output LayerNorm confines to a narrow band -- expect it to look flat. "
            "The `distance (pre-norm)` column is the same measurement one layer earlier, where the "
            "magnitude is free to move."
        )

    if extra:
        report.section("Run")
        for key, value in extra:
            report.kv(key, value)

    report.emit(save_path, save=not args.no_report)


def main() -> None:
    args = parse_args()
    theme.apply_theme()

    ensure_local_checkpoint(args.ckpt_path)
    model = load_goal_conditioned_diffusion_policy(args.ckpt_path)
    cfg = require_run_config(args.ckpt_path)
    slug = args.run_label or run_slug(args.ckpt_path, model, cfg, args.seed)

    if args.source == "rollout":
        for _, _, env_kwargs in iter_env_kwargs(cfg, args.env_id):
            (
                episode_label,
                target,
                background,
                successes,
                transform,
                goal_raw,
            ) = load_from_rollout(model, cfg, env_kwargs, args)
            analyse(
                model,
                cfg,
                env_kwargs["env_id"],
                episode_label,
                target,
                background,
                [
                    ("source", "rollout"),
                    ("success_once", f"{sum(successes)}/{len(successes)}"),
                ],
                args,
                slug,
                transform,
                # The heuristic goal is a property of the environment, so every episode in this
                # env is measured against the same one.
                lambda _episode: goal_raw,
            )
        return

    configured_env = str(cfg.get("datamodule", {}).get("env_id", "") or "")
    env_id = (args.env_id or [None])[0] or configured_env or "StackCubeLockedRotation-v1"
    dataset_path = ensure_local_dataset(args.dataset_path or default_demo_path(env_id))
    env_id = resolve_env_id(dataset_path, env_id)

    episode_label, target, background = load_from_dataset(
        dataset_path, args.episode_idx, args.num_background_episodes, args.seed
    )

    transform = observation_pipeline(
        env_id=env_id,
        canonicalize=bool(cfg.get("datamodule", {}).get("canonicalize", True)),
    )

    def goal_from_episode(episode: np.ndarray):
        index = (
            args.goal_frame_idx if args.goal_frame_idx >= 0 else len(episode) + args.goal_frame_idx
        )
        index = max(0, min(index, len(episode) - 1))
        return to_tensor(episode[index], device=model.device, dtype=torch.float32)

    analyse(
        model,
        cfg,
        env_id,
        episode_label,
        target,
        background,
        [("source", "dataset"), ("dataset", str(dataset_path))],
        args,
        slug,
        transform,
        goal_from_episode,
    )


if __name__ == "__main__":
    main()
