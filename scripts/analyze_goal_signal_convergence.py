"""Checks whether the goal-conditioning signal fades as a rollout reaches its goal.

`z` is the literal conditioning tensor that reaches the diffusion network's FiLM layers. If goal
conditioning is working as intended, `||z||` should shrink as the robot closes in on the goal.

Runs live episodes driven by the checkpoint's own policy and records. Produces
    - `||z||` over time,
    - `||z||` against true distance
    - a per-episode first-to-last convergence ratio

By default it sweeps the whole LockedRotation family, so one invocation covers the training task
and the three zero-shot targets.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

import policy.environments  # noqa: F401  (registers the project's envs as a side effect)
from policy.algorithms.goal_conditioned_diffusion_policy import GoalConditionedDiffusionPolicy
from policy.utils.typing_utils import TensorTree, get_subtree, get_tensor
from scripts.utils import cli, theme
from scripts.utils.checkpoints import (
    describe_model_config,
    load_goal_conditioned_diffusion_policy,
    require_run_config,
    run_slug,
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
from scripts.utils.taps import (
    capture_pre_norm,
    embedder_output_norm,
    magnitude_band,
    relative_spread,
)

SCRIPT_NAME = Path(__file__).stem


@dataclass
class StepRecord:
    """One environment step's conditioning signal and true distance to goal."""

    episode_idx: int
    step_idx: int
    z_norm: float
    """||z||, the conditioning vector the network receives."""
    h_norm: float | None
    """||h||, the same representation one layer earlier, before the output normalisation."""
    gt_distance: float


@dataclass
class EpisodeResult:
    """Per-episode step records plus ManiSkill's own success bookkeeping."""

    episode_idx: int
    steps: list[StepRecord]
    success_once: bool
    success_at_end: bool
    episode_len: int


def compute_z(
    model: GoalConditionedDiffusionPolicy, external_cond: dict[str, TensorTree]
) -> torch.Tensor:
    """The per-step goal-delta vector `z`, shape [B, D]."""

    # The generated tensor can get structured in quite diifferent ways depending on the embedder and pooling,
    # so we need to do extra work to fetch the correct tensor.
    pools_time = model._embedder_pools_time()
    task = (
        get_tensor(external_cond, "task")
        if pools_time
        else get_tensor(get_subtree(external_cond, "obs"), "task")
    )
    current = task[:, -1] if task.ndim == 3 else task  # most recently observed frame

    if model.goal_delta is None:
        goal_value = external_cond["goal"]
        goal_task = (
            goal_value if isinstance(goal_value, torch.Tensor) else get_tensor(goal_value, "task")
        )
        return current - goal_task
    return current


def z_norm(model: GoalConditionedDiffusionPolicy, external_cond: dict[str, TensorTree]) -> float:
    """`||z||` for a single-env rollout."""
    return float(torch.linalg.norm(compute_z(model, external_cond), dim=-1).item())


def ground_truth_distance(
    obs_canonical: Mapping[str, TensorTree], goal_canonical: Mapping[str, TensorTree]
) -> float:
    """Straight-line distance from the manipulated object to its goal position."""
    obs_pose = get_tensor(obs_canonical, "a_pose")
    obs_position = obs_pose[:, -1, :3] if obs_pose.ndim == 3 else obs_pose[..., :3]
    goal_position = get_tensor(goal_canonical, "a_pose")[..., :3]
    return float(torch.linalg.norm(goal_position - obs_position, dim=-1).item())


def collect_episodes(
    model, cfg, env_kwargs: dict, args: argparse.Namespace
) -> list[EpisodeResult]:
    """Runs the requested episodes in one env, recording a step trace for each."""
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

    results: list[EpisodeResult] = []
    try:
        for episode in range(args.num_episodes):
            with capture_pre_norm(model.embedder) as pre_norm:

                def record(step, pre_norm=pre_norm) -> StepRecord:
                    h = float(torch.linalg.norm(pre_norm[0].flatten())) if pre_norm else None
                    pre_norm.clear()
                    return StepRecord(
                        episode_idx=step.episode_idx,
                        step_idx=step.step_idx,
                        z_norm=z_norm(model, step.external_cond),
                        h_norm=h,
                        gt_distance=ground_truth_distance(step.obs, step.goal),
                    )

                rollout = run_episode(
                    model,
                    env,
                    inner_env,
                    transform,
                    episode_idx=episode,
                    seed=args.seed + episode,
                    num_inference_steps=args.num_inference_steps,
                    clamp_action=args.clamp_action,
                    render_mode=args.render_mode,
                    on_step=record,
                )
            results.append(
                EpisodeResult(
                    episode_idx=episode,
                    steps=rollout.records,
                    success_once=rollout.success_once,
                    success_at_end=rollout.success_at_end,
                    episode_len=rollout.episode_len,
                )
            )
            print(
                f"  episode {episode}: {len(rollout.records)} steps, "
                f"success_once={rollout.success_once}, success_at_end={rollout.success_at_end}"
            )
    finally:
        env.close()

    return results


SIGNALS = (
    ("z", "||z|| (post-norm, what FiLM receives)"),
    ("h", "||h|| (pre-norm)"),
)


def signal_values(result: EpisodeResult, which: str) -> list[float]:
    """The chosen signal's per-step magnitudes, dropping steps where it is unavailable."""
    if which == "z":
        return [step.z_norm for step in result.steps]
    return [step.h_norm for step in result.steps if step.h_norm is not None]


def has_signal(results: list[EpisodeResult], which: str) -> bool:
    return any(signal_values(result, which) for result in results)


def convergence_ratio(result: EpisodeResult, which: str = "z") -> float:
    """Last-step magnitude relative to the first.

    Below 1.0 means the signal shrank.
    """
    values = signal_values(result, which)
    if not values or values[0] <= 0:
        return float("nan")
    return values[-1] / values[0]


def correlation(signal: np.ndarray, distance: np.ndarray) -> str:
    """Pearson correlation, or a note when there is not enough variance to compute one."""
    if len(signal) < 2 or np.unique(signal).size < 2 or np.unique(distance).size < 2:
        return "n/a (insufficient variance)"
    return f"{np.corrcoef(signal, distance)[0, 1]:.4f}"


def outcome_color(result: EpisodeResult) -> str:
    return theme.SUCCESS if result.success_once else theme.FAILURE


def draw_band(ax: plt.Axes, band: tuple[float, float] | None) -> Line2D | None:
    """Shades the range the output normalisation confines this signal to."""
    if band is None:
        return None
    ax.axhspan(band[0], band[1], color=theme.TEXT_MUTED, alpha=0.10, zorder=0)
    return Line2D([], [], color=theme.TEXT_MUTED, lw=6, alpha=0.3, label="reachable by LayerNorm")


def plot_over_time(
    results: list[EpisodeResult], band, fields, save_path: Path, *, show, dpi
) -> None:
    """One panel per signal, each line an episode coloured by its outcome."""
    signals = [(key, label) for key, label in SIGNALS if has_signal(results, key)]
    fig, axes = plt.subplots(1, len(signals), figsize=(6.5 * len(signals), 5.5), squeeze=False)

    for index, (key, label) in enumerate(signals):
        ax = axes[0][index]
        for result in results:
            values = signal_values(result, key)
            if not values:
                continue
            ax.plot(
                range(len(values)),
                values,
                color=outcome_color(result),
                alpha=0.75,
                linewidth=1.4,
            )
        handles = list(theme.outcome_legend_handles())
        if key == "z":
            band_handle = draw_band(ax, band)
            if band_handle is not None:
                handles.append(band_handle)
        ax.set_xlabel("environment step")
        ax.set_ylabel(label)
        theme.style_axes(ax)
        ax.legend(handles=handles, loc="best", facecolor=theme.SURFACE, edgecolor=theme.GRID)

    theme.set_title(fig, "Goal-conditioning signal over a rollout", fields)
    save_figure(fig, save_path, show=show, dpi=dpi)
    print(f"Figure saved: {save_path}")


def plot_vs_distance(
    results: list[EpisodeResult], num_bins: int, band, fields, save_path: Path, *, show, dpi
) -> None:
    """All steps pooled against true distance, with a binned mean through the scatter."""
    signals = [(key, label) for key, label in SIGNALS if has_signal(results, key)]
    fig, axes = plt.subplots(1, len(signals), figsize=(6.5 * len(signals), 5.5), squeeze=False)

    for index, (key, label) in enumerate(signals):
        ax = axes[0][index]
        for result in results:
            values = signal_values(result, key)
            if not values:
                continue
            ax.scatter(
                [step.gt_distance for step in result.steps][: len(values)],
                values,
                color=outcome_color(result),
                alpha=0.35,
                s=12,
                linewidths=0,
            )

        distances = np.array(
            [step.gt_distance for r in results for step in r.steps[: len(signal_values(r, key))]]
        )
        magnitudes = np.array([v for r in results for v in signal_values(r, key)])

        handles = list(theme.outcome_legend_handles())
        if key == "z":
            band_handle = draw_band(ax, band)
            if band_handle is not None:
                handles.append(band_handle)

        if distances.size and distances.max() > distances.min():
            edges = np.linspace(distances.min(), distances.max(), num_bins + 1)
            which_bin = np.clip(np.digitize(distances, edges) - 1, 0, num_bins - 1)
            centers, means, deviations = [], [], []
            for bin_index in range(num_bins):
                mask = which_bin == bin_index
                if not mask.any():
                    continue
                centers.append((edges[bin_index] + edges[bin_index + 1]) / 2)
                means.append(magnitudes[mask].mean())
                deviations.append(magnitudes[mask].std())
            centers, means, deviations = map(np.array, (centers, means, deviations))
            ax.plot(centers, means, color=theme.TEXT_PRIMARY, linewidth=2)
            ax.fill_between(
                centers,
                means - deviations,
                means + deviations,
                color=theme.TEXT_PRIMARY,
                alpha=0.15,
            )
            handles.append(Line2D([], [], color=theme.TEXT_PRIMARY, lw=2, label="binned mean"))

        ax.set_xlabel("true distance to goal")
        ax.set_ylabel(label)
        theme.style_axes(ax)
        ax.legend(handles=handles, loc="best", facecolor=theme.SURFACE, edgecolor=theme.GRID)

    theme.set_title(fig, "Conditioning signal against true distance to goal", fields)
    save_figure(fig, save_path, show=show, dpi=dpi)
    print(f"Figure saved: {save_path}")


def plot_convergence_ratio(results: list[EpisodeResult], fields, save_path: Path, *, show, dpi):
    """Per-episode last/first ratio for each signal; at or above 1.0 means it never converged."""
    valid = [result for result in results if result.steps]
    signals = [(key, label) for key, label in SIGNALS if has_signal(results, key)]

    fig, ax = plt.subplots(figsize=(max(7, len(valid) * 1.1), 6))
    positions = np.arange(len(valid))
    width = 0.8 / max(1, len(signals))

    for index, (key, label) in enumerate(signals):
        ax.bar(
            positions + index * width,
            [convergence_ratio(result, key) - 1.0 for result in valid],
            width=width * 0.9,
            color=theme.SERIES[index],
            label=label,
        )

    ax.axhline(0.0, color=theme.TEXT_MUTED, linewidth=1)
    ax.set_xticks(positions + width * (len(signals) - 1) / 2)
    ax.set_xticklabels([f"ep {result.episode_idx}" for result in valid], fontsize=9)
    for tick, result in zip(ax.get_xticklabels(), valid, strict=False):
        tick.set_color(outcome_color(result))

    ax.set_xlabel("episode (label colour = success / failure)")
    ax.set_ylabel("change in magnitude:  last / first  -  1")
    theme.style_axes(ax)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=len(signals),
        facecolor=theme.SURFACE,
        edgecolor=theme.GRID,
    )
    theme.set_title(fig, "Per-episode convergence of the conditioning signal", fields)
    save_figure(fig, save_path, show=show, dpi=dpi)
    print(f"Figure saved: {save_path}")


def build_report(results: list[EpisodeResult], fields, ckpt_path: Path, band, norm_name) -> Report:
    """Per-episode traces, how much each signal actually moves, and the correlations."""
    report = Report("Goal-signal convergence", [("ckpt", str(ckpt_path)), *fields])
    signals = [(key, label) for key, label in SIGNALS if has_signal(results, key)]

    report.section("Per-episode")
    report.table(
        ["episode", "steps", "signal", "first", "last", "ratio", "success_once", "success_end"],
        [
            [
                result.episode_idx,
                len(signal_values(result, key)),
                key,
                f"{signal_values(result, key)[0]:.4f}" if signal_values(result, key) else "-",
                f"{signal_values(result, key)[-1]:.4f}" if signal_values(result, key) else "-",
                f"{convergence_ratio(result, key):.4f}",
                result.success_once,
                result.success_at_end,
            ]
            for result in results
            for key, _ in signals
        ],
    )

    distances = np.array([step.gt_distance for r in results for step in r.steps])
    succeeded = np.array([r.success_once for r in results for _ in r.steps], dtype=bool)

    report.section("How much each signal actually moves")
    rows = []
    for key, label in signals:
        values = np.array([v for r in results for v in signal_values(r, key)])
        rows.append(
            [
                label,
                f"{values.mean():.4f}",
                f"{values.std():.4f}",
                f"{relative_spread(values):.5f}",
            ]
        )
    report.table(["signal", "mean", "std", "relative spread"], rows)

    if band is not None:
        report.kv(f"{norm_name} reachable band", f"[{band[0]:.3f}, {band[1]:.3f}]")
        report.note(
            "The embedder's output passes through a normalisation, which fixes the sum of squares "
            "of its normalised input. ||z|| is therefore confined to the band above no matter what "
            "the observation was: a ratio near 1.0 and a tiny relative spread are what that "
            "constraint produces, not evidence about the policy. Read ||h|| for whether the signal "
            "genuinely shrinks, and ||z|| for what the network is actually handed."
        )

    report.section("Correlation with true distance")
    for key, label in signals:
        values = np.array([v for r in results for v in signal_values(r, key)])
        usable = min(len(values), len(distances))
        report.kv(f"{label} — all steps", correlation(values[:usable], distances[:usable]))
        if succeeded[:usable].any():
            mask = succeeded[:usable]
            report.kv(
                f"{label} — successful",
                correlation(values[:usable][mask], distances[:usable][mask]),
            )
    report.note(
        "A correlation can look convincing even for a signal whose magnitude barely moves — check "
        "it against the relative spread above before reading anything into it."
    )

    successes = sum(result.success_once for result in results)
    report.section("Outcome").kv("success_once", f"{successes}/{len(results)}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        parents=[cli.checkpoint_args(), cli.output_args(), cli.rollout_args()],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=20,
        help="Bins used for the mean trend line over true distance. (default: 20)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    theme.apply_theme()

    if not args.ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {args.ckpt_path}")

    model = load_goal_conditioned_diffusion_policy(args.ckpt_path)
    cfg = require_run_config(args.ckpt_path)
    slug = args.run_label or run_slug(args.ckpt_path, model, cfg, args.seed)
    print(
        f"Loaded {type(model).__name__}: goal_delta={model.goal_delta!r} "
        f"obs_horizon={model.obs_horizon} act_horizon={model.act_horizon}"
    )

    norm = embedder_output_norm(model.embedder)
    band = magnitude_band(norm) if norm is not None else None
    norm_name = type(norm).__name__ if norm is not None else "none"
    if norm is None:
        print("Embedder has no output normalisation; ||z|| is unconstrained.")
    else:
        print(f"Embedder output passes through {norm_name}; ||z|| confined to {band}.")

    for _, _, env_kwargs in iter_env_kwargs(cfg, args.env_id):
        env_id = env_kwargs["env_id"]
        results = collect_episodes(model, cfg, env_kwargs, args)
        fields = describe_model_config(model, cfg, extra=[("env", env_id)])

        def path_for(name: str) -> Path:
            return figure_path(
                SCRIPT_NAME, name, env_id=env_id, run_slug=slug, out_dir=args.out_dir
            )

        plot_over_time(results, band, fields, path_for("z-vs-time"), show=args.show, dpi=args.dpi)
        plot_vs_distance(
            results,
            args.num_bins,
            band,
            fields,
            path_for("z-vs-distance"),
            show=args.show,
            dpi=args.dpi,
        )
        summary_path = path_for("z-convergence-ratio")
        plot_convergence_ratio(results, fields, summary_path, show=args.show, dpi=args.dpi)

        build_report(results, fields, args.ckpt_path, band, norm_name).emit(
            summary_path, save=not args.no_report
        )


if __name__ == "__main__":
    main()
