"""Measures how close a policy's state embedder is to being a plain affine map.

This script captures the embedder's (input, output) pairs and scores:

- **R^2** of the best-fit affine map: how much of the output an affine map explains.
- **linear CKA**: how linearly related the two representations are overall, independent of width.
- **cosine similarity** between the true output and the affine map's prediction.

Only the embedder is used; the diffusion network is never run.

With `--source rollout` the same measurement is taken on live rollouts instead of recorded data,
including the zero-shot environments.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import r2_score

from policy.utils import map_leaves
from scripts.utils import cli, theme
from scripts.utils.checkpoints import (
    describe_model_config,
    ensure_local_checkpoint,
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
from scripts.utils.taps import capture_pre_norm, embedder_output_norm

SCRIPT_NAME = Path(__file__).stem

NEAR_LINEAR = 0.95
MOSTLY_LINEAR = 0.80


class EmbedderCapture:
    """Collects what the embedder sees and produces, via forward hooks."""

    def __init__(self, model) -> None:
        if model.encoder is None:
            raise RuntimeError("configure_model() must run before the embedder can be captured.")
        self.model = model
        self.embedder = model.encoder.embedder
        self.inputs: list[torch.Tensor] = []
        self.outputs: list[torch.Tensor] = []
        self.pre_norm_outputs: list[torch.Tensor] = []
        self._handle = None
        self._pre_norm_ctx = None

    def __enter__(self) -> EmbedderCapture:
        def hook(_module, inputs, output):
            self.inputs.append(inputs[0].detach().cpu())
            self.outputs.append(output.detach().cpu())

        self._handle = self.embedder.register_forward_hook(hook)
        self._pre_norm_ctx = capture_pre_norm(self.embedder)
        self._captured_pre_norm = self._pre_norm_ctx.__enter__()
        return self

    def __exit__(self, *_exc) -> None:
        if self._handle is not None:
            self._handle.remove()
        if self._pre_norm_ctx is not None:
            # Move the captures out before the hook is torn down, so they survive the context.
            self.pre_norm_outputs = [t.cpu() for t in self._captured_pre_norm]
            self._pre_norm_ctx.__exit__(None, None, None)

    def stacked(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Returns the captured pairs as `(inputs, outputs)`, one row per sample."""
        if not self.inputs:
            raise RuntimeError("The embedder was never called; nothing was captured.")
        x = torch.cat([t.reshape(t.shape[0], -1) for t in self.inputs], dim=0)
        y = torch.cat([t.reshape(t.shape[0], -1) for t in self.outputs], dim=0)
        if x.shape[0] != y.shape[0]:
            raise RuntimeError(
                f"Captured {x.shape[0]} embedder inputs but {y.shape[0]} outputs; they cannot be "
                "paired. This embedder changes the batch size internally."
            )

        pre = None
        if self.pre_norm_outputs:
            stacked_pre = torch.cat(
                [t.reshape(t.shape[0], -1) for t in self.pre_norm_outputs], dim=0
            )
            if stacked_pre.shape[0] == x.shape[0]:
                pre = stacked_pre.double().numpy()

        return x.double().numpy(), y.double().numpy(), pre


def capture_from_dataset(model, cfg, split: str, num_batches: int | None, seed: int):
    """Replays recorded batches through the embedder."""
    datamodule = hydra.utils.instantiate(cfg.datamodule, num_workers=0)
    datamodule.setup(stage="fit")
    dataloader = datamodule.train_dataloader() if split == "train" else datamodule.val_dataloader()

    # HER goal relabelling draws from the global torch RNG on every item, so seed here to keep the
    # captured pairs reproducible between runs.
    torch.manual_seed(seed)

    with EmbedderCapture(model) as capture, torch.no_grad():
        for index, batch in enumerate(dataloader):
            if num_batches is not None and index >= num_batches:
                break
            obs = map_leaves(lambda t: t.to(model.device), batch["obs_seq"])
            goal = batch.get("goal")
            if goal is not None:
                goal = map_leaves(lambda t: t.to(model.device), goal)
            if model.obs_normalizer is not None:
                obs = model.obs_normalizer.normalize(obs)
                if goal is not None:
                    goal = model.obs_normalizer.normalize(goal)
            # Goes through the algorithm's ConditioningEncoder directly (not
            # model._build_external_cond, which just passes obs/goal through raw now) so the
            # embedder's forward hook actually fires.
            model.encoder(obs, goal)

    return capture.stacked()


def capture_from_rollout(model, cfg, env_kwargs: dict, args: argparse.Namespace):
    """Drives the policy live and captures the embedder as the rollout proceeds."""
    max_episode_steps = resolve_max_episode_steps(cfg, args.max_episode_steps)
    env, inner_env = build_rollout_env(
        env_kwargs, model.obs_horizon, max_episode_steps, args.render_mode, args.video_dir
    )
    transform = build_obs_transform(env, env_kwargs, cfg)

    successes = 0
    try:
        with EmbedderCapture(model) as capture:
            for episode in range(args.num_episodes):
                result = run_episode(
                    model,
                    env,
                    inner_env,
                    transform,
                    episode_idx=episode,
                    seed=args.seed + episode,
                    num_inference_steps=args.num_inference_steps,
                    clamp_action=args.clamp_action,
                    render_mode=args.render_mode,
                )
                successes += int(result.success_once)
                print(
                    f"  episode {episode}: {result.num_steps} steps, "
                    f"success_once={result.success_once}"
                )
        return capture.stacked(), successes
    finally:
        env.close()


def fit_affine_map(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares affine map `y ~= x @ w + b`."""
    augmented = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    coefficients, *_ = np.linalg.lstsq(augmented, y, rcond=None)
    return coefficients[:-1], coefficients[-1]


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA (Kornblith et al., 2019).

    Bounded in [0, 1], invariant to rotation, and defined for representations of different width.
    It reaches 1.0 exactly when the two are related by an orthogonal transform.
    """
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    hsic = np.linalg.norm(y.T @ x, ord="fro") ** 2  # codespell:ignore fro
    norm_x = np.linalg.norm(x.T @ x, ord="fro")  # codespell:ignore fro
    norm_y = np.linalg.norm(y.T @ y, ord="fro")  # codespell:ignore fro
    return float(hsic / (norm_x * norm_y))


def mean_cosine_similarity(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Row-wise cosine similarity between two equal-shaped matrices, as `(mean, std)`."""
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    valid = denominator > 0
    cosine = np.full(a.shape[0], np.nan)
    cosine[valid] = np.sum(a[valid] * b[valid], axis=1) / denominator[valid]
    return float(np.nanmean(cosine)), float(np.nanstd(cosine))


def verdict(r2: float, cka: float) -> str:
    """A one-line reading of the numbers."""
    if r2 > NEAR_LINEAR and cka > NEAR_LINEAR:
        return "near-linear: a single Linear layer would likely do the same job."
    if r2 > MOSTLY_LINEAR and cka > MOSTLY_LINEAR:
        return "mostly linear: some nonlinear contribution, but an affine map explains the bulk."
    return "meaningfully nonlinear: the embedder's nonlinear capacity is doing real work."


def score(x: np.ndarray, y: np.ndarray, test_split: float, seed: int) -> dict:
    """Fits the affine map on a random split and scores it on the held-out remainder."""
    n = x.shape[0]
    permutation = np.random.default_rng(seed).permutation(n)
    n_test = max(1, int(round(n * test_split)))
    test_idx, train_idx = permutation[:n_test], permutation[n_test:]

    w, b = fit_affine_map(x[train_idx], y[train_idx])
    predicted = x[test_idx] @ w + b

    cosine_mean, cosine_std = mean_cosine_similarity(y[test_idx], predicted)
    return {
        "num_samples": n,
        "task_dim": x.shape[1],
        "output_dim": y.shape[1],
        "r2": float(r2_score(y[test_idx], predicted)),
        "cka": linear_cka(x, y),
        "cosine_mean": cosine_mean,
        "cosine_std": cosine_std,
        "y_true": y[test_idx],
        "y_pred": predicted,
    }


def plot_linearity(
    results: dict[str, dict], title_fields: list[tuple[str, str]], save_path: Path, *, show, dpi
) -> None:
    """Left: how well an affine map predicts the embedder. Right: the scores side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axes[0]
    for index, (label, result) in enumerate(results.items()):
        true_values = result["y_true"].ravel()
        predicted = result["y_pred"].ravel()
        # Subsample so a long rollout does not draw millions of near-identical points.
        if true_values.size > 4000:
            pick = np.random.default_rng(0).choice(true_values.size, 4000, replace=False)
            true_values, predicted = true_values[pick], predicted[pick]
        ax.scatter(
            true_values,
            predicted,
            s=4,
            alpha=0.35,
            color=theme.SERIES[index % len(theme.SERIES)],
            label=label,
            edgecolors="none",
        )

    limits = np.array(ax.get_xlim())
    ax.plot(limits, limits, color=theme.TEXT_MUTED, linestyle="--", linewidth=1)
    ax.set_xlabel("embedder output")
    ax.set_ylabel("best affine map's prediction")
    ax.set_title(
        "Points on the diagonal mean affine-explainable",
        fontsize=10,
        color=theme.TEXT_SECONDARY,
    )
    theme.style_axes(ax)
    ax.legend(facecolor=theme.SURFACE, edgecolor=theme.GRID, fontsize=8)

    ax = axes[1]
    labels = list(results)
    metrics = [("R²", "r2"), ("CKA", "cka"), ("cos", "cosine_mean")]
    width = 0.8 / len(metrics)
    positions = np.arange(len(labels))
    for index, (metric_label, key) in enumerate(metrics):
        ax.bar(
            positions + index * width,
            [results[label][key] for label in labels],
            width=width * 0.9,
            color=theme.SERIES[index],
            label=metric_label,
        )
    ax.axhline(NEAR_LINEAR, color=theme.TEXT_MUTED, linestyle="--", linewidth=1)
    ax.set_xticks(positions + width)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score (1.0 = perfectly affine)")
    ax.set_title("Higher means more linear", fontsize=10, color=theme.TEXT_SECONDARY)
    theme.style_axes(ax)
    ax.legend(facecolor=theme.SURFACE, edgecolor=theme.GRID, fontsize=8)

    theme.set_title(fig, "Embedder linearity", title_fields)
    save_figure(fig, save_path, show=show, dpi=dpi)
    print(f"Figure saved: {save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        parents=[
            cli.checkpoint_args(),
            cli.output_args(),
            cli.source_args(),
            cli.rollout_args(default_num_episodes=3),
        ],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="val",
        help="Dataset split to draw embedder inputs from, in --source dataset. (default: val)",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=None,
        help="Cap on batches drawn from the dataloader. (default: the whole split)",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Fraction of captured samples held out to score the affine fit. (default: 0.2)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    theme.apply_theme()

    ensure_local_checkpoint(args.ckpt_path)
    model = load_goal_conditioned_diffusion_policy(args.ckpt_path)
    cfg = require_run_config(args.ckpt_path)
    slug = args.run_label or run_slug(args.ckpt_path, model, cfg, args.seed)

    results: dict[str, dict] = {}
    extra_report_rows: list[list[str]] = []

    def add(label: str, x, y, pre) -> None:
        """Scores the affine fit against the embedder output, and against its pre-norm form."""
        suffix = " (post-norm)" if pre is not None else ""
        results[label + suffix] = score(x, y, args.test_split, args.seed)
        if pre is not None:
            results[label + " (pre-norm)"] = score(x, pre, args.test_split, args.seed)

    if args.source == "dataset":
        x, y, pre = capture_from_dataset(model, cfg, args.split, args.num_batches, args.seed)
        add(f"dataset/{args.split}", x, y, pre)
    else:
        for _, _, env_kwargs in iter_env_kwargs(cfg, args.env_id):
            (x, y, pre), successes = capture_from_rollout(model, cfg, env_kwargs, args)
            add(env_kwargs["env_id"], x, y, pre)
            extra_report_rows.append([env_kwargs["env_id"], f"{successes}/{args.num_episodes}"])

    if model.encoder is None:
        raise RuntimeError("configure_model() must run before the embedder is available.")
    norm = embedder_output_norm(model.encoder.embedder)
    fields = describe_model_config(
        model,
        cfg,
        extra=[("source", args.source), ("output_norm", type(norm).__name__ if norm else "none")],
    )
    save_path = figure_path(
        SCRIPT_NAME,
        f"linearity-{args.source}",
        run_slug=slug,
        out_dir=args.out_dir,
    )
    plot_linearity(results, fields, save_path, show=args.show, dpi=args.dpi)

    report = Report("Embedder linearity", [("ckpt", str(args.ckpt_path)), *fields])
    report.section("Scores")
    report.table(
        ["source", "samples", "in-dim", "out-dim", "R^2", "CKA", "cos(affine)"],
        [
            [
                label,
                result["num_samples"],
                result["task_dim"],
                result["output_dim"],
                f"{result['r2']:.4f}",
                f"{result['cka']:.4f}",
                f"{result['cosine_mean']:.4f} +/- {result['cosine_std']:.4f}",
            ]
            for label, result in results.items()
        ],
    )

    if extra_report_rows:
        report.section("Rollout success").table(["env", "success_once"], extra_report_rows)

    report.section("Reading")
    for label, result in results.items():
        report.kv(label, verdict(result["r2"], result["cka"]))
    if norm is not None:
        report.note(
            f"The embedder ends in a {type(norm).__name__}, which is itself nonlinear. The "
            "post-norm score therefore describes the embedder and the normalisation together; the "
            "pre-norm score isolates the embedder's own map."
        )

    report.emit(save_path, save=not args.no_report)


if __name__ == "__main__":
    main()
