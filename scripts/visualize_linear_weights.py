"""Plots the weight matrices and bias vectors of a checkpoint's linear layers.

Pass `--list-modules` to discover which prefixes a checkpoint actually contains.

Alongside each figure it writes a report of per-layer weight statistics (spread, saturation, how
much of the layer is effectively zero).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.utils import cli, theme
from scripts.utils.checkpoints import checkpoint_slug, ensure_local_checkpoint
from scripts.utils.figures import figure_path, save_figure, slugify
from scripts.utils.report import Report

SCRIPT_NAME = Path(__file__).stem

# Matrices at most this wide/tall get their values written into the cells.
ANNOTATE_MAX_DIM = 12
# Above this many rows/columns, per-index ticks become unreadable.
TICK_MAX_DIM = 50


def get_layer_index(key: str) -> int:
    """Extracts the numeric layer index from a state-dict key, e.g. `net.0.weight` -> 0."""
    matches = re.findall(r"\.(\d+)\.", key)
    return int(matches[-1]) if matches else 0


def extract_state_dict(ckpt_path: Path) -> dict[str, torch.Tensor]:
    """Loads a checkpoint and returns its state dict."""
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict):
        return checkpoint.get("state_dict", checkpoint)
    return checkpoint


# Module-name fragments that mark a normalisation layer. Matched by name because this script works
# on a raw state dict and never instantiates the model, so the module classes are not available.
NORM_MODULE_HINTS = ("norm", "layernorm", "batchnorm", "groupnorm", "_ln", "_bn")

# Parameters that are learned lookups rather than maps: positional embeddings, learned queries.
VECTOR_PARAM_HINTS = ("emb", "query")


def is_weight_key(key: str) -> bool:
    """Whether a state-dict key holds a learned weight rather than a bias."""
    name = key.rsplit(".", 1)[-1]
    return name == "weight" or (name.endswith("_weight") and not name.endswith("bias"))


def bias_key_for(weight_key: str) -> str:
    """The bias that pairs with a weight, for both the `.weight` and `in_proj_weight` spellings."""
    return weight_key[: -len("weight")] + "bias"


def parameter_kind(key: str, tensor: torch.Tensor) -> str:
    """Classifies a parameter as `linear`, `norm` or `vector`."""
    *path, name = key.split(".")
    module = path[-1].lower() if path else ""

    if any(hint in name.lower() for hint in VECTOR_PARAM_HINTS):
        return "vector"
    if any(hint in module for hint in NORM_MODULE_HINTS):
        return "norm"
    return "linear" if tensor.ndim >= 2 else "vector"


def all_parameters(state_dict: dict[str, torch.Tensor]) -> list[tuple[str, str]]:
    """Every non-bias parameter with its kind, sorted by key."""
    return [
        (key, parameter_kind(key, state_dict[key]))
        for key in sorted(state_dict)
        if not key.rsplit(".", 1)[-1].endswith("bias")
    ]


def list_available_modules(state_dict: dict[str, torch.Tensor]) -> list[str]:
    """Returns candidate `--prefix` values: every module path that owns a weight tensor."""
    prefixes = set()
    for key in (k for k in state_dict if is_weight_key(k)):
        parts = key.split(".")
        if len(parts) > 1:
            prefixes.add(".".join(parts[:-1]) + ".")
            if len(parts) > 2:
                prefixes.add(".".join(parts[:-2]) + ".")
    return sorted(prefixes)


def report_available_modules(state_dict: dict[str, torch.Tensor], report: Report) -> None:
    """Records the discoverable prefixes and every parameter with its kind."""
    report.section("Available prefixes")
    for prefix in list_available_modules(state_dict):
        report.note(f"--prefix {prefix}")

    report.section("Parameters")
    report.table(
        ["key", "shape", "kind"],
        [
            [key, str(tuple(state_dict[key].shape)), kind]
            for key, kind in all_parameters(state_dict)
        ],
    )
    report.note(
        "Only `linear` parameters are plotted by default; pass --include-norms for the rest."
    )


def find_layers(
    prefix: str, state_dict: dict[str, Any], include_norms: bool = False
) -> list[tuple[str, str | None, str]]:
    """Finds `(weight_key, bias_key, kind)` under a module prefix, ordered by layer index."""

    def matching(candidate_prefix: str) -> list[str]:
        return [
            key
            for key in state_dict
            if (key.startswith(candidate_prefix) or f".{candidate_prefix}" in key)
            and is_weight_key(key)
        ]

    weight_keys = matching(prefix)
    if not weight_keys and not prefix.endswith("."):
        weight_keys = matching(f"{prefix}.")

    layers = []
    for key in weight_keys:
        kind = parameter_kind(key, state_dict[key])
        if kind != "linear" and not include_norms:
            continue
        bias = bias_key_for(key)
        layers.append((key, bias if bias in state_dict else None, kind))

    layers.sort(key=lambda entry: get_layer_index(entry[0]))
    return layers


def as_matrix(tensor: torch.Tensor) -> np.ndarray:
    """Flattens a parameter to 2D so it can be shown as a heatmap."""
    matrix: np.ndarray = tensor.detach().cpu().numpy()
    if matrix.ndim == 1:
        return matrix.reshape(1, -1)
    if matrix.ndim > 2:
        return matrix.reshape(matrix.shape[0], -1)
    return matrix


def summarise(matrix: np.ndarray) -> dict[str, float]:
    """Per-layer statistics worth reading before squinting at the heatmap."""
    magnitude = float(np.max(np.abs(matrix))) if matrix.size else 0.0
    threshold = 0.01 * magnitude
    return {
        "mean": float(np.mean(matrix)) if matrix.size else 0.0,
        "std": float(np.std(matrix)) if matrix.size else 0.0,
        "max_abs": magnitude,
        "near_zero_frac": float(np.mean(np.abs(matrix) <= threshold)) if matrix.size else 0.0,
    }


def draw_parameter(
    ax: plt.Axes,
    matrix: np.ndarray,
    title: str,
    cmap,
    *,
    snap: bool,
    colorbar_label: str,
    annotate: bool,
    show_xticks: bool = True,
) -> None:
    """Draws one weight matrix or bias vector, with symmetric colour limits around zero."""
    if snap:
        vmin, vmax = -1.1, 1.1
    else:
        magnitude = float(np.max(np.abs(matrix))) if matrix.size else 1.0
        vmin, vmax = -magnitude, magnitude or 1.0

    image = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar(image, ax=ax, label=colorbar_label)
    ax.set_title(title, fontsize=10, color=theme.TEXT_SECONDARY)

    if show_xticks and matrix.shape[1] <= TICK_MAX_DIM:
        ax.set_xticks(range(matrix.shape[1]))
    elif not show_xticks:
        ax.set_xticks([])
    if matrix.shape[0] <= TICK_MAX_DIM:
        ax.set_yticks(range(matrix.shape[0]))

    ax.grid(which="both", color=theme.GRID, linestyle="-", linewidth=0.1, alpha=0.3)

    if annotate and max(matrix.shape) <= ANNOTATE_MAX_DIM:
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                ax.text(
                    col,
                    row,
                    f"{matrix[row, col]:.2f}",
                    ha="center",
                    va="center",
                    color=theme.TEXT_PRIMARY,
                    fontsize=8,
                )


def visualize_weights(
    ckpt_path: Path,
    prefix: str,
    layers: list[tuple[str, str | None, str]],
    state_dict: dict[str, torch.Tensor],
    save_path: Path,
    report: Report,
    *,
    snap_weights: bool,
    show: bool,
    dpi: int,
) -> None:
    """Renders one row per layer: the weight matrix, and its bias vector beside it."""
    cmap = theme.diverging_cmap()
    num_layers = len(layers)

    fig, axes = plt.subplots(
        num_layers,
        2,
        figsize=(12, max(4, 4 * num_layers)),
        squeeze=False,
        gridspec_kw={"width_ratios": [10, 1]},
    )
    theme.set_title(
        fig,
        "Layer weights and biases",
        [("ckpt", ckpt_path.name), ("prefix", prefix), ("layers", str(num_layers))],
    )

    stats_rows = []
    for index, (weight_key, bias_key, kind) in enumerate(layers):
        weight = as_matrix(state_dict[weight_key])
        if snap_weights:
            weight = np.round(weight)

        draw_parameter(
            axes[index, 0],
            weight,
            f"weights: {weight_key} {tuple(weight.shape)}",
            cmap,
            snap=snap_weights,
            colorbar_label="weight",
            annotate=True,
        )
        axes[index, 0].set_xlabel("input dim", fontsize=9)
        axes[index, 0].set_ylabel("output dim", fontsize=9)

        stats = summarise(weight)
        stats_rows.append(
            [
                weight_key,
                kind,
                str(tuple(weight.shape)),
                f"{stats['mean']:+.4f}",
                f"{stats['std']:.4f}",
                f"{stats['max_abs']:.4f}",
                f"{stats['near_zero_frac']:.1%}",
            ]
        )

        if bias_key is not None:
            bias = as_matrix(state_dict[bias_key]).reshape(-1, 1)
            if snap_weights:
                bias = np.round(bias)
            draw_parameter(
                axes[index, 1],
                bias,
                f"bias: {bias_key.split('.')[-2]}",
                cmap,
                snap=snap_weights,
                colorbar_label="bias",
                annotate=False,
                show_xticks=False,
            )
        else:
            axes[index, 1].axis("off")
            axes[index, 1].set_title("no bias", fontsize=10, color=theme.TEXT_MUTED)

    report.section("Per-layer weight statistics")
    report.table(["layer", "kind", "shape", "mean", "std", "max|w|", "near-zero"], stats_rows)
    report.note(
        "near-zero counts weights below 1% of that layer's own largest weight; a high value "
        "means most of the layer contributes little relative to its strongest connection."
    )

    save_figure(fig, save_path, show=show, dpi=dpi)
    print(f"Figure saved: {save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        parents=[cli.checkpoint_args(), cli.output_args()],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--prefix",
        "-p",
        type=str,
        default=None,
        help="Module prefix to plot, e.g. 'embedder.' or 'network.'. Omit to list what the "
        "checkpoint contains.",
    )
    parser.add_argument(
        "--list-modules",
        "-l",
        action="store_true",
        help="List the weight modules in the checkpoint and exit.",
    )
    parser.add_argument(
        "--include-norms",
        action="store_true",
        help="Also plot normalisation gains and learned vectors. Off by default: they are "
        "per-feature scales, not maps, so a heatmap of them invites reading structure that is not "
        "there.",
    )
    parser.add_argument(
        "--snap-weights",
        action="store_true",
        help="Round weights to the nearest integer before plotting, to check for quantised or "
        "near-binary structure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    theme.apply_theme()

    ensure_local_checkpoint(args.ckpt_path)
    state_dict = extract_state_dict(args.ckpt_path)
    slug = args.run_label or checkpoint_slug(args.ckpt_path)

    if args.list_modules or args.prefix is None:
        report = Report(
            "Checkpoint weight inventory",
            [("ckpt", str(args.ckpt_path)), ("tensors", str(len(state_dict)))],
        )
        report_available_modules(state_dict, report)
        if args.prefix is None and not args.list_modules:
            report.section("Next step").note("Pick one of the prefixes above and pass --prefix.")
        report.emit(
            figure_path(SCRIPT_NAME, "modules", run_slug=slug, out_dir=args.out_dir, ext="png"),
            save=not args.no_report,
        )
        return

    layers = find_layers(args.prefix, state_dict, include_norms=args.include_norms)
    save_path = figure_path(
        SCRIPT_NAME,
        f"weights-{slugify(args.prefix.strip('.'))}",
        run_slug=slug,
        out_dir=args.out_dir,
    )

    report = Report(
        "Layer weights and biases",
        [("ckpt", str(args.ckpt_path)), ("prefix", args.prefix)],
    )

    if not layers:
        report.section("No match").note(
            f"No plottable weight tensors matched prefix {args.prefix!r}. If the module contains "
            "only normalisation gains or learned vectors, pass --include-norms."
        )
        report_available_modules(state_dict, report)
        report.emit(save_path, save=not args.no_report)
        return

    visualize_weights(
        args.ckpt_path,
        args.prefix,
        layers,
        state_dict,
        save_path,
        report,
        snap_weights=args.snap_weights,
        show=args.show,
        dpi=args.dpi,
    )
    report.emit(save_path, save=not args.no_report)


if __name__ == "__main__":
    main()
