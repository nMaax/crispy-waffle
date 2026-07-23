"""Visualization script for linear layer weights and biases from PyTorch checkpoints.

Loads a .ckpt file and plots weight matrices and bias vectors using matplotlib.
"""

import argparse
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


def get_layer_index(key: str) -> int:
    """Extract numeric layer index from state dict key (e.g., 'net.0.weight' -> 0)."""
    matches = re.findall(r"\.(\d+)\.", key)
    return int(matches[-1]) if matches else 0


def extract_state_dict(ckpt_path: Path) -> dict[str, torch.Tensor]:
    """Load checkpoint file and extract state_dict."""
    print(f"Loading checkpoint from: {ckpt_path}")
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict):
        return checkpoint.get("state_dict", checkpoint)
    return checkpoint


def list_available_modules(state_dict: dict[str, torch.Tensor]) -> list[str]:
    """Inspect state dict and return candidate prefixes containing weight tensors."""
    weight_keys = [k for k in state_dict.keys() if k.endswith(".weight")]
    prefixes = set()
    for wk in weight_keys:
        # e.g., 'agent.model.net.0.weight' -> 'agent.model.net.'
        parts = wk.split(".")
        if len(parts) > 1:
            prefix = ".".join(parts[:-1]) + "."
            prefixes.add(prefix)
            # also add parent module prefix (e.g., 'agent.model.')
            if len(parts) > 2:
                prefixes.add(".".join(parts[:-2]) + ".")

    return sorted(prefixes)


def find_layers(prefix: str, state_dict: dict[str, Any]) -> list[tuple[str, str | None]]:
    """Find matching weight and bias key pairs for a given module prefix."""
    # Match keys starting with prefix or containing prefix
    weight_keys = [
        k for k in state_dict.keys()
        if (k.startswith(prefix) or f".{prefix}" in k) and k.endswith(".weight")
    ]

    if not weight_keys and not prefix.endswith("."):
        prefix_dot = f"{prefix}."
        weight_keys = [
            k for k in state_dict.keys()
            if (k.startswith(prefix_dot) or f".{prefix_dot}" in k) and k.endswith(".weight")
        ]

    layers: list[tuple[str, str | None]] = []
    for wk in weight_keys:
        bk = wk.replace(".weight", ".bias")
        layers.append((wk, bk if bk in state_dict else None))

    layers.sort(key=lambda x: get_layer_index(x[0]))
    return layers


def visualize_linear_weights(
    ckpt_path: Path,
    prefix: str,
    save_path: Path | None = None,
    snap_weights: bool = False,
    cmap: str = "seismic",
    annotate_small: bool = True,
) -> None:
    """Load checkpoint and plot weights/biases for linear layers matching prefix."""
    state_dict = extract_state_dict(ckpt_path)
    layers = find_layers(prefix, state_dict)

    if not layers:
        print(f"\n[Error] Could not find any weight tensors matching prefix/module: '{prefix}'")
        print("\nAvailable weight prefixes in checkpoint:")
        avail_prefixes = list_available_modules(state_dict)
        for p in avail_prefixes:
            print(f"  --prefix {p}")
        print("\nAvailable weight keys:")
        for k in sorted(state_dict.keys()):
            if k.endswith(".weight"):
                print(f"  - {k} (shape: {tuple(state_dict[k].shape)})")
        return

    num_layers = len(layers)
    print(f"Found {num_layers} layer(s) matching prefix '{prefix}'.")

    if save_path is None:
        clean_prefix = re.sub(r"[^a-zA-Z0-9_-]", "_", prefix.strip("."))
        save_path = Path(f"scripts/figures/linear_weights_{ckpt_path.stem}_{clean_prefix}.png")

    fig, axes = plt.subplots(
        num_layers,
        2,
        figsize=(12, max(4, 4 * num_layers)),
        squeeze=False,
        gridspec_kw={"width_ratios": [10, 1]},
    )
    fig.suptitle(f"Weight & Bias Visualization: {ckpt_path.name}\nPrefix: '{prefix}'", fontsize=14, y=0.99)

    for i, (w_key, b_key) in enumerate(layers):
        ax_w = axes[i, 0]
        ax_b = axes[i, 1]

        # Weight matrix visualization
        weight_tensor = state_dict[w_key]
        weight_matrix: np.ndarray = weight_tensor.detach().cpu().numpy()
        if weight_matrix.ndim == 1:
            weight_matrix = weight_matrix.reshape(1, -1)
        elif weight_matrix.ndim > 2:
            # Flatten spatial / conv dimensions into 2D matrix
            weight_matrix = weight_matrix.reshape(weight_matrix.shape[0], -1)

        if snap_weights:
            weight_matrix = np.round(weight_matrix)

        print(f"Layer {i+1}/{num_layers} ({w_key}): shape {weight_matrix.shape}")

        if snap_weights:
            vmin, vmax = -1.1, 1.1
        else:
            mag = float(np.max(np.abs(weight_matrix))) if weight_matrix.size > 0 else 1.0
            vmin, vmax = -mag, mag

        im_w = ax_w.imshow(weight_matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        plt.colorbar(im_w, ax=ax_w, label="Weight Value")

        ax_w.set_title(f"Weights: {w_key} {tuple(weight_matrix.shape)}", fontsize=11)
        ax_w.set_xlabel("Input Dim", fontsize=10)
        ax_w.set_ylabel("Output Dim", fontsize=10)

        if weight_matrix.shape[1] <= 50:
            ax_w.set_xticks(range(weight_matrix.shape[1]))
        if weight_matrix.shape[0] <= 50:
            ax_w.set_yticks(range(weight_matrix.shape[0]))
        ax_w.grid(which="both", color="black", linestyle="-", linewidth=0.1, alpha=0.3)

        # Annotate small matrices with values
        if annotate_small and weight_matrix.shape[0] <= 12 and weight_matrix.shape[1] <= 12:
            for r in range(weight_matrix.shape[0]):
                for c in range(weight_matrix.shape[1]):
                    val = weight_matrix[r, c]
                    color = "white" if abs(val) > 0.5 * max(abs(vmin), abs(vmax)) else "black"
                    ax_w.text(c, r, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

        # Bias vector visualization
        if b_key is not None:
            bias_tensor = state_dict[b_key]
            bias_vec: np.ndarray = bias_tensor.detach().cpu().numpy().reshape(-1, 1)
            if snap_weights:
                bias_vec = np.round(bias_vec)

            print(f"Layer {i+1}/{num_layers} ({b_key}): shape {bias_vec.shape}")

            if snap_weights:
                bvmin, bvmax = -1.1, 1.1
            else:
                bmag = float(np.max(np.abs(bias_vec))) if bias_vec.size > 0 else 1.0
                bvmin, bvmax = -bmag, bmag

            im_b = ax_b.imshow(bias_vec, cmap=cmap, aspect="auto", vmin=bvmin, vmax=bvmax)
            plt.colorbar(im_b, ax=ax_b, label="Bias Value")

            ax_b.set_title(f"Bias: {b_key}", fontsize=11)
            ax_b.set_xticks([])
            if bias_vec.shape[0] <= 50:
                ax_b.set_yticks(range(bias_vec.shape[0]))
            ax_b.grid(which="both", color="black", linestyle="-", linewidth=0.1, alpha=0.3)
        else:
            ax_b.axis("off")
            ax_b.set_title("No Bias", fontsize=11)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"\n[Success] Weight visualization saved to: {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize weights and biases of linear/neural network layers from PyTorch checkpoints."
    )
    parser.add_argument("ckpt_path", type=Path, help="Path to PyTorch .ckpt file.")
    parser.add_argument(
        "--prefix", "-p",
        type=str,
        required=False,
        default=None,
        help="Module prefix/name to visualize (e.g. 'model', 'agent.mlp', 'encoder'). Required unless --list-modules is passed.",
    )
    parser.add_argument(
        "--save_path", "-s",
        type=Path,
        default=None,
        help="Output path for plot. Default: scripts/figures/linear_weights_<ckpt>_<prefix>.png",
    )
    parser.add_argument(
        "--list-modules", "-l",
        action="store_true",
        help="List available weight modules in checkpoint and exit.",
    )
    parser.add_argument(
        "--snap_weights",
        action="store_true",
        help="Snap weights to nearest integer values before plotting.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="seismic",
        help="Matplotlib colormap (default: seismic).",
    )
    args = parser.parse_args()

    state_dict = extract_state_dict(args.ckpt_path)

    if args.list_modules or args.prefix is None:
        print("\nAvailable weight prefixes in checkpoint:")
        avail_prefixes = list_available_modules(state_dict)
        for p in avail_prefixes:
            print(f"  --prefix {p}")
        print("\nAll weight keys in state dict:")
        for k in sorted(state_dict.keys()):
            if k.endswith(".weight"):
                print(f"  - {k} (shape: {tuple(state_dict[k].shape)})")

        if args.prefix is None and not args.list_modules:
            print("\n[Notice] Please specify a module prefix using --prefix / -p.")
        return

    visualize_linear_weights(
        ckpt_path=args.ckpt_path,
        prefix=args.prefix,
        save_path=args.save_path,
        snap_weights=args.snap_weights,
        cmap=args.cmap,
    )


if __name__ == "__main__":
    main()
