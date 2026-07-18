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


def find_layers(prefix: str, state_dict: dict[str, Any]) -> list[tuple[str, str | None]]:
    """Find matching weight and bias key pairs for a given module prefix."""
    weight_keys = [k for k in state_dict.keys() if k.startswith(prefix) and k.endswith(".weight")]

    if not weight_keys and not prefix.endswith(".net."):
        fallback_prefix = prefix.rstrip(".") + ".net."
        weight_keys = [
            k for k in state_dict.keys() if k.startswith(fallback_prefix) and k.endswith(".weight")
        ]

    layers: list[tuple[str, str | None]] = []
    for wk in weight_keys:
        bk = wk.replace(".weight", ".bias")
        layers.append((wk, bk if bk in state_dict else None))
    return layers


def visualize_linear_weights(
    ckpt_path: Path,
    save_path: Path,
    prefix: str = "network.net.",
    snap_weights: bool = False,
) -> None:
    """Load checkpoint and plot weights/biases for linear layers matching prefix."""
    print(f"Loading checkpoint from: {ckpt_path}")
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    state_dict: dict[str, torch.Tensor] = (
        checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    )

    layers = find_layers(prefix, state_dict)

    if not layers:
        print(f"Error: Could not find any weights with prefix '{prefix}' in state dict.")
        potential_prefixes = {
            k.split(".net.")[0]
            for k in state_dict.keys()
            if ".net." in k and k.endswith(".weight")
        }

        if potential_prefixes:
            print("\nFound other potential modules. Try one of these:")
            for p in sorted(potential_prefixes):
                print(f"  --prefix {p}")

        print("\nAvailable keys:")
        for k in sorted(state_dict.keys()):
            print(f"  - {k}")
        return

    layers.sort(key=lambda x: get_layer_index(x[0]))
    num_layers = len(layers)
    print(f"Found {num_layers} linear layers.")

    fig, axes = plt.subplots(
        num_layers,
        2,
        figsize=(12, 5 * num_layers),
        squeeze=False,
        gridspec_kw={"width_ratios": [10, 1]},
    )

    for i, (w_key, b_key) in enumerate(layers):
        ax_w = axes[i, 0]
        ax_b = axes[i, 1]

        # Weight matrix visualization
        weight_tensor = state_dict[w_key]
        weight_matrix: np.ndarray = weight_tensor.detach().cpu().numpy()
        if snap_weights:
            weight_matrix = np.round(weight_matrix)

        print(f"Layer {i} ({w_key}): shape {weight_matrix.shape}")

        if snap_weights:
            vmin, vmax = -1.1, 1.1
        else:
            mag = float(np.max(np.abs(weight_matrix))) if weight_matrix.size > 0 else 1.0
            vmin, vmax = -mag, mag

        im_w = ax_w.imshow(weight_matrix, cmap="seismic", aspect="auto", vmin=vmin, vmax=vmax)
        plt.colorbar(im_w, ax=ax_w, label="Weight Value")

        ax_w.set_title(f"Weights: {w_key}", fontsize=14)
        ax_w.set_xlabel("Input Dimension", fontsize=12)
        ax_w.set_ylabel("Output Dimension", fontsize=12)

        if weight_matrix.shape[1] <= 50:
            ax_w.set_xticks(range(weight_matrix.shape[1]))
        if weight_matrix.shape[0] <= 50:
            ax_w.set_yticks(range(weight_matrix.shape[0]))
        ax_w.grid(which="both", color="black", linestyle="-", linewidth=0.1, alpha=0.3)

        # Bias vector visualization
        if b_key is not None:
            bias_tensor = state_dict[b_key]
            bias_vec: np.ndarray = bias_tensor.detach().cpu().numpy().reshape(-1, 1)
            if snap_weights:
                bias_vec = np.round(bias_vec)

            print(f"Layer {i} ({b_key}): shape {bias_vec.shape}")

            if snap_weights:
                bvmin, bvmax = -1.1, 1.1
            else:
                bmag = float(np.max(np.abs(bias_vec))) if bias_vec.size > 0 else 1.0
                bvmin, bvmax = -bmag, bmag

            im_b = ax_b.imshow(bias_vec, cmap="seismic", aspect="auto", vmin=bvmin, vmax=bvmax)
            plt.colorbar(im_b, ax=ax_b, label="Bias Value")

            ax_b.set_title(f"Bias: {b_key}", fontsize=14)
            ax_b.set_xticks([])
            if bias_vec.shape[0] <= 50:
                ax_b.set_yticks(range(bias_vec.shape[0]))
            ax_b.grid(which="both", color="black", linestyle="-", linewidth=0.1, alpha=0.3)
        else:
            ax_b.axis("off")
            ax_b.set_title("No Bias", fontsize=14)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Successfully saved weight visualization to {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize weights and biases of linear layers from PyTorch checkpoints."
    )
    parser.add_argument("ckpt_path", type=Path, help="Path to .ckpt file.")
    parser.add_argument(
        "--save_path",
        type=Path,
        default=Path("scripts/figures/linear_weights.png"),
        help="Path where plot will be saved.",
    )
    parser.add_argument(
        "--snap_weights",
        action="store_true",
        help="Snap weights to nearest integer values before plotting.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="network.net.",
        help="Prefix for weight keys in state dict.",
    )
    args = parser.parse_args()

    visualize_linear_weights(
        ckpt_path=args.ckpt_path,
        save_path=args.save_path,
        prefix=args.prefix,
        snap_weights=args.snap_weights,
    )


if __name__ == "__main__":
    main()
