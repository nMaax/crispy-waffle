import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(
        description="Visualize the linear weights of the MLP Adapter layers."
    )
    parser.add_argument("ckpt_path", type=str, help="Path to the .ckpt file")
    parser.add_argument(
        "--save_path",
        type=str,
        default="scripts/figures/mlp_weights.png",
        help="Where to save the plot",
    )
    parser.add_argument(
        "--snap_weights",
        action="store_true",
        help="If set, snap weights to nearest integer values before plotting",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="network.net.",
        help="Prefix for the weight keys in the state dict",
    )
    args = parser.parse_args()

    print(f"Loading checkpoint from: {args.ckpt_path}")
    # Checkpoints often contain OmegaConf objects in hyperparameters, which weights_only=True blocks by default.
    try:
        checkpoint = torch.load(args.ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        # Fallback to weights_only=False if secure load fails (common with OmegaConf/Hydra)
        checkpoint = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)

    state_dict = checkpoint["state_dict"]

    # Robustly find the weight keys
    def get_layer_index(key):
        import re
        # Find all numbers in the key and take the last one.
        # e.g. 'trickster_mlp.net.0.weight' -> 0
        matches = re.findall(r"\.(\d+)\.", key)
        return int(matches[-1]) if matches else 0

    def find_weight_keys(prefix, state_dict):
        keys = [
            k
            for k in state_dict.keys()
            if k.startswith(prefix) and k.endswith(".weight")
        ]
        # Fallback: if no keys found, try adding '.net.' (common for our MLP class)
        if not keys and not prefix.endswith(".net."):
            fallback_prefix = prefix.rstrip(".") + ".net."
            keys = [
                k
                for k in state_dict.keys()
                if k.startswith(fallback_prefix) and k.endswith(".weight")
            ]
        return keys

    weight_keys = find_weight_keys(args.prefix, state_dict)

    if not weight_keys:
        print(f"Error: Could not find any weights with prefix '{args.prefix}' in state dict.")

        # Smart discovery: find other potential MLP prefixes
        potential_prefixes = set()
        for k in state_dict.keys():
            if ".net." in k and k.endswith(".weight"):
                potential_prefixes.add(k.split(".net.")[0])

        if potential_prefixes:
            print("\nFound other potential MLP modules. Try one of these:")
            for p in sorted(potential_prefixes):
                print(f"  --prefix {p}")

        print("\nAvailable keys (first 20):")
        for k in list(state_dict.keys())[:20]:
            print(f"  - {k}")
        return

    weight_keys.sort(key=get_layer_index)

    num_layers = len(weight_keys)
    print(f"Found {num_layers} linear layers.")

    # Determine layout: if many layers, maybe use multiple columns?
    # For now, stick to one column as requested for "small MLPs"
    fig, axes = plt.subplots(
        num_layers, 1, figsize=(10, 5 * num_layers), squeeze=False
    )
    axes = axes.flatten()

    for i, (key, ax) in enumerate(zip(weight_keys, axes)):
        weight_matrix = state_dict[key].numpy()
        if args.snap_weights:
            weight_matrix = np.round(weight_matrix)

        print(f"Layer {i} ({key}): weight matrix of shape {weight_matrix.shape}")

        # Use a dynamic vmin/vmax if not snapping, or keep fixed if looking for permutations
        if args.snap_weights:
            vmin, vmax = -1.1, 1.1
        else:
            # Dynamic range but centered at 0
            mag = np.max(np.abs(weight_matrix))
            vmin, vmax = -mag, mag

        im = ax.imshow(weight_matrix, cmap="seismic", aspect="auto", vmin=vmin, vmax=vmax)

        plt.colorbar(im, ax=ax, label="Weight Value")

        ax.set_title(f"Weights: {key}", fontsize=14)
        ax.set_xlabel("Input Dimension", fontsize=12)
        ax.set_ylabel("Output Dimension", fontsize=12)

        # Only show ticks if the matrix is small
        if weight_matrix.shape[1] <= 50:
            ax.set_xticks(range(weight_matrix.shape[1]), minor=False)
        if weight_matrix.shape[0] <= 50:
            ax.set_yticks(range(weight_matrix.shape[0]), minor=False)

        ax.grid(which="both", color="black", linestyle="-", linewidth=0.1, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.save_path, dpi=300)
    print(f"Successfully saved weight visualization to {args.save_path}")


if __name__ == "__main__":
    main()
