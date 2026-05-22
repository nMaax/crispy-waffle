import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(
        description="Visualize the weights and biases of linear layers."
    )
    parser.add_argument("ckpt_path", type=str, help="Path to the .ckpt file")
    parser.add_argument(
        "--save_path",
        type=str,
        default="scripts/figures/linear_plot.png",
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
    try:
        checkpoint = torch.load(args.ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        # Fallback to weights_only=False if secure load fails
        checkpoint = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)

    state_dict = checkpoint["state_dict"]

    def get_layer_index(key):
        import re

        # Find all numbers in the key and take the last one.
        # e.g. 'trickster_mlp.net.0.weight' -> 0
        matches = re.findall(r"\.(\d+)\.", key)
        return int(matches[-1]) if matches else 0

    def find_layers(prefix, state_dict):
        weight_keys = [
            k for k in state_dict.keys() if k.startswith(prefix) and k.endswith(".weight")
        ]
        # Fallback: if no keys found, try adding '.net.'
        if not weight_keys and not prefix.endswith(".net."):
            fallback_prefix = prefix.rstrip(".") + ".net."
            weight_keys = [
                k
                for k in state_dict.keys()
                if k.startswith(fallback_prefix) and k.endswith(".weight")
            ]

        layers = []
        for wk in weight_keys:
            bk = wk.replace(".weight", ".bias")
            if bk not in state_dict:
                bk = None
            layers.append((wk, bk))
        return layers

    layers = find_layers(args.prefix, state_dict)

    if not layers:
        print(f"Error: Could not find any weights with prefix '{args.prefix}' in state dict.")

        # Find other potential prefixes
        potential_prefixes = set()
        for k in state_dict.keys():
            if ".net." in k and k.endswith(".weight"):
                potential_prefixes.add(k.split(".net.")[0])

        if potential_prefixes:
            print("\nFound other potential modules. Try one of these:")
            for p in sorted(potential_prefixes):
                print(f"  --prefix {p}")

        print("\nAvailable keys:")
        for k in list(state_dict.keys()):
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

    for i, ((w_key, b_key), (ax_w, ax_b)) in enumerate(zip(layers, axes)):
        # Plot Weight Matrix
        weight_matrix = state_dict[w_key].numpy()
        if args.snap_weights:
            weight_matrix = np.round(weight_matrix)

        print(f"Layer {i} ({w_key}): weight matrix of shape {weight_matrix.shape}")

        if args.snap_weights:
            vmin, vmax = -1.1, 1.1
        else:
            mag = np.max(np.abs(weight_matrix))
            vmin, vmax = -mag, mag

        im_w = ax_w.imshow(weight_matrix, cmap="seismic", aspect="auto", vmin=vmin, vmax=vmax)
        plt.colorbar(im_w, ax=ax_w, label="Weight Value")

        ax_w.set_title(f"Weights: {w_key}", fontsize=14)
        ax_w.set_xlabel("Input Dimension", fontsize=12)
        ax_w.set_ylabel("Output Dimension", fontsize=12)

        if weight_matrix.shape[1] <= 50:
            ax_w.set_xticks(range(weight_matrix.shape[1]), minor=False)
        if weight_matrix.shape[0] <= 50:
            ax_w.set_yticks(range(weight_matrix.shape[0]), minor=False)
        ax_w.grid(which="both", color="black", linestyle="-", linewidth=0.1, alpha=0.3)

        # Plot Bias Vector
        if b_key:
            bias_vec = state_dict[b_key].numpy().reshape(-1, 1)
            if args.snap_weights:
                bias_vec = np.round(bias_vec)

            print(f"Layer {i} ({b_key}): bias vector of shape {bias_vec.shape}")

            if args.snap_weights:
                bvmin, bvmax = -1.1, 1.1
            else:
                bmag = np.max(np.abs(bias_vec)) if bias_vec.size > 0 else 1.0
                bvmin, bvmax = -bmag, bmag

            im_b = ax_b.imshow(bias_vec, cmap="seismic", aspect="auto", vmin=bvmin, vmax=bvmax)
            plt.colorbar(im_b, ax=ax_b, label="Bias Value")

            ax_b.set_title(f"Bias: {b_key}", fontsize=14)
            ax_b.set_xticks([])  # Hide X axis for bias as it is 1D
            if bias_vec.shape[0] <= 50:
                ax_b.set_yticks(range(bias_vec.shape[0]), minor=False)
            ax_b.grid(which="both", color="black", linestyle="-", linewidth=0.1, alpha=0.3)
        else:
            ax_b.axis("off")
            ax_b.set_title("No Bias", fontsize=14)

    plt.tight_layout()
    plt.savefig(args.save_path, dpi=300)
    print(f"Successfully saved weight visualization to {args.save_path}")


if __name__ == "__main__":
    main()
