import argparse

import matplotlib.pyplot as plt
import torch


def main():
    parser = argparse.ArgumentParser(
        description="Visualize the linear weights of the MLP Adapter."
    )
    parser.add_argument("ckpt_path", type=str, help="Path to the .ckpt file")
    parser.add_argument(
        "--save_path",
        type=str,
        default="scripts/figures/permutation_matrix.png",
        help="Where to save the plot",
    )
    args = parser.parse_args()

    print(f"Loading checkpoint from: {args.ckpt_path}")
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]

    weight_key = "network.net.0.weight"

    if weight_key not in state_dict:
        print(f"Error: Could not find '{weight_key}' in state dict.")
        print("Available keys:")
        for k in state_dict.keys():
            print(f"  - {k}")
        return

    weight_matrix = state_dict[weight_key].numpy()

    print(f"Extracted weight matrix of shape: {weight_matrix.shape}")

    plt.figure(figsize=(10, 8))

    im = plt.imshow(weight_matrix, cmap="seismic", aspect="auto", vmin=-1.1, vmax=1.1)

    plt.colorbar(im, label="Weight Value")

    plt.title("Learned Permutation Matrix (0-Layer MLP)", fontsize=14)
    plt.xlabel("Input State Index", fontsize=12)
    plt.ylabel("Output State Index", fontsize=12)

    plt.grid(which="both", color="black", linestyle="-", linewidth=0.1, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.save_path, dpi=300)
    print(f"Successfully saved weight visualization to {args.save_path}")


if __name__ == "__main__":
    main()
