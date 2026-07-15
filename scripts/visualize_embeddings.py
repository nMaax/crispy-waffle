import argparse
import json
import random
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection

from policy.algorithms.goal_conditioned_diffusion_policy_mlp import (
    GoalConditionedDiffusionPolicyMLP,
)


def detect_key_moments(obs):
    # obs shape: [L, 48]
    tcp_to_cubeA = obs[:, 39:42]
    cubeA_pos = obs[:, 25:28]
    # cubeB_pos = obs[:, 32:35]
    cubeA_to_cubeB = obs[:, 45:48]

    L = len(obs)

    # 1. Grab moment
    # Find when Cube A starts rising
    z_A = cubeA_pos[:, 2]
    z_A_init = z_A[0]
    z_A_diff = z_A - z_A_init

    # Find first frame where z_A rises by more than 5mm
    lift_frames = np.where(z_A_diff > 0.005)[0]
    if len(lift_frames) > 0:
        t_lift = lift_frames[0]
    else:
        t_lift = L // 2

    # Grab frame is the one with minimal TCP-to-CubeA distance before t_lift
    dist_tcp_A = np.linalg.norm(tcp_to_cubeA, axis=-1)
    t_grab = int(np.argmin(dist_tcp_A[:t_lift]))

    # 2. Placement moment
    # We want Cube A to be on top of Cube B: cubeA_pos - cubeB_pos ≈ [0, 0, 0.04]
    # So cubeA_to_cubeB (which is cubeB - cubeA) ≈ [0, 0, -0.04]
    target_rel = np.array([0.0, 0.0, -0.04])
    place_err = np.linalg.norm(cubeA_to_cubeB - target_rel, axis=-1)

    # Find frames after t_lift where the error is small
    valid_place_frames = np.where((place_err < 0.02) & (np.arange(L) > t_grab))[0]
    if len(valid_place_frames) > 0:
        t_place = valid_place_frames[0]
    else:
        # Fallback to the minimum place error after t_grab
        t_place = int(t_grab + np.argmin(place_err[t_grab:]))

    # 3. Mid-air moment
    # Somewhere between grab and place where Cube A is highest
    if t_place > t_grab + 1:
        t_midair = int(t_grab + np.argmax(z_A[t_grab:t_place]))
    else:
        t_midair = (t_grab + t_place) // 2

    return {"grab": t_grab, "midair": t_midair, "place": t_place}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize Goal-Conditioned Diffusion Policy state embeddings using PCA, t-SNE, and UMAP."
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="logs/GoalConditionedDiffusionPolicyMLP__StackCube-v1__default__train/runs/2026-07-07/16-54-32/checkpoints/last.ckpt",
        help="Path to the policy checkpoint.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=str(
            Path.home()
            / ".maniskill/demos/StackCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cuda.h5"
        ),
        help="Path to the HDF5 dataset file.",
    )
    parser.add_argument(
        "--episode_idx", type=int, default=0, help="Index of the target episode to visualize."
    )
    parser.add_argument(
        "--frames",
        type=str,
        default="0%,25%,50%,75%,100%",
        help="Comma-separated list of frame indices (e.g. 0,50,-1) or relative percentages (e.g. 0%%,25%%,50%%,75%%,100%%) to highlight.",
    )
    parser.add_argument(
        "--frame_idx", type=int, default=None, help="Deprecated: use --frames instead."
    )
    parser.add_argument(
        "--goal_frame_idx",
        type=int,
        default=-1,
        help="Frame index to use as the goal. Defaults to -1 (last frame).",
    )
    parser.add_argument(
        "--num_background_episodes",
        type=int,
        default=15,
        help="Number of background episodes to load for context.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save the generated plot. If None, it is dynamically generated to avoid overwriting.",
    )
    parser.add_argument(
        "--equal_aspect",
        action="store_true",
        default=False,
        help="Ensure equal aspect ratio for the axes to prevent shape distortion (defaults to False).",
    )
    parser.add_argument(
        "--xlim",
        type=str,
        default=None,
        help="Custom X-axis limits as 'min,max' (e.g. '-10,10'). If not specified, limits are auto-scaled.",
    )
    parser.add_argument(
        "--ylim",
        type=str,
        default=None,
        help="Custom Y-axis limits as 'min,max' (e.g. '-10,10'). If not specified, limits are auto-scaled.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Spawn an interactive window to view and explore the plots.",
    )
    parser.add_argument(
        "--reducer",
        type=str,
        default="all",
        choices=["all", "pca", "tsne", "umap"],
        help="Dimensionality reduction method to run and plot (choices: all, pca, tsne, umap; default: all).",
    )
    parser.add_argument(
        "--highlight_stages",
        action="store_true",
        default=True,
        help="Automatically detect and highlight key semantic stages (Grab, Mid-air, Place) of the task.",
    )
    parser.add_argument(
        "--no_highlight_stages",
        action="store_false",
        dest="highlight_stages",
        help="Disable automatic highlight of key stages and fallback to relative frames.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env_id = "UnknownTask"
    status_str = "unknown"
    t_place = -1

    # Parse custom axis limits
    xlim = None
    if args.xlim:
        try:
            xlim = [float(x) for x in args.xlim.split(",")]
            if len(xlim) != 2:
                raise ValueError
        except ValueError:
            print("Error: --xlim must be in the format 'min,max'")
            sys.exit(1)

    ylim = None
    if args.ylim:
        try:
            ylim = [float(y) for y in args.ylim.split(",")]
            if len(ylim) != 2:
                raise ValueError
        except ValueError:
            print("Error: --ylim must be in the format 'min,max'")
            sys.exit(1)

    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found at {ckpt_path}")
        sys.exit(1)

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"Error: Dataset file not found at {dataset_path}")
        sys.exit(1)

    print(f"Loading checkpoint from: {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        act_dim = ckpt["hyper_parameters"]["act_dim"]
        network_config = dict(ckpt["hyper_parameters"]["network"])
        network_config["act_dim"] = act_dim

        model = GoalConditionedDiffusionPolicyMLP.load_from_checkpoint(
            ckpt_path,
            network=network_config,
        )
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Look for hydra config next to checkpoint
    hydra_dir = ckpt_path.parent.parent / ".hydra"
    config_file = hydra_dir / "config.yaml"

    seed = args.seed
    val_split = 0.1
    load_count = -1
    success_only = False

    if config_file.exists():
        try:
            from omegaconf import OmegaConf

            cfg = OmegaConf.load(config_file)
            print(f"Loaded Hydra run config from: {config_file}")

            # Extract parameters with fallback
            if "seed" in cfg:
                seed = cfg.seed
            if "datamodule" in cfg:
                val_split = cfg.datamodule.get("val_split", val_split)
                load_count = cfg.datamodule.get("load_count", load_count)
                success_only = cfg.datamodule.get("success_only", success_only)

            print(
                f"Extracted split parameters: seed={seed}, val_split={val_split}, load_count={load_count}, success_only={success_only}"
            )
        except Exception as e:
            print(f"Warning: Could not parse Hydra config. Using CLI defaults. Error: {e}")

    train_ids = []
    val_ids = []

    # Initialize env_id based on folder path as a smart default
    env_id = dataset_path.parent.parent.name

    json_path = dataset_path.with_suffix(".json")
    if json_path.exists():
        try:
            with open(json_path) as f:
                meta = json.load(f)
                all_episodes = meta["episodes"]
                env_id = meta.get("env_info", {}).get("env_id", env_id)

            if success_only:
                all_episodes = [ep for ep in all_episodes if ep.get("success", False)]

            rng = random.Random(seed)
            rng.shuffle(all_episodes)

            if load_count > 0:
                all_episodes = all_episodes[:load_count]

            val_size = int(len(all_episodes) * val_split)
            train_size = len(all_episodes) - val_size

            train_episodes = all_episodes[:train_size]
            val_episodes = all_episodes[train_size:]

            train_ids = [int(ep["episode_id"]) for ep in train_episodes]
            val_ids = [int(ep["episode_id"]) for ep in val_episodes]

            print(
                f"Deterministic split: {len(train_ids)} train episodes, {len(val_ids)} validation episodes."
            )
            print(f"Available Validation Episode IDs (first 15): {sorted(val_ids)[:15]}")
        except Exception as e:
            print(f"Warning: Error during train/val split calculation: {e}")

    print(f"Loading dataset from: {dataset_path}")
    try:
        with h5py.File(dataset_path, "r") as f:
            traj_keys = sorted(
                [k for k in f.keys() if k.startswith("traj_")], key=lambda x: int(x.split("_")[1])
            )

        print(f"Found {len(traj_keys)} episodes in the dataset.")

        # Check if episode_idx refers to an actual episode_id
        target_key = f"traj_{args.episode_idx}"
        if target_key in traj_keys:
            print(f"Selected target episode by ID: {target_key}")
        else:
            # Fallback to index in list
            if args.episode_idx >= len(traj_keys) or args.episode_idx < -len(traj_keys):
                print(
                    f"Error: episode_idx {args.episode_idx} is out of bounds (0 to {len(traj_keys) - 1})."
                )
                sys.exit(1)
            target_key = traj_keys[args.episode_idx]
            print(f"Selected target episode by index in list: {target_key}")

        # Print train/val status of target episode
        target_id = int(target_key.split("_")[1])
        if train_ids or val_ids:
            if target_id in val_ids:
                status = "VALIDATION (unseen/test)"
                status_str = "validation"
            elif target_id in train_ids:
                status = "TRAINING (seen)"
                status_str = "training"
            else:
                status = "OUT OF SPLIT (unused/filtered)"
                status_str = "unused"
            print(f"Target episode status: {status}")

        with h5py.File(dataset_path, "r") as f:
            target_obs = np.array(f[target_key]["obs"])  # type: ignore

        # Select background keys
        bg_keys = [k for k in traj_keys if k != target_key]
        if len(bg_keys) > args.num_background_episodes:
            bg_keys = random.sample(bg_keys, args.num_background_episodes)

        print(f"Loading {len(bg_keys)} background episodes...")
        bg_obs_list = []
        with h5py.File(dataset_path, "r") as f:
            for k in bg_keys:
                bg_obs_list.append(np.array(f[k]["obs"]))  # type: ignore

    except Exception as e:
        print(f"Error reading dataset: {e}")
        sys.exit(1)

    # Process target episode indices
    L = len(target_obs)

    goal_frame_idx = args.goal_frame_idx
    if goal_frame_idx < 0:
        goal_frame_idx = L + goal_frame_idx
    goal_frame_idx = max(0, min(goal_frame_idx, L - 1))

    highlight_frames = []
    semantic_stages = []

    if args.highlight_stages:
        print("Detecting key semantic stages...")
        moments = detect_key_moments(target_obs)
        t_grab = moments["grab"]
        t_midair = moments["midair"]
        t_place = moments["place"]

        print(f"Detected stages: Grab={t_grab}, Mid-air={t_midair}, Place={t_place}")
        semantic_stages.append((0, "Start", "#94a3b8"))
        semantic_stages.append((t_grab, "Grab", "#06b6d4"))
        semantic_stages.append((t_midair, "Mid-air", "#3b82f6"))

        # Check if placement is very close to the goal (avoid overlaps)
        if (goal_frame_idx - t_place) >= max(5, int(0.15 * L)):
            semantic_stages.append((t_place, "Place", "#10b981"))
    else:
        # If --frames is specified, use it. Otherwise if frame_idx is specified, use that.
        frames_str = args.frames
        if args.frame_idx is not None and not (
            args.frames and args.frames != "0%,25%,50%,75%,100%"
        ):
            # If user explicitly passed frame_idx, we override the default frames
            highlight_frames.append(
                (
                    max(
                        0,
                        min(args.frame_idx if args.frame_idx >= 0 else L + args.frame_idx, L - 1),
                    ),
                    f"t={args.frame_idx}",
                )
            )
        else:
            for f_str in frames_str.split(","):
                f_str = f_str.strip()
                if not f_str:
                    continue
                if f_str.endswith("%"):
                    try:
                        pct = float(f_str[:-1])
                        idx = int(round((pct / 100.0) * (L - 1)))
                        idx = max(0, min(idx, L - 1))
                        highlight_frames.append((idx, f_str))
                    except ValueError:
                        print(f"Warning: Could not parse percentage '{f_str}', skipping.")
                else:
                    try:
                        idx = int(f_str)
                        if idx < 0:
                            idx = L + idx
                        idx = max(0, min(idx, L - 1))
                        highlight_frames.append((idx, f"t={idx}"))
                    except ValueError:
                        print(f"Warning: Could not parse frame index '{f_str}', skipping.")

    print(f"Target episode length: {L} frames.")
    if args.highlight_stages:
        print("Semantic stages:")
        for idx, label, color in semantic_stages:
            print(f"  - {label} -> index {idx}")
    else:
        print("Highlighted frames:")
        for idx, label in highlight_frames:
            print(f"  - {label} -> index {idx}")
    print(f"Goal frame index: {goal_frame_idx}")

    # Extract embeddings
    print("Extracting embeddings...")
    with torch.no_grad():
        # Target episode
        target_obs_t = torch.tensor(target_obs, dtype=torch.float32)
        target_emb_dict = model.extract_embeddings(target_obs_t)
        target_embs = target_emb_dict["obs_embeddings"].numpy()  # Shape: [L, state_embedding_dim]

        # Background episodes
        bg_embs_list = []
        for bg_obs in bg_obs_list:
            bg_obs_t = torch.tensor(bg_obs, dtype=torch.float32)
            bg_embs_list.append(model.extract_embeddings(bg_obs_t)["obs_embeddings"].numpy())
        bg_embs = np.concatenate(
            bg_embs_list, axis=0
        )  # Shape: [Total_bg_len, state_embedding_dim]

    print(f"Extracted target embeddings shape: {target_embs.shape}")
    print(f"Extracted background embeddings shape: {bg_embs.shape}")

    # Prepare data for dimensionality reduction
    all_embs = np.concatenate([bg_embs, target_embs], axis=0)
    split_idx = len(bg_embs)

    # Convert reducer choice to lowercase
    reducer_choice = args.reducer.lower()

    # Dimensionality Reduction
    embeddings_2d_dict = {}

    # PCA
    if reducer_choice in ("all", "pca"):
        print("Running PCA...")
        try:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=2, random_state=args.seed)
            embeddings_2d_dict["PCA"] = pca.fit_transform(all_embs)
            print("PCA completed.")
        except Exception as e:
            print(f"Failed to run PCA: {e}")

    # t-SNE
    if reducer_choice in ("all", "tsne"):
        print("Running t-SNE...")
        try:
            from sklearn.manifold import TSNE

            tsne = TSNE(
                n_components=2, random_state=args.seed, perplexity=min(30, len(all_embs) // 3)
            )
            embeddings_2d_dict["t-SNE"] = tsne.fit_transform(all_embs)
            print("t-SNE completed.")
        except Exception as e:
            print(f"Failed to run t-SNE: {e}")

    # UMAP
    if reducer_choice in ("all", "umap"):
        print("Running UMAP...")
        try:
            import umap

            reducer = umap.UMAP(
                n_components=2, random_state=args.seed, n_neighbors=min(15, len(all_embs) // 2)
            )
            embeddings_2d_dict["UMAP"] = reducer.fit_transform(all_embs)
            print("UMAP completed.")
        except Exception as e:
            print(f"Failed to run UMAP (make sure 'umap-learn' is installed): {e}")
            print("UMAP plot will be skipped.")

    if not embeddings_2d_dict:
        print("Error: No dimensionality reduction method succeeded.")
        sys.exit(1)

    # Setup styling (dark mode)
    plt.rcParams["figure.facecolor"] = "#0f172a"
    plt.rcParams["axes.facecolor"] = "#1e293b"
    plt.rcParams["text.color"] = "#f8fafc"
    plt.rcParams["axes.labelcolor"] = "#94a3b8"
    plt.rcParams["xtick.color"] = "#64748b"
    plt.rcParams["ytick.color"] = "#64748b"
    plt.rcParams["grid.color"] = "#334155"
    plt.rcParams["grid.alpha"] = 0.5

    num_plots = len(embeddings_2d_dict)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6.5), squeeze=False)
    axes = axes[0]  # flatten to 1D array of axes

    # Colormap for time
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(0, L - 1)

    plot_idx = 0
    for name, coords_2d in embeddings_2d_dict.items():
        ax = axes[plot_idx]

        bg_coords = coords_2d[:split_idx]
        target_coords = coords_2d[split_idx:]

        # Plot background points, coloring each background episode subtly and drawing a thin line
        current_idx = 0
        bg_cmap = plt.get_cmap("tab20")
        for bg_i, bg_obs in enumerate(bg_obs_list):
            bg_len = len(bg_obs)
            bg_ep_coords = bg_coords[current_idx : current_idx + bg_len]
            current_idx += bg_len

            color = bg_cmap(bg_i % 20)

            # Draw subtle path line
            ax.plot(
                bg_ep_coords[:, 0],
                bg_ep_coords[:, 1],
                color=color,
                alpha=0.08,
                linewidth=0.5,
                zorder=1,
            )
            # Draw scattered points
            ax.scatter(
                bg_ep_coords[:, 0],
                bg_ep_coords[:, 1],
                color=color,
                alpha=0.12,
                s=5,
                marker=".",
                label="Background Episodes" if (bg_i == 0 and plot_idx == 0) else "",
                zorder=2,
            )

        # 2. Plot target episode path as a line with time gradient
        points = target_coords.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=1.5, alpha=0.85, zorder=3)
        lc.set_array(np.arange(L - 1))

        # Highlight key frames or semantic stages
        if args.highlight_stages:
            plotted_stage = False
            for idx, label, color in semantic_stages:
                if idx == goal_frame_idx:
                    continue
                ax.scatter(
                    target_coords[idx, 0],
                    target_coords[idx, 1],
                    color=color,
                    marker="o",
                    s=35,
                    edgecolors="#f8fafc",
                    linewidths=1.0,
                    label="Key Stages" if not plotted_stage else "",
                    zorder=6,
                )
                plotted_stage = True

                # Annotate stage (slim text with matching color)
                ax.annotate(
                    label,
                    (target_coords[idx, 0], target_coords[idx, 1]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    ha="left",
                    va="bottom",
                    fontsize=6,
                    color=color,
                    zorder=7,
                )
        else:
            plotted_keyframe = False
            for idx, label in highlight_frames:
                if idx == goal_frame_idx:
                    continue
                ax.scatter(
                    target_coords[idx, 0],
                    target_coords[idx, 1],
                    color="#f97316",
                    marker="o",
                    s=35,
                    edgecolors="#f8fafc",
                    linewidths=1.0,
                    label="Key Frames" if not plotted_keyframe else "",
                    zorder=6,
                )
                plotted_keyframe = True

                # Annotate key frame (slim text with no border/box)
                ax.annotate(
                    label,
                    (target_coords[idx, 0], target_coords[idx, 1]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    ha="left",
                    va="bottom",
                    fontsize=6,
                    color="#cbd5e1",
                    zorder=7,
                )

        # Highlight goal frame
        ax.scatter(
            target_coords[goal_frame_idx, 0],
            target_coords[goal_frame_idx, 1],
            color="#ec4899",
            marker="X",
            s=75,
            edgecolors="#f8fafc",
            linewidths=1.2,
            label="Goal State" if plot_idx == 0 else "",
            zorder=8,
        )

        # Determine the label for the goal state
        goal_label = f"Goal (Frame {goal_frame_idx})"
        if args.highlight_stages:
            if (goal_frame_idx - t_place) < max(5, int(0.15 * L)):
                goal_label = "Place / Goal"
        else:
            for idx, label in highlight_frames:
                if idx == goal_frame_idx:
                    goal_label = f"Goal ({label})"
                    break

        # Annotate goal state (slim text with no border/box)
        ax.annotate(
            goal_label,
            (target_coords[goal_frame_idx, 0], target_coords[goal_frame_idx, 1]),
            textcoords="offset points",
            xytext=(4, 4),
            ha="left",
            va="bottom",
            fontsize=6.5,
            color="#f472b6",
            zorder=9,
        )

        ax.set_title(f"{name} Embedding Projection", fontsize=14, fontweight="bold", pad=15)
        ax.grid(True, linestyle="--", linewidth=0.5)
        ax.set_xlabel("Dimension 1", fontsize=11, labelpad=8)
        ax.set_ylabel("Dimension 2", fontsize=11, labelpad=8)

        # Apply aspect ratio
        if args.equal_aspect:
            ax.set_aspect("equal", adjustable="box")

        # Set axis limits
        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])
        else:
            all_x = coords_2d[:, 0]
            x_pad = (all_x.max() - all_x.min()) * 0.05
            ax.set_xlim(all_x.min() - x_pad, all_x.max() + x_pad)

        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
        else:
            all_y = coords_2d[:, 1]
            y_pad = (all_y.max() - all_y.min()) * 0.05
            ax.set_ylim(all_y.min() - y_pad, all_y.max() + y_pad)

        plot_idx += 1

    # Create a single legend at the bottom of the figure
    handles, labels = axes[0].get_legend_handles_labels()
    # Also add the path colormap legend item
    from matplotlib.lines import Line2D

    path_handle = Line2D(
        [0], [0], color=cmap(0.5), lw=2.5, label="Target Episode Path (Time $\\rightarrow$)"
    )
    handles.insert(1, path_handle)

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(handles),
        frameon=True,
        facecolor="#1e293b",
        edgecolor="#334155",
        fontsize=11,
    )

    # Title and Metadata text
    dataset_name = dataset_path.name
    fig.suptitle(
        f"Goal-Conditioned Diffusion Policy ({env_id}) MLP Latent Embeddings\n"
        f"Dataset: {dataset_name} | Split: {status_str.upper()} | Episode: {target_key} | Goal Frame: {goal_frame_idx}",
        fontsize=15,
        fontweight="bold",
        color="#f8fafc",
        y=0.96,
    )

    # Adjust layout
    plt.tight_layout(rect=(0, 0.08, 1, 0.88))

    # Determine save path and ensure directory exists
    if args.save_path is None:
        if args.highlight_stages:
            slug = "stages"
        else:
            slug = args.frames.replace("%", "pct").replace(",", "_").replace(" ", "")
        save_path = (
            Path("scripts/figures") / f"embeddings_ep{args.episode_idx}_{status_str}_{slug}.png"
        )
    else:
        save_path = Path(args.save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    print(f"Visualization saved successfully to: {save_path.resolve()}")

    if args.show:
        print("Spawning interactive plot window. Close the window to exit...")
        plt.show()

    plt.close()


if __name__ == "__main__":
    main()
