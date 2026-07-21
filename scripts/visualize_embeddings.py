"""Visualize Goal-Conditioned Diffusion Policy state embeddings using PCA, t-SNE, and UMAP."""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from policy.algorithms.goal_conditioned_diffusion_policy import (
    GoalConditionedDiffusionPolicy,
)


def detect_key_moments(obs: np.ndarray) -> dict[str, int]:
    """Detect frame indices for grab, midair, and placement semantic stages."""
    tcp_to_cube_a = obs[:, 39:42]
    cube_a_pos = obs[:, 25:28]
    cube_a_to_cube_b = obs[:, 45:48]

    seq_len = len(obs)

    z_a = cube_a_pos[:, 2]
    z_a_init = z_a[0]
    z_a_diff = z_a - z_a_init

    lift_frames = np.where(z_a_diff > 0.005)[0]
    t_lift = int(lift_frames[0]) if len(lift_frames) > 0 else seq_len // 2

    dist_tcp_a = np.linalg.norm(tcp_to_cube_a, axis=-1)
    t_grab = int(np.argmin(dist_tcp_a[:t_lift]))

    target_rel = np.array([0.0, 0.0, -0.04])
    place_err = np.linalg.norm(cube_a_to_cube_b - target_rel, axis=-1)

    valid_place_frames = np.where((place_err < 0.02) & (np.arange(seq_len) > t_grab))[0]
    if len(valid_place_frames) > 0:
        t_place = int(valid_place_frames[0])
    else:
        t_place = int(t_grab + np.argmin(place_err[t_grab:]))

    if t_place > t_grab + 1:
        t_midair = int(t_grab + np.argmax(z_a[t_grab:t_place]))
    else:
        t_midair = (t_grab + t_place) // 2

    return {"grab": t_grab, "midair": t_midair, "place": t_place}


def parse_args() -> argparse.Namespace:
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
        help="Comma-separated list of frame indices or percentages (e.g., 0%%,25%%,50%%,75%%,100%%).",
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
        help="Path to save generated plot.",
    )
    parser.add_argument(
        "--equal_aspect",
        action="store_true",
        default=False,
        help="Ensure equal aspect ratio for plot axes.",
    )
    parser.add_argument(
        "--xlim",
        type=str,
        default=None,
        help="Custom X-axis limits as 'min,max'.",
    )
    parser.add_argument(
        "--ylim",
        type=str,
        default=None,
        help="Custom Y-axis limits as 'min,max'.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Display plot in an interactive window.",
    )
    parser.add_argument(
        "--reducer",
        type=str,
        default="all",
        choices=["all", "pca", "tsne", "umap"],
        help="Dimensionality reduction method (all, pca, tsne, umap).",
    )
    parser.add_argument(
        "--highlight_stages",
        action="store_true",
        default=True,
        help="Highlight key semantic stages (Grab, Mid-air, Place).",
    )
    parser.add_argument(
        "--no_highlight_stages",
        action="store_false",
        dest="highlight_stages",
        help="Disable automatic stage highlights.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def parse_limits(limit_str: str | None, name: str) -> list[float] | None:
    """Parse comma-separated min,max limits."""
    if not limit_str:
        return None
    try:
        limits = [float(x) for x in limit_str.split(",")]
        if len(limits) != 2:
            raise ValueError
        return limits
    except ValueError:
        print(f"Error: --{name} must be in the format 'min,max'")
        sys.exit(1)


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    status_str = "unknown"
    t_place = -1

    xlim = parse_limits(args.xlim, "xlim")
    ylim = parse_limits(args.ylim, "ylim")

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
        checkpoint_data: dict[str, Any] = torch.load(
            ckpt_path, map_location="cpu", weights_only=False
        )
        hparams = checkpoint_data.get("hyper_parameters", {})
        act_dim = hparams.get("act_dim")
        network_config = dict(hparams.get("network", {}))
        if act_dim is not None:
            network_config["act_dim"] = act_dim

        embedder_config = hparams.get("embedder")
        if embedder_config is None and "state_embedding_dim" in hparams:
            # Checkpoint predates the configurable embedder (was trained as the now-deleted
            # GoalConditionedDiffusionPolicyMLP, which hard-coded an MLP embedder).
            print("Checkpoint predates the embedder config; reconstructing its MLP embedder.")
            embedder_config = {
                "_target_": "policy.algorithms.networks.mlp.MLP",
                "input_dim": hparams.get("task_dim"),
                "output_dim": hparams["state_embedding_dim"],
                "hidden_dims": hparams.get("hidden_dims", [128, 128, 128]),
            }

        model = GoalConditionedDiffusionPolicy.load_from_checkpoint(
            ckpt_path,
            network=network_config,
            embedder=embedder_config,
        )
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

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

            if "seed" in cfg:
                seed = int(cfg.seed)
            if "datamodule" in cfg:
                val_split = float(cfg.datamodule.get("val_split", val_split))
                load_count = int(cfg.datamodule.get("load_count", load_count))
                success_only = bool(cfg.datamodule.get("success_only", success_only))
        except Exception as e:
            print(f"Warning: Could not parse Hydra config ({e}). Using CLI defaults.")

    train_ids: list[int] = []
    val_ids: list[int] = []
    env_id = dataset_path.parent.parent.name

    json_path = dataset_path.with_suffix(".json")
    if json_path.exists():
        try:
            with open(json_path) as f:
                meta = json.load(f)
                all_episodes = meta.get("episodes", [])
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
                f"Deterministic split: {len(train_ids)} train episodes, {len(val_ids)} val episodes."
            )
        except Exception as e:
            print(f"Warning: Error during train/val split calculation: {e}")

    print(f"Loading dataset from: {dataset_path}")
    try:
        with h5py.File(dataset_path, "r") as f:
            traj_keys = sorted(
                [k for k in f.keys() if k.startswith("traj_")], key=lambda x: int(x.split("_")[1])
            )

        print(f"Found {len(traj_keys)} episodes in dataset.")

        target_key = f"traj_{args.episode_idx}"
        if target_key not in traj_keys:
            if args.episode_idx >= len(traj_keys) or args.episode_idx < -len(traj_keys):
                print(f"Error: episode_idx {args.episode_idx} is out of bounds.")
                sys.exit(1)
            target_key = traj_keys[args.episode_idx]

        target_id = int(target_key.split("_")[1])
        if train_ids or val_ids:
            if target_id in val_ids:
                status_str = "validation"
            elif target_id in train_ids:
                status_str = "training"
            else:
                status_str = "unused"
            print(f"Target episode status: {status_str.upper()}")

        with h5py.File(dataset_path, "r") as f:
            target_grp = f[target_key]
            if not isinstance(target_grp, h5py.Group):
                raise ValueError(f"Expected HDF5 group for {target_key}")

            target_obs: np.ndarray = np.asarray(target_grp["obs"])

            bg_keys = [k for k in traj_keys if k != target_key]
            if len(bg_keys) > args.num_background_episodes:
                bg_keys = random.sample(bg_keys, args.num_background_episodes)

            bg_obs_list: list[np.ndarray] = []
            for k in bg_keys:
                bg_grp = f[k]
                if isinstance(bg_grp, h5py.Group):
                    bg_obs_list.append(np.asarray(bg_grp["obs"]))

    except Exception as e:
        print(f"Error reading dataset: {e}")
        sys.exit(1)

    seq_len = len(target_obs)
    goal_frame_idx = args.goal_frame_idx
    if goal_frame_idx < 0:
        goal_frame_idx = seq_len + goal_frame_idx
    goal_frame_idx = max(0, min(goal_frame_idx, seq_len - 1))

    highlight_frames: list[tuple[int, str]] = []
    semantic_stages: list[tuple[int, str, str]] = []

    if args.highlight_stages:
        moments = detect_key_moments(target_obs)
        t_grab = moments["grab"]
        t_midair = moments["midair"]
        t_place = moments["place"]

        semantic_stages.append((0, "Start", "#94a3b8"))
        semantic_stages.append((t_grab, "Grab", "#06b6d4"))
        semantic_stages.append((t_midair, "Mid-air", "#3b82f6"))

        if (goal_frame_idx - t_place) >= max(5, int(0.15 * seq_len)):
            semantic_stages.append((t_place, "Place", "#10b981"))
    else:
        frames_str = args.frames
        if args.frame_idx is not None and not (
            args.frames and args.frames != "0%,25%,50%,75%,100%"
        ):
            idx = args.frame_idx if args.frame_idx >= 0 else seq_len + args.frame_idx
            highlight_frames.append((max(0, min(idx, seq_len - 1)), f"t={args.frame_idx}"))
        else:
            for f_str in frames_str.split(","):
                f_str = f_str.strip()
                if not f_str:
                    continue
                if f_str.endswith("%"):
                    try:
                        pct = float(f_str[:-1])
                        idx = int(round((pct / 100.0) * (seq_len - 1)))
                        highlight_frames.append((max(0, min(idx, seq_len - 1)), f_str))
                    except ValueError:
                        pass
                else:
                    try:
                        idx = int(f_str)
                        if idx < 0:
                            idx = seq_len + idx
                        highlight_frames.append((max(0, min(idx, seq_len - 1)), f"t={idx}"))
                    except ValueError:
                        pass

    print(f"Extracting embeddings for episode length {seq_len}...")
    with torch.no_grad():
        target_obs_t = torch.tensor(target_obs, dtype=torch.float32)
        target_emb_dict = model.extract_embeddings(target_obs_t)
        target_embs: np.ndarray = target_emb_dict["obs_embeddings"].detach().cpu().numpy()

        bg_embs_list: list[np.ndarray] = [
            model.extract_embeddings(torch.tensor(obs, dtype=torch.float32))["obs_embeddings"]
            .detach()
            .cpu()
            .numpy()
            for obs in bg_obs_list
        ]
        bg_embs: np.ndarray = (
            np.concatenate(bg_embs_list, axis=0)
            if bg_embs_list
            else np.empty((0, target_embs.shape[1]))
        )

    all_embs: np.ndarray = np.concatenate([bg_embs, target_embs], axis=0)
    split_idx = len(bg_embs)
    reducer_choice = args.reducer.lower()
    embeddings_2d_dict: dict[str, np.ndarray] = {}

    if reducer_choice in ("all", "pca"):
        try:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=2, random_state=args.seed)
            embeddings_2d_dict["PCA"] = np.asarray(pca.fit_transform(all_embs))
        except Exception as e:
            print(f"PCA failed: {e}")

    if reducer_choice in ("all", "tsne"):
        try:
            from sklearn.manifold import TSNE

            tsne = TSNE(
                n_components=2,
                random_state=args.seed,
                perplexity=min(30, max(1, len(all_embs) // 3)),
            )
            embeddings_2d_dict["t-SNE"] = np.asarray(tsne.fit_transform(all_embs))
        except Exception as e:
            print(f"t-SNE failed: {e}")

    if reducer_choice in ("all", "umap"):
        try:
            import umap

            reducer = umap.UMAP(
                n_components=2,
                random_state=args.seed,
                n_neighbors=min(15, max(2, len(all_embs) // 2)),
            )
            embeddings_2d_dict["UMAP"] = np.asarray(reducer.fit_transform(all_embs))
        except Exception as e:
            print(f"UMAP failed: {e}")

    if not embeddings_2d_dict:
        print("Error: No dimensionality reduction method succeeded.")
        sys.exit(1)

    plt.rcParams.update(
        {
            "figure.facecolor": "#0f172a",
            "axes.facecolor": "#1e293b",
            "text.color": "#f8fafc",
            "axes.labelcolor": "#94a3b8",
            "xtick.color": "#64748b",
            "ytick.color": "#64748b",
            "grid.color": "#334155",
            "grid.alpha": 0.5,
        }
    )

    num_plots = len(embeddings_2d_dict)
    fig, axes_raw = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6.5), squeeze=False)
    axes = axes_raw[0]

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(0, seq_len - 1)

    for plot_idx, (name, coords_2d) in enumerate(embeddings_2d_dict.items()):
        ax = axes[plot_idx]
        bg_coords = coords_2d[:split_idx]
        target_coords = coords_2d[split_idx:]

        current_idx = 0
        bg_cmap = plt.get_cmap("tab20")
        for bg_i, bg_obs in enumerate(bg_obs_list):
            bg_len = len(bg_obs)
            bg_ep_coords = bg_coords[current_idx : current_idx + bg_len]
            current_idx += bg_len
            color = bg_cmap(bg_i % 20)

            ax.plot(
                bg_ep_coords[:, 0],
                bg_ep_coords[:, 1],
                color=color,
                alpha=0.08,
                linewidth=0.5,
                zorder=1,
            )
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

        points = target_coords.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(
            segments.tolist(), cmap=cmap, norm=norm, linewidths=1.5, alpha=0.85, zorder=3
        )
        lc.set_array(np.arange(seq_len - 1))
        ax.add_collection(lc)

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

        goal_label = f"Goal (Frame {goal_frame_idx})"
        if args.highlight_stages:
            if (goal_frame_idx - t_place) < max(5, int(0.15 * seq_len)):
                goal_label = "Place / Goal"
        else:
            for idx, label in highlight_frames:
                if idx == goal_frame_idx:
                    goal_label = f"Goal ({label})"
                    break

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

        if args.equal_aspect:
            ax.set_aspect("equal", adjustable="box")

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

    handles, labels = axes[0].get_legend_handles_labels()
    path_handle = Line2D(
        [0], [0], color=cmap(0.5), lw=2.5, label=r"Target Episode Path (Time $\rightarrow$)"
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

    dataset_name = dataset_path.name
    fig.suptitle(
        f"Goal-Conditioned Diffusion Policy ({env_id}) MLP Latent Embeddings\n"
        f"Dataset: {dataset_name} | Split: {status_str.upper()} | Episode: {target_key} | Goal Frame: {goal_frame_idx}",
        fontsize=15,
        fontweight="bold",
        color="#f8fafc",
        y=0.96,
    )

    plt.tight_layout(rect=(0, 0.08, 1, 0.88))

    if args.save_path is None:
        slug = (
            "stages"
            if args.highlight_stages
            else args.frames.replace("%", "pct").replace(",", "_").replace(" ", "")
        )
        save_path = (
            Path("scripts/figures") / f"embeddings_ep{args.episode_idx}_{status_str}_{slug}.png"
        )
    else:
        save_path = Path(args.save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    print(f"Visualization saved successfully to: {save_path.resolve()}")

    if args.show:
        print("Spawning interactive plot window...")
        plt.show()

    plt.close()


if __name__ == "__main__":
    main()
