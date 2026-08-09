"""Analyzes whether the goal-conditioning signal ("z" -- the literal conditioning tensor that
reaches the UNet's FiLM layers) shrinks toward zero as a live GoalConditionedDiffusionPolicy
rollout converges to its goal.

Runs a handful of live ManiSkill episodes driven by the checkpoint's own `get_action()` policy (no
offline HDF5 replay), recording the exact goal-delta tensor `_build_external_cond()` produces at
every environment step -- not just at replanning boundaries, since that call alone never invokes
the expensive diffusion denoising loop -- alongside a task-agnostic ground-truth distance to goal
derived from the canonicalized `a_pose`. Plots ||z|| vs. time and ||z|| vs. ground-truth distance
across all collected episodes.

Strictly scoped to GoalConditionedDiffusionPolicy: BESO/BESO++ condition through DiffusionGPT (a
transformer), not ConditionalUnet1D/FiLM, so there is no analogous standalone "z" to extract there.
"""

import argparse
import time
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium.spaces import Box
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers import FrameStack, RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from matplotlib.lines import Line2D
from omegaconf import DictConfig, OmegaConf

import policy.environments  # noqa: F401
from policy.algorithms.goal_conditioned_diffusion_policy import GoalConditionedDiffusionPolicy
from policy.transforms import observation_pipeline
from policy.utils import to_tensor
from policy.utils.checkpoint_utils import load_goal_conditioned_diffusion_policy
from policy.utils.typing_utils import (
    GoalConditionedEnvProtocol,
    TensorTree,
    get_subtree,
    get_tensor,
)

SUCCESS_COLOR = "#10b981"
FAILURE_COLOR = "#f87171"


@dataclass
class StepRecord:
    """One environment step's recorded convergence signal."""

    episode_idx: int
    step_idx: int
    z_norm: float
    gt_distance: float


@dataclass
class EpisodeResult:
    """Per-episode step records plus ManiSkill's own success bookkeeping."""

    episode_idx: int
    steps: list[StepRecord]
    success_once: bool
    success_at_end: bool
    episode_len: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to a GoalConditionedDiffusionPolicy checkpoint (e.g. logs/.../last.ckpt).",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=8, help="Number of live rollout episodes to collect."
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=None,
        help="Override for max episode length. Default: the env's registered default.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=None,
        help="Diffusion denoising steps used for the actual action. Default: the model's own.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Base env reset seed, offset per episode."
    )
    parser.add_argument(
        "--no_clamp_action",
        action="store_false",
        dest="clamp_action",
        default=True,
        help="Disable clamping actions to the action space bounds (enabled by default, "
        "matching RolloutEvaluationCallback).",
    )
    parser.add_argument(
        "--render_mode",
        type=str,
        default=None,
        choices=["human", "rgb_array"],
        help="Optional render mode; 'human' sleeps briefly per step, like the training callback.",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="If set, records episode videos via ManiSkill's RecordEpisode wrapper.",
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=25,
        help="Number of bins for the pooled ||z|| vs. ground-truth-distance trend line.",
    )
    parser.add_argument(
        "--save_path_prefix",
        type=str,
        default=None,
        help="Prefix for saved figures under scripts/figures/. Default: derived from ckpt_path.",
    )
    parser.add_argument(
        "--show", action="store_true", default=False, help="Also display plots interactively."
    )
    return parser.parse_args()


def load_env_config(ckpt_path: Path) -> DictConfig:
    """Loads the checkpoint's saved Hydra run config, the same way `analyze_embedder_linearity.py`
    reconstructs the training datamodule -- there is no live `trainer.datamodule` here."""
    config_file = ckpt_path.parent.parent / ".hydra" / "config.yaml"
    if not config_file.exists():
        raise FileNotFoundError(
            f"No saved Hydra run config found at {config_file}; cannot recover the environment "
            "settings (env_id, obs_mode, control_mode, physx_backend, ...) this checkpoint was "
            "trained with."
        )
    cfg = OmegaConf.load(config_file)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected a DictConfig at {config_file}, got {type(cfg).__name__}.")
    return cfg


def resolve_env_kwargs(cfg: DictConfig) -> dict:
    """Reads the rollout-relevant env settings off the saved datamodule config (mirrors the
    attribute names `RolloutEvaluationCallback._resolve_param` pulls off `trainer.datamodule`).

    `physx_backend` is only used to warn about a CUDA/CPU mismatch, exactly like
    `RolloutEvaluationCallback.setup()` -- it is never threaded into `gym.make()` itself (nor is it
    there, upstream): ManiSkill infers CPU vs. GPU simulation from `num_envs`, which this script
    hardcodes to 1.
    """
    dm = cfg.datamodule
    physx_backend = dm.physx_backend
    if "cuda" in str(physx_backend).lower() and not torch.cuda.is_available():
        warnings.warn(
            f"Checkpoint was trained with physx_backend={physx_backend!r}, but CUDA is not "
            "available on this machine. Proceeding with a single CPU env for this analysis "
            "rollout regardless (num_envs=1 already implies a CPU-backed simulation).",
            stacklevel=2,
        )

    return {
        "env_id": dm.env_id,
        "obs_mode": dm.obs_mode,
        "control_mode": dm.control_mode,
        "robot_uids": dm.get("robot_uids", None),
        "no_proprio_vel": bool(dm.get("no_proprio_vel", False)),
    }


def _describe_model_config(
    model: GoalConditionedDiffusionPolicy, cfg: DictConfig
) -> list[tuple[str, str]]:
    """`(key, value)` pairs describing this checkpoint's tokenizer/embedder/pooling architecture,
    goal-conditioning mode, and HER ratio -- the single source of truth for both the human-readable
    title line (`build_metadata_str`) and the filesystem-safe filename slug
    (`build_metadata_slug`)."""
    pooling = getattr(model.embedder, "pooling", None)
    fields = [
        ("tokenizer", type(model.tokenizer).__name__ if model.tokenizer is not None else "none"),
        ("embedder", type(model.embedder).__name__ if model.embedder is not None else "none"),
        ("pooling", type(pooling).__name__ if pooling is not None else "none"),
        ("goal_delta", str(model.goal_delta)),
    ]
    her_ratio = cfg.get("datamodule", {}).get("her_ratio", None)
    if her_ratio is not None:
        fields.append(("her_ratio", str(her_ratio)))
    return fields


def build_metadata_str(model: GoalConditionedDiffusionPolicy, cfg: DictConfig) -> str:
    """A compact `key=value` summary of the config this checkpoint was trained with -- env,
    tokenizer/embedder/pooling architecture, goal-conditioning mode, HER ratio -- appended under
    every figure's title so a saved PNG identifies its own provenance without cross-referencing the
    checkpoint path."""
    fields = [("env", cfg.datamodule.env_id), *_describe_model_config(model, cfg)]
    return " | ".join(f"{k}={v}" for k, v in fields)


def build_metadata_slug(model: GoalConditionedDiffusionPolicy, cfg: DictConfig) -> str:
    """Filesystem-safe counterpart to `build_metadata_str` (env_id omitted -- it's already folded
    into the filename's checkpoint-derived prefix), appended to every saved figure's filename so
    checkpoints that differ only in HER ratio, pooling, tokenizer, embedder, or goal-delta mode
    (but happen to share a Hydra experiment/run directory name) don't overwrite each other's
    PNGs."""
    return "_".join(f"{k}-{v}" for k, v in _describe_model_config(model, cfg))


def build_rollout_env(
    env_kwargs: dict,
    obs_horizon: int,
    max_episode_steps: int | None,
    render_mode: str | None,
    video_dir: str | None,
):
    """Constructs gym_env -> FrameStack -> [RecordEpisode] -> ManiSkillVectorEnv, mirroring
    `RolloutEvaluationCallback.setup()` (policy/algorithms/callbacks/rollout_evaluation.py) as a
    free function, hardcoded to a single CPU env (no Lightning Trainer/pl_module involved).

    Returns (vector_env, inner_env) -- `inner_env` is kept around to call
    `generate_heuristic_goal()` on, exactly as the callback does.
    """
    make_kwargs = {}
    if env_kwargs["robot_uids"] is not None:
        make_kwargs["robot_uids"] = env_kwargs["robot_uids"]

    gym_env = gym.make(
        id=env_kwargs["env_id"],
        obs_mode=env_kwargs["obs_mode"],
        control_mode=env_kwargs["control_mode"],
        render_mode=render_mode,
        num_envs=1,
        max_episode_steps=max_episode_steps,
        **make_kwargs,
    )
    inner_env = gym_env.unwrapped
    frame_stack_env = FrameStack(gym_env, num_stack=obs_horizon)

    if video_dir:
        # RecordEpisode's stub expects a BaseEnv; FrameStack wraps one but isn't typed as one --
        # same cast RolloutEvaluationCallback.setup() uses for its own equivalent wrapping.
        frame_stack_as_base_env = cast(BaseEnv, frame_stack_env)
        max_steps = gym_utils.find_max_episode_steps_value(frame_stack_as_base_env)
        recorded_env = RecordEpisode(
            frame_stack_as_base_env,
            output_dir=video_dir,
            save_trajectory=False,
            save_video=True,
            max_steps_per_video=max_steps,
            source_type="diffusion_policy",
            source_desc="analyze_goal_signal_convergence rollout",
        )
        vector_env = ManiSkillVectorEnv(
            recorded_env, ignore_terminations=True, record_metrics=True
        )
    else:
        vector_env = ManiSkillVectorEnv(
            frame_stack_env, ignore_terminations=True, record_metrics=True
        )
    return vector_env, inner_env


def build_external_cond_only(
    model: GoalConditionedDiffusionPolicy, obs: TensorTree, goal: TensorTree
) -> dict[str, TensorTree]:
    """Normalizes obs/goal and builds the network's conditioning tree -- no diffusion loop.

    Cheap:
    safe to call at every environment step to keep the recorded z-trace dense.
    """
    if model.obs_normalizer is not None:
        obs = model.obs_normalizer.normalize(obs)
        goal = model.obs_normalizer.normalize(goal)
    return model._build_external_cond(obs, goal)


def get_action_and_external_cond(
    model: GoalConditionedDiffusionPolicy,
    obs: TensorTree,
    goal: TensorTree,
    num_inference_steps: int | None,
) -> tuple[torch.Tensor, dict[str, TensorTree]]:
    """Replicates `get_action()`'s body but also returns the `external_cond` that produced the
    action, so the caller records the exact conditioning tensor without a second (expensive)
    diffusion pass.

    `GoalConditionedDiffusionPolicy.get_action()` does exactly this sequence with no transform in
    between, so this is behaviorally identical to calling `get_action()` once.
    """
    external_cond = build_external_cond_only(model, obs, goal)
    action_seq = model._run_diffusion_loop(
        external_cond=external_cond,
        num_inference_steps=num_inference_steps,
        output_clip_range=None,
    )
    return action_seq, external_cond


def compute_z(
    model: GoalConditionedDiffusionPolicy, external_cond: dict[str, TensorTree]
) -> torch.Tensor:
    """The literal per-step goal-delta vector z, shape [B, D], uniform across all three
    `goal_delta` modes:

    - `goal_delta in ("input", "embedding")`: the "task" entry already IS the delta fed to the
      network (`embed(goal - obs)` or `embed(goal) - embed(obs)`).
    - `goal_delta is None`: no delta tensor exists in `external_cond`; z is the explicit
      "distance-in-embedding-space to goal" proxy, `embed(obs) - embed(goal)`.

    A pooling embedder (`model._embedder_pools_time()`) collapses the obs time axis, moving the
    "task" entry from `external_cond["obs"]["task"]` up to a sibling `external_cond["task"]".
    """
    pools_time = model._embedder_pools_time()
    task = (
        get_tensor(external_cond, "task")
        if pools_time
        else get_tensor(get_subtree(external_cond, "obs"), "task")
    )
    current = task[:, -1] if task.ndim == 3 else task  # most recent observed frame

    if model.goal_delta is None:
        goal_val = external_cond["goal"]
        goal_task = (
            goal_val if isinstance(goal_val, torch.Tensor) else get_tensor(goal_val, "task")
        )
        return current - goal_task
    return current


def z_norm(
    model: GoalConditionedDiffusionPolicy, external_cond: dict[str, TensorTree]
) -> torch.Tensor:
    """||z||, shape [B]."""
    return torch.linalg.norm(compute_z(model, external_cond), dim=-1)


def ground_truth_distance(
    obs_canon: Mapping[str, TensorTree], goal_canon: Mapping[str, TensorTree]
) -> torch.Tensor:
    """Task-agnostic distance-to-goal: L2 norm of the position-only difference between the
    canonicalized obs's and goal's `a_pose` (works uniformly across StackCube*/PlaceCube*/
    PlaceSphere* variants, all of which populate `a_pose` per `Canonicalizer.DIM_SPEC`).

    `obs_canon` still carries its FrameStack time axis (only the most recent frame is used);
    `goal_canon` (from `generate_heuristic_goal()`) never does. Shape [B].
    """
    obs_a_pose = get_tensor(obs_canon, "a_pose")
    obs_pos = obs_a_pose[:, -1, :3] if obs_a_pose.ndim == 3 else obs_a_pose[..., :3]
    goal_pos = get_tensor(goal_canon, "a_pose")[..., :3]
    return torch.linalg.norm(goal_pos - obs_pos, dim=-1)


def extract_episode_metrics(info: dict) -> tuple[bool, bool, int]:
    """Pulls success_once/success_at_end/episode_len out of `info["final_info"]["episode"]`, with
    the same fallbacks `RolloutEvaluationCallback._run_rollouts` uses.

    With num_envs=1 there is no batched-index bookkeeping to do.
    """
    ep_dict = info.get("final_info", {}).get("episode", {})

    if "success_once" in ep_dict:
        success_once = bool(ep_dict["success_once"][0].item())
    else:
        success_once = bool(
            info.get("final_info", {}).get("success", torch.tensor([False]))[0].item()
        )

    success_at_end = (
        bool(ep_dict["success_at_end"][0].item()) if "success_at_end" in ep_dict else False
    )
    episode_len = int(ep_dict["episode_len"][0].item()) if "episode_len" in ep_dict else -1

    return success_once, success_at_end, episode_len


def collect_episode(
    model: GoalConditionedDiffusionPolicy,
    env,
    inner_env,
    apply_transforms,
    episode_idx: int,
    seed: int,
    num_inference_steps: int | None,
    clamp_action: bool,
    render_mode: str | None,
) -> EpisodeResult:
    """Runs one episode to completion, recording a StepRecord at every environment step (cheap:

    `build_external_cond_only` alone), replanning (the expensive
    `get_action_and_external_cond`) every `model.act_horizon` steps -- mirroring
    `RolloutEvaluationCallback._run_rollouts`'s inner loop structure for a single fixed episode.
    """
    if hasattr(model, "reset"):
        model.reset()

    action_space = env.action_space
    if not isinstance(action_space, Box):
        raise ValueError(f"Expected Box action space, got {type(action_space)}")
    action_low = torch.as_tensor(action_space.low, device=model.device, dtype=torch.float32)
    action_high = torch.as_tensor(action_space.high, device=model.device, dtype=torch.float32)

    obs, info = env.reset(seed=seed)
    obs = apply_transforms(to_tensor(obs, device=model.device, dtype=torch.float32))

    assert isinstance(inner_env, GoalConditionedEnvProtocol)
    goal_state = apply_transforms(
        to_tensor(inner_env.generate_heuristic_goal(), device=model.device, dtype=torch.float32)
    )

    if render_mode == "human":
        env.render()

    steps: list[StepRecord] = []
    step_idx = 0
    truncated_all = False

    with torch.no_grad():
        while not truncated_all:
            action_seq, external_cond = get_action_and_external_cond(
                model, obs, goal_state, num_inference_steps
            )

            for i in range(model.act_horizon):
                steps.append(
                    StepRecord(
                        episode_idx=episode_idx,
                        step_idx=step_idx,
                        z_norm=z_norm(model, external_cond).item(),
                        gt_distance=ground_truth_distance(obs, goal_state).item(),
                    )
                )

                action = action_seq[:, i]
                if clamp_action:
                    action = torch.clamp(
                        action, action_low.to(action.dtype), action_high.to(action.dtype)
                    )

                obs_raw, _reward, _terminated, truncated, info = env.step(action)
                if render_mode == "human":
                    time.sleep(0.05)
                    env.render()

                obs = apply_transforms(
                    to_tensor(obs_raw, device=model.device, dtype=torch.float32)
                )
                goal_state = apply_transforms(
                    to_tensor(
                        inner_env.generate_heuristic_goal(),
                        device=model.device,
                        dtype=torch.float32,
                    )
                )
                step_idx += 1

                truncated_all = torch.as_tensor(
                    truncated, device=model.device, dtype=torch.bool
                ).all()
                if truncated_all:
                    break

                # Refresh (cheaply) so the trace stays dense at every step, not just at
                # replanning boundaries.
                external_cond = build_external_cond_only(model, obs, goal_state)

    success_once, success_at_end, episode_len = extract_episode_metrics(info)
    if episode_len < 0:
        episode_len = step_idx

    return EpisodeResult(
        episode_idx=episode_idx,
        steps=steps,
        success_once=success_once,
        success_at_end=success_at_end,
        episode_len=episode_len,
    )


def collect_all_episodes(
    model: GoalConditionedDiffusionPolicy, args: argparse.Namespace, env_kwargs: dict
) -> list[EpisodeResult]:
    """Builds the env once, runs `args.num_episodes` episodes with per-episode seed offsets, then
    closes it."""
    env, inner_env = build_rollout_env(
        env_kwargs,
        model.obs_horizon,
        args.max_episode_steps,
        args.render_mode,
        args.video_dir,
    )
    apply_transforms = observation_pipeline(
        env_id=env_kwargs["env_id"],
        is_flat=not isinstance(env.observation_space, gym.spaces.Dict),
        canonicalize=True,
        as_dict=True,
        no_proprio_vel=env_kwargs["no_proprio_vel"],
    )

    results: list[EpisodeResult] = []
    try:
        for i in range(args.num_episodes):
            result = collect_episode(
                model,
                env,
                inner_env,
                apply_transforms,
                episode_idx=i,
                seed=args.seed + i,
                num_inference_steps=args.num_inference_steps,
                clamp_action=args.clamp_action,
                render_mode=args.render_mode,
            )
            results.append(result)
            print(
                f"  Episode {i}: {len(result.steps)} steps, "
                f"success_once={result.success_once}, success_at_end={result.success_at_end}"
            )
    finally:
        env.close()

    return results


def print_summary(results: list[EpisodeResult]) -> None:
    print(f"\n{'=' * 88}\nSummary across {len(results)} episode(s)")

    all_z = np.array([s.z_norm for r in results for s in r.steps])
    all_gt = np.array([s.gt_distance for r in results for s in r.steps])
    all_success = np.array([r.success_once for r in results for _s in r.steps])

    for r in results:
        if not r.steps:
            continue
        ratio = r.steps[-1].z_norm / r.steps[0].z_norm if r.steps[0].z_norm > 0 else float("nan")
        print(
            f"  Episode {r.episode_idx}: z_norm[0]={r.steps[0].z_norm:.4f}  "
            f"z_norm[-1]={r.steps[-1].z_norm:.4f}  ratio={ratio:.4f}  "
            f"success_once={r.success_once}  success_at_end={r.success_at_end}"
        )

    def _corr(z: np.ndarray, gt: np.ndarray) -> str:
        if len(z) < 2 or np.unique(z).size < 2 or np.unique(gt).size < 2:
            return "N/A (insufficient variance)"
        return f"{np.corrcoef(z, gt)[0, 1]:.4f}"

    print(f"\n  Pearson corr(||z||, gt_distance), all steps:       {_corr(all_z, all_gt)}")
    if all_success.any():
        print(
            f"  Pearson corr(||z||, gt_distance), successful only: "
            f"{_corr(all_z[all_success], all_gt[all_success])}"
        )
    if (~all_success).any():
        print(
            f"  Pearson corr(||z||, gt_distance), failed only:     "
            f"{_corr(all_z[~all_success], all_gt[~all_success])}"
        )


def _apply_dark_theme() -> None:
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


def _success_legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=SUCCESS_COLOR, lw=2, label="Success"),
        Line2D([0], [0], color=FAILURE_COLOR, lw=2, label="Failure"),
    ]


def plot_z_norm_vs_time(
    results: list[EpisodeResult], metadata_str: str, save_path: Path, show: bool
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for r in results:
        if not r.steps:
            continue
        color = SUCCESS_COLOR if r.success_once else FAILURE_COLOR
        ax.plot(
            [s.step_idx for s in r.steps],
            [s.z_norm for s in r.steps],
            color=color,
            alpha=0.7,
            linewidth=1.2,
        )

    ax.set_title(
        f"Goal-Conditioning Signal ||z|| over a Live Rollout\n{metadata_str}",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Environment step")
    ax.set_ylabel("||z||")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(handles=_success_legend_handles(), loc="best", frameon=True)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    print(f"Saved: {save_path.resolve()}")
    if show:
        plt.show()
    plt.close(fig)


def plot_z_norm_vs_gt_distance(
    results: list[EpisodeResult],
    num_bins: int,
    metadata_str: str,
    save_path: Path,
    show: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    for r in results:
        if not r.steps:
            continue
        color = SUCCESS_COLOR if r.success_once else FAILURE_COLOR
        ax.scatter(
            [s.gt_distance for s in r.steps],
            [s.z_norm for s in r.steps],
            color=color,
            alpha=0.35,
            s=12,
            linewidths=0,
        )

    all_gt = np.array([s.gt_distance for r in results for s in r.steps])
    all_z = np.array([s.z_norm for r in results for s in r.steps])
    if len(all_gt) > 0 and all_gt.max() > all_gt.min():
        bin_edges = np.linspace(all_gt.min(), all_gt.max(), num_bins + 1)
        bin_idx = np.clip(np.digitize(all_gt, bin_edges) - 1, 0, num_bins - 1)
        bin_centers, bin_means, bin_stds = [], [], []
        for b in range(num_bins):
            mask = bin_idx == b
            if not mask.any():
                continue
            bin_centers.append((bin_edges[b] + bin_edges[b + 1]) / 2)
            bin_means.append(all_z[mask].mean())
            bin_stds.append(all_z[mask].std())
        bin_centers, bin_means, bin_stds = map(np.array, (bin_centers, bin_means, bin_stds))
        ax.plot(bin_centers, bin_means, color="#f8fafc", linewidth=2, label="Binned mean")
        ax.fill_between(
            bin_centers, bin_means - bin_stds, bin_means + bin_stds, color="#f8fafc", alpha=0.15
        )

    ax.set_title(
        f"||z|| vs. Ground-Truth Distance to Goal (pooled across episodes)\n{metadata_str}",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Ground-truth distance to goal (||goal_pos - obs_pos||)")
    ax.set_ylabel("||z||")
    ax.grid(True, linestyle="--", linewidth=0.5)
    handles = [
        *_success_legend_handles(),
        Line2D([0], [0], color="#f8fafc", lw=2, label="Binned mean"),
    ]
    ax.legend(handles=handles, loc="best", frameon=True)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    print(f"Saved: {save_path.resolve()}")
    if show:
        plt.show()
    plt.close(fig)


def plot_summary_bars(
    results: list[EpisodeResult], metadata_str: str, save_path: Path, show: bool
) -> None:
    valid = [r for r in results if r.steps]
    fig, ax = plt.subplots(figsize=(max(6, len(valid) * 0.8), 6))

    ratios = [
        (r.steps[-1].z_norm / r.steps[0].z_norm if r.steps[0].z_norm > 0 else 0.0) for r in valid
    ]
    colors = [SUCCESS_COLOR if r.success_once else FAILURE_COLOR for r in valid]
    ax.bar([r.episode_idx for r in valid], ratios, color=colors, alpha=0.85)
    ax.axhline(1.0, color="#64748b", linestyle="--", linewidth=1)

    ax.set_title(
        f"Per-Episode ||z|| Convergence Ratio (last / first step)\n{metadata_str}",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("||z||[-1] / ||z||[0]")
    ax.grid(True, linestyle="--", linewidth=0.5, axis="y")
    ax.legend(handles=_success_legend_handles(), loc="best", frameon=True)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    print(f"Saved: {save_path.resolve()}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    print(f"Loading checkpoint from: {ckpt_path}")
    model = load_goal_conditioned_diffusion_policy(ckpt_path)
    print(
        f"Loaded {type(model).__name__}: goal_delta={model.goal_delta!r}  "
        f"obs_horizon={model.obs_horizon}  act_horizon={model.act_horizon}"
    )

    cfg = load_env_config(ckpt_path)
    env_kwargs = resolve_env_kwargs(cfg)
    print(
        f"Env: {env_kwargs['env_id']}  obs_mode={env_kwargs['obs_mode']}  "
        f"control_mode={env_kwargs['control_mode']}"
    )

    print(f"\nRunning {args.num_episodes} live rollout episode(s)...")
    results = collect_all_episodes(model, args, env_kwargs)

    print_summary(results)

    metadata_str = build_metadata_str(model, cfg)
    metadata_slug = build_metadata_slug(model, cfg)
    _apply_dark_theme()
    base_prefix = args.save_path_prefix or ckpt_path.parent.parent.parent.name
    prefix = f"{base_prefix}_{metadata_slug}"
    save_dir = Path("scripts/figures")
    plot_z_norm_vs_time(results, metadata_str, save_dir / f"{prefix}_z_vs_time.png", args.show)
    plot_z_norm_vs_gt_distance(
        results,
        args.num_bins,
        metadata_str,
        save_dir / f"{prefix}_z_vs_gt_distance.png",
        args.show,
    )
    plot_summary_bars(
        results, metadata_str, save_dir / f"{prefix}_z_convergence_ratio.png", args.show
    )


if __name__ == "__main__":
    main()
