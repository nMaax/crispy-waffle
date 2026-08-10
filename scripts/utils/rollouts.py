from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import gymnasium as gym
import hydra
import torch
from gymnasium.spaces import Box
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers import FrameStack, RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from omegaconf import DictConfig, OmegaConf

from policy.transforms.pipelines import observation_pipeline
from policy.utils import to_tensor
from policy.utils.typing_utils import TensorTree
from policy.utils.typing_utils.protocols import GoalConditionedEnvProtocol
from scripts.utils.checkpoints import build_external_cond

__all__ = [
    "DEFAULT_LOCKED_ROTATION_ENV_IDS",
    "EpisodeRollout",
    "RolloutStep",
    "build_obs_transform",
    "build_rollout_env",
    "extract_episode_metrics",
    "iter_env_kwargs",
    "load_env_config",
    "resolve_env_kwargs",
    "resolve_max_episode_steps",
    "run_episode",
]


# TODO: this could be re-used in the callback rollout


# --- Building the environment -------------------------------------------------------------------


DEFAULT_LOCKED_ROTATION_ENV_IDS = (
    "StackCubeLockedRotation-v1",
    "PlaceCubeLeftLockedRotation-v1",
    "PlaceCubeRightLockedRotation-v1",
    "StackCubeSwappedLockedRotation-v1",
)


def load_env_config(ckpt_path: Path) -> DictConfig:
    """Loads the checkpoint's saved Hydra run config."""
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


def resolve_env_kwargs(cfg: DictConfig, env_id_override: str | None = None) -> dict:
    """Reads the rollout-relevant env settings off the checkpoint's saved datamodule config
    (mirrors the attribute names `RolloutEvaluationCallback._resolve_param` pulls off
    `trainer.datamodule`), with `env_id` independently overridable.

    `env_id`/`obs_mode`/`control_mode`/`physx_backend`/`robot_uids` are NOT Hydra config fields of
    `TrajectoryDataModule`; they're computed after construction, inside `_prepare_local_dataset()`
    /`_load_metadata_from_json()`, by reading the dataset's sibling `.json` metadata file.
    Thus we actually instantiate the datamodule, which populates them a JSON-metadata-derived
    instance attributes.
    """
    dm = hydra.utils.instantiate(cfg.datamodule, num_workers=0)
    dm.prepare_data()  # no-op if hf_dataset_repo is None (__init__ already ran this then)

    physx_backend = dm.physx_backend
    if "cuda" in str(physx_backend).lower() and not torch.cuda.is_available():
        raise RuntimeError(
            f"Checkpoint was trained with physx_backend={physx_backend!r}, but CUDA is not "
            "available on this machine. Cannot run a CUDA-backed rollout here -- rerun on a "
            "CUDA-capable machine (matches RolloutEvaluationCallback.setup()'s own check)."
        )

    return {
        "env_id": env_id_override if env_id_override is not None else dm.env_id,
        "obs_mode": dm.obs_mode,
        "control_mode": dm.control_mode,
        "robot_uids": dm.robot_uids,
        "no_proprio_vel": bool(dm.no_proprio_vel),
        "physx_backend": physx_backend,
    }


def resolve_max_episode_steps(cfg: DictConfig, override: int | None) -> int | None:
    """Defaults to the checkpoint's own training-time rollout-evaluation `max_episode_steps`."""
    if override is not None:
        return override
    rollout_cfg = cfg.get("trainer", {}).get("callbacks", {}).get("rollout_evaluation", None)
    if rollout_cfg is not None:
        return rollout_cfg.get("max_episode_steps", None)
    return None


def build_rollout_env(
    env_kwargs: dict,
    obs_horizon: int,
    max_episode_steps: int | None,
    render_mode: str | None,
    video_dir: str | None,
):
    """Constructs gym_env -> FrameStack -> [RecordEpisode] -> ManiSkillVectorEnv, mirroring
    `RolloutEvaluationCallback.setup()`."""
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
        sim_backend=env_kwargs["physx_backend"],
        **make_kwargs,
    )
    inner_env = gym_env.unwrapped
    frame_stack_env = FrameStack(gym_env, num_stack=obs_horizon)

    if video_dir:
        frame_stack_as_base_env = cast(BaseEnv, frame_stack_env)
        max_steps = gym_utils.find_max_episode_steps_value(frame_stack_as_base_env)
        recorded_env = RecordEpisode(
            frame_stack_as_base_env,
            output_dir=video_dir,
            save_trajectory=False,
            save_video=True,
            max_steps_per_video=max_steps,
            source_type="diffusion_policy",
            source_desc="live rollout analysis",
        )
        vector_env = ManiSkillVectorEnv(
            recorded_env, ignore_terminations=True, record_metrics=True
        )
    else:
        vector_env = ManiSkillVectorEnv(
            frame_stack_env, ignore_terminations=True, record_metrics=True
        )
    return vector_env, inner_env


def extract_episode_metrics(info: dict) -> tuple[bool, bool, int]:
    """Pulls success_once/success_at_end/episode_len out of `info["final_info"]["episode"]`, with
    the same fallbacks `RolloutEvaluationCallback._run_rollouts` uses."""
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


# --- Driving it ----------------------------------------------------------------


@dataclass
class RolloutStep:
    """One environment step, with everything an analysis might want to record."""

    episode_idx: int
    step_idx: int
    obs: TensorTree
    """Transformed observation, still carrying the FrameStack time axis."""
    goal: TensorTree
    """Transformed goal state."""
    obs_raw: TensorTree
    """Untransformed observation, exactly as the env emitted it."""
    goal_raw: TensorTree
    """Untransformed goal state, before the observation pipeline is applied."""
    external_cond: Mapping[str, TensorTree]
    """The conditioning tree the network sees for this step."""
    action: torch.Tensor


@dataclass
class EpisodeRollout:
    """Outcome of one episode."""

    episode_idx: int
    success_once: bool
    success_at_end: bool
    episode_len: int
    num_steps: int
    records: list = field(default_factory=list)
    """Whatever `on_step` returned, in step order (None returns are dropped)."""


def build_obs_transform(
    env, env_kwargs: dict, cfg: DictConfig | None = None
) -> Callable[[TensorTree], TensorTree]:
    """Builds the observation pipeline for a live env."""
    datamodule_cfg = cfg.get("datamodule", {}) if cfg is not None else {}

    return observation_pipeline(
        env_id=env_kwargs["env_id"],
        is_flat=not isinstance(env.observation_space, gym.spaces.Dict),
        canonicalize=bool(datamodule_cfg.get("canonicalize", True)),
        as_dict=bool(datamodule_cfg.get("as_dict", True)),
        no_proprio_vel=bool(env_kwargs["no_proprio_vel"]),
    )


def _action_bounds(env, device) -> tuple[torch.Tensor, torch.Tensor]:
    action_space = env.action_space
    if not isinstance(action_space, Box):
        raise ValueError(f"Expected a Box action space, got {type(action_space).__name__}.")
    low = torch.as_tensor(action_space.low, device=device, dtype=torch.float32)
    high = torch.as_tensor(action_space.high, device=device, dtype=torch.float32)
    return low, high


def _plan(model, obs, goal, num_inference_steps):
    """Produces an action chunk plus the conditioning that produced it."""
    external_cond = build_external_cond(model, obs, goal)
    if hasattr(model, "_run_diffusion_loop"):
        action_seq = model._run_diffusion_loop(
            external_cond=external_cond,
            num_inference_steps=num_inference_steps,
            output_clip_range=None,
        )
    else:
        action_seq = model.get_action(obs, goal, num_inference_steps=num_inference_steps)
    return action_seq, external_cond


def run_episode(
    model,
    env,
    inner_env,
    transform: Callable[[TensorTree], TensorTree],
    *,
    episode_idx: int = 0,
    seed: int = 0,
    num_inference_steps: int | None = None,
    clamp_action: bool = True,
    render_mode: str | None = None,
    refresh_goal_each_step: bool = True,
    on_step: Callable[[RolloutStep], object] | None = None,
) -> EpisodeRollout:
    """Runs one episode to completion, replanning every `model.act_horizon` steps."""
    if hasattr(model, "reset"):
        model.reset()

    action_low, action_high = _action_bounds(env, model.device)

    obs_raw, info = env.reset(seed=seed)
    obs_raw = to_tensor(obs_raw, device=model.device, dtype=torch.float32)
    obs = transform(obs_raw)

    if not isinstance(inner_env, GoalConditionedEnvProtocol):
        raise TypeError(
            f"{type(inner_env).__name__} does not implement generate_heuristic_goal(); "
            "goal-conditioned analysis needs a goal-conditioned environment."
        )

    goal_raw = to_tensor(
        inner_env.generate_heuristic_goal(), device=model.device, dtype=torch.float32
    )
    goal = transform(goal_raw)

    if render_mode == "human":
        env.render()

    records: list = []
    step_idx = 0
    truncated_all = False

    with torch.no_grad():
        while not truncated_all:
            action_seq, external_cond = _plan(model, obs, goal, num_inference_steps)

            for i in range(model.act_horizon):
                action = action_seq[:, i]
                if clamp_action:
                    action = torch.clamp(
                        action, action_low.to(action.dtype), action_high.to(action.dtype)
                    )

                if on_step is not None:
                    record = on_step(
                        RolloutStep(
                            episode_idx=episode_idx,
                            step_idx=step_idx,
                            obs=obs,
                            goal=goal,
                            obs_raw=obs_raw,
                            goal_raw=goal_raw,
                            external_cond=external_cond,
                            action=action,
                        )
                    )
                    if record is not None:
                        records.append(record)

                obs_raw, _reward, _terminated, truncated, info = env.step(action)
                obs_raw = to_tensor(obs_raw, device=model.device, dtype=torch.float32)
                obs = transform(obs_raw)

                if render_mode == "human":
                    time.sleep(0.05)
                    env.render()

                if refresh_goal_each_step:
                    goal_raw = to_tensor(
                        inner_env.generate_heuristic_goal(),
                        device=model.device,
                        dtype=torch.float32,
                    )
                    goal = transform(goal_raw)

                step_idx += 1

                truncated_all = bool(
                    torch.as_tensor(truncated, device=model.device, dtype=torch.bool).all()
                )
                if truncated_all:
                    break

                # Refresh cheaply so per-step captures stay dense between replans.
                external_cond = build_external_cond(model, obs, goal)

    success_once, success_at_end, episode_len = extract_episode_metrics(info)

    return EpisodeRollout(
        episode_idx=episode_idx,
        success_once=success_once,
        success_at_end=success_at_end,
        episode_len=episode_len if episode_len >= 0 else step_idx,
        num_steps=step_idx,
        records=records,
    )


def iter_env_kwargs(
    cfg: DictConfig, env_ids: Sequence[str] | None = None
) -> Iterator[tuple[int, int, dict]]:
    """Yields `(index, total, env_kwargs)` for each env in a sweep."""
    ids = list(env_ids) if env_ids else list(DEFAULT_LOCKED_ROTATION_ENV_IDS)
    base = resolve_env_kwargs(cfg)

    for index, env_id in enumerate(ids):
        env_kwargs = dict(base)
        env_kwargs["env_id"] = env_id
        if len(ids) > 1:
            print(f"\n{'#' * 88}\n# Env {index + 1}/{len(ids)}: {env_id}\n{'#' * 88}")
        yield index, len(ids), env_kwargs
