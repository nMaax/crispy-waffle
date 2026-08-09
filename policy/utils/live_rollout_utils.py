"""Shared plumbing for standalone analysis scripts (`scripts/analyze_goal_signal_convergence.py`,
`scripts/visualize_attention.py`) that drive a checkpoint's own live policy in a ManiSkill rollout
env, rather than replaying an offline HDF5 dataset.

TODO: `policy/algorithms/callbacks/rollout_evaluation.py`'s `RolloutEvaluationCallback` builds this
exact `gym.make -> FrameStack -> [RecordEpisode] -> ManiSkillVectorEnv` env (`setup()`/
`_run_rollouts()`) via its own independent, hand-rolled copy of this pattern -- a third
implementation alongside the two these scripts used to have before this module existed. Its
`_resolve_param()` is also the same override-with-fallback idiom as `resolve_env_kwargs()` below,
just resolving off a *live* `trainer.datamodule` instance instead of reconstructing one from a
saved `.hydra/config.yaml`. Worth eventually having the callback delegate to `build_rollout_env`/
`resolve_env_kwargs()` here instead of maintaining a third copy -- but not a drop-in swap: the
callback additionally supports `num_envs > 1` (batched parallel envs; `build_rollout_env` here
hardcodes `num_envs=1`), per-phase (`val`/`test`) video subdirs, and CPU/GPU backend-mismatch
validation/logging that don't exist here. Unifying would mean generalizing `build_rollout_env` to
take `num_envs`, and giving `resolve_env_kwargs` an alternate path that resolves off an
already-live datamodule object rather than always instantiating a fresh one from disk.
"""

from pathlib import Path
from typing import cast

import gymnasium as gym
import hydra
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers import FrameStack, RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from omegaconf import DictConfig, OmegaConf


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


def resolve_env_kwargs(cfg: DictConfig, env_id_override: str | None = None) -> dict:
    """Reads the rollout-relevant env settings off the checkpoint's saved datamodule config
    (mirrors the attribute names `RolloutEvaluationCallback._resolve_param` pulls off
    `trainer.datamodule`), with `env_id` independently overridable -- the same override-with-
    fallback idiom `RolloutEvaluationCallback` itself uses, and the same shape as the `__ZeroShot`
    experiment YAMLs' extra rollout callbacks (only `env_id`/`name` overridden, every other setting
    inherited from the training datamodule).

    `env_id`/`obs_mode`/`control_mode`/`physx_backend`/`robot_uids` are NOT Hydra config fields of
    `TrajectoryDataModule` -- they're computed after construction, inside
    `_prepare_local_dataset()`/`_load_metadata_from_json()`, by reading the dataset's sibling
    `.json` metadata file. A saved `.hydra/config.yaml`'s `datamodule:` block therefore never
    contains them; reading `cfg.datamodule.physx_backend` directly raises
    `ConfigAttributeError` unconditionally. Fix: actually instantiate the datamodule (mirrors
    `analyze_embedder_linearity.py::build_datamodule`), which populates them as genuine
    JSON-metadata-derived instance attributes.

    `physx_backend` is threaded through to `build_rollout_env`'s `gym.make(sim_backend=...)` call --
    ManiSkill's own default (`sim_backend="auto"`) picks CPU whenever `num_envs=1` (which
    `build_rollout_env` hardcodes), *regardless* of what backend the checkpoint was actually
    trained/evaluated with. Demos generated on `physx_cuda` (GPU sim) are common in this repo, and
    `RolloutEvaluationCallback` itself picks `physx_cuda` during training's own validation rollouts
    whenever the datamodule isn't CPU-backed -- silently forcing CPU here would evaluate the policy
    under different simulation dynamics than it was ever trained/validated against, which can tank
    an otherwise-working policy's live success rate for reasons that have nothing to do with the
    policy itself.
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


def build_rollout_env(
    env_kwargs: dict,
    obs_horizon: int,
    max_episode_steps: int | None,
    render_mode: str | None,
    video_dir: str | None,
):
    """Constructs gym_env -> FrameStack -> [RecordEpisode] -> ManiSkillVectorEnv, mirroring
    `RolloutEvaluationCallback.setup()` (policy/algorithms/callbacks/rollout_evaluation.py) as a
    free function, hardcoded to a single env (no Lightning Trainer/pl_module involved) -- but
    explicitly on `env_kwargs["physx_backend"]` (not ManiSkill's `num_envs`-driven "auto" default,
    which would silently pick CPU here since `num_envs=1`), so the live rollout runs under the same
    simulation backend the checkpoint was actually trained/evaluated with.

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
        sim_backend=env_kwargs["physx_backend"],
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
