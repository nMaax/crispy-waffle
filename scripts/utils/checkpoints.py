from __future__ import annotations

import functools
import re
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

from policy.algorithms.goal_conditioned_diffusion_policy import GoalConditionedDiffusionPolicy
from policy.utils.hydra_utils import find_checkpoint_hydra_config
from policy.utils.typing_utils import TensorTree

# Stripped from experiment names when building filename slugs. The full name still appears in the
# report and the figure subtitle, so nothing is lost -- this only keeps filenames manageable.
_SLUG_NOISE = (
    "GoalConditioned",
    "__default__train",
    "__default__test",
)


# --- Run config -------------------------------------------------------------------------------


def resolve_run_config(ckpt_path: Path) -> DictConfig | None:
    """Finds the Hydra config for the run that produced `ckpt_path`, or None with a warning."""
    config = find_checkpoint_hydra_config(str(ckpt_path))
    if config is None:
        warnings.warn(
            f"No .hydra/config.yaml found above {ckpt_path}. Provenance fields and env settings "
            "recovered from the training run will be unavailable.",
            stacklevel=2,
        )
    return config


def require_run_config(ckpt_path: Path) -> DictConfig:
    """Like `resolve_run_config`, but raises when the config is missing."""
    config = find_checkpoint_hydra_config(str(ckpt_path))
    if config is None:
        raise FileNotFoundError(
            f"No .hydra/config.yaml found above {ckpt_path}; cannot recover the environment "
            "settings this checkpoint was trained with. Place the run's config.yaml in a "
            f".hydra/ directory beside the checkpoint (e.g. {ckpt_path.parent / '.hydra'})."
        )
    return config


# --- Checkpoint identity ----------------------------------------------------------------------


@functools.lru_cache(maxsize=8)
def _checkpoint_metadata(ckpt_path_str: str) -> dict[str, Any]:
    """Reads the non-tensor metadata out of a checkpoint."""
    data: dict[str, Any] = torch.load(ckpt_path_str, map_location="cpu", weights_only=False)
    return {
        "epoch": data.get("epoch"),
        "global_step": data.get("global_step"),
        "callbacks": data.get("callbacks", {}),
        "hyper_parameters": data.get("hyper_parameters", {}),
    }


def _experiment_from_parts(parts: tuple[str, ...]) -> str | None:
    """Extracts `<experiment>` from a `.../<experiment>/{runs,multiruns}/<date>/...` path."""
    for i, part in enumerate(parts):
        if part in ("runs", "multiruns") and i > 0:
            return parts[i - 1]
    return None


def experiment_name(ckpt_path: Path) -> str | None:
    """Recovers the experiment name a checkpoint came from.

    Prefers the run directory in the checkpoint's own path; falls back to the `dirpath` recorded by
    the model-checkpoint callback.
    """
    from_path = _experiment_from_parts(ckpt_path.resolve().parts)
    if from_path:
        return from_path

    try:
        callbacks = _checkpoint_metadata(str(ckpt_path)).get("callbacks", {})
    except Exception:
        return None

    for state in callbacks.values():
        if not isinstance(state, dict):
            continue
        dirpath = state.get("dirpath")
        if not dirpath:
            continue
        from_callback = _experiment_from_parts(Path(str(dirpath)).parts)
        if from_callback:
            return from_callback
    return None


def compact_name(name: str) -> str:
    """Shortens a long experiment name for use in a filename."""
    for noise in _SLUG_NOISE:
        name = name.replace(noise, "")
    # Drop the env segment; it is already a directory level in the output path.
    name = re.sub(r"__[A-Za-z]+-v\d+", "", name)
    return name.strip("_") or "checkpoint"


def checkpoint_slug(ckpt_path: Path) -> str:
    """A short identifier for a checkpoint, used in output filenames."""
    name = experiment_name(ckpt_path)
    if name:
        return compact_name(name)
    else:
        return ckpt_path.stem


# --- Model / config description ---------------------------------------------------------------


def describe_model_config(
    model: Any,
    cfg: DictConfig | None = None,
    extra: list[tuple[str, str]] | None = None,
) -> list[tuple[str, str]]:
    """Summarises the parts of a model that distinguish one experiment from another."""
    pooling = getattr(getattr(model, "embedder", None), "pooling", None)
    tokenizer = getattr(model, "tokenizer", None)
    embedder = getattr(model, "embedder", None)

    fields: list[tuple[str, str]] = list(extra or [])
    fields.extend(
        [
            ("tokenizer", type(tokenizer).__name__ if tokenizer is not None else "none"),
            ("embedder", type(embedder).__name__ if embedder is not None else "none"),
            ("pooling", type(pooling).__name__ if pooling is not None else "none"),
            ("goal_delta", str(getattr(model, "goal_delta", "n/a"))),
        ]
    )

    if cfg is not None:
        her_ratio = cfg.get("datamodule", {}).get("her_ratio", None)
        if her_ratio is not None:
            fields.append(("her_ratio", str(her_ratio)))

    return fields


def metadata_slug(model: Any, cfg: DictConfig | None = None) -> str:
    """Condenses the distinguishing config into a filename fragment."""
    abbreviations = {"goal_delta": "gd", "her_ratio": "her"}
    parts = []
    for key, value in describe_model_config(model, cfg):
        if key in abbreviations:
            parts.append(f"{abbreviations[key]}-{value}")
    return "_".join(parts)


def run_slug(
    ckpt_path: Path, model: Any, cfg: DictConfig | None = None, seed: int | None = None
) -> str:
    """The full per-run filename prefix: checkpoint, seed, config."""
    parts = [checkpoint_slug(ckpt_path)]
    if seed is not None:
        parts.append(f"seed{seed}")
    meta = metadata_slug(model, cfg)
    if meta:
        parts.append(meta)
    return "_".join(parts)


# --- Conditioning -----------------------------------------------------------------------------


def build_external_cond(model: Any, obs: TensorTree, goal: TensorTree) -> Mapping[str, TensorTree]:
    """Normalises obs/goal and builds the conditioning tree the network sees."""
    if model.obs_normalizer is not None:
        obs = model.obs_normalizer.normalize(obs)
        goal = model.obs_normalizer.normalize(goal)
    return model._build_external_cond(obs, goal)


# --- Loading ----------------------------------------------------------------------------------


def load_goal_conditioned_diffusion_policy(ckpt_path: Path) -> GoalConditionedDiffusionPolicy:
    """Rebuilds a `GoalConditionedDiffusionPolicy` from a checkpoint's own hyperparameters."""
    checkpoint_data: dict[str, Any] = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = checkpoint_data.get("hyper_parameters", {})

    act_dim = hparams.get("act_dim")
    network_config = dict(hparams.get("network", {}))
    if act_dim is not None:
        network_config["act_dim"] = act_dim

    embedder_config = hparams.get("embedder")
    if embedder_config is None and "state_embedding_dim" in hparams:
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
    return model
