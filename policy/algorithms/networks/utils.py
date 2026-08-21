from __future__ import annotations

from collections.abc import Mapping

import hydra.utils
import torch.nn as nn

from policy.utils import drop_key, get_total_dim
from policy.utils.typing_utils import DimSpec


def resolve_activation(activation: type[nn.Module] | str) -> type[nn.Module]:
    """Resolves an activation given either as a class or a string (looked up on ``torch.nn``,
    falling back to a fully-qualified Hydra target)."""
    if isinstance(activation, str):
        if hasattr(nn, activation):
            return getattr(nn, activation)
        return hydra.utils.get_class(activation)
    return activation


def resolve_proprio_dim(obs_dim: DimSpec, proprio_dim: int | None = None) -> int:
    """Resolves ``proprio_dim`` against ``obs_dim``.

    An explicit value is validated against `obs_dim` (via :func:`validate_proprio_dim`) and
    returned as-is. When omitted (``None``), it is derived from ``obs_dim["proprio"]`` when
    `obs_dim` is a dict. A flat `obs_dim` carries no field names, so it cannot be inferred
    and requires an explicit value.
    """
    if proprio_dim is not None:
        validate_proprio_dim(obs_dim, proprio_dim)
        return proprio_dim

    if not isinstance(obs_dim, Mapping):
        raise ValueError(
            "proprio_dim must be provided explicitly when obs_dim is flat (not a dict), since "
            "it cannot be inferred from a flat observation spec."
        )

    if "proprio" not in obs_dim:
        raise ValueError("Observation dictionary spec must contain 'proprio' key.")

    return get_total_dim(obs_dim["proprio"])


def validate_proprio_dim(obs_dim: DimSpec, proprio_dim: int) -> None:
    """Validates that `proprio_dim` is consistent with `obs_dim`.

    A Mapping `obs_dim` must contain a `"proprio"` key whose width matches `proprio_dim`; a flat
    `obs_dim` must be at least `proprio_dim` wide (proprio occupies the leading features).
    """
    if isinstance(obs_dim, Mapping):
        if "proprio" not in obs_dim:
            raise ValueError("Observation dictionary spec must contain 'proprio' key.")
        if obs_dim["proprio"] != proprio_dim:
            raise ValueError(
                f"Proprioception dimension in spec ({obs_dim['proprio']}) does not match "
                f"proprio_dim ({proprio_dim})."
            )
    elif isinstance(obs_dim, int):
        if obs_dim < proprio_dim:
            raise ValueError(
                f"Observation dimension ({obs_dim}) must be >= proprio_dim ({proprio_dim})."
            )
    else:
        raise ValueError(
            f"Observation dimensionality must be an integer or dict, but got {type(obs_dim)}."
        )


def derive_task_dim(obs_dim: DimSpec, proprio_dim: int, task_dim: int | None = None) -> int:
    """Derives the task-only (non-proprio) width from `obs_dim`, optionally cross-checking it
    against an explicitly provided `task_dim`."""
    if isinstance(obs_dim, Mapping):
        calc_task_dim = sum(get_total_dim(v) for v in drop_key(obs_dim, "proprio").values())
    else:
        calc_task_dim = get_total_dim(obs_dim) - proprio_dim

    if task_dim is not None and calc_task_dim != task_dim:
        raise ValueError(
            f"Task dimension calculated from spec ({calc_task_dim}) does not match task_dim ({task_dim})."
        )

    return calc_task_dim
