"""Utilities to help annotate the types of values in the policy."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, NewType, TypeAlias, TypeGuard

import numpy as np
import torch
from hydra_zen.typing import Builds
from typing_extensions import TypeVar

from .protocols import (
    AdapterProtocol,
    DataModule,
    DiffusionNetworkProtocol,
    DiffusionSchedulerProtocol,
    EnvProtocol,
    GoalConditionedEnvProtocol,
    GoalConditionedPolicyProtocol,
    PolicyProtocol,
)

# These are used to show which dim is which.
C = NewType("C", int)
H = NewType("H", int)
W = NewType("W", int)


T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")

HydraConfigFor = Builds[type[T]]
"""Type annotation to say "a hydra config that returns an object of type T when instantiated"."""


NestedMapping = Mapping[K, V | "NestedMapping[K, V]"]
PyTree = T | Iterable["PyTree[T]"] | Mapping[Any, "PyTree[T]"]

TensorTree: TypeAlias = torch.Tensor | Mapping[str, "TensorTree"]
"""A tensor, or an arbitrarily nested mapping of tensors."""

NestedTensorMapping: TypeAlias = Mapping[str, Mapping[str, torch.Tensor]]
"""A 2-level nested mapping of tensors, e.g., environment observation state dicts."""

RawTree: TypeAlias = torch.Tensor | np.ndarray | Sequence[Any] | Mapping[str, "RawTree"]
"""A raw array, sequence, or nested mapping of raw data prior to tensor conversion."""

DimSpec: TypeAlias = int | torch.Tensor | Mapping[str, "DimSpec"]
"""A dimension specification: an integer, tensor, or nested mapping of dimensions."""


def is_sequence_of(
    object: Any, item_type: type[V] | tuple[type[V], ...]
) -> TypeGuard[Sequence[V]]:
    """Used to check (and tell the type checker) that `object` is a sequence of items of this
    type."""
    return isinstance(object, Sequence) and all(isinstance(value, item_type) for value in object)


def is_mapping_of(object: Any, key_type: type[K], value_type: type[V]) -> TypeGuard[Mapping[K, V]]:
    """Used to check (and tell the type checker) that `object` is a mapping with keys and values of
    the given types."""
    return isinstance(object, Mapping) and all(
        isinstance(key, key_type) and isinstance(value, value_type)
        for key, value in object.items()
    )


__all__ = [
    "DataModule",
    "DiffusionSchedulerProtocol",
    "AdapterProtocol",
    "PolicyProtocol",
    "GoalConditionedPolicyProtocol",
    "DiffusionNetworkProtocol",
    "GoalConditionedEnvProtocol",
    "EnvProtocol",
    "TensorTree",
    "NestedTensorMapping",
    "RawTree",
    "DimSpec",
]
