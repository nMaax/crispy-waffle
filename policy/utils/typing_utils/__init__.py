"""Utilities to help annotate the types of values in the policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NewType, TypeAlias, TypeGuard

import numpy as np
import torch
from hydra_zen.typing import Builds
from typing_extensions import TypeVar

from .protocols import (
    DataModule,
    DiffusionNetworkProtocol,
    DiffusionSchedulerProtocol,
    EnvProtocol,
    GoalConditionedEnvProtocol,
    GoalConditionedPolicyProtocol,
    PolicyProtocol,
)

# These are used to show which dim is which in image data
C = NewType("C", int)
"""NewType annotation representing a image/feature channel dimension."""

H = NewType("H", int)
"""NewType annotation representing an image height dimension."""

W = NewType("W", int)
"""NewType annotation representing an image width dimension."""


T = TypeVar("T")
HydraConfigFor = Builds[type[T]]
"""Type annotation to say "a hydra config that returns an object of type T when instantiated"."""


K = TypeVar("K")
V = TypeVar("V")
NestedMapping: TypeAlias = Mapping[K, V | "NestedMapping[K, V]"]
"""A mapping with keys of type K and values that are either of type V or recursively nested
mappings."""

Leaf = TypeVar("Leaf")
Tree: TypeAlias = Leaf | NestedMapping[str, Leaf]
"""A generic tree structure mapping string keys to either leaf values of type Leaf or nested
subtrees."""

TensorLeaf: TypeAlias = torch.Tensor
TensorTree: TypeAlias = Tree[TensorLeaf]
"""A tensor, or an arbitrarily nested mapping of tensors."""

RawLeaf: TypeAlias = torch.Tensor | np.ndarray | Sequence[Any]
RawTree: TypeAlias = Tree[RawLeaf]
"""A raw array, sequence, or nested mapping of raw data prior to tensor conversion."""

DimSpec: TypeAlias = int | torch.Tensor | Sequence[int] | Mapping[str, "DimSpec"]
"""A dimension specification: an integer, tensor, shape sequence, or nested mapping of dimensions."""

IndexRange: TypeAlias = tuple[int, int]
"""An index range tuple (start, end) defining a slice along an axis."""

StateSchema: TypeAlias = NestedMapping[str, IndexRange]
"""A nested mapping schema of string keys to index range tuples."""


def is_sequence_of(
    object: Any, item_type: type[T] | tuple[type[T], ...]
) -> TypeGuard[Sequence[T]]:
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


def get_tensor(tree: Mapping[str, TensorTree], key: str) -> torch.Tensor:
    """Look up `key`, asserting the result is a leaf tensor (not a further nested mapping)."""
    value = tree[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected a Tensor at key {key!r}, got {type(value).__name__}.")
    return value


def get_subtree(tree: Mapping[str, TensorTree], key: str) -> Mapping[str, TensorTree]:
    """Look up `key`, asserting the result is a nested mapping (not a leaf tensor)."""
    value = tree[key]
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected a nested mapping at key {key!r}, got {type(value).__name__}.")
    return value


__all__ = [
    "C",
    "H",
    "W",
    "T",
    "DataModule",
    "DiffusionSchedulerProtocol",
    "PolicyProtocol",
    "GoalConditionedPolicyProtocol",
    "DiffusionNetworkProtocol",
    "GoalConditionedEnvProtocol",
    "EnvProtocol",
    "HydraConfigFor",
    "NestedMapping",
    "Tree",
    "TensorTree",
    "RawLeaf",
    "RawTree",
    "DimSpec",
    "IndexRange",
    "StateSchema",
    "is_sequence_of",
    "is_mapping_of",
    "get_tensor",
    "get_subtree",
]
