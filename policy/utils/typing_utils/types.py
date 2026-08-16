from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

import numpy as np
import torch
from hydra_zen.typing import Builds
from typing_extensions import TypeVar

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

__all__ = [
    "DimSpec",
    "HydraConfigFor",
    "K",
    "Leaf",
    "NestedMapping",
    "RawLeaf",
    "RawTree",
    "T",
    "TensorLeaf",
    "TensorTree",
    "Tree",
    "V",
]
