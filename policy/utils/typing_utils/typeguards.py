from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypeGuard

from .types import K, T, V


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


__all__ = [
    "is_mapping_of",
    "is_sequence_of",
]
