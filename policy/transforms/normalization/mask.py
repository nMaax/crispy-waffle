from collections.abc import Mapping, Sequence

import torch

from policy.utils.typing_utils import TensorTree


def validate_mask(mask: TensorTree | None, dim: int | Sequence[int]) -> torch.Tensor | None:
    """Coerces a leaf normalization mask to a bool tensor shaped like the leaf it masks."""
    if mask is None:
        return None

    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"Leaf normalization mask must be a tensor, got {type(mask).__name__}.")

    shape = (dim,) if isinstance(dim, int) else tuple(dim)
    if tuple(mask.shape) != shape:
        raise ValueError(
            f"Normalization mask has shape {tuple(mask.shape)}, expected {shape} to match the "
            "width of the leaf it masks."
        )

    return mask.to(torch.bool)


def apply_mask(
    mask: torch.Tensor | None, transformed: torch.Tensor, original: torch.Tensor
) -> torch.Tensor:
    """Keeps `original` wherever `mask` is False, so non-normalizable channels pass through."""
    return transformed if mask is None else torch.where(mask, transformed, original)


def child_mask(mask: TensorTree | None, key: str) -> TensorTree | None:
    """Selects the sub-mask for `key`, treating an absent key as 'normalize this leaf in full'."""
    if mask is None:
        return None

    if not isinstance(mask, Mapping):
        raise TypeError(
            f"Normalization mask must be a mapping where the spec is a mapping, got "
            f"{type(mask).__name__} while descending into {key!r}."
        )

    return mask.get(key)
