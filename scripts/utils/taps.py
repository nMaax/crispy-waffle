from __future__ import annotations

import contextlib
import math
from collections.abc import Generator

import torch
import torch.nn as nn

# Layers whose output magnitude is constrained by construction, so measuring it says more about the
# layer than about the data flowing through it.
NORMALISATION_LAYERS: tuple[type[nn.Module], ...] = tuple(
    layer
    for layer in (
        nn.LayerNorm,
        nn.BatchNorm1d,
        nn.GroupNorm,
        getattr(nn, "RMSNorm", None),
    )
    if layer is not None
)


def final_output_module(embedder: nn.Module) -> nn.Module:
    """The submodule that produces the embedder's final output."""
    pooling = getattr(embedder, "pooling", None)
    return pooling if isinstance(pooling, nn.Module) else embedder


def output_norm(module: nn.Module) -> nn.Module | None:
    """The normalisation layer a module's output passes through, or None if there isn't one."""
    candidate = getattr(module, "norm", None)
    if isinstance(candidate, NORMALISATION_LAYERS):
        return candidate
    return None


def embedder_output_norm(embedder: nn.Module | None) -> nn.Module | None:
    """The normalisation on the embedder's final output, following through any pooling."""
    if embedder is None:
        return None
    return output_norm(final_output_module(embedder))


def magnitude_band(norm: nn.Module) -> tuple[float, float] | None:
    """The range the output magnitude of a normalisation layer is confined to."""
    gain = getattr(norm, "weight", None)
    if gain is None:
        return None

    gain = gain.detach()
    scale = math.sqrt(gain.numel())
    bias = getattr(norm, "bias", None)
    slack = float(torch.linalg.norm(bias.detach())) if bias is not None else 0.0

    low = scale * float(gain.abs().min()) - slack
    high = scale * float(gain.abs().max()) + slack
    return max(0.0, low), high


@contextlib.contextmanager
def capture_pre_norm(embedder: nn.Module | None) -> Generator[list[torch.Tensor], None, None]:
    """Collects the representation feeding the embedder's final normalisation."""
    captured: list[torch.Tensor] = []
    norm = embedder_output_norm(embedder)

    if norm is None:
        yield captured
        return

    def pre_hook(_module: nn.Module, args: tuple) -> None:
        if args:
            captured.append(args[0].detach())

    handle = norm.register_forward_pre_hook(pre_hook)
    try:
        yield captured
    finally:
        handle.remove()


def relative_spread(values) -> float:
    """`std / mean` of a signal through time."""
    import numpy as np

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return float("nan")
    mean = float(np.mean(array))
    if mean == 0.0:
        return float("nan")
    return float(np.std(array) / abs(mean))
