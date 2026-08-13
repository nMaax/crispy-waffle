from __future__ import annotations

import torch
import torch.nn as nn


def pool_tokens(x: torch.Tensor, pooling: nn.Module) -> torch.Tensor:
    """Flattens an optional per-timestep token axis into the sequence axis pooling expects, then
    pools.

    Shapes:
        x: [B, T, dim] or [B, T, K, dim] (K tokens per timestep).
        returns: [B, dim].
    """
    if x.ndim == 4:
        b, t, k, dim = x.shape
        x = x.reshape(b, t * k, dim)
    return pooling(x)
