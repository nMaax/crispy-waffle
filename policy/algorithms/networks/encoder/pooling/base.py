from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import torch
import torch.nn as nn

PoolingMode = Literal["all", "objects", "time"]


class BasePooling(nn.Module, ABC):
    """Abstract base class for token sequence pooling modules."""

    def __init__(self, mode: PoolingMode = "all"):
        super().__init__()
        if mode not in ("all", "objects", "time"):
            raise ValueError(
                f"Unknown pooling mode: {mode!r}. Expected 'all', 'objects', or 'time'."
            )
        self.mode: PoolingMode = mode

    @property
    def pools_time(self) -> bool:
        return self.mode in ("all", "time")

    @property
    def pools_objects(self) -> bool:
        return self.mode in ("all", "objects")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pools tokens across the dimensions configured by ``self.mode``.

        Shapes:
            x: [B, T, dim] or [B, T, K, dim] (K tokens per timestep).
            returns:
                - mode="all": [B, dim]
                - mode="objects": [B, T, dim]
                - mode="time": [B, K, dim]
        """
        if x.ndim == 3:
            return self._pool(x)

        if x.ndim == 4:
            b, t, k, dim = x.shape
            if self.mode == "all":
                return self._pool(x.reshape(b, t * k, dim))
            if self.mode == "objects":
                pooled = self._pool(x.reshape(b * t, k, dim))
                return pooled.reshape(b, t, -1)
            if self.mode == "time":
                pooled = self._pool(x.permute(0, 2, 1, 3).reshape(b * k, t, dim))
                return pooled.reshape(b, k, -1)

        raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(x.shape)}")

    @abstractmethod
    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        """Pools a 3D sequence [B, L, dim] -> [B, out_dim]."""
        raise NotImplementedError
