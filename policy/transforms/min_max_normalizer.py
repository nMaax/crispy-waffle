from collections.abc import Iterable
from typing import overload

import torch
import torch.nn as nn


class MinMaxNormalizer(nn.Module):
    """Min-Max normalizer (scales to [min_val, max_val]) for a tensor or nested dict of tensors.

    Structure of the tree of sub-normalizers is fixed at construction time via `spec`.
    """

    _running_min: torch.Tensor | None
    _running_max: torch.Tensor | None

    def __init__(self, spec: int | torch.Tensor | dict, min_val: float = -1.0, max_val: float = 1.0):
        super().__init__()

        self.min_val = min_val
        self.max_val = max_val

        if not isinstance(spec, dict):
            if isinstance(spec, torch.Tensor):
                dim = spec.shape[-1]
            else:
                dim = spec

            self.register_buffer("min", torch.zeros(dim))
            self.register_buffer("max", torch.ones(dim))
            self.register_buffer("is_fit", torch.tensor(False))
        else:
            self.norms = nn.ModuleDict(
                {
                    key: MinMaxNormalizer(child_spec, min_val, max_val)
                    for key, child_spec in spec.items()
                }
            )

    def fit(self, data: torch.Tensor | dict):
        """Fits the normalizer to the data, computing min and max."""
        if not isinstance(data, dict):
            data_flat = data.reshape(-1, data.shape[-1])
            self.min.copy_(data_flat.min(dim=0).values)
            self.max.copy_(data_flat.max(dim=0).values)
            self.is_fit.fill_(True)
        else:
            for key, norm in self.norms.items():
                norm.fit(data[key])

    def fit_incremental(self, data_iterator: Iterable[torch.Tensor | dict]) -> None:
        """Fits the normalizer incrementally using an iterator of data (e.g. trajectories).

        This avoids loading the entire dataset into memory.
        """
        self._init_running_stats()
        for item in data_iterator:
            self._update_running_stats(item)
        self._finalize_running_stats()

    def _init_running_stats(self) -> None:
        if not hasattr(self, "norms"):
            self._running_min = None
            self._running_max = None
        else:
            for norm in self.norms.values():
                norm._init_running_stats()

    def _update_running_stats(self, data: torch.Tensor | dict) -> None:
        if not hasattr(self, "norms"):
            if not isinstance(data, torch.Tensor):
                data = torch.as_tensor(data)
            x_flat = data.reshape(-1, data.shape[-1]).to(torch.float32)
            if x_flat.shape[0] == 0:
                return

            batch_min = x_flat.min(dim=0).values
            batch_max = x_flat.max(dim=0).values

            if self._running_min is None or self._running_max is None:
                self._running_min = batch_min.to(torch.float64)
                self._running_max = batch_max.to(torch.float64)
            else:
                self._running_min = torch.minimum(self._running_min, batch_min.to(torch.float64))
                self._running_max = torch.maximum(self._running_max, batch_max.to(torch.float64))
        else:
            assert isinstance(data, dict), "Expected dict data for nested normalizer"
            for key, norm in self.norms.items():
                norm._update_running_stats(data[key])

    def _finalize_running_stats(self) -> None:
        if not hasattr(self, "norms"):
            if self._running_min is not None and self._running_max is not None:
                self.min.copy_(self._running_min.to(self.min.dtype).to(self.min.device))
                self.max.copy_(self._running_max.to(self.max.dtype).to(self.max.device))
                self.is_fit.fill_(True)

            if hasattr(self, "_running_min"):
                del self._running_min
            if hasattr(self, "_running_max"):
                del self._running_max
        else:
            for norm in self.norms.values():
                norm._finalize_running_stats()

    @overload
    def normalize(self, x: torch.Tensor) -> torch.Tensor: ...

    @overload
    def normalize(self, x: dict) -> dict: ...

    def normalize(self, x: torch.Tensor | dict) -> torch.Tensor | dict:
        """Normalizes the input tensor or dict of tensors using the fitted min and max."""
        if not isinstance(x, dict):
            if self.is_fit:
                diff = (self.max - self.min).clamp(min=1e-6)
                return self.min_val + (self.max_val - self.min_val) * (x - self.min) / diff
            else:
                return x
        else:
            return {key: norm.normalize(x[key]) for key, norm in self.norms.items()}

    @overload
    def unnormalize(self, x: torch.Tensor) -> torch.Tensor: ...

    @overload
    def unnormalize(self, x: dict) -> dict: ...

    def unnormalize(self, x: torch.Tensor | dict) -> torch.Tensor | dict:
        """Unnormalizes the input tensor or dict of tensors using the fitted min and max."""
        if not isinstance(x, dict):
            if self.is_fit:
                diff = (self.max - self.min).clamp(min=1e-6)
                return (x - self.min_val) * diff / (self.max_val - self.min_val) + self.min
            else:
                return x
        else:
            return {key: norm.unnormalize(x[key]) for key, norm in self.norms.items()}

    @overload
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    @overload
    def forward(self, x: dict) -> dict: ...

    def forward(self, x: torch.Tensor | dict) -> torch.Tensor | dict:
        """Forward pass for the normalizer, which normalizes the input tensor or dict of
        tensors."""
        return self.normalize(x)
