from collections.abc import Iterable, Mapping
from typing import overload

import torch
import torch.nn as nn


class ZScoreNormalizer(nn.Module):
    """Z-score normalizer (mean=0, std=1) for a tensor or a nested mapping of tensors.

    A dict is treated as a tree; each leaf tensor gets its own independent
    normalizer, fit/applied over its last (feature) dimension. Structure of
    the tree of sub-normalizers is fixed at construction time via `spec`.

    Example:
        spec = {
            "proprio": {"qpos": 9, "qvel": 9},
            "states": 30,
        }
        normalizer = ZScoreNormalizer(spec)
        normalizer.fit(batch)          # batch matches the tree structure
        normed = normalizer(batch)     # same tree structure, normalized leaves
        original = normalizer.unnormalize(normed)
    """

    _n: int
    _running_mean: torch.Tensor | None
    _running_M2: torch.Tensor | None

    def __init__(self, spec: int | torch.Tensor | Mapping):
        super().__init__()

        if not isinstance(spec, Mapping):
            if isinstance(spec, torch.Tensor):
                dim = spec.shape[-1]
            else:
                dim = spec

            self.register_buffer("mean", torch.zeros(dim))
            self.register_buffer("std", torch.ones(dim))
            self.register_buffer("_is_fit", torch.tensor(False))
        else:
            self.norms = nn.ModuleDict(
                {key: ZScoreNormalizer(child_spec) for key, child_spec in spec.items()}
            )

    @property
    def is_fit(self) -> torch.Tensor:
        if hasattr(self, "norms"):
            all_fit = all(bool(norm.is_fit.item()) for norm in self.norms.values())
            return torch.tensor(all_fit)
        else:
            return self._is_fit

    def fit(self, data: torch.Tensor | dict):
        """Fits the normalizer to the data, computing mean and std."""
        if not isinstance(data, Mapping):
            data_flat = data.reshape(-1, data.shape[-1])
            self.mean.copy_(data_flat.mean(dim=0))
            self.std.copy_(data_flat.std(dim=0).clamp(min=1e-6))
            self._is_fit.fill_(True)
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
            self._n = 0
            self._running_mean = None
            self._running_M2 = None
        else:
            for norm in self.norms.values():
                norm._init_running_stats()

    def _update_running_stats(self, data: torch.Tensor | dict) -> None:
        if not hasattr(self, "norms"):
            if not isinstance(data, torch.Tensor):
                data = torch.as_tensor(data)
            x_flat = data.reshape(-1, data.shape[-1]).to(torch.float64)
            n_B = x_flat.shape[0]
            if n_B == 0:
                return

            mean_B = x_flat.mean(dim=0)
            M2_B = torch.sum((x_flat - mean_B) ** 2, dim=0)

            if self._n == 0:
                self._n = n_B
                self._running_mean = mean_B
                self._running_M2 = M2_B
            else:
                n_X = self._n + n_B
                delta = mean_B - self._running_mean
                self._running_mean = self._running_mean + delta * (n_B / n_X)
                self._running_M2 = self._running_M2 + M2_B + (delta**2) * (self._n * n_B / n_X)
                self._n = n_X
        else:
            assert isinstance(data, Mapping), "Expected dict data for nested normalizer"
            for key, norm in self.norms.items():
                norm._update_running_stats(data[key])

    def _finalize_running_stats(self) -> None:
        if not hasattr(self, "norms"):
            if self._n > 1:
                assert self._running_mean is not None
                assert self._running_M2 is not None
                mean = self._running_mean.to(self.mean.dtype)
                var = self._running_M2 / (self._n - 1)
                std = torch.sqrt(var).to(self.std.dtype).clamp(min=1e-6)
                self.mean.copy_(mean.to(self.mean.device))
                self.std.copy_(std.to(self.std.device))
                self._is_fit.fill_(True)
            elif self._n == 1:
                assert self._running_mean is not None
                mean = self._running_mean.to(self.mean.dtype)
                self.mean.copy_(mean.to(self.mean.device))
                self.std.fill_(1.0)
                self._is_fit.fill_(True)

            if hasattr(self, "_n"):
                del self._n
            if hasattr(self, "_running_mean"):
                del self._running_mean
            if hasattr(self, "_running_M2"):
                del self._running_M2
        else:
            for norm in self.norms.values():
                norm._finalize_running_stats()

    @overload
    def normalize(self, x: torch.Tensor) -> torch.Tensor: ...

    @overload
    def normalize(self, x: Mapping) -> dict: ...

    def normalize(self, x: torch.Tensor | Mapping) -> torch.Tensor | dict:
        """Normalizes the input tensor or mapping of tensors using the fitted mean and std."""
        if not isinstance(x, Mapping):
            if self.is_fit:
                return (x - self.mean) / self.std
            else:
                return x
        else:
            return {key: norm.normalize(x[key]) for key, norm in self.norms.items()}

    @overload
    def unnormalize(self, x: torch.Tensor) -> torch.Tensor: ...

    @overload
    def unnormalize(self, x: dict) -> dict: ...

    def unnormalize(self, x: torch.Tensor | dict) -> torch.Tensor | dict:
        """Unnormalizes the input tensor or dict of tensors using the fitted mean and std."""
        if not isinstance(x, Mapping):
            if self.is_fit:
                return (x * self.std) + self.mean
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
