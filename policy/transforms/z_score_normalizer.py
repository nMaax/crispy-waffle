from collections.abc import Iterable, Mapping
from typing import overload

import torch
import torch.nn as nn

from policy.utils.typing_utils import DimSpec, TensorTree


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

    def __init__(self, spec: DimSpec):
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

    def fit(self, data: TensorTree):
        """Fits the normalizer on the provided dataset or batch of data."""
        if not isinstance(data, Mapping):
            if self.mean.ndim == 0:
                raise ValueError("Cannot fit normalizer with zero-dimensional target buffer")

            data_flat = data.reshape(-1, data.shape[-1])
            mean = data_flat.mean(dim=0)
            std = data_flat.std(dim=0)

            # Avoid divide-by-zero for constant features by setting std=1
            std = torch.where(std < 1e-6, torch.ones_like(std), std)

            self.mean.copy_(mean)
            self.std.copy_(std)
            self._is_fit.fill_(True)
        else:
            for key, norm in self.norms.items():
                norm.fit(data[key])

    def fit_incremental(self, data: Iterable[TensorTree]):
        """Fits the normalizer incrementally on batches using Welford's algorithm."""
        self._init_running_stats()
        for batch in data:
            self._update_running_stats(batch)
        self._finalize_running_stats()

    def _init_running_stats(self) -> None:
        if not hasattr(self, "norms"):
            self._n = 0
            self._running_mean = None
            self._running_M2 = None
        else:
            for norm in self.norms.values():
                norm._init_running_stats()

    def _update_running_stats(self, data: TensorTree):
        if not isinstance(data, Mapping):
            assert isinstance(data, torch.Tensor), "Expected Tensor data for leaf normalizer"
            data_flat = data.reshape(-1, data.shape[-1])

            count_batch = data_flat.shape[0]
            if count_batch == 0:
                return

            mean_batch = data_flat.mean(dim=0)
            M2_batch = ((data_flat - mean_batch) ** 2).sum(dim=0)

            if not hasattr(self, "_n") or self._running_mean is None or self._running_M2 is None:
                self._n = count_batch
                self._running_mean = mean_batch
                self._running_M2 = M2_batch
            else:
                n_old = self._n
                n_new = n_old + count_batch

                delta = mean_batch - self._running_mean
                self._running_mean = self._running_mean + delta * (count_batch / n_new)

                self._running_M2 = (
                    self._running_M2 + M2_batch + (delta**2) * (n_old * count_batch / n_new)
                )
                self._n = n_new
        else:
            assert isinstance(data, Mapping), "Expected dict data for nested normalizer"
            for key, norm in self.norms.items():
                norm._update_running_stats(data[key])

    def _finalize_running_stats(self):
        if not hasattr(self, "norms"):
            if not hasattr(self, "_n") or self._n == 0 or self._running_mean is None:
                return

            if self._n == 1:
                self.mean.copy_(self._running_mean)
                self.std.fill_(1.0)
                self._is_fit.fill_(True)
            elif self._running_M2 is not None:
                mean = self._running_mean
                var = self._running_M2 / (self._n - 1)
                std = torch.sqrt(var)
                std = torch.where(std < 1e-6, torch.ones_like(std), std)

                self.mean.copy_(mean)
                self.std.copy_(std)
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
    def normalize(self, x: Mapping[str, TensorTree]) -> dict[str, TensorTree]: ...

    def normalize(self, x: TensorTree) -> TensorTree:
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
    def unnormalize(self, x: Mapping[str, TensorTree]) -> dict[str, TensorTree]: ...

    def unnormalize(self, x: TensorTree) -> TensorTree:
        """Unnormalizes the input tensor or mapping of tensors using the fitted mean and std."""
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
    def forward(self, x: Mapping[str, TensorTree]) -> dict[str, TensorTree]: ...

    def forward(self, x: TensorTree) -> TensorTree:
        """Forward pass for the normalizer, which normalizes the input tensor or mapping of
        tensors."""
        return self.normalize(x)
