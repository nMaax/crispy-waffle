from collections.abc import Iterable, Mapping
from typing import overload

import torch
import torch.nn as nn

from policy.utils.typing_utils import DimSpec, TensorTree


class MinMaxNormalizer(nn.Module):
    """Min-Max normalizer (scales to [min_val, max_val]) for a tensor or nested dict of tensors.

    Structure of the tree of sub-normalizers is fixed at construction time via `spec`.
    """

    _running_min: torch.Tensor | None
    _running_max: torch.Tensor | None

    def __init__(
        self, spec: DimSpec, min_val: float = -1.0, max_val: float = 1.0
    ):
        super().__init__()

        self.min_val = min_val
        self.max_val = max_val

        if not isinstance(spec, Mapping):
            if isinstance(spec, torch.Tensor):
                dim = spec.shape[-1]
            else:
                dim = spec

            self.register_buffer("min", torch.zeros(dim))
            self.register_buffer("max", torch.ones(dim))
            self.register_buffer("_is_fit", torch.tensor(False))
        else:
            self.norms = nn.ModuleDict(
                {
                    key: MinMaxNormalizer(child_spec, min_val, max_val)
                    for key, child_spec in spec.items()
                }
            )

    @property
    def is_fit(self) -> torch.Tensor:
        if hasattr(self, "norms"):
            all_fit = all(bool(norm.is_fit.item()) for norm in self.norms.values())
            return torch.tensor(all_fit)
        else:
            return self._is_fit

    def fit(self, data: TensorTree):
        """Fits the normalizer to the data, computing min and max."""
        if not isinstance(data, Mapping):
            data_flat = data.reshape(-1, data.shape[-1])
            self.min.copy_(data_flat.min(dim=0).values)
            self.max.copy_(data_flat.max(dim=0).values)
            self._is_fit.fill_(True)
        else:
            for key, norm in self.norms.items():
                norm.fit(data[key])

    def fit_incremental(self, data_iterator: Iterable[TensorTree]) -> None:
        """Fits the normalizer incrementally using an iterator of data."""
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

    def _update_running_stats(self, data: TensorTree) -> None:
        if not hasattr(self, "norms"):
            if not isinstance(data, torch.Tensor):
                data = torch.as_tensor(data)
            x_flat = data.reshape(-1, data.shape[-1])
            if x_flat.shape[0] == 0:
                return

            min_B = x_flat.min(dim=0).values
            max_B = x_flat.max(dim=0).values

            if self._running_min is None or self._running_max is None:
                self._running_min = min_B
                self._running_max = max_B
            else:
                self._running_min = torch.minimum(self._running_min, min_B)
                self._running_max = torch.maximum(self._running_max, max_B)
        else:
            assert isinstance(data, Mapping), "Expected dict data for nested normalizer"
            for key, norm in self.norms.items():
                norm._update_running_stats(data[key])

    def _finalize_running_stats(self) -> None:
        if not hasattr(self, "norms"):
            if self._running_min is not None and self._running_max is not None:
                self.min.copy_(self._running_min.to(self.min.device))
                self.max.copy_(self._running_max.to(self.max.device))
                self._is_fit.fill_(True)

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
    def normalize(self, x: Mapping[str, TensorTree]) -> dict[str, TensorTree]: ...

    def normalize(self, x: TensorTree) -> TensorTree:
        """Normalizes the input tensor or mapping of tensors using the fitted min and max."""
        if not isinstance(x, Mapping):
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
    def unnormalize(self, x: Mapping[str, TensorTree]) -> dict[str, TensorTree]: ...

    def unnormalize(self, x: TensorTree) -> TensorTree:
        """Unnormalizes the input tensor or dict of tensors using the fitted min and max."""
        if not isinstance(x, Mapping):
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
    def forward(self, x: Mapping[str, TensorTree]) -> dict[str, TensorTree]: ...

    def forward(self, x: TensorTree) -> TensorTree:
        """Forward pass for the normalizer, which normalizes the input tensor or dict of
        tensors."""
        return self.normalize(x)
