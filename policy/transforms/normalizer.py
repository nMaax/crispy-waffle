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

    def __init__(self, spec: int | torch.Tensor | dict):
        super().__init__()

        if not isinstance(spec, dict):
            if isinstance(spec, int):
                dim = spec
            elif isinstance(spec, torch.Tensor):
                dim = spec.shape[-1] if isinstance(spec, torch.Tensor) else spec

            self.register_buffer("mean", torch.zeros(dim))
            self.register_buffer("std", torch.ones(dim))
            self.register_buffer("is_fit", torch.tensor(False))
        else:
            self.norms = nn.ModuleDict(
                {key: ZScoreNormalizer(child_spec) for key, child_spec in spec.items()}
            )

    def fit(self, data: torch.Tensor | dict):
        if not isinstance(data, dict):
            data_flat = data.reshape(-1, data.shape[-1])
            self.mean.copy_(data_flat.mean(dim=0))
            self.std.copy_(data_flat.std(dim=0).clamp(min=1e-6))
            self.is_fit.fill_(True)
        else:
            for key, norm in self.norms.items():
                norm.fit(data[key])

    def normalize(self, x: torch.Tensor | dict) -> torch.Tensor | dict:
        if not isinstance(x, dict):
            if self.is_fit:
                return (x - self.mean) / self.std
            else:
                return x
        else:
            return {key: norm.normalize(x[key]) for key, norm in self.norms.items()}

    def unnormalize(self, x: torch.Tensor | dict) -> torch.Tensor | dict:
        if not isinstance(x, dict):
            if self.is_fit:
                return (x * self.std) + self.mean
            else:
                return x
        else:
            return {key: norm.unnormalize(x[key]) for key, norm in self.norms.items()}

    def forward(self, x: torch.Tensor | dict) -> torch.Tensor | dict:
        return self.normalize(x)
