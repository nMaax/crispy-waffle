import torch
import torch.nn as nn


class TensorNormalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.zeros(1))
        self.register_buffer("std", torch.ones(1))
        self.register_buffer("is_fit", torch.tensor(False))

    def fit(self, data: torch.Tensor):
        self.mean = data.mean(dim=0, keepdim=True)
        self.std = data.std(dim=0, keepdim=True).clamp(min=1e-6)
        self.is_fit.fill_(True)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_fit:
            return x
        return (x - self.mean) / self.std

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_fit:
            return x
        return (x * self.std) + self.mean
