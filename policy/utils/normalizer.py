import torch
import torch.nn as nn


class TensorNormalizer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        self.register_buffer("is_fit", torch.tensor(False))

    def fit(self, data: torch.Tensor):

        # A shape of (Batch, Horizon, 48) becomes (Batch * Horizon, 48).
        data_flat = data.reshape(-1, data.shape[-1])

        self.register_buffer("mean", data_flat.mean(dim=0))
        self.register_buffer("std", data_flat.std(dim=0).clamp(min=1e-6))
        self.is_fit.fill_(True)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_fit:
            return x
        return (x - self.mean) / self.std

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_fit:
            return x
        return (x * self.std) + self.mean
