import torch
import torch.nn as nn


class ZScoreNormalizer(nn.Module):
    """Z-score normalizer for tensors (mean=0, std=1).

    Registers statistics as PyTorch buffers so they are automatically saved/loaded inside
    state_dict.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        self.register_buffer("is_fit", torch.tensor(False))

    def fit(self, data: torch.Tensor):
        # Flatten batches and sequence horizons (if any) to calculate stats over feature dimensions
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalize(x)
