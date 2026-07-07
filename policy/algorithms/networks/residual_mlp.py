from collections.abc import Sequence

import torch
import torch.nn as nn

from policy.algorithms.networks import MLP


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Sequence[int] = (256, 256)):
        super().__init__()
        self.mlp = MLP(input_dim=input_dim, output_dim=output_dim, hidden_dims=hidden_dims)

        if input_dim == output_dim:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x) + self.shortcut(x)
