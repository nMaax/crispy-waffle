from collections.abc import Sequence

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        bias: bool = True,
    ):
        super().__init__()

        layers = []
        current_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim, bias=bias))
            layers.append(nn.ReLU())
            current_dim = h_dim

        layers.append(nn.Linear(current_dim, output_dim, bias=bias))

        self.net = nn.Sequential(*layers)

        if not hidden_dims and not bias:
            nn.init.zeros_(self.net[0].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
