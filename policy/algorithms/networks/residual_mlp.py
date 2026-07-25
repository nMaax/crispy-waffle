from collections.abc import Sequence

import torch
import torch.nn as nn

from policy.algorithms.networks.mlp import MLP


class ZeroPadShortcut(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_dim < self.output_dim:
            pad_dim = self.output_dim - self.input_dim
            return torch.nn.functional.pad(x, (0, pad_dim))
        else:
            return x[..., : self.output_dim]


class ResidualMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int | None = None,
        hidden_dims: Sequence[int] = (256, 256),
        bias: bool = True,
        use_linear_shortcut: bool = True,
    ):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_linear_shortcut = use_linear_shortcut
        self.mlp = MLP(
            input_dim=input_dim, output_dim=output_dim, hidden_dims=hidden_dims, bias=bias
        )

        if input_dim == output_dim:
            self.shortcut: nn.Module = nn.Identity()
        elif use_linear_shortcut:
            self.shortcut = nn.Linear(input_dim, output_dim, bias=bias)
        else:
            self.shortcut = ZeroPadShortcut(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x) + self.shortcut(x)
