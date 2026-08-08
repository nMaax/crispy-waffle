from collections.abc import Sequence

import torch
import torch.nn as nn

from policy.algorithms.networks.mlp import MLP
from policy.algorithms.networks.pooling import pool_tokens


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
        pooling: nn.Module | None = None,
    ):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_linear_shortcut = use_linear_shortcut
        self.pooling = pooling
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
        """
        Shapes:
            x: [B, T, input_dim] or [B, T, K, input_dim] (K tokens per timestep).
            returns: same leading shape as ``x`` with ``input_dim`` -> ``output_dim``, or
                [B, output_dim] if ``pooling`` is set.
        """
        out = self.mlp(x) + self.shortcut(x)
        return pool_tokens(out, self.pooling) if self.pooling is not None else out
