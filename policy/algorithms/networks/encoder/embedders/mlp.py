from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from policy.algorithms.networks.utils import resolve_activation


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        activation: type[nn.Module] | str = nn.ReLU,
        bias: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.bias = bias

        act_cls = resolve_activation(activation)

        layers: list[nn.Module] = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim, bias=bias))
            layers.append(act_cls())
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim, bias=bias))

        self.net = nn.Sequential(*layers)

        if not hidden_dims and not bias:
            nn.init.zeros_(self.net[0].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Shapes:

        x: [B, T, input_dim] or [B, T, K, input_dim] (K tokens per timestep).
        returns: same leading shape as ``x`` with ``input_dim`` -> ``output_dim``.
        """
        return self.net(x)
