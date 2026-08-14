from __future__ import annotations

from collections.abc import Sequence

import hydra.utils
import torch
import torch.nn as nn


def _resolve_activation(activation: type[nn.Module] | str) -> type[nn.Module]:
    if isinstance(activation, str):
        if hasattr(nn, activation):
            return getattr(nn, activation)
        return hydra.utils.get_class(activation)
    return activation


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

        act_cls = _resolve_activation(activation)

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
