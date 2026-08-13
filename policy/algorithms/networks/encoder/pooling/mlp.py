from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class MLPPooling(nn.Module):
    """Pools a token sequence into a fixed-size vector via flatten + MLP."""

    def __init__(
        self,
        dim: int,
        obs_horizon: int,
        output_dim: int | None = None,
        hidden_dims: Sequence[int] = (),
        tokens_per_step: int = 1,
        activation: type[nn.Module] = nn.Mish,
    ):
        super().__init__()
        out_dim = output_dim if output_dim is not None else dim
        in_dim = obs_horizon * tokens_per_step * dim

        layers: list[nn.Module] = [nn.Flatten(start_dim=1)]
        curr_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(activation())
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            x: [B, T, dim] (T == obs_horizon)
            returns: [B, dim]
        """
        return self.net(x)
