from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from policy.algorithms.networks.encoder.pooling.base import BasePooling, PoolingMode
from policy.algorithms.networks.utils import resolve_activation


class MLPPooling(BasePooling):
    """Pools a token sequence into a fixed-size vector via flatten + MLP."""

    def __init__(
        self,
        dim: int,
        obs_horizon: int = 1,
        output_dim: int | None = None,
        hidden_dims: Sequence[int] = (),
        tokens_per_step: int = 1,
        activation: type[nn.Module] | str = nn.Mish,
        mode: PoolingMode = "all",
    ):
        super().__init__(mode=mode)
        if self.mode in ("all", "objects") and (tokens_per_step is None or tokens_per_step <= 0):
            raise ValueError(
                f"MLPPooling with mode={self.mode!r} requires a positive integer for tokens_per_step, "
                f"got {tokens_per_step}."
            )
        out_dim = output_dim if output_dim is not None else dim
        seq_len = {
            "all": obs_horizon * tokens_per_step,
            "objects": tokens_per_step,
            "time": obs_horizon,
        }[self.mode]
        in_dim = seq_len * dim

        act_cls = resolve_activation(activation)

        layers: list[nn.Module] = [nn.Flatten(start_dim=1)]
        curr_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(act_cls())
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, out_dim))

        self.net = nn.Sequential(*layers)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            x: [B, L, dim]
            returns: [B, out_dim]
        """
        return self.net(x)
