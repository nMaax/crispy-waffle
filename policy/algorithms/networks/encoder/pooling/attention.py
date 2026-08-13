from __future__ import annotations

import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    """Pools a token sequence into a fixed-size vector via a learned query attending over it."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        nn.init.normal_(self.query, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            x: [B, T, dim]
            returns: [B, dim]
        """
        query = self.query.expand(x.shape[0], -1, -1)
        pooled, _ = self.attn(query, x, x, need_weights=False)
        return pooled.squeeze(1)
