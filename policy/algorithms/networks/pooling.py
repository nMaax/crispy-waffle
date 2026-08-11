import torch
import torch.nn as nn


def pool_tokens(x: torch.Tensor, pooling: nn.Module) -> torch.Tensor:
    """Flattens an optional per-timestep token axis into the sequence axis pooling expects, then
    pools.

    Shapes:
        x: [B, T, dim] or [B, T, K, dim] (K tokens per timestep).
        returns: [B, dim].
    """
    if x.ndim == 4:
        b, t, k, dim = x.shape
        x = x.reshape(b, t * k, dim)
    return pooling(x)


class MLPPooling(nn.Module):
    """Pools a token sequence into a fixed-size vector via flatten + MLP."""

    def __init__(
        self,
        dim: int,
        obs_horizon: int,
        hidden_dim: int | None = None,
        tokens_per_step: int = 1,
    ):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(obs_horizon * tokens_per_step * dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            x: [B, T, dim] (T == obs_horizon)
            returns: [B, dim]
        """
        return self.net(x)


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
