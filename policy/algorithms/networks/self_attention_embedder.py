import torch
import torch.nn as nn


class SelfAttentionEmbedder(nn.Module):
    """Embeds a window of per-timestep tokens by self-attending across them."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        obs_horizon: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.obs_horizon = obs_horizon

        self.input_proj = nn.Linear(input_dim, output_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, obs_horizon, output_dim))
        self.attn = nn.MultiheadAttention(output_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(output_dim)

        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            x: [B, T, input_dim] (T <= obs_horizon)
            returns: [B, T, output_dim]
        """
        T = x.shape[1]
        if T > self.obs_horizon:
            raise ValueError(
                f"Got a window of length {T}, but this embedder was configured with "
                f"obs_horizon={self.obs_horizon}."
            )

        tokens = self.input_proj(x) + self.pos_emb[:, :T, :]
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        return self.norm(tokens + attn_out)
