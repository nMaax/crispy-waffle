import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """Embeds a window of per-timestep tokens with one (post-norm) transformer block: self-
    attention across the window, then a position-wise feed-forward network."""

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
        self.ln1 = nn.LayerNorm(output_dim)
        self.attn = nn.MultiheadAttention(output_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(output_dim)
        self.mlp = nn.Sequential(
            nn.Linear(output_dim, 4 * output_dim),
            nn.GELU(),
            nn.Linear(4 * output_dim, output_dim),
            nn.Dropout(dropout),
        )
        self.ln_f = nn.LayerNorm(output_dim)

        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Shapes:

        x: [B, T, input_dim] (T <= obs_horizon), or [B, T, K, input_dim] (K tokens per
            timestep).
            Grouping order for the attended sequence is t-major, k-minor: index = t * K + k.
        returns: same leading shape as ``x`` with ``input_dim`` -> ``output_dim``.
        """
        squeeze_k = x.ndim == 3
        if squeeze_k:
            x = x.unsqueeze(2)  # [B, T, 1, input_dim]
        elif x.ndim != 4:
            raise ValueError(f"Expected a 3D or 4D input, got shape {tuple(x.shape)}.")

        B, T, K, _ = x.shape
        if T > self.obs_horizon:
            raise ValueError(
                f"Got a window of length {T}, but this embedder was configured with "
                f"obs_horizon={self.obs_horizon}."
            )

        # All K tokens at a given timestep share that time positional embedding
        pos = self.pos_emb[:, :T, :].unsqueeze(2)  # [1, T, 1, output_dim]
        tokens = (self.input_proj(x) + pos).reshape(B, T * K, self.output_dim)

        normed = self.ln1(tokens)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        tokens = tokens + attn_out

        tokens = tokens + self.mlp(self.ln2(tokens))

        out = self.ln_f(tokens).reshape(B, T, K, self.output_dim)
        return out.squeeze(2) if squeeze_k else out
