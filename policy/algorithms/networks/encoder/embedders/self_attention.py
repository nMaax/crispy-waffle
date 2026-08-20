import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """Embeds a window of per-timestep tokens with one pre-norm transformer block: self-attention
    across the window, then a position-wise feed-forward network.

    Norm placement and sublayer naming mirror :class:`DiffusionGPT`'s ``Block`` (``ln1``/``ln2``
    around each residual sublayer, ``ln_f`` on the way out). ``include_feedforward=False`` drops the
    feed-forward sublayer entirely, leaving attention alone -- the shape this embedder had before
    ``6dde21e``. ``pre_norm=False`` normalizes after each residual sublayer instead of before,
    which together with ``include_feedforward=False`` reproduces the embedder as of ``91cf38f``.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        obs_horizon: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        include_feedforward: bool = True,
        pre_norm: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.obs_horizon = obs_horizon
        self.pre_norm = pre_norm

        self.input_proj = nn.Linear(input_dim, output_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, obs_horizon, output_dim))
        self.ln1 = nn.LayerNorm(output_dim)
        self.attn = nn.MultiheadAttention(output_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2: nn.LayerNorm | None = None
        self.mlp: nn.Sequential | None = None
        if include_feedforward:
            self.ln2 = nn.LayerNorm(output_dim)
            self.mlp = nn.Sequential(
                nn.Linear(output_dim, 4 * output_dim),
                nn.GELU(),
                nn.Linear(4 * output_dim, output_dim),
                nn.Dropout(dropout),
            )
        # Post-norm normalizes at the end of each sublayer, so it needs no separate output norm.
        self.ln_f: nn.LayerNorm | None = nn.LayerNorm(output_dim) if pre_norm else None

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

        if self.pre_norm:
            normed = self.ln1(tokens)
            attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
            tokens = tokens + attn_out
        else:
            attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
            tokens = self.ln1(tokens + attn_out)

        if self.ln2 is not None and self.mlp is not None:
            if self.pre_norm:
                tokens = tokens + self.mlp(self.ln2(tokens))
            else:
                tokens = self.ln2(tokens + self.mlp(tokens))

        if self.ln_f is not None:
            tokens = self.ln_f(tokens)
        out = tokens.reshape(B, T, K, self.output_dim)
        return out.squeeze(2) if squeeze_k else out
