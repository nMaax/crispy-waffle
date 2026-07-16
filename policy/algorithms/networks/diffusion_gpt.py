import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from policy.utils.typing_utils.protocols import DiffusionNetworkProtocol


class CausalSelfAttention(nn.Module):
    """A vanilla multi-head masked self-attention layer, adapted from BESO."""

    def __init__(
        self, n_embd: int, n_heads: int, attn_pdrop: float, resid_pdrop: float, block_size: int
    ):
        super().__init__()
        assert n_embd % n_heads == 0
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)

        # Causal mask to ensure attention is only applied to the left in the sequence
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )
        self.n_head = n_heads

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """Transformer block, adapted from BESO."""

    def __init__(
        self, n_embd: int, n_heads: int, attn_pdrop: float, resid_pdrop: float, block_size: int
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_heads, attn_pdrop, resid_pdrop, block_size)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DiffusionGPT(nn.Module, DiffusionNetworkProtocol):
    """GPT architecture adapted from BESO for Action Sequence generation."""

    def __init__(
        self,
        act_dim: int,
        external_cond_dim: int,
        embed_dim: int = 256,
        external_cond_horizon: int = 8,
        pred_horizon: int = 8,
        n_layers: int = 4,
        n_heads: int = 8,
        embed_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        goal_seq_len: int = 0,
    ):
        super().__init__()

        if external_cond_horizon != pred_horizon:
            raise ValueError(
                "Observation horizon and act horizon must be equal for DiffusionGPT. (For now)"
            )

        # Dimension and horizons
        self.obs_dim = external_cond_dim
        self.act_dim = act_dim
        self.embed_dim = embed_dim

        self.obs_horizon = external_cond_horizon
        self.pred_horizon = pred_horizon
        self.goal_seq_len = goal_seq_len

        # NOTE: Causal mask 'self.mask' is registered as a static buffer.
        # To support dynamic/arbitrary sequence lengths at runtime, this could be generalized
        # to generate the mask dynamically at forward time based on the current sequence length.
        # Maximum sequence length: 1 (sigma) + goal_seq_len + obs_horizon + pred_horizon
        self.block_size = 1 + goal_seq_len + external_cond_horizon + pred_horizon
        self.seq_len = 1 + goal_seq_len + external_cond_horizon

        # Encoders
        self.obs_emb = nn.Linear(external_cond_dim, embed_dim)
        self.act_emb = nn.Linear(act_dim, embed_dim)
        self.sigma_emb = nn.Linear(1, embed_dim)

        # Positional Embedding
        self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_len, embed_dim))
        self.drop = nn.Dropout(embed_pdrop)

        # Transformer Blocks
        self.blocks = nn.Sequential(
            *[
                Block(embed_dim, n_heads, attn_pdrop, resid_pdrop, self.block_size)
                for _ in range(n_layers)
            ]
        )

        # Decoder Head
        self.ln_f = nn.LayerNorm(embed_dim)
        self.action_pred = nn.Sequential(
            nn.Linear(embed_dim, 100), nn.SiLU(), nn.Linear(100, act_dim)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear | nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, DiffusionGPT):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)

    def forward(
        self, sample: torch.Tensor, timestep: torch.Tensor, obs: torch.Tensor, goal: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            sample: [B, pred_horizon, act_dim] (Noisy actions)
            timestep: [B] (Continuous sigma values in BESO)
            obs: [B, obs_horizon * obs_dim] (Flattened context)
            goal: [B, goal_seq_len * obs_dim] or [B, goal_seq_len, obs_dim] (Optional goal context)
        """
        B = sample.size(0)

        # Embed Sigma
        sigma = timestep.view(B, 1, 1)
        sigma_log = sigma.log() / 4.0
        sigma_token = self.sigma_emb(sigma_log.to(torch.float32))  # [B, 1, embed_dim]

        # Embed Observations and Actions
        if obs.ndim == 3:
            obs_seq = obs
            cur_obs_horizon = obs.shape[1]
        else:
            cur_obs_horizon = obs.shape[1] // self.obs_dim
            obs_seq = obs.view(B, cur_obs_horizon, -1)
        obs_tokens = self.obs_emb(obs_seq)  # [B, cur_obs_horizon, embed_dim]
        act_tokens = self.act_emb(sample)  # [B, pred_horizon, embed_dim]

        cur_pred_horizon = sample.shape[1]
        if cur_obs_horizon != cur_pred_horizon:
            raise ValueError(
                f"Observation sequence length {cur_obs_horizon} and action sequence length {cur_pred_horizon} must be equal."
            )

        # Apply Positional Embeddings
        # pos_emb covers [1, 1 (sigma) + goal_seq_len + obs_horizon, embed_dim]
        sigma_token = self.drop(sigma_token + self.pos_emb[:, 0:1, :])

        if self.goal_seq_len > 0:
            if goal is None:
                raise ValueError("goal must be provided for goal-conditioned DiffusionGPT")
            if goal.ndim == 2:
                goal_seq = goal.view(B, self.goal_seq_len, -1)
            else:
                goal_seq = goal

            if goal_seq.shape[1] != self.goal_seq_len:
                raise ValueError(
                    f"Expected goal sequence length {self.goal_seq_len}, but got {goal_seq.shape[1]}"
                )

            goal_tokens = self.obs_emb(goal_seq)  # [B, goal_seq_len, embed_dim]
            goal_tokens = self.drop(goal_tokens + self.pos_emb[:, 1 : 1 + self.goal_seq_len, :])
            pos_emb_sa = self.pos_emb[:, 1 + self.goal_seq_len :, :]
        else:
            goal_tokens = None
            pos_emb_sa = self.pos_emb[:, 1 :, :]

        obs_tokens = self.drop(obs_tokens + pos_emb_sa[:, :cur_obs_horizon, :])
        act_tokens = self.drop(act_tokens + pos_emb_sa[:, :cur_pred_horizon, :])

        # Interleave the Sequence
        # torch.stack creates [B, cur_obs_horizon, 2, embed_dim]
        # .view flattens it to [B, cur_obs_horizon * 2, embed_dim] resulting in [s1, a1, s2, a2...]
        sa_seq = torch.stack([obs_tokens, act_tokens], dim=2)
        sa_seq = sa_seq.view(B, cur_obs_horizon * 2, self.embed_dim)

        # Assemble Final Sequence
        if goal_tokens is not None:
            x = torch.cat([sigma_token, goal_tokens, sa_seq], dim=1)  # [B, block_size, embed_dim]
        else:
            x = torch.cat([sigma_token, sa_seq], dim=1)  # [B, block_size, embed_dim]

        # Pass through Transformer
        x = self.blocks(x)
        x = self.ln_f(x)

        # Extract Action Tokens
        # Because we interleaved [sigma, goal, s1, a1, s2, a2...], the actions are now evenly spaced.
        # First, strip off the sigma token and goal tokens:
        if self.goal_seq_len > 0:
            x_sa = x[:, 1 + self.goal_seq_len :, :]
        else:
            x_sa = x[:, 1 :, :]

        # Reshape back to pairs [B, cur_obs_horizon, 2, embed_dim]
        x_sa = x_sa.view(B, cur_obs_horizon, 2, self.embed_dim)
        # Extract index 1 from the pairs (which corresponds to the actions)
        act_outputs = x_sa[:, :, 1, :]

        # Decode back to action space
        predicted_actions = self.action_pred(act_outputs)

        return predicted_actions
