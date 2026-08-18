from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from policy.transforms.canonicalization.spec import RELATIVE_SE3_DIM
from policy.utils import get_tensor
from policy.utils.typing_utils import TensorTree

# Edge kinds, in the order their learned embeddings are indexed.
EDGE_NONE, EDGE_SPATIAL, EDGE_TEMPORAL, EDGE_GOAL = range(4)
NUM_EDGE_KINDS = 4


class GraphTransformer(nn.Module):
    """Attends over a scene graph, with the topology applied as an attention mask.

    The node grid ``[B, T_all, K, D]`` is flattened t-major/k-minor (``index = t * K + k``) into
    one sequence, exactly as :class:`SelfAttention` does, and a mask decides which of those pairs
    are genuine edges:

    - **spatial**: every pair sharing a timestep, in both directions (self-loops included).
    - **temporal**: a node attending to *itself one step earlier*; the present may look at the
      past, never the reverse.
    - **goal**: only the most recent observed node attends to its own goal node -- the goal is a
      known future, so it is read by the present rather than updated by it.

    Goal nodes are therefore keys but never queries. Each edge carries its endpoints' SE(3) delta,
    which -- with a learned per-kind embedding added -- becomes a per-head additive attention bias,
    so one attention stack sees both the geometry and the relation type.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        obs_horizon: int,
        goal_horizon: int = 1,
        edge_dim: int = RELATIVE_SE3_DIM,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.obs_horizon = obs_horizon
        self.goal_horizon = goal_horizon
        self.num_heads = num_heads

        self.input_proj = nn.Linear(input_dim, output_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, obs_horizon + goal_horizon, output_dim))

        self.edge_kind_emb = nn.Embedding(NUM_EDGE_KINDS, edge_dim)
        self.edge_bias = nn.Sequential(
            nn.Linear(edge_dim, 2 * edge_dim),
            nn.GELU(),
            nn.Linear(2 * edge_dim, num_heads),
        )

        self.attn = nn.ModuleList(
            nn.MultiheadAttention(output_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        )
        self.attn_norm = nn.ModuleList(nn.LayerNorm(output_dim) for _ in range(num_layers))
        self.ffn = nn.ModuleList(
            nn.Sequential(
                nn.Linear(output_dim, 4 * output_dim),
                nn.GELU(),
                nn.Linear(4 * output_dim, output_dim),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        )
        self.ffn_norm = nn.ModuleList(nn.LayerNorm(output_dim) for _ in range(num_layers))

        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    def forward(self, task: Mapping[str, TensorTree]) -> torch.Tensor:
        """Shapes:

        task: ``{"nodes": [B, T_all, K, input_dim], "valid": [B, T_all, K],
            "edge_feat": [B, S, S, edge_dim]}`` with ``S = T_all * K``.
        returns: ``[B, T_all, K, output_dim]``.
        """
        nodes = get_tensor(task, "nodes")
        valid = get_tensor(task, "valid")
        edge_feat = get_tensor(task, "edge_feat")

        batch, time, num_slots, _ = nodes.shape
        if time != self.obs_horizon + self.goal_horizon:
            raise ValueError(
                f"Got {time} timesteps, but this embedder was configured with "
                f"obs_horizon={self.obs_horizon} + goal_horizon={self.goal_horizon}."
            )

        # All K nodes of a timestep share that time's positional embedding.
        pos = self.pos_emb[:, :time, :].unsqueeze(2)
        x = (self.input_proj(nodes) + pos).reshape(batch, time * num_slots, self.output_dim)

        attn_mask = self._attention_mask(edge_feat, valid, time, num_slots)

        for attn, attn_norm, ffn, ffn_norm in zip(
            self.attn, self.attn_norm, self.ffn, self.ffn_norm, strict=True
        ):
            attended, _ = attn(x, x, x, attn_mask=attn_mask, need_weights=False)
            x = attn_norm(x + attended)
            x = ffn_norm(x + ffn(x))

        return x.reshape(batch, time, num_slots, self.output_dim)

    def _attention_mask(
        self, edge_feat: torch.Tensor, valid: torch.Tensor, time: int, num_slots: int
    ) -> torch.Tensor:
        """The additive ``[B * num_heads, S, S]`` mask carrying both topology and geometry."""
        kinds = self._edge_kinds(time, num_slots, edge_feat.device)
        bias = self.edge_bias(edge_feat + self.edge_kind_emb(kinds))  # [B, S, S, num_heads]

        connected = kinds != EDGE_NONE
        # A key that is not a real object must never be attended to, whatever the topology says.
        allowed = connected.unsqueeze(0) & valid.flatten(1).bool().unsqueeze(1)
        # An all-masked row makes softmax emit NaN, and an invalid node's row is exactly that;
        # keeping its self-loop costs nothing since its output is discarded downstream.
        eye = torch.eye(allowed.shape[-1], dtype=torch.bool, device=allowed.device)
        allowed = allowed | eye.unsqueeze(0)

        bias = bias.permute(0, 3, 1, 2)  # [B, num_heads, S, S]
        bias = bias.masked_fill(~allowed.unsqueeze(1), float("-inf"))
        return bias.reshape(-1, *bias.shape[-2:])

    def _edge_kinds(self, time: int, num_slots: int, device: torch.device) -> torch.Tensor:
        """Static ``[S, S]`` relation type per node pair; depends only on the grid, not the
        data."""
        steps = torch.arange(time, device=device).repeat_interleave(num_slots)
        slots = torch.arange(num_slots, device=device).repeat(time)

        query_step, key_step = steps.unsqueeze(1), steps.unsqueeze(0)
        same_slot = slots.unsqueeze(1) == slots.unsqueeze(0)
        is_goal_step = key_step >= self.obs_horizon

        kinds = torch.full(
            (time * num_slots, time * num_slots), EDGE_NONE, dtype=torch.long, device=device
        )
        kinds[query_step == key_step] = EDGE_SPATIAL
        kinds[same_slot & (key_step == query_step - 1) & (query_step < self.obs_horizon)] = (
            EDGE_TEMPORAL
        )
        kinds[same_slot & is_goal_step & (query_step == self.obs_horizon - 1)] = EDGE_GOAL
        return kinds
