import math
from collections.abc import Mapping

import torch
import torch.nn as nn
from torch.nn import functional as F

from policy.utils import concat_leaf_tensors, get_total_dim, split_leaf_key
from policy.utils.typing_utils import DimSpec, TensorTree
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
        cond_dims: DimSpec,
        embed_dim: int = 256,
        obs_horizon: int = 8,
        goal_horizon: int = 0,
        proprio_dim: int | None = None,
        use_proprio_token: bool = False,
        pred_horizon: int = 8,
        n_layers: int = 4,
        n_heads: int = 8,
        embed_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
    ):
        """
        Extra Args:
            proprio_dim: per-timestep proprioception width. Only meaningful -- and required --
                when ``use_proprio_token`` is set. ``None`` (default) means this instance doesn't
                reason about proprioception at all: proprio, if any, stays glued to the rest of
                the state features, undifferentiated.
            use_proprio_token: ``False`` (default) is equivalent to original BESO". ``True`` opts for
                a "robot-agnostic BESO" variant to better compare with GCDP: proprioception gets its
                own dedicated per-timestep token and related embedding, separate from the task-only
                ``obs_emb`` shared by obs-state and goal tokens. Context then becomes:
                [simga, goal_1, ..., goal_G, proprio_1, obs_1, act_1, proprio_2, obs_2, act_2, ...]
        """
        super().__init__()

        if obs_horizon != pred_horizon:
            raise ValueError(
                "Observation horizon and act horizon must be equal for DiffusionGPT. (For now)"
            )

        # Dimension and horizons
        self.obs_dim = get_total_dim(
            cond_dims["obs"] if isinstance(cond_dims, Mapping) else cond_dims
        )

        self.proprio_dim = proprio_dim
        self.use_proprio_token = use_proprio_token
        self.proprio_emb: nn.Linear | None = None
        if use_proprio_token:
            if proprio_dim is None:
                raise ValueError("use_proprio_token=True requires proprio_dim to be set.")
            if proprio_dim >= self.obs_dim:
                raise ValueError(
                    f"proprio_dim ({proprio_dim}) must be smaller than the per-timestep obs "
                    f"width ({self.obs_dim}) -- there must be a nonempty task remainder."
                )
            self.task_dim = self.obs_dim - proprio_dim
            self.proprio_emb = nn.Linear(proprio_dim, embed_dim)
        else:
            self.task_dim = self.obs_dim

        if isinstance(cond_dims, Mapping) and "goal" in cond_dims:
            goal_cond_dim = get_total_dim(cond_dims["goal"])
            if goal_cond_dim != self.task_dim:
                raise ValueError(
                    f"cond_dims['goal'] ({goal_cond_dim}) must match the per-timestep task width "
                    f"({self.task_dim}), since goal tokens share obs_emb with obs-task tokens."
                )

        self.act_dim = act_dim
        self.embed_dim = embed_dim

        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.goal_horizon = goal_horizon

        # NOTE: here obs_horizon === obs_seq_len in BESO code, see:
        # https://github.com/intuitive-robots/beso/blob/ef68824e533802ec0d7a5368ae21d013ce0df5c3/beso/agents/diffusion_agents/k_diffusion/score_gpts.py#L148
        # In our case, since obs_horizon and pred_horizon are sssumed equal, obs_horizon + pred_horizon = 2 * obs_horizon
        # With use_proprio_token=True, each obs timestep contributes with one extra dedicated token (3 * obs_seq_len)

        tokens_per_step = 3 if use_proprio_token else 2
        self.block_size = 1 + goal_horizon + tokens_per_step * obs_horizon

        # NOTE: Position embedding sequence length aligns with original BESO score_gpts.py:
        # seq_size = goal_horizon + obs_seq_len + 1.

        # NOTE: seq_len simply is the number of positional embedding we need (consecutive tokens p1, s1, a1 share the same position)
        self.seq_len = goal_horizon + obs_horizon + 1

        # Encoders
        self.obs_emb = nn.Linear(self.task_dim, embed_dim)
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
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        external_cond: Mapping[str, TensorTree],
    ) -> torch.Tensor:
        """
        Args:
            sample: [B, pred_horizon, act_dim] (Noisy actions)
            timestep: [B] (Continuous sigma values in BESO)
            external_cond: conditioning tensor tree with an ``"obs"`` key
                (``[B, obs_horizon * obs_dim]`` or ``[B, obs_horizon, obs_dim]``, possibly a
                nested mapping of components that gets merged on the feature axis) and an
                optional ``"goal"`` key (``[B, goal_horizon * obs_dim]`` or
                ``[B, goal_horizon, obs_dim]``). When ``self.use_proprio_token`` is set, ``"obs"``
                must carry a ``"proprio"`` key (Mapping form) and must be 3D (or a Mapping of 3D
                tensors) -- a pre-flattened 2D obs tensor's proprio/task boundary is ambiguous
                once multiple timesteps are packed into one flat vector. A ``"proprio"`` key in
                ``"goal"``, if present, is simply discarded (goal tokens are always task-only).
        """
        obs = external_cond["obs"]
        proprio = None
        if self.use_proprio_token:
            assert self.proprio_dim is not None
            proprio, obs = split_leaf_key(obs, "proprio", self.proprio_dim)
            if proprio is None:
                raise ValueError(
                    "use_proprio_token=True requires external_cond['obs'] to carry a 'proprio' key."
                )

        if isinstance(obs, Mapping):
            # e.g. with concat_leaf_tensors(dim=-1) a external_cond["obs"] tree like
            #       "obs": {
            #           "proprio": torch.Tensor[B, T, 18],
            #           "tcp": torch.Tensor[B, T, 8],
            #           "extras": torch.Tensor[B, T, 12]
            #       } will be flattened as :
            #   obs = torch.Tensor[B, T, (18 + 8 + 12)]
            obs = concat_leaf_tensors(obs, dim=-1)

        if not isinstance(obs, torch.Tensor):
            raise ValueError(
                f"Expected external_cond['obs'] to be a torch.Tensor or tensor-like tree structure, but got {type(obs)}"
            )

        goal = external_cond.get("goal", None)
        if self.use_proprio_token and isinstance(goal, Mapping):
            assert self.proprio_dim is not None
            _, goal = split_leaf_key(goal, "proprio", self.proprio_dim)
        if isinstance(goal, Mapping):
            # e.g. with concat_leaf_tensors(dim=-1) a external_cond["goal"] tree like
            #   torch.Tensor[B, T, 20] (degenerate tree with one tensor leaf only)
            #   simply becomes goal = torch.Tensor[B, T, 20]
            #   (if goal is a proper mapping with multiple leaves, it will behave just like obs above.)
            goal = concat_leaf_tensors(goal, dim=-1)

        if goal is not None and not isinstance(goal, torch.Tensor):
            raise ValueError(
                f"Expected external_cond['goal'] to be a torch.Tensor or None, but got {type(goal)}"
            )

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
            if self.use_proprio_token:
                raise ValueError(
                    "use_proprio_token=True requires external_cond['obs'] to be 3D (or a Mapping "
                    "of 3D tensors); a pre-flattened 2D obs tensor's proprio/task boundary is "
                    "ambiguous once multiple timesteps are packed into one flat vector."
                )
            cur_obs_horizon = obs.shape[1] // self.obs_dim
            obs_seq = obs.view(B, cur_obs_horizon, -1)

        if proprio is not None and proprio.ndim != 3:
            raise ValueError("external_cond['obs']['proprio'] must be 3D ([B, T, proprio_dim]).")

        obs_tokens = self.obs_emb(obs_seq)  # [B, cur_obs_horizon, embed_dim]
        act_tokens = self.act_emb(sample)  # [B, pred_horizon, embed_dim]

        cur_pred_horizon = sample.shape[1]
        if cur_obs_horizon != cur_pred_horizon:
            raise ValueError(
                f"Observation sequence length {cur_obs_horizon} and action sequence length {cur_pred_horizon} must be equal."
            )

        # Apply Positional Embeddings, pos_emb covers [1, goal_horizon + obs_horizon + 1, embed_dim]

        # NOTE: In the original BESO score_gpts.py, they did not add a positional embedding
        # to the sigma token (but still they reserved such parameter in the positional embedding vector).
        # They just concatenated sigma token raw in the context, and used pos_emb[:, 0:goal_len] for the goals.
        # We align with this choice by not adding positional embeddings to the sigma token as well.

        sigma_token = self.drop(sigma_token)

        if self.goal_horizon > 0:
            if goal is None:
                raise ValueError("goal must be provided for goal-conditioned DiffusionGPT")
            if goal.ndim == 2:
                goal_seq = goal.view(B, self.goal_horizon, -1)
            else:
                goal_seq = goal

            if goal_seq.shape[1] != self.goal_horizon:
                raise ValueError(
                    f"Expected goal sequence length {self.goal_horizon}, but got {goal_seq.shape[1]}"
                )

            if self.use_proprio_token and goal_seq.shape[-1] == self.obs_dim:
                # A flat/3D Tensor goal can't be tolerantly checked for a "proprio" key the way a
                # Mapping can; disambiguate by width instead -- if it's exactly obs-width (task +
                # proprio, e.g. real or zeroed proprio from an upstream goal-crafter), strip the
                # leading proprio slice. A goal that's already task-width is left untouched.
                goal_seq = goal_seq[..., self.proprio_dim :]

            if goal_seq.shape[-1] != self.task_dim:
                raise ValueError(
                    f"Expected goal width {self.task_dim} (task-only), but got {goal_seq.shape[-1]}"
                )

            goal_tokens = self.obs_emb(goal_seq)  # [B, goal_horizon, embed_dim]
            goal_tokens = self.drop(goal_tokens + self.pos_emb[:, : self.goal_horizon, :])
            pos_emb_sa = self.pos_emb[
                :, self.goal_horizon : cur_obs_horizon + self.goal_horizon, :
            ]
        else:
            goal_tokens = None
            pos_emb_sa = self.pos_emb[:, :cur_obs_horizon, :]

        obs_tokens = self.drop(obs_tokens + pos_emb_sa[:, :cur_obs_horizon, :])
        act_tokens = self.drop(act_tokens + pos_emb_sa[:, :cur_pred_horizon, :])

        # Interleave the Sequence
        # torch.stack creates [B, cur_obs_horizon, N, embed_dim] (N=2, or N=3 when use_proprio_token)
        # .view flattens it to [B, cur_obs_horizon * N, embed_dim], e.g. N=2: [s1, a1, s2, a2...],
        # N=3: [p1, s1, a1, p2, s2, a2...] -- action always sits last in each timestep's group.
        if proprio is not None:
            assert self.proprio_emb is not None
            proprio_tokens = self.proprio_emb(proprio)
            proprio_tokens = self.drop(proprio_tokens + pos_emb_sa[:, :cur_obs_horizon, :])
            interleaved = torch.stack([proprio_tokens, obs_tokens, act_tokens], dim=2)
        else:
            interleaved = torch.stack([obs_tokens, act_tokens], dim=2)
        interleave_width = interleaved.shape[2]
        sa_seq = interleaved.view(B, cur_obs_horizon * interleave_width, self.embed_dim)

        # Assemble Final Sequence
        if goal_tokens is not None:
            x = torch.cat([sigma_token, goal_tokens, sa_seq], dim=1)  # [B, block_size, embed_dim]
        else:
            x = torch.cat([sigma_token, sa_seq], dim=1)  # [B, block_size, embed_dim]

        # Pass through Transformer
        x = self.blocks(x)
        x = self.ln_f(x)

        # Extract Action Tokens
        # Because we interleaved [sigma, goal, (p1,) s1, a1, (p2,) s2, a2...], the actions are now
        # evenly spaced. First, strip off the sigma token and goal tokens:
        if self.goal_horizon > 0:
            x_sa = x[:, 1 + self.goal_horizon :, :]
        else:
            x_sa = x[:, 1:, :]

        # Reshape back to groups [B, cur_obs_horizon, interleave_width, embed_dim]
        x_sa = x_sa.view(B, cur_obs_horizon, interleave_width, self.embed_dim)
        # Actions always sit last within each timestep's group.
        act_outputs = x_sa[:, :, interleave_width - 1, :]

        # Decode back to action space
        predicted_actions = self.action_pred(act_outputs)

        return predicted_actions
