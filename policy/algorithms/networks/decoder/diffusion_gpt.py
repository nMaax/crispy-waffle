# Reference: https://github.com/intuitive-robots/beso/blob/main/beso/agents/diffusion_agents/k_diffusion/score_gpts.py

from collections.abc import Mapping

import torch
import torch.nn as nn

from policy.transforms.canonicalization.spec import dim_shape
from policy.utils import concat_leaf_tensors, get_subtree, get_tensor, get_total_dim
from policy.utils.typing_utils import DimSpec, TensorTree
from policy.utils.typing_utils.protocols import DiffusionNetworkProtocol


def init_gpt_weights(module: nn.Module) -> None:
    """Shared weight init for both GPT decoders; each root handles its own ``pos_emb``."""
    if isinstance(module, nn.Linear | nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        torch.nn.init.zeros_(module.bias)
        torch.nn.init.ones_(module.weight)
    elif isinstance(module, nn.MultiheadAttention):
        # out_proj is reached via the nn.Linear branch above; only the packed in-projection
        # is owned directly by this module (add_bias_kv=False, embed_dim==kdim==vdim here,
        # so bias_k/bias_v/q_proj_weight etc. stay None).
        torch.nn.init.normal_(module.in_proj_weight, mean=0.0, std=0.02)
        if module.in_proj_bias is not None:
            torch.nn.init.zeros_(module.in_proj_bias)


class CausalSelfAttention(nn.Module):
    """A multi-head masked self-attention layer, adapted from BESO.

    Thin wrapper over ``nn.MultiheadAttention``: the packed QKV projection, the softmax and the
    attention-weight dropout all live there; only the causal mask and the post-projection
    (residual) dropout stay here.
    """

    mask: torch.Tensor

    def __init__(
        self, n_embd: int, n_heads: int, attn_pdrop: float, resid_pdrop: float, block_size: int
    ):
        super().__init__()
        assert n_embd % n_heads == 0
        self.attn = nn.MultiheadAttention(
            embed_dim=n_embd, num_heads=n_heads, dropout=attn_pdrop, batch_first=True
        )
        self.resid_drop = nn.Dropout(resid_pdrop)

        # nn.MultiheadAttention's attn_mask polarity: True == "may not attend" (opposite of the
        # old 1=allowed/0=disallowed float buffer). 2D so it broadcasts across batch and heads.
        self.register_buffer(
            "mask", torch.ones(block_size, block_size, dtype=torch.bool).triu(diagonal=1)
        )

    def forward(self, x):
        T = x.size(1)
        y, _ = self.attn(x, x, x, attn_mask=self.mask[:T, :T], need_weights=False)
        return self.resid_drop(y)


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
        pred_horizon: int = 8,
        n_layers: int = 4,
        n_heads: int = 8,
        embed_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
    ):
        super().__init__()

        if obs_horizon != pred_horizon:
            raise ValueError(
                "Observation horizon and act horizon must be equal for DiffusionGPT. (For now)"
            )

        # Dimension and horizons
        self.obs_dim = get_total_dim(
            cond_dims["obs"] if isinstance(cond_dims, Mapping) else cond_dims
        )

        if isinstance(cond_dims, Mapping) and "goal" in cond_dims:
            goal_cond_dim = get_total_dim(cond_dims["goal"])
            if goal_cond_dim != self.obs_dim:
                raise ValueError(
                    f"cond_dims['goal'] ({goal_cond_dim}) must match the per-timestep obs width "
                    f"({self.obs_dim}), since goal tokens share obs_emb with obs tokens."
                )

        self.act_dim = act_dim
        self.embed_dim = embed_dim

        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.goal_horizon = goal_horizon

        # NOTE: here obs_horizon === obs_seq_len in BESO code, see:
        # https://github.com/intuitive-robots/beso/blob/ef68824e533802ec0d7a5368ae21d013ce0df5c3/beso/agents/diffusion_agents/k_diffusion/score_gpts.py#L148
        # In our case, since obs_horizon and pred_horizon are sssumed equal, obs_horizon + pred_horizon = 2 * obs_horizon
        self.block_size = 1 + goal_horizon + 2 * obs_horizon

        # NOTE: Position embedding sequence length aligns with original BESO score_gpts.py:
        # seq_size = goal_horizon + obs_seq_len + 1.

        # NOTE: seq_len simply is the number of positional embedding we need (consecutive tokens s1, a1 share the same position)
        self.seq_len = goal_horizon + obs_horizon + 1

        # Encoders
        self.obs_emb = nn.Linear(self.obs_dim, embed_dim)
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
        if module is self:
            torch.nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        else:
            init_gpt_weights(module)

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
                ``[B, goal_horizon, obs_dim]``, merged the same way). Goal tokens share
                ``obs_emb`` with observation tokens, so both sides carry the same width.
        """
        obs = external_cond["obs"]
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
            cur_obs_horizon = obs.shape[1] // self.obs_dim
            obs_seq = obs.view(B, cur_obs_horizon, -1)

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

            if goal_seq.shape[-1] != self.obs_dim:
                raise ValueError(
                    f"Expected goal width {self.obs_dim}, but got {goal_seq.shape[-1]}"
                )

            goal_tokens = self.obs_emb(goal_seq)  # [B, goal_horizon, embed_dim]
            # Even if there should be the sigma token position embedding before the goal tokens, we didn't add it before (following what done in original BESO --- see NOTE above)
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
        # torch.stack creates [B, cur_obs_horizon, 2, embed_dim], .view flattens it to
        # [B, cur_obs_horizon * 2, embed_dim], i.e. [s1, a1, s2, a2, ...].
        interleaved = torch.stack([obs_tokens, act_tokens], dim=2)
        sa_seq = interleaved.view(B, cur_obs_horizon * 2, self.embed_dim)

        # Assemble Final Sequence
        if goal_tokens is not None:
            x = torch.cat([sigma_token, goal_tokens, sa_seq], dim=1)  # [B, block_size, embed_dim]
        else:
            x = torch.cat([sigma_token, sa_seq], dim=1)  # [B, block_size, embed_dim]

        # Pass through Transformer
        x = self.blocks(x)
        x = self.ln_f(x)

        # Extract Action Tokens
        # Because we interleaved [sigma, goal, s1, a1, s2, a2...], the actions are now evenly
        # spaced. First, strip off the sigma token and goal tokens
        x_sa = x[:, 1 + self.goal_horizon :, :]

        # Reshape back to groups [B, cur_obs_horizon, 2, embed_dim]
        x_sa = x_sa.view(B, cur_obs_horizon, 2, self.embed_dim)
        act_outputs = x_sa[:, :, 1, :]

        # Decode back to action space
        predicted_actions = self.action_pred(act_outputs)

        return predicted_actions


class ObjectDiffusionGPT(nn.Module, DiffusionNetworkProtocol):
    """DiffusionGPT variant giving proprioception and every object token its own sequence slot.

    The context becomes ``[sigma, g_1, ..., g_{G*K}, p_1, o_1^1, ..., o_1^K, a_1, p_2, ...]``,
    where ``K`` is the tokenizer's ``tokens_per_step`` -- 1 for ``StateTokenizer`` (one flat
    task token per step, which reproduces the original per-timestep layout plus a proprioception
    token), one token per object for ``ObjectTokenizer``.
    """

    def __init__(
        self,
        act_dim: int,
        cond_dims: Mapping[str, DimSpec],
        embed_dim: int = 256,
        obs_horizon: int = 8,
        goal_horizon: int = 0,
        pred_horizon: int = 8,
        n_layers: int = 4,
        n_heads: int = 8,
        embed_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
    ):
        super().__init__()

        if obs_horizon != pred_horizon:
            raise ValueError(
                "Observation horizon and act horizon must be equal for ObjectDiffusionGPT. (For now)"
            )

        # Dimension and horizons
        obs_dims = cond_dims["obs"]
        if not isinstance(obs_dims, Mapping):
            raise TypeError(
                "ObjectDiffusionGPT requires a cond_dims['obs'] mapping with 'proprio' and 'task' "
                f"entries, got {type(obs_dims).__name__}."
            )

        self.proprio_dim = get_total_dim(obs_dims["proprio"])
        self.tokens_per_step, self.token_dim = self._token_shape(obs_dims["task"])

        if "goal" in cond_dims:
            goal_dims = cond_dims["goal"]
            if not isinstance(goal_dims, Mapping):
                raise TypeError(
                    "ObjectDiffusionGPT requires a cond_dims['goal'] mapping with a 'task' entry, "
                    f"got {type(goal_dims).__name__}."
                )
            if self._token_shape(goal_dims["task"]) != (self.tokens_per_step, self.token_dim):
                raise ValueError(
                    f"cond_dims['goal']['task'] {self._token_shape(goal_dims['task'])} must match "
                    f"the observation's ({self.tokens_per_step}, {self.token_dim}), since goal "
                    "tokens share obj_emb with observation tokens."
                )

        self.act_dim = act_dim
        self.embed_dim = embed_dim

        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.goal_horizon = goal_horizon

        # Each frame provides K object tokens + proprioception and action
        self.block_size = (
            1 + goal_horizon * self.tokens_per_step + (self.tokens_per_step + 2) * obs_horizon
        )

        # One position per frame, shared by every token of that frame
        self.seq_len = goal_horizon + obs_horizon + 1

        # Encoders
        self.obj_emb = nn.Linear(self.token_dim, embed_dim)
        self.proprio_emb = nn.Linear(self.proprio_dim, embed_dim)
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
        if module is self:
            torch.nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        else:
            init_gpt_weights(module)

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
            external_cond: ``{"obs": {"proprio": [B, T, proprio_dim], "task": task tokens}}``, plus
                a ``"goal"`` entry of the same shape when goal-conditioned. Task tokens are
                ``[B, T, K, token_dim]``, or ``[B, T, token_dim]`` when the tokenizer emits a
                single token per step. A ``"proprio"`` entry on the goal side is ignored: goal
                tokens are task-only, since proprioception has its own token stream.
        """
        obs = get_subtree(external_cond, "obs")
        proprio = get_tensor(obs, "proprio")
        task = self._as_token_sequence(get_tensor(obs, "task"))

        B, cur_obs_horizon = task.shape[0], task.shape[1]

        cur_pred_horizon = sample.shape[1]
        if cur_obs_horizon != cur_pred_horizon:
            raise ValueError(
                f"Observation sequence length {cur_obs_horizon} and action sequence length {cur_pred_horizon} must be equal."
            )

        # Embed Sigma; no positional embedding on it, as in the original BESO.
        sigma_log = timestep.view(B, 1, 1).log() / 4.0
        sigma_token = self.drop(self.sigma_emb(sigma_log.to(torch.float32)))

        # Assemble each frame as [proprio, obj_1, ..., obj_K, action], action always last so its
        # output token attends over everything else in the frame.
        frame = torch.cat(
            [
                self.proprio_emb(proprio).unsqueeze(2),
                self.obj_emb(task),
                self.act_emb(sample).unsqueeze(2),
            ],
            dim=2,
        )  # [B, T, K + 2, embed_dim]

        pos_emb_sa = self.pos_emb[
            :, self.goal_horizon : self.goal_horizon + cur_obs_horizon, :
        ].unsqueeze(2)
        frame = self.drop(frame + pos_emb_sa)

        interleave_width = self.tokens_per_step + 2
        sa_seq = frame.reshape(B, cur_obs_horizon * interleave_width, self.embed_dim)

        if self.goal_horizon > 0:
            x = torch.cat([sigma_token, self._goal_tokens(external_cond, B), sa_seq], dim=1)
        else:
            x = torch.cat([sigma_token, sa_seq], dim=1)

        # Pass through Transformer
        x = self.blocks(x)
        x = self.ln_f(x)

        # Strip the sigma and goal tokens, then read the trailing (action) token of every frame.
        x_sa = x[:, 1 + self.goal_horizon * self.tokens_per_step :, :]
        x_sa = x_sa.view(B, cur_obs_horizon, interleave_width, self.embed_dim)

        return self.action_pred(x_sa[:, :, -1, :])

    def _goal_tokens(self, external_cond: Mapping[str, TensorTree], batch_size: int):
        """Embeds the goal into ``[B, goal_horizon * K, embed_dim]``."""
        if "goal" not in external_cond:
            raise ValueError("goal must be provided for goal-conditioned ObjectDiffusionGPT")

        goal_task = self._as_token_sequence(get_tensor(get_subtree(external_cond, "goal"), "task"))
        if goal_task.shape[1] != self.goal_horizon:
            raise ValueError(
                f"Expected goal sequence length {self.goal_horizon}, but got {goal_task.shape[1]}"
            )

        tokens = self.obj_emb(goal_task) + self.pos_emb[:, : self.goal_horizon, :].unsqueeze(2)
        return self.drop(
            tokens.reshape(batch_size, self.goal_horizon * self.tokens_per_step, self.embed_dim)
        )

    def _as_token_sequence(self, task: torch.Tensor) -> torch.Tensor:
        """Restores the slot axis a single-token-per-step tokenizer does not emit."""
        return task.unsqueeze(-2) if self.tokens_per_step == 1 else task

    @staticmethod
    def _token_shape(spec: DimSpec) -> tuple[int, int]:
        """Reads ``(tokens_per_step, token_dim)`` off a task dim spec."""
        shape = dim_shape(spec)
        if len(shape) != 2:
            raise ValueError(
                "ObjectDiffusionGPT expects a (tokens_per_step, token_dim) task dim spec, got "
                f"{spec!r}."
            )
        return shape[0], shape[1]
