import torch

from policy.algorithms.goal_conditioned_diffusion_policy import GoalConditionedDiffusionPolicy
from policy.utils.typing_utils import DimSpec, StateTokenizer, TensorTree


class CrossAttentionGoalConditionedDiffusionPolicy(GoalConditionedDiffusionPolicy):
    """Goal-conditioned diffusion policy whose task conditioning is cross-attended over by the
    network (e.g.
    :class:`~policy.algorithms.networks.unet1d_cross_attention.CrossAttentionConditionalUnet1D`),
    Stable-Diffusion-style, instead of being flattened into one global FiLM vector.

    Requires a tokenizer with ``tokens_per_step > 1`` (e.g. ``PerObjectStateTokenizer``) and
    ``goal_delta="input"``, so that the per-object relative-displacement tokens
    (``r_k,t = g_k - o_k,t``) form a genuine token sequence to attend over. The embedder must not
    pool over time (``pooling=None``), or there would be nothing left for the network to attend
    to.

    This class overrides exactly the seams :class:`GoalConditionedDiffusionPolicy` exposes for
    this purpose: :meth:`_validate_tokenizer` (the extra compatibility check above),
    :meth:`_fold_multi_token_embedding` (kept as a sequence axis instead of folded into the
    feature axis), :meth:`_package_task`, and :meth:`_package_task_dims` (both routed to a
    top-level ``"context"`` entry instead of ``"obs"]["task"]``/``"task"``). None of these
    re-resolve the tokenizer or re-implement ``configure_model()``'s guard -- they take the
    already-resolved tokenizer/values as arguments, so ``_resolve_tokenizer()``'s single-instance
    caching stays intact. Everything else -- ``configure_model``, ``_shared_step``, ``get_action``,
    ``_build_external_cond``, ``_get_cond_dims``, ``_embed_states``, ``_tokenize_delta``, etc. --
    is inherited unchanged.
    """

    def _validate_tokenizer(self, tokenizer: StateTokenizer) -> None:
        if self.goal_delta != "input" or tokenizer.tokens_per_step <= 1:
            raise ValueError(
                f"{type(self).__name__} requires goal_delta='input' and a tokenizer with "
                f"tokens_per_step > 1 (e.g. PerObjectStateTokenizer); got "
                f"goal_delta={self.goal_delta!r}, tokens_per_step={tokenizer.tokens_per_step} "
                f"({type(tokenizer).__name__}). Cross-attention needs a genuine per-object token "
                "sequence to attend over."
            )

    def _fold_multi_token_embedding(self, task_embedded: torch.Tensor) -> torch.Tensor:
        """Keeps the K per-object embeddings as separate sequence entries (``[B, T*K, D]``) instead
        of folding them into a wider per-timestep vector -- the network cross-attends over this
        sequence rather than having it flattened for FiLM."""
        b, t, k, d = task_embedded.shape
        return task_embedded.reshape(b, t * k, d)

    def _package_task(self, proprio: torch.Tensor, task: torch.Tensor) -> dict[str, TensorTree]:
        return {"obs": {"proprio": proprio}, "context": task}

    def _package_task_dims(self, embed_dim: int, tokens_per_step: int) -> dict[str, DimSpec]:
        return {"obs": {"proprio": self.proprio_dim}, "context": embed_dim}
