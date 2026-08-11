import torch

from policy.algorithms.goal_conditioned_diffusion_policy import GoalConditionedDiffusionPolicy
from policy.utils.typing_utils import DimSpec, StateTokenizer, TensorTree


class CrossAttentionGoalConditionedDiffusionPolicy(GoalConditionedDiffusionPolicy):
    """Goal-conditioned diffusion policy that cross-attends over a per-object token sequence
    instead of flattening it into a single per-timestep vector."""

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
        b, t, k, d = task_embedded.shape
        return task_embedded.reshape(b, t * k, d)

    def _package_task(self, proprio: torch.Tensor, task: torch.Tensor) -> dict[str, TensorTree]:
        return {"obs": {"proprio": proprio}, "context": task}

    def _package_task_dims(self, embed_dim: int, tokens_per_step: int) -> dict[str, DimSpec]:
        return {"obs": {"proprio": self.proprio_dim}, "context": embed_dim}
