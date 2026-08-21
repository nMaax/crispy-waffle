from collections.abc import Mapping
from typing import Any, Literal

import torch

from policy.algorithms.diffusion_policy import DiffusionPolicy
from policy.utils import to_device
from policy.utils.typing_utils import GoalConditionedPolicyProtocol, TensorTree


class GoalConditionedDiffusionPolicy(DiffusionPolicy, GoalConditionedPolicyProtocol):
    """Goal-conditioned diffusion policy using diffusers noise schedulers."""

    def __init__(self, *args, goal_horizon: int = 1, **kwargs):
        if goal_horizon < 1:
            raise ValueError(
                f"GoalConditionedDiffusionPolicy requires goal_horizon >= 1 (got {goal_horizon}). "
                "Use DiffusionPolicy for unconditioned diffusion."
            )
        super().__init__(*args, goal_horizon=goal_horizon, **kwargs)

    def extract_embeddings(
        self,
        obs: torch.Tensor | Mapping[str, Any],
        goal: torch.Tensor | Mapping[str, Any],
    ) -> tuple[dict[str, torch.Tensor], Literal["absolute", "goal-relative"]]:
        """Helper for visualizing the embeddings."""
        if self.encoder is None:
            raise ValueError("Encoder not initialized. Call configure_model() first.")

        with torch.no_grad():
            payload = self._encode(
                {"obs": to_device(obs, self.device), "goal": to_device(goal, self.device)}
            )
            embeddings = {"obs_embeddings": self.encoder.unpack_task(payload)}
            if "goal" in payload:
                goal_embedding = payload["goal"]
                assert isinstance(goal_embedding, torch.Tensor), (
                    f"Expected the embedded goal to be a Tensor, got {type(goal_embedding)}."
                )
                embeddings["goal_embedding"] = goal_embedding

        return embeddings, "goal-relative" if self.relative_goal else "absolute"

    def get_action(
        self,
        obs_seq: torch.Tensor | Mapping[str, Any],
        goal: torch.Tensor | Mapping[str, Any],
        num_inference_steps: int | None = None,
        output_clip_range: tuple | None = None,
    ) -> torch.Tensor:
        """Runs the reverse diffusion process to predict an action sequence from the observation
        and goal.

        Shapes:
            obs_seq: [B, obs_horizon * obs_dim] or dict
            goal: [B, obs_dim] or dict
            returns: [B, act_horizon, act_dim] (denoised actions to execute)
        """
        external_cond = self._build_external_cond(obs_seq, goal)

        return self._run_diffusion_loop(
            external_cond=external_cond,
            num_inference_steps=num_inference_steps,
            output_clip_range=output_clip_range,
        )

    def _build_external_cond_from_batch(self, batch: dict[str, Any]) -> dict[str, TensorTree]:
        """Extracts and normalizes observation and goal from a batch."""
        obs_seq = batch["obs_seq"]
        goal = batch.get("goal", None)

        if goal is None or not isinstance(goal, torch.Tensor | Mapping):
            raise ValueError(
                f"Expected batch['goal'] to be a torch.Tensor or Mapping, but got {type(goal)}."
            )

        return self._build_external_cond(obs_seq, goal)

    def _build_external_cond(self, obs: TensorTree, goal: TensorTree) -> dict[str, TensorTree]:
        return {"obs": obs, "goal": goal}
