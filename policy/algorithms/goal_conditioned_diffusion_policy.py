from collections.abc import Mapping
from typing import Any, Literal

import torch

from policy.algorithms.diffusion_policy import DiffusionPolicy
from policy.utils import map_leaves
from policy.utils.typing_utils import GoalConditionedPolicyProtocol, TensorTree


class GoalConditionedDiffusionPolicy(DiffusionPolicy, GoalConditionedPolicyProtocol):
    """Goal-conditioned diffusion policy using diffusers noise schedulers."""

    def __init__(self, *args, goal_horizon: int = 1, **kwargs):
        if goal_horizon < 1:
            raise ValueError(
                f"GoalConditionedDiffusionPolicy requires goal_horizon >= 1 (got {goal_horizon}). "
                "Use DiffusionPolicy for unconditioned diffusion."
            )
        super().__init__(*args, **kwargs)
        self.goal_horizon = goal_horizon
        self.goal_conditioned = True

    def _encoder_extra_kwargs(self) -> dict[str, Any]:
        return {**super()._encoder_extra_kwargs(), "goal_conditioned": self.goal_conditioned}

    def extract_embeddings(
        self,
        obs: torch.Tensor | Mapping[str, Any],
        goal: torch.Tensor | Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, torch.Tensor], Literal["absolute", "goal-relative"]]:
        """Helper for visualizing the embeddings."""
        if self.encoder is None:
            raise ValueError("Encoder not initialized. Call configure_model() first.")

        obs = self._normalize_obs(map_leaves(lambda t: t.to(self.device), obs))
        if goal is not None:
            goal = self._normalize_obs(map_leaves(lambda t: t.to(self.device), goal))

        with torch.no_grad():
            return self.encoder.extract_embeddings(obs, goal=goal)

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
        """Extracts and normalizes observation and goal conditioning from a training/validation
        batch."""
        obs_seq = batch["obs_seq"]
        goal = batch.get("goal", None)

        if not isinstance(obs_seq, torch.Tensor | Mapping):
            raise ValueError(
                f"Expected batch['obs_seq'] to be a torch.Tensor or Mapping, but got {type(obs_seq)}."
            )

        if goal is None or not isinstance(goal, torch.Tensor | Mapping):
            raise ValueError(
                f"Expected batch['goal'] to be a torch.Tensor or Mapping, but got {type(goal)}."
            )

        return self._build_external_cond(obs_seq, goal)

    def _build_external_cond(self, obs: TensorTree, goal: TensorTree) -> dict[str, TensorTree]:
        return {"obs": self._normalize_obs(obs), "goal": self._normalize_obs(goal)}
