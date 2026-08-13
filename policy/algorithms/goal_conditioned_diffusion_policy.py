from collections.abc import Mapping
from typing import Any

import torch

from policy.algorithms.diffusion_policy import DiffusionPolicy
from policy.utils.typing_utils import GoalConditionedPolicyProtocol, TensorTree


class GoalConditionedDiffusionPolicy(DiffusionPolicy, GoalConditionedPolicyProtocol):
    """Goal-conditioned diffusion policy using diffusers noise schedulers."""

    def __init__(self, *args, goal_horizon: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.goal_horizon = goal_horizon
        self.goal_conditioned = goal_horizon > 0

    def _encoder_extra_kwargs(self) -> dict[str, Any]:
        return {"obs_dim": self.obs_dim, "goal_conditioned": self.goal_conditioned}

    def extract_embeddings(
        self,
        obs: torch.Tensor | Mapping[str, Any],
        goal: torch.Tensor | Mapping[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Helper for visualizing the embeddings.

        Extracts embedder outputs for observations (and optionally a goal).
        """
        if self.encoder is None:
            raise ValueError("Encoder not initialized. Call configure_model() first.")

        obs = (
            {k: v.to(self.device) for k, v in obs.items()}
            if isinstance(obs, Mapping)
            else obs.to(self.device)
        )
        if goal is not None:
            goal = (
                {k: v.to(self.device) for k, v in goal.items()}
                if isinstance(goal, Mapping)
                else goal.to(self.device)
            )

        if self.obs_normalizer is not None:
            obs = self.obs_normalizer.normalize(obs)
            if goal is not None:
                goal = self.obs_normalizer.normalize(goal)

        with torch.no_grad():
            return self.encoder.extract_embeddings(obs, goal=goal)

    def get_action(
        self,
        obs_seq: torch.Tensor | Mapping[str, Any],
        goal: torch.Tensor | Mapping[str, Any] | None = None,
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
        if self.obs_normalizer is not None:
            obs_seq = self.obs_normalizer.normalize(obs_seq)
            if goal is not None:
                goal = self.obs_normalizer.normalize(goal)

        external_cond = self._build_external_cond(obs_seq, goal)

        return self._run_diffusion_loop(
            external_cond=external_cond,
            num_inference_steps=num_inference_steps,
            output_clip_range=output_clip_range,
        )

    def _shared_step(self, batch: dict[str, Any], batch_idx: int, phase: str) -> torch.Tensor:
        """Main step logic for training and validation."""
        obs_seq = batch["obs_seq"]
        action_seq = batch["act_seq"]
        goal = batch.get("goal", None)

        if not isinstance(obs_seq, torch.Tensor | Mapping):
            raise ValueError(
                f"Expected batch['obs_seq'] to be a torch.Tensor or Mapping, but got {type(obs_seq)}."
            )

        if goal is not None and not isinstance(goal, torch.Tensor | Mapping):
            raise ValueError(
                f"Expected batch['goal'] to be a torch.Tensor or Mapping, but got {type(goal)}."
            )

        if self.obs_normalizer is not None:
            obs_seq = self.obs_normalizer.normalize(obs_seq)
            if goal is not None:
                goal = self.obs_normalizer.normalize(goal)

        if self.act_normalizer is not None:
            action_seq = self.act_normalizer.normalize(action_seq)

        external_cond = self._build_external_cond(obs_seq, goal)

        loss = self._compute_loss(external_cond, action_seq)

        self.log(f"{phase}/loss", loss, prog_bar=True, sync_dist=(phase == "val"))
        return loss

    def _build_external_cond(
        self, obs: TensorTree, goal: TensorTree | None
    ) -> dict[str, TensorTree]:
        if self.goal_conditioned and goal is None:
            raise ValueError(
                f"{type(self).__name__} is configured with goal_horizon={self.goal_horizon} > 0, "
                "but received goal=None."
            )

        if self.goal_conditioned:
            assert goal is not None
            return {"obs": obs, "goal": goal}
        else:
            return {"obs": obs}
