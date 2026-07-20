from collections.abc import Mapping
from typing import Any, cast

import torch

from policy.algorithms.diffusion_policy import DiffusionPolicy
from policy.utils import concat_leaf_tensors, flatten_and_concat_leaf_tensors
from policy.utils.typing_utils import GoalConditionedPolicyProtocol


class GoalConditionedDiffusionPolicy(DiffusionPolicy, GoalConditionedPolicyProtocol):
    """Goal-conditioned diffusion policy using diffusers noise schedulers."""

    def __init__(
        self,
        *args,
        proprio_dim: int = 18,
        task_dim: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.goal_conditioned = True

        if isinstance(self.obs_dim, Mapping):
            if "proprio" not in self.obs_dim:
                raise ValueError("Observation dictionary spec must contain 'proprio' key.")
            if self.obs_dim["proprio"] != proprio_dim:
                raise ValueError(
                    f"Proprioception dimension in spec ({self.obs_dim['proprio']}) does not match proprio_dim ({proprio_dim})."
                )

            calc_task_dim = sum(cast(int, v) for k, v in self.obs_dim.items() if k != "proprio")
            if task_dim is not None and calc_task_dim != task_dim:
                raise ValueError(
                    f"Task dimension calculated from spec ({calc_task_dim}) does not match task_dim ({task_dim})."
                )
        elif isinstance(self.obs_dim, int):
            if self.obs_dim < proprio_dim:
                raise ValueError(
                    f"Observation dimension ({self.obs_dim}) must be >= proprio_dim ({proprio_dim})."
                )
            calc_task_dim = self.obs_dim - proprio_dim
            if task_dim is not None and calc_task_dim != task_dim:
                raise ValueError(
                    f"Proprioception dimensionality ({proprio_dim}) + Task dimensionality ({task_dim}) "
                    f"do not match observation dimensionality ({self.obs_dim}). "
                    f"{proprio_dim} + {task_dim} != {self.obs_dim}."
                )
        else:
            raise ValueError(
                f"Observation dimensionality must be an integer or dict, but got {type(self.obs_dim)}."
            )

        self.proprio_dim = proprio_dim
        self.task_dim = calc_task_dim
        self.goal_dim = self.task_dim

        self.network_cond_dim = self.network_cond_dim + self.goal_dim

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

        obs_cond = self._prepare_obs(obs_seq)
        goal_cond = self._prepare_goal(goal) if goal is not None else None

        return self._run_diffusion_loop(
            obs_cond=obs_cond,
            goal_cond=goal_cond,
            num_inference_steps=num_inference_steps,
            output_clip_range=output_clip_range,
        )

    def _shared_step(self, batch: dict[str, Any], batch_idx: int, phase: str) -> torch.Tensor:
        """Main step logic for training and validation step logging in goal-conditioned
        policies."""
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

        obs_cond = self._prepare_obs(obs_seq)
        goal_cond = self._prepare_goal(goal) if goal is not None else None

        loss = self._compute_loss(obs_seq=obs_cond, act_seq=action_seq, goal=goal_cond)

        self.log(f"{phase}/loss", loss, prog_bar=True, sync_dist=(phase == "val"))
        return loss

    def _prepare_goal(self, goal: Mapping[str, Any] | torch.Tensor) -> torch.Tensor:
        """Prepares the goal conditioning tensor by excluding proprioception entries."""

        if isinstance(goal, Mapping):
            goal_filtered = {k: v for k, v in goal.items() if k != "proprio"}
        elif isinstance(goal, torch.Tensor):
            goal_filtered = goal[..., self.proprio_dim :]
        else:
            raise ValueError(
                f"Expected goal to be a torch.Tensor or Mapping, but got {type(goal)}."
            )

        if self.flatten_obs:
            return flatten_and_concat_leaf_tensors(goal_filtered, device=self.device)
        else:
            return concat_leaf_tensors(goal_filtered, device=self.device)

    def _compute_loss(
        self, obs_seq: torch.Tensor, act_seq: torch.Tensor, goal: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Computes reconstruction loss by passing observation and optional goal to network."""
        return super()._compute_loss(obs_seq=obs_seq, act_seq=act_seq, goal=goal)

    def _run_diffusion_loop(
        self,
        obs_cond: torch.Tensor,
        goal_cond: torch.Tensor | None = None,
        num_inference_steps: int | None = None,
        output_clip_range: tuple | None = None,
    ) -> torch.Tensor:
        """Reverse diffusion process loop supporting observation and optional goal conditioning."""
        return super()._run_diffusion_loop(
            obs_cond=obs_cond,
            num_inference_steps=num_inference_steps,
            output_clip_range=output_clip_range,
            goal=goal_cond,
        )
