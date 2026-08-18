from collections.abc import Mapping
from typing import Any, Literal

import torch

from policy.algorithms.diffusion_policy import DiffusionPolicy
from policy.transforms.canonicalization.spec import dim_shape
from policy.utils import as_batch, get_tensor, pop_leaf_key, to_device
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

    def _tokenize(self, obs: TensorTree, goal: TensorTree | None = None) -> dict[str, TensorTree]:
        """Splits proprioception off and tokenizes, routing absolute vs goal-relative."""
        if self.tokenizer is None:
            raise ValueError(
                f"{type(self).__name__} has no tokenizer; _tokenize() should not be reached."
            )

        if goal is None:
            raise ValueError("goal_conditioned=True, but received goal=None.")

        obs_proprio, obs_task = pop_leaf_key(obs, "proprio", self.proprio_dim)
        if obs_proprio is None:
            raise ValueError("Observation mapping must contain a 'proprio' key.")

        # The goal's own proprioception never enters conditioning: only the historical (observed)
        # proprioception is used, on both the absolute and the relative path.
        _, goal_task = pop_leaf_key(goal, "proprio", self.proprio_dim)

        if not isinstance(obs_task, Mapping) or not isinstance(goal_task, Mapping):
            raise TypeError(
                "Tokenized conditioning requires canonical dict observation and goal trees, got "
                f"{type(obs_task).__name__} and {type(goal_task).__name__}."
            )

        if not self.relative_goal:
            return {
                "proprio": obs_proprio,
                "task": self.tokenizer.tokenize(obs_task, None),
                "goal_task": self.tokenizer.tokenize(None, goal_task),
            }

        # An obs without a time axis would not raise below: it would broadcast against the
        # unsqueezed goal into [B, B, ...], silently mistaking the batch axis for the time axis.
        self._validate_obs_time_axis(obs_task)

        # goal_task's own missing time axis must be added on every leaf,
        goal_task = self._add_goal_time_axis(goal_task)

        return {"proprio": obs_proprio, "task": self.tokenizer.tokenize(obs_task, goal_task)}

    def _validate_obs_time_axis(self, obs_task: Mapping[str, TensorTree]) -> None:
        """Requires a [B, T, ...] prefix on every task leaf."""
        for key, spec_ndim in self._task_leaf_ndims().items():
            ndim = get_tensor(obs_task, key).ndim
            if ndim != spec_ndim + 2:
                raise ValueError(
                    "relative_goal=True expects observations of shape [B, T, F], but task leaf "
                    f"{key!r} has {ndim} axes where {spec_ndim + 2} were expected."
                )

    def _add_goal_time_axis(self, goal_task: Mapping[str, TensorTree]) -> TensorTree:
        """Inserts the goal's missing time axis, leaving an already-timed goal untouched."""
        return {
            key: leaf.unsqueeze(1) if leaf.ndim == spec_ndim + 1 else leaf
            for key, spec_ndim in self._task_leaf_ndims().items()
            for leaf in (get_tensor(goal_task, key),)
        }

    def _task_leaf_ndims(self) -> dict[str, int]:
        """Per-key axis counts before the [B, T] prefix."""
        return {key: len(dim_shape(dim)) for key, dim in self._tokenizer_task_dim().items()}

    def _obs_normalizer_view(self, item: dict[str, Any]) -> TensorTree:
        return self._tokenize(as_batch(item["obs_seq"]), as_batch(item["goal"]))

    def extract_embeddings(
        self,
        obs: torch.Tensor | Mapping[str, Any],
        goal: torch.Tensor | Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, torch.Tensor], Literal["absolute", "goal-relative"]]:
        """Helper for visualizing the embeddings."""
        if self.encoder is None:
            raise ValueError("Encoder not initialized. Call configure_model() first.")

        obs = to_device(obs, self.device)
        if goal is not None:
            goal = to_device(goal, self.device)

        with torch.no_grad():
            payload = self.encoder(self._normalize_obs(self._tokenize(obs, goal)))
            embeddings = {"obs_embeddings": self.encoder.unpack_task(payload)}
            if "goal" in payload:
                embeddings["goal_embedding"] = payload["goal"]

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
