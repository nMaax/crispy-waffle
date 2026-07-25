from collections.abc import Mapping
from typing import Any, cast

import torch

from policy.algorithms.embedded_diffusion_policy import EmbeddedDiffusionPolicy
from policy.utils import merge_dicts
from policy.utils.typing_utils import (
    DimSpec,
    GoalConditionedPolicyProtocol,
    TensorTree,
)


class GoalConditionedDiffusionPolicy(EmbeddedDiffusionPolicy, GoalConditionedPolicyProtocol):
    """Goal-conditioned diffusion policy using diffusers noise schedulers."""

    def __init__(
        self,
        *args,
        goal_horizon: int = 1,
        exclude_proprio_from_goal: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.goal_horizon = goal_horizon
        self.goal_conditioned = goal_horizon > 0

        self.goal_dim = self.task_dim
        self.exclude_proprio_from_goal = exclude_proprio_from_goal

    def _get_cond_dims(self) -> DimSpec:
        """Reports the per-timestep conditioning dimensionality passed to the network's
        ``cond_dims``."""
        cond_dims = super()._get_cond_dims()
        if self.goal_horizon == 0:
            return cond_dims

        if not isinstance(cond_dims, Mapping):
            raise TypeError(
                f"Expected a Mapping cond_dims from {EmbeddedDiffusionPolicy.__name__}, "
                f"but got {type(cond_dims)}."
            )

        embed_dim = self._embedder_output_dim()
        if self.exclude_proprio_from_goal:
            goal_spec = embed_dim
        else:
            goal_spec = {"proprio": self.proprio_dim, "task": embed_dim}

        return {**cond_dims, "goal": goal_spec}

    @torch.no_grad()
    def extract_embeddings(
        self,
        obs: torch.Tensor | dict,
        goal: torch.Tensor | dict | None = None,
    ):
        """Extracts embedder outputs for observations (and optionally a goal).

        Helper function for visualizing the embeddings.
        """
        res = super().extract_embeddings(obs)

        if goal is not None:
            if isinstance(goal, Mapping):
                goal = {k: v.to(self.device) for k, v in goal.items()}
            else:
                goal = goal.to(self.device)

            if self.obs_normalizer is not None:
                goal = self.obs_normalizer.normalize(goal)

            goal_embedding = self._build_goal_external_cond(cast(TensorTree, goal))["goal"]
            if isinstance(goal_embedding, Mapping):
                goal_task_embedding = goal_embedding.get("task")
            else:
                goal_task_embedding = goal_embedding

            if not isinstance(goal_task_embedding, torch.Tensor):
                raise ValueError(
                    f"Expected goal_task_embedding to be a torch.Tensor, but got {type(goal_task_embedding)}."
                )
            res["goal_embedding"] = goal_task_embedding.cpu()

        return res

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

        # NOTE: Proprioception is always kept raw and never routed through the embedder, so embedders stay
        # robot-agnostic. The "no embedding" variant is simply ``embedder=None`` (an identity
        # embedder); other embedders (e.g. an MLP) are selected via config.
        # Propriception of the historical states is the re-routed to be concatenated alongside the embedder outputs to
        # condition the network denoising process. Among such proprioception we can optionally include the one associated
        # to the goal by turning exclude_proprio_from_goal to False if our inference setting provides reasonable proprioception data.

        external_cond = self._build_obs_external_cond(obs)
        if self.goal_horizon > 0:
            if goal is None:
                raise ValueError(
                    f"{type(self).__name__} is configured with goal_horizon={self.goal_horizon} > 0, "
                    "but received goal=None."
                )
            goal_external_cond = self._build_goal_external_cond(goal)
            external_cond = merge_dicts([external_cond, goal_external_cond])

        return external_cond

    def _build_goal_external_cond(self, goal: TensorTree) -> dict[str, TensorTree]:
        proprio, goal_embedded = self._embed_states(goal)
        if self.exclude_proprio_from_goal:
            return {"goal": goal_embedded}
        else:
            return {"goal": {"proprio": proprio, "task": goal_embedded}}
