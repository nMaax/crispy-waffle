from collections.abc import Mapping
from typing import Any, cast

import hydra_zen
import torch
import torch.nn as nn

from policy.algorithms.diffusion_policy import DiffusionPolicy
from policy.utils import merge_dicts
from policy.utils.typing_utils import (
    DimSpec,
    GoalConditionedPolicyProtocol,
    HydraConfigFor,
    TensorTree,
)


class GoalConditionedDiffusionPolicy(DiffusionPolicy, GoalConditionedPolicyProtocol):
    """Goal-conditioned diffusion policy using diffusers noise schedulers.

    Proprioception is always kept raw and never routed through the embedder, so embedders stay
    robot-agnostic. The "no embedding" variant is simply ``embedder=None`` (an identity
    embedder); other embedders (e.g. an MLP) are selected via config.
    """

    def __init__(
        self,
        *args,
        proprio_dim: int = 18,
        task_dim: int | None = None,
        embedder: HydraConfigFor[nn.Module] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.goal_conditioned = True

        proprio_dim, task_dim = self._validate_obs_dim(proprio_dim, task_dim)

        self.proprio_dim = proprio_dim
        self.task_dim = task_dim
        self.goal_dim = task_dim

        self.embedder_config = embedder
        self.embedder: nn.Module | None = None

    def _validate_obs_dim(self, proprio_dim: int, task_dim: int | None) -> tuple[int, int]:
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

        return proprio_dim, calc_task_dim

    def configure_model(self) -> None:
        if self.network is not None:
            return
        self.embedder = (
            hydra_zen.instantiate(self.embedder_config)
            if self.embedder_config is not None
            else nn.Identity()
        )
        super().configure_model()

    def _get_cond_dims(self) -> DimSpec:
        """Reports the per-timestep conditioning dimensionality passed to the network's
        ``cond_dims``.

        Widths here are *not* multiplied by ``obs_horizon`` -- each network knows its own horizon
        (via config) and is responsible for resolving how it consumes the time axis of each key.
        """
        embed_dim = self._embedder_output_dim()
        return {"obs": {"proprio": self.proprio_dim, "task": embed_dim}, "goal": embed_dim}

    def _embedder_output_dim(self) -> int:
        """Lookup of the embedder's output dim.

        Reads config only, never an instantiated module, so that
        :meth:`_get_cond_dims` remains safe to call before :meth:`configure_model`.
        """
        if self.embedder_config is None:
            return self.task_dim

        return self.embedder_config.get("output_dim")

    @torch.no_grad()
    def extract_embeddings(
        self,
        obs: torch.Tensor | dict,
        goal: torch.Tensor | dict | None = None,
    ):
        """Extracts embedder outputs for observations (and optionally a goal).

        Helper function for visualizing the embeddings.
        """
        if isinstance(obs, Mapping):
            obs = {k: v.to(self.device) for k, v in obs.items()}
        else:
            obs = obs.to(self.device)

        if goal is not None:
            if isinstance(goal, Mapping):
                goal = {k: v.to(self.device) for k, v in goal.items()}
            else:
                goal = goal.to(self.device)

        if self.obs_normalizer is not None:
            obs = self.obs_normalizer.normalize(obs)
            if goal is not None:
                goal = self.obs_normalizer.normalize(goal)

        external_cond = self._build_external_cond(obs, goal)
        obs_embeddings = external_cond.get("obs")
        if obs_embeddings is None:
            raise ValueError("Failed to extract observation embeddings from external_cond.")

        if isinstance(obs_embeddings, Mapping):
            obs_task_embeddings = obs_embeddings.get("task")
        else:
            obs_task_embeddings = obs_embeddings

        if not isinstance(obs_task_embeddings, torch.Tensor):
            raise ValueError(
                f"Expected obs_task_embeddings to be a torch.Tensor, but got {type(obs_task_embeddings)}."
            )

        res = {"obs_embeddings": obs_task_embeddings.cpu()}

        goal_embedding = external_cond.get("goal", None)
        if goal_embedding is not None:
            if not isinstance(goal_embedding, torch.Tensor):
                raise ValueError(
                    f"Expected goal_embedding to be a torch.Tensor, but got {type(goal_embedding)}."
                )
            res["goal_embedding"] = goal_embedding.cpu()

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

        external_cond = self._build_external_cond(obs_seq, goal)

        loss = self._compute_loss(external_cond, action_seq)

        self.log(f"{phase}/loss", loss, prog_bar=True, sync_dist=(phase == "val"))
        return loss

    def _build_external_cond(
        self, obs: TensorTree, goal: TensorTree | None
    ) -> dict[str, TensorTree]:
        external_cond = self._build_obs_external_cond(obs)
        if goal is not None:
            goal_external_cond = self._build_goal_external_cond(goal)
            external_cond = merge_dicts([external_cond, goal_external_cond])

        return external_cond

    def _build_obs_external_cond(self, obs: TensorTree) -> dict[str, TensorTree]:
        proprio, task_embedded = self._embed_states(obs)
        return {"obs": {"proprio": proprio, "task": task_embedded}}

    def _build_goal_external_cond(self, goal: TensorTree) -> dict[str, TensorTree]:
        _, goal_embedded = self._embed_states(goal)
        return {"goal": goal_embedded}

    def _embed_states(self, states: TensorTree) -> tuple[torch.Tensor, torch.Tensor]:
        """Splits proprio/task and embeds the task components.

        Handles both a horizon window (``task`` is ``[B, T, task_dim]``, e.g. obs) and a single
        timestep with no time axis at all (``task`` is ``[B, task_dim]``, e.g. goal) uniformly:
        a missing time axis is unsqueezed to ``T=1`` before embedding, then squeezed back out of
        the result so the returned shape matches whatever was passed in.
        """
        if self.embedder is None:
            raise ValueError(
                "Embedder not initialized. Call configure_model() before using the embedder."
            )
        proprio, task = self._split_proprio_task(states)

        had_no_time_axis = task.ndim == 2
        if had_no_time_axis:
            task = task.unsqueeze(1)

        B, T = task.shape[0], task.shape[1]
        task_flat = task.reshape(B * T, self.task_dim)
        task_embedded = self.embedder(task_flat).reshape(B, T, -1)

        if had_no_time_axis:
            task_embedded = task_embedded.squeeze(1)

        return proprio, task_embedded

    def _split_proprio_task(
        self, x: torch.Tensor | Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Splits proprioception from the (concatenated) task-relevant components."""
        if isinstance(x, Mapping):
            proprio = x["proprio"]
            task_components = [v for k, v in x.items() if k != "proprio"]
            task = torch.cat(task_components, dim=-1)
        else:
            proprio = x[..., : self.proprio_dim]
            task = x[..., self.proprio_dim :]
        return proprio, task
