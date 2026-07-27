from collections.abc import Mapping
from typing import Any

import hydra_zen
import torch
import torch.nn as nn

from policy.algorithms.diffusion_policy import DiffusionPolicy
from policy.utils import (
    concat_leaf_tensors,
    derive_task_dim,
    merge_dicts,
    resolve_proprio_dim,
    split_leaf_key,
)
from policy.utils.typing_utils import (
    DimSpec,
    GoalConditionedPolicyProtocol,
    HydraConfigFor,
    TensorTree,
)


class GoalConditionedDiffusionPolicy(DiffusionPolicy, GoalConditionedPolicyProtocol):
    """Goal-conditioned diffusion policy using diffusers noise schedulers."""

    def __init__(
        self,
        *args,
        goal_horizon: int = 1,
        proprio_dim: int | None = None,
        task_dim: int | None = None,
        embedder: HydraConfigFor[nn.Module] | None = None,
        mixer: HydraConfigFor[nn.Module] | None = None,
        exclude_proprio_from_goal: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.goal_horizon = goal_horizon
        self.goal_conditioned = goal_horizon > 0

        if mixer is not None and goal_horizon == 0:
            raise ValueError(
                "mixer requires goal_horizon > 0: it mixes the observation task-embedding "
                "sequence with a goal embedding, so there must be a goal to mix with."
            )

        proprio_dim, task_dim = self._validate_obs_dim(proprio_dim, task_dim)

        self.proprio_dim = proprio_dim
        self.task_dim = task_dim
        self.goal_dim = task_dim
        self.exclude_proprio_from_goal = exclude_proprio_from_goal

        self.embedder_config = embedder
        self.embedder: nn.Module | None = None

        self.mixer_config = mixer
        self.mixer: nn.Module | None = None

    def _validate_obs_dim(self, proprio_dim: int | None, task_dim: int | None) -> tuple[int, int]:
        proprio_dim = resolve_proprio_dim(self.obs_dim, proprio_dim)
        task_dim = derive_task_dim(self.obs_dim, proprio_dim, task_dim)
        return proprio_dim, task_dim

    def configure_model(self) -> None:
        if self.network is not None:
            return
        self.embedder = (
            hydra_zen.instantiate(self.embedder_config, input_dim=self.task_dim)
            if self.embedder_config is not None
            else nn.Identity()
        )
        self.mixer = (
            hydra_zen.instantiate(self.mixer_config, input_dim=self._mixer_input_dim())
            if self.mixer_config is not None
            else None
        )
        super().configure_model()

    def _get_cond_dims(self) -> DimSpec:
        """Reports the per-timestep conditioning dimensionality passed to the network's
        ``cond_dims``."""
        embed_dim = self._embedder_output_dim()
        obs_spec = {"proprio": self.proprio_dim, "task": embed_dim}
        if self.goal_horizon == 0:
            return {"obs": obs_spec}

        if self.mixer_config is not None:
            cond_dims: dict[str, DimSpec] = {
                "obs": {"proprio": self.proprio_dim},
                "plan": self._mixer_output_dim(),
            }
            if not self.exclude_proprio_from_goal:
                cond_dims["goal"] = {"proprio": self.proprio_dim}
            return cond_dims

        if self.exclude_proprio_from_goal:
            goal_spec = embed_dim
        else:
            goal_spec = {"proprio": self.proprio_dim, "task": embed_dim}

        return {"obs": obs_spec, "goal": goal_spec}

    def _embedder_output_dim(self) -> int:
        """Lookup of the embedder's output dim.

        Reads config only, never an instantiated module, so that
        :meth:`_get_cond_dims` remains safe to call before :meth:`configure_model`.
        """
        if self.embedder_config is None:
            return self.task_dim

        return self.embedder_config.get("output_dim")

    def _mixer_output_dim(self) -> int:
        """Lookup of the mixer's output dim.

        Reads config only, never an instantiated module, so that
        :meth:`_get_cond_dims` remains safe to call before :meth:`configure_model`.
        """
        if self.mixer_config is None:
            raise ValueError("mixer_config is not set.")
        return self.mixer_config.get("output_dim")

    def _mixer_input_dim(self) -> int:
        """Width of the mixer's input: the flattened obs task-embedding sequence concatenated with
        the goal task-embedding."""
        embed_dim = self._embedder_output_dim()
        return self.obs_horizon * embed_dim + embed_dim

    def _mix(self, task_embeddings_seq: torch.Tensor, goal_embedding: torch.Tensor) -> torch.Tensor:
        """Fuses the observation task-embedding sequence and the goal task-embedding into a single
        "plan" embedding via the mixer module.

        Shapes:
            task_embeddings_seq: [B, obs_horizon, embed_dim]
            goal_embedding: [B, embed_dim]
            returns: [B, mixer_output_dim]
        """
        if self.mixer is None:
            raise ValueError(
                "Mixer not initialized. Call configure_model() before using the mixer."
            )
        B = task_embeddings_seq.shape[0]
        mixer_input = torch.cat([task_embeddings_seq.reshape(B, -1), goal_embedding], dim=-1)
        return self.mixer(mixer_input)

    @torch.no_grad()
    def extract_embeddings(
        self,
        obs: torch.Tensor | dict,
        goal: torch.Tensor | dict | None = None,
    ):
        """Extracts embedder (and, if configured, mixer) outputs for observations (and optionally a
        goal).

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

        _, obs_task_embeddings = self._embed_states(obs)
        res = {"obs_embeddings": obs_task_embeddings.cpu()}

        if goal is not None:
            _, goal_task_embedding = self._embed_states(goal)
            res["goal_embedding"] = goal_task_embedding.cpu()

            if self.mixer is not None:
                res["plan_embedding"] = self._mix(obs_task_embeddings, goal_task_embedding).cpu()

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

        if self.goal_horizon == 0:
            return self._build_obs_external_cond(obs)

        if goal is None:
            raise ValueError(
                f"{type(self).__name__} is configured with goal_horizon={self.goal_horizon} > 0, "
                "but received goal=None."
            )

        if self.mixer is not None:
            return self._build_mixed_external_cond(obs, goal)

        external_cond = self._build_obs_external_cond(obs)
        goal_external_cond = self._build_goal_external_cond(goal)
        return merge_dicts([external_cond, goal_external_cond])

    def _build_obs_external_cond(self, obs: TensorTree) -> dict[str, TensorTree]:
        proprio, task_embedded = self._embed_states(obs)
        return {"obs": {"proprio": proprio, "task": task_embedded}}

    def _build_mixed_external_cond(
        self, obs: TensorTree, goal: TensorTree
    ) -> dict[str, TensorTree]:
        """Fuses the obs task-embedding sequence with the goal task-embedding via the mixer,
        keeping proprioception raw and unmixed (same principle as
        :meth:`_build_goal_external_cond`)."""
        proprio_seq, task_embeddings_seq = self._embed_states(obs)
        goal_proprio, goal_embedding = self._embed_states(goal)

        plan = self._mix(task_embeddings_seq, goal_embedding)

        external_cond: dict[str, TensorTree] = {"obs": {"proprio": proprio_seq}, "plan": plan}
        if not self.exclude_proprio_from_goal:
            external_cond["goal"] = {"proprio": goal_proprio}
        return external_cond

    def _build_goal_external_cond(self, goal: TensorTree) -> dict[str, TensorTree]:
        proprio, goal_embedded = self._embed_states(goal)
        if self.exclude_proprio_from_goal:
            return {"goal": goal_embedded}
        else:
            return {"goal": {"proprio": proprio, "task": goal_embedded}}

    def _embed_states(self, states: TensorTree) -> tuple[torch.Tensor, torch.Tensor]:
        """Splits proprio/task and embeds the task components."""
        if self.embedder is None:
            raise ValueError(
                "Embedder not initialized. Call configure_model() before using the embedder."
            )
        proprio, task = self._split_proprio_task(states)

        # Handles both a horizon window (``task`` is ``[B, T, task_dim]``, e.g. obs) and a single
        # timestep with no time axis at all (``task`` is ``[B, task_dim]``, e.g. goal) uniformly:
        # a missing time axis is unsqueezed to ``T=1`` before embedding, then squeezed back out of
        # the result so the returned shape matches whatever was passed in.

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
        proprio, remainder = split_leaf_key(x, "proprio", self.proprio_dim)
        if proprio is None:
            raise ValueError("Observation/goal mapping must contain a 'proprio' key.")
        task = (
            concat_leaf_tensors(remainder, dim=-1) if isinstance(remainder, Mapping) else remainder
        )
        return proprio, task
