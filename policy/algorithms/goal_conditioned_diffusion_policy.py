from collections.abc import Mapping
from typing import Any

import hydra_zen
import torch
import torch.nn as nn

from policy.algorithms.diffusion_policy import DiffusionPolicy
from policy.utils import (
    concat_leaf_tensors,
    derive_task_dim,
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
            else nn.Identity()
        )
        super().configure_model()

    def _get_cond_dims(self) -> DimSpec:
        """Reports the per-timestep conditioning dimensionality passed to the network's
        ``cond_dims``.

        The obs task-embedding sequence (and, when goal-conditioned, the goal embedding) are always
        fused into a single "plan" entry (see _build_mixed_external_cond). With no mixer
        configured, the mixer is an identity/concatenation, so "plan" is just the available
        embeddings concatenated -- numerically equivalent to the old separate obs["task"] / goal
        entries, just reported under one key. With goal_horizon=0 there's no goal to fuse in, so
        "plan" is just the flattened obs task-embedding sequence on its own.
        """
        cond_dims: dict[str, DimSpec] = {
            "obs": {"proprio": self.proprio_dim},
            "plan": self._mixer_output_dim(),
        }
        if self.goal_horizon > 0 and not self.exclude_proprio_from_goal:
            cond_dims["goal"] = {"proprio": self.proprio_dim}
        return cond_dims

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
        :meth:`_get_cond_dims` remains safe to call before :meth:`configure_model`. With no
        mixer configured, the (identity) mixer preserves its input width.
        """
        if self.mixer_config is None:
            return self._mixer_input_dim()

        return self.mixer_config.get("output_dim")

    def _mixer_input_dim(self) -> int:
        """Width of the mixer's input: the flattened obs task-embedding sequence concatenated with
        the flattened goal task-embedding (goal_horizon may be > 1)."""
        embed_dim = self._embedder_output_dim()
        return (self.obs_horizon + self.goal_horizon) * embed_dim

    def _mix(
        self, task_embeddings_seq: torch.Tensor, goal_embedding: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Fuses the observation task-embedding sequence with the goal task-embedding (when
        present) into a single "plan" embedding via the mixer module.

        Shapes:
            task_embeddings_seq: [B, obs_horizon, embed_dim]
            goal_embedding: [B, embed_dim] or [B, goal_horizon, embed_dim], or None if unconditioned
            returns: [B, mixer_output_dim]
        """
        if self.mixer is None:
            raise ValueError(
                "Mixer not initialized. Call configure_model() before using the mixer."
            )

        B = task_embeddings_seq.shape[0]
        parts = [task_embeddings_seq.reshape(B, -1)]
        if goal_embedding is not None:
            parts.append(goal_embedding.reshape(B, -1))
        return self.mixer(torch.cat(parts, dim=-1))

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

        goal_task_embedding = None
        if goal is not None:
            _, goal_task_embedding = self._embed_states(goal)
            res["goal_embedding"] = goal_task_embedding.cpu()

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
        # The obs task-embedding sequence (and, when goal-conditioned, the goal embedding) are
        # always fused via the mixer (an identity/concatenation when no ``mixer`` is configured)
        # into a single "plan" entry -- proprioception itself is never routed through the mixer
        # either, and stays raw under "obs"/"goal". Among such proprioception we can optionally
        # include the one associated to the goal by turning exclude_proprio_from_goal to False if
        # our inference setting provides reasonable proprioception data.

        if self.goal_horizon == 0:
            # No goal configured: ignore whatever was passed (mirrors the unconditioned contract
            # used elsewhere, e.g. get_action()/get_cond_dims() never reserving room for a goal).
            goal = None
        elif goal is None:
            raise ValueError(
                f"{type(self).__name__} is configured with goal_horizon={self.goal_horizon} > 0, "
                "but received goal=None."
            )

        return self._build_mixed_external_cond(obs, goal)

    def _build_mixed_external_cond(
        self, obs: TensorTree, goal: TensorTree | None
    ) -> dict[str, TensorTree]:
        """Fuses the obs task-embedding sequence with the goal task-embedding (when present) via
        the mixer, keeping proprioception raw and unmixed."""
        proprio_seq, task_embeddings_seq = self._embed_states(obs)

        goal_proprio, goal_embeddings_seq = (None, None)
        if goal is not None:
            goal_proprio, goal_embeddings_seq = self._embed_states(goal)

        plan = self._mix(task_embeddings_seq, goal_embeddings_seq)

        external_cond: dict[str, TensorTree] = {"obs": {"proprio": proprio_seq}, "plan": plan}
        if goal_proprio is not None and not self.exclude_proprio_from_goal:
            external_cond["goal"] = {"proprio": goal_proprio}
        return external_cond

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
