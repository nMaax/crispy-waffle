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
    """Goal-conditioned diffusion policy using diffusers noise schedulers.

    Two conditioning modes, selected by ``goal_delta``:

    - ``goal_delta=False`` (default, absolute): the network sees the observation embeddings and
      the goal embedding as separate conditioning entries, i.e. ``s_1, ..., s_T`` plus ``g``.
    - ``goal_delta=True`` (relative): the goal is folded into the observation window, so the
      network sees one difference per observed timestep, i.e. ``g - s_1, ..., g - s_T``, and no
      standalone goal entry. On this branch the difference is taken in input space and then
      embedded (``embed(g - s_t)``); the embedding-space variant (``embed(g) - embed(s_t)``) is
      on ``feature/goal-state-delta-conditioning``.
    """

    def __init__(
        self,
        *args,
        goal_horizon: int = 1,
        proprio_dim: int | None = None,
        task_dim: int | None = None,
        embedder: HydraConfigFor[nn.Module] | None = None,
        exclude_proprio_from_goal: bool = True,
        goal_delta: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.goal_horizon = goal_horizon
        self.goal_conditioned = goal_horizon > 0

        if goal_delta and goal_horizon == 0:
            raise ValueError(
                "goal_delta=True requires goal_horizon > 0: there is no goal to difference the "
                "observations against when goal-conditioning is disabled."
            )

        proprio_dim, task_dim = self._validate_obs_dim(proprio_dim, task_dim)

        self.proprio_dim = proprio_dim
        self.task_dim = task_dim
        self.goal_dim = task_dim
        self.exclude_proprio_from_goal = exclude_proprio_from_goal
        self.goal_delta = goal_delta

        self.embedder_config = embedder
        self.embedder: nn.Module | None = None

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
        super().configure_model()

    def _get_cond_dims(self) -> DimSpec:
        """Reports the per-timestep conditioning dimensionality passed to the network's
        ``cond_dims``."""
        embed_dim = self._embedder_output_dim()
        obs_spec = {"proprio": self.proprio_dim, "task": embed_dim}
        if self.goal_horizon == 0 or self.goal_delta:
            # In delta mode the goal lives inside the obs entry (as ``g - s_t``), so it adds no
            # conditioning width of its own and the widths match the unconditioned case.
            return {"obs": obs_spec}
        else:
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



    @torch.no_grad()
    def extract_embeddings(
        self,
        obs: torch.Tensor | dict,
        goal: torch.Tensor | dict | None = None,
    ):
        """Extracts embedder outputs for observations (and optionally a goal).

        Helper function for visualizing the embeddings. Always reports the absolute embeddings,
        independently of ``goal_delta``, so visualizations stay comparable across both modes.
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

        if self.goal_delta:
            return self._build_delta_external_cond(obs, goal)

        return merge_dicts(
            [self._build_obs_external_cond(obs), self._build_goal_external_cond(goal)]
        )

    def _build_obs_external_cond(self, obs: TensorTree) -> dict[str, TensorTree]:
        proprio, task_embedded = self._embed_states(obs)
        return {"obs": {"proprio": proprio, "task": task_embedded}}

    def _build_goal_external_cond(self, goal: TensorTree) -> dict[str, TensorTree]:
        proprio, goal_embedded = self._embed_states(goal)
        if self.exclude_proprio_from_goal:
            return {"goal": goal_embedded}
        else:
            return {"goal": {"proprio": proprio, "task": goal_embedded}}

    def _build_delta_external_cond(
        self, obs: TensorTree, goal: TensorTree
    ) -> dict[str, TensorTree]:
        """Conditions on goal-state differences (``g - s_t``) instead of ``g`` and ``s_t`` apart.

        The obs "task" slot carries one difference per observed timestep, ``g - s_t``, with the
        goal broadcast across the observation window, and no separate "goal" entry is emitted.
        The subtraction happens in *input* space and the difference is what gets embedded
        (``embed(g - s_t)``, not ``embed(g) - embed(s_t)``), so the embedder sees goal-relative
        features and never an absolute state. The embedding-space variant lives on its own branch
        (``feature/goal-state-delta-conditioning``); the two coincide only for a linear embedder
        without bias, and exactly for the identity embedder.

        Proprioception follows ``exclude_proprio_from_goal`` exactly as in absolute mode: when
        True (default) the raw per-timestep proprio is passed through untouched, and only the task
        components are differenced; when False the goal's proprio is differenced against the
        observed proprio too (so no absolute proprio reaches the network). Proprio never goes
        through the embedder, so its difference is unaffected by this choice of space.
        """
        obs_proprio, obs_task = self._split_proprio_task(obs)
        goal_proprio, goal_task = self._split_proprio_task(goal)

        goal_task = self._broadcast_goal_over_obs(goal_task, obs_task)
        task_delta = self._embed_task(goal_task - obs_task)
        if self.exclude_proprio_from_goal:
            proprio = obs_proprio
        else:
            proprio = self._broadcast_goal_over_obs(goal_proprio, obs_proprio) - obs_proprio

        return {"obs": {"proprio": proprio, "task": task_delta}}

    @staticmethod
    def _broadcast_goal_over_obs(goal: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """Aligns a goal tensor's time axis with the observation window's so they can be
        subtracted.

        A goal is a single timestep, so it usually arrives without a time axis (``[B, D]`` against
        an obs window of ``[B, T, D]``) and gets one inserted to broadcast over the window.
        """
        if goal.ndim == obs.ndim - 1:
            return goal.unsqueeze(1)

        if goal.ndim != obs.ndim:
            raise ValueError(
                f"Cannot align goal of shape {tuple(goal.shape)} with observations of shape "
                f"{tuple(obs.shape)}: expected the goal to have the same number of dimensions, "
                "or one fewer (no time axis)."
            )

        if obs.ndim == 3 and goal.shape[1] not in (1, obs.shape[1]):
            raise ValueError(
                f"Cannot align a goal window of length {goal.shape[1]} with an observation window "
                f"of length {obs.shape[1]}: goal_delta=True expects either a single goal timestep "
                "(broadcast over the window) or one goal per observed timestep."
            )

        return goal

    def _embed_states(self, states: TensorTree) -> tuple[torch.Tensor, torch.Tensor]:
        """Splits proprio/task and embeds the task components."""
        proprio, task = self._split_proprio_task(states)
        return proprio, self._embed_task(task)

    def _embed_task(self, task: torch.Tensor) -> torch.Tensor:
        """Runs already-split task components through the embedder."""
        if self.embedder is None:
            raise ValueError(
                "Embedder not initialized. Call configure_model() before using the embedder."
            )

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

        return task_embedded

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
