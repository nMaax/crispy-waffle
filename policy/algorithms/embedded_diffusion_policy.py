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
from policy.utils.typing_utils import DimSpec, HydraConfigFor, TensorTree


class EmbeddedDiffusionPolicy(DiffusionPolicy):
    """DiffusionPolicy with an optional per-timestep state embedder.

    Splits each observation into proprioception (always kept raw) and the remaining "task"
    features, optionally passed through an embedder module (e.g. MLP/ResidualMLP).
    """

    def __init__(
        self,
        *args,
        proprio_dim: int | None = None,
        task_dim: int | None = None,
        embedder: HydraConfigFor[nn.Module] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        proprio_dim, task_dim = self._validate_obs_dim(proprio_dim, task_dim)

        self.proprio_dim = proprio_dim
        self.task_dim = task_dim

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
        return {"obs": {"proprio": self.proprio_dim, "task": embed_dim}}

    def _embedder_output_dim(self) -> int:
        """Lookup of the embedder's output dim.

        Reads config only, never an instantiated module, so that
        :meth:`_get_cond_dims` remains safe to call before :meth:`configure_model`.
        """
        if self.embedder_config is None:
            return self.task_dim

        return self.embedder_config.get("output_dim")

    @torch.no_grad()
    def extract_embeddings(self, obs: torch.Tensor | dict):
        """Extracts embedder outputs for observations.

        Helper function for visualizing the embeddings.
        """
        if isinstance(obs, Mapping):
            obs = {k: v.to(self.device) for k, v in obs.items()}
        else:
            obs = obs.to(self.device)

        if self.obs_normalizer is not None:
            obs = self.obs_normalizer.normalize(obs)

        external_cond = self._build_obs_external_cond(obs)
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

        return {"obs_embeddings": obs_task_embeddings.cpu()}

    def _build_external_cond(self, obs: TensorTree) -> dict[str, TensorTree]:
        return self._build_obs_external_cond(obs)

    def _build_obs_external_cond(self, obs: TensorTree) -> dict[str, TensorTree]:
        proprio, task_embedded = self._embed_states(obs)
        return {"obs": {"proprio": proprio, "task": task_embedded}}

    def _embed_states(self, states: TensorTree) -> tuple[torch.Tensor, torch.Tensor]:
        """Splits proprio/task and embeds the task components."""
        if self.embedder is None:
            raise ValueError(
                "Embedder not initialized. Call configure_model() before using the embedder."
            )
        proprio, task = self._split_proprio_task(states)

        # Handles both a horizon window (``task`` is ``[B, T, task_dim]``, e.g. obs) and a single
        # timestep with no time axis at all (``task`` is ``[B, task_dim]``), uniformly: a missing
        # time axis is unsqueezed to ``T=1`` before embedding, then squeezed back out of the
        # result so the returned shape matches whatever was passed in.

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
            raise ValueError("Observation mapping must contain a 'proprio' key.")
        task = (
            concat_leaf_tensors(remainder, dim=-1) if isinstance(remainder, Mapping) else remainder
        )
        return proprio, task
