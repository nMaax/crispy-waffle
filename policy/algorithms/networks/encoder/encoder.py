from collections.abc import Mapping
from typing import Literal

import hydra_zen
import torch
import torch.nn as nn

from policy.algorithms.networks.encoder.spec import ConditioningContract
from policy.utils import get_subtree, get_tensor, merge_dicts
from policy.utils.typing_utils import HydraConfigFor, PoolingProtocol, TensorTree


class ConditioningEncoder(nn.Module):
    """Embeds conditioning into the tensors a downstream network consumes."""

    def __init__(
        self,
        proprio_dim: int,
        token_dim: int,
        tokens_per_step: int | None,
        goal_conditioned: bool,
        relative_goal: bool,
        decoder_type: Literal["film", "cross_attention"] = "film",
        embedder: HydraConfigFor[nn.Module] | None = None,
        pooling: HydraConfigFor[PoolingProtocol] | None = None,
    ):
        super().__init__()

        self.proprio_dim = proprio_dim
        self.token_dim = token_dim
        self.tokens_per_step = tokens_per_step

        self.goal_conditioned = goal_conditioned
        self.relative_goal = relative_goal

        self.decoder_type = decoder_type

        # Embedder
        if isinstance(embedder, nn.Module):
            self.embedder = embedder
        elif embedder is not None:
            self.embedder = hydra_zen.instantiate(embedder, input_dim=token_dim)
        else:
            self.embedder = None

        if self.embedder is not None:
            self.output_dim = self.embedder.output_dim
        else:
            self.output_dim = token_dim

        # Pooling
        if isinstance(pooling, nn.Module | PoolingProtocol):
            self.pooling = pooling
        elif pooling is not None:
            self.pooling = hydra_zen.instantiate(pooling, dim=self.output_dim)
        else:
            self.pooling = None

        # Validation and extra options
        self._validate_config()
        self.cond_dims = self._compute_cond_dims()

    def _validate_config(self) -> None:
        if self.relative_goal and not self.goal_conditioned:
            raise ValueError(
                "relative_goal=True requires goal_conditioned=True: there is no goal to "
                "difference the observations against otherwise."
            )

        if self.decoder_type == "cross_attention":
            if self.tokens_per_step is not None and self.tokens_per_step <= 1:
                raise ValueError(
                    "mode='cross_attention' requires a tokenizer with tokens_per_step > 1; "
                    f"got relative_goal={self.relative_goal!r}, tokens_per_step={self.tokens_per_step}. "
                    "Cross-attention needs a genuine per-object token sequence to attend over."
                )

        if self.decoder_type == "film" and self.tokens_per_step is None and not self.pools_objects:
            raise ValueError(
                "decoder_type='film' with a dynamic/variable number of object tokens requires "
                "pooling across objects (pools_objects=True, e.g. AttentionPooling(mode='objects'))."
            )

    def _compute_cond_dims(self) -> ConditioningContract:
        step_task_dim = (
            self.output_dim
            if self.pools_objects
            else self.output_dim * (self.tokens_per_step or 1)
        )
        goal_dim = step_task_dim if self.has_standalone_goal else 0

        if self.decoder_type == "cross_attention":
            return ConditioningContract(
                step_dim=self.proprio_dim,
                global_dim=goal_dim,
                context_dim=self.output_dim,
                context_key="context",
            )

        if self.pools_time:
            return ConditioningContract(
                step_dim=self.proprio_dim,
                global_dim=self.output_dim + goal_dim,
            )
        return ConditioningContract(
            step_dim=self.proprio_dim + step_task_dim,
            global_dim=goal_dim,
        )

    @property
    def has_standalone_goal(self) -> bool:
        """Whether conditioning carries its own goal token stream.

        A relative goal is already folded into the task tokens as a delta, so it has no separate
        entry to embed.
        """
        return self.goal_conditioned and not self.relative_goal

    @property
    def is_multi_token(self) -> bool:
        """Whether the encoder processes multiple or dynamic tokens per timestep."""
        return self.tokens_per_step is None or self.tokens_per_step > 1

    @property
    def pooling_mode(self) -> Literal["all", "objects", "time"] | None:
        return self.pooling.mode if self.pooling is not None else None

    @property
    def pools_time(self) -> bool:
        return self.pooling.pools_time if self.pooling is not None else False

    @property
    def pools_objects(self) -> bool:
        return self.pooling.pools_objects if self.pooling is not None else False

    def forward(self, tokens: Mapping[str, TensorTree]) -> dict[str, TensorTree]:
        """Embeds a ``{"obs"[, "goal"]}`` tree of ``{"proprio", "task"}`` tokens into a payload."""
        obs = get_subtree(tokens, "obs")
        proprio = get_tensor(obs, "proprio")
        task = obs["task"]

        if not isinstance(task, Mapping):
            payload = self._pack_task(proprio, self._embed_tokens(task))
        else:
            payload = self._pack_graph(proprio, task)

        if self.has_standalone_goal:
            # The goal's own proprioception never enters conditioning, only the observed history's.
            goal_task = get_tensor(get_subtree(tokens, "goal"), "task")
            payload = merge_dicts([payload, {"goal": self._embed_tokens(goal_task)}])

        self.cond_dims.validate_payload(payload)
        return payload

    def _pack_graph(
        self, proprio: torch.Tensor, task: Mapping[str, TensorTree]
    ) -> dict[str, TensorTree]:
        """Packs a graph token subtree: nodes become the attended sequence, validity its key mask.

        The embedder consumes the whole subtree (it needs the edges and the mask to attend at
        all), so this bypasses ``_embed_tokens``' single-tensor path entirely.
        """
        if self.embedder is None:
            raise ValueError(
                "A graph token subtree requires an embedder that consumes it (e.g. "
                "GraphTransformer); got embedder=None."
            )

        embedded = self.embedder(task)  # [B, T_all, K, output_dim]
        batch, time, num_slots, dim = embedded.shape
        valid = get_tensor(task, "valid").reshape(batch, time * num_slots).bool()

        return {
            "obs": {"proprio": proprio},
            self.cond_dims.context_key: embedded.reshape(batch, time * num_slots, dim),
            # key_padding_mask semantics: True marks a token to ignore.
            self.cond_dims.context_mask_key: ~valid,
        }

    def _pack_task(self, proprio: torch.Tensor, task: torch.Tensor) -> dict[str, TensorTree]:
        if self.decoder_type == "cross_attention":
            return {"obs": {"proprio": proprio}, "context": task}
        elif self.pools_time:
            return {"obs": {"proprio": proprio}, "task": task}
        else:
            return {"obs": {"proprio": proprio, "task": task}}

    def unpack_task(self, payload: Mapping[str, TensorTree]) -> torch.Tensor:
        """Recovers the embedded task stream from a payload built by :meth:`forward`."""
        if self.decoder_type == "cross_attention":
            return get_tensor(payload, self.cond_dims.context_key)
        elif self.pools_time:
            return get_tensor(payload, "task")
        else:
            return get_tensor(get_subtree(payload, "obs"), "task")

    def _embed_tokens(self, task: torch.Tensor) -> torch.Tensor:
        """Runs raw tokens through the embedder and pooling into their packed form."""
        had_no_time_axis = task.ndim == (3 if self.is_multi_token else 2)
        if had_no_time_axis:
            task = task.unsqueeze(1)

        task_embedded = self.embedder(task) if self.embedder is not None else task

        if self.pooling is not None:
            task_embedded = self.pooling(task_embedded)
        elif task_embedded.ndim == 4:
            b, t, k, d = task_embedded.shape
            # Cross-attention keeps K as a sequence axis (t-major, k-minor) for the network to
            # attend over; everything else folds it back into one wider per-timestep vector so
            # downstream FiLM conditioning keeps seeing "one vector per timestep".
            task_embedded = (
                task_embedded.reshape(b, t * k, d)
                if self.decoder_type == "cross_attention"
                else task_embedded.reshape(b, t, k * d)
            )

        # A pooling embedder already drops the time axis it was given, so there's nothing left to
        # squeeze; squeeze(1) would then operate on output_dim, which generally is not size 1
        # (thus squeeze would be a no-op); however we avoid it to keep it clean.
        if had_no_time_axis and not self.pools_time:
            task_embedded = task_embedded.squeeze(1)

        return task_embedded
