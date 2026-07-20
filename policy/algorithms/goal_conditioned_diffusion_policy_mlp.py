from collections.abc import Mapping
from typing import Any, cast

import torch

from policy.algorithms import GoalConditionedDiffusionPolicy
from policy.algorithms.networks import MLP
from policy.utils import get_batch_size


class GoalConditionedDiffusionPolicyMLP(GoalConditionedDiffusionPolicy):
    def __init__(
        self,
        *args,
        proprio_dim: int = 18,  # panda's qpos(9) + qvel(9), including fingers
        task_dim: int = 30,  # TCP, A, B, A-B, TCP-A, TCP-B
        hidden_dims: list[int] = [128, 128, 128],
        state_embedding_dim: int = 64,
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
            if calc_task_dim != task_dim:
                raise ValueError(
                    f"Task dimension calculated from spec ({calc_task_dim}) does not match task_dim ({task_dim})."
                )
        else:
            if not isinstance(self.obs_dim, int):
                raise ValueError(
                    f"Observation dimensionality must be an integer or dict, but got {type(self.obs_dim)}."
                )

            if proprio_dim + task_dim != self.obs_dim:
                raise ValueError(
                    f"Proprioception dimensionality ({proprio_dim}) + Task dimensionality ({task_dim}) "
                    f"do not match observation dimensionality ({self.obs_dim}). "
                    f"{proprio_dim} + {task_dim} != {self.obs_dim}."
                )

        self.proprio_dim = proprio_dim
        self.task_dim = task_dim
        self.hidden_dims = hidden_dims
        self.state_embedding_dim = state_embedding_dim

        self.state_embedder = MLP(
            input_dim=task_dim,
            output_dim=state_embedding_dim,
            hidden_dims=hidden_dims,
        )

        if self.flatten_obs:
            self.network_cond_dim = (
                self.obs_horizon * (proprio_dim + state_embedding_dim) + state_embedding_dim
            )  # (proprioception + embedded observation) for each timestep in the past + the embedded goal (no proprio)
        else:
            self.network_cond_dim = (
                proprio_dim + 2 * state_embedding_dim
            )  # (proprioception + embedded observation + embedded goal) for each timestep in sequence

    @torch.no_grad()
    def extract_embeddings(
        self,
        obs: torch.Tensor | dict,
        goal: torch.Tensor | dict | None = None,
    ):
        """Extracts MLP state embeddings for observations (and optionally a goal).

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

        if isinstance(obs, Mapping):
            task_components = [v for k, v in obs.items() if k != "proprio"]
            obs_task = torch.cat(task_components, dim=-1)
        else:
            obs_task = obs[..., self.proprio_dim :]

        obs_embeddings = self.state_embedder(obs_task)

        res = {"obs_embeddings": obs_embeddings.cpu()}
        if goal is not None:
            res["goal_embedding"] = self._prepare_goal(goal).cpu()

        return res

    def _prepare_obs(self, obs_seq: torch.Tensor | Mapping[str, Any]) -> torch.Tensor:
        """Prepares observation conditioning by embedding task states."""

        B = get_batch_size(obs_seq)

        if isinstance(obs_seq, Mapping):
            proprio = obs_seq["proprio"]
            task_components = [v for k, v in obs_seq.items() if k != "proprio"]
            task_state = torch.cat(task_components, dim=-1)
        else:
            proprio = obs_seq[..., : self.proprio_dim]
            task_state = obs_seq[..., self.proprio_dim :]

        flatten_task_state = task_state.reshape(B * self.obs_horizon, self.task_dim)
        flatten_obs_embedding = self.state_embedder(flatten_task_state)
        state_embeddings = flatten_obs_embedding.reshape(
            B, self.obs_horizon, self.state_embedding_dim
        )  # B, horizon, state_embedding_dim

        if self.flatten_obs:
            proprio_seq = proprio.reshape(
                B, self.obs_horizon * self.proprio_dim
            )  # B, horizon * proprio_dim
            flat_state_embeddings = state_embeddings.reshape(
                B, self.obs_horizon * self.state_embedding_dim
            )  # B, horizon * embedding_dim

            return torch.cat([proprio_seq, flat_state_embeddings], dim=-1)
        else:
            return torch.cat([proprio, state_embeddings], dim=-1)

    def _prepare_goal(self, goal: torch.Tensor | Mapping[str, Any]) -> torch.Tensor:
        """Prepares the goal conditioning for the network by embedding it."""
        if isinstance(goal, Mapping):
            goal_task_components = [v for k, v in goal.items() if k != "proprio"]
            goal_task_state = torch.cat(goal_task_components, dim=-1)
        else:
            goal_task_state = goal[..., self.proprio_dim :]

        return self.state_embedder(goal_task_state)  # B, embedding_dim
