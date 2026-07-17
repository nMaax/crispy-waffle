from collections.abc import Mapping
from typing import Any

import torch

from policy.algorithms.diffusion_policy import DiffusionPolicy
from policy.algorithms.networks import MLP
from policy.utils import get_batch_size


# TODO: make this work also for non-unet architectures, like done in DiffusionPolicy
class GoalConditionedDiffusionPolicyMLP(DiffusionPolicy):
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

        if "ConditionalUnet" not in self.network_config.get("_target_", None):
            raise ValueError(
                f"BesoPolicy requires a ConditionalUnet architecture for the diffusion model, but got {self.network_config.get('_target_')}."
            )

        if isinstance(self.obs_dim, Mapping):
            if "proprio" not in self.obs_dim:
                raise ValueError("Observation dictionary spec must contain 'proprio' key.")
            if self.obs_dim["proprio"] != proprio_dim:
                raise ValueError(
                    f"Proprioception dimension in spec ({self.obs_dim['proprio']}) does not match proprio_dim ({proprio_dim})."
                )

            calc_task_dim = sum(v for k, v in self.obs_dim.items() if k != "proprio")
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

        self.network_cond_dim = (
            self.obs_horizon * (proprio_dim + state_embedding_dim) + state_embedding_dim
        )  # (proprioception + embedded observation) for each timestep in the past + the embedded goal (no proprio)

    def get_action(
        self,
        obs_seq: torch.Tensor | dict,
        goal: torch.Tensor | dict,
        num_inference_timesteps: int | None = None,
        output_clip_range: tuple | None = None,
    ):
        """Runs the reverse diffusion process to predict an action sequence from the current
        observation.

        Shapes:
            obs_seq: [B, obs_horizon * obs_dim] or dict
            goal: [B, obs_dim] or dict
            returns: [B, act_horizon, act_dim] (denoised actions to execute)
        """
        if self.normalizer is not None:
            obs_seq = self.normalizer.normalize(obs_seq)
            goal = self.normalizer.normalize(goal)

        # network_cond: B, horizon * (proprio_dim + embedding_dim) + embedding_dim
        network_cond = self._prepare_network_cond(obs_seq, goal)

        return self._run_diffusion_loop(network_cond, num_inference_timesteps, output_clip_range)

    @torch.no_grad()
    def extract_embeddings(
        self,
        obs: torch.Tensor | dict,
        goal: torch.Tensor | dict | None = None,
    ):
        """Extracts MLP state embeddings for observations (and optionally a goal)."""
        if isinstance(obs, Mapping):
            obs = {k: v.to(self.device) for k, v in obs.items()}
        else:
            obs = obs.to(self.device)

        if goal is not None:
            if isinstance(goal, Mapping):
                goal = {k: v.to(self.device) for k, v in goal.items()}
            else:
                goal = goal.to(self.device)

        if self.normalizer is not None:
            obs = self.normalizer.normalize(obs)
            if goal is not None:
                goal = self.normalizer.normalize(goal)

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

    def _prepare_goal(self, goal: torch.Tensor | Mapping[str, Any]) -> torch.Tensor:
        """Prepares the goal conditioning for the network by embedding it."""
        if isinstance(goal, Mapping):
            goal_task_components = [v for k, v in goal.items() if k != "proprio"]
            goal_task_state = torch.cat(goal_task_components, dim=-1)
        else:
            goal_task_state = goal[..., self.proprio_dim :]

        return self.state_embedder(goal_task_state)  # B, embedding_dim

    def _shared_step(self, batch: dict[str, Any], batch_idx: int, phase: str) -> torch.Tensor:
        """Main step logic, it doesn't differ between training and validation except for the
        logging.

        Shapes:
            batch["obs_seq"]: [B, obs_horizon, obs_dim] or dict
            batch["goal"]: [B, obs_dim] or dict
            batch["act_seq"]: [B, pred_horizon, act_dim]
            returns: scalar loss tensor []
        """

        obs_seq = batch["obs_seq"]
        goal = batch["goal"]

        if not isinstance(obs_seq, torch.Tensor | Mapping):
            raise ValueError(
                f"Expected batch['obs_seq'] to be a torch.Tensor or Mapping, but got {type(obs_seq)}."
            )

        if not isinstance(goal, torch.Tensor | Mapping):
            raise ValueError(
                f"Expected batch['goal'] to be a torch.Tensor or Mapping, but got {type(goal)}."
            )

        if self.normalizer is not None:
            obs_seq = self.normalizer.normalize(obs_seq)
            goal = self.normalizer.normalize(goal)

        action_seq = batch["act_seq"]
        if self.action_normalizer is not None:
            action_seq = self.action_normalizer.normalize(action_seq)
        network_cond = self._prepare_network_cond(obs_seq, goal)

        # network_cond: B, horizon * (proprio_dim + embedding_dim) + embedding_dim
        loss = self._compute_loss(network_cond, action_seq)

        self.log(f"{phase}/loss", loss, prog_bar=True, sync_dist=(phase == "val"))
        return loss

    def _prepare_network_cond(
        self, obs_seq: torch.Tensor | Mapping[str, Any], goal: torch.Tensor | Mapping[str, Any]
    ) -> torch.Tensor:
        """Prepares the conditioning for the diffusion model by embedding the observations and
        goal."""

        B = get_batch_size(obs_seq)

        if isinstance(obs_seq, Mapping):
            proprio = obs_seq["proprio"]
            task_components = [v for k, v in obs_seq.items() if k != "proprio"]
            task_state = torch.cat(task_components, dim=-1)
        else:
            proprio = obs_seq[..., : self.proprio_dim]
            task_state = obs_seq[..., self.proprio_dim :]

        proprio_seq = proprio.reshape(
            B, self.obs_horizon * self.proprio_dim
        )  # B, horizon * proprio_dim

        flatten_task_state = task_state.reshape(B * self.obs_horizon, self.task_dim)
        flatten_obs_embedding = self.state_embedder(flatten_task_state)
        state_embeddings = flatten_obs_embedding.reshape(
            B, self.obs_horizon * self.state_embedding_dim
        )  # B, horizon * embedding_dim

        goal_embedding = self._prepare_goal(goal)  # B, embedding_dim

        # Concatenate all together
        network_cond = torch.cat(
            [proprio_seq, state_embeddings, goal_embedding], dim=-1
        )  # B, horizon * (proprio_dim + embedding_dim) + embedding_dim

        return network_cond
