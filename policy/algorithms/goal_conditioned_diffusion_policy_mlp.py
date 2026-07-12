from typing import Any

import torch

from policy.algorithms.diffusion_policy import DiffusionPolicy
from policy.algorithms.networks import MLP
from policy.utils import get_batch_size

# TODO: make this work also for non-unet architectures, like done in DiffusionPolicy
# TODO: make this work also for dictionary observations (see type: ignore, isinstance() checks and typing in method signatures)


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

        if "ConditionalUnet" not in self.network_config.get("_target_", None):
            raise ValueError(
                f"BesoPolicy requires a ConditoinalUnet architecture for the diffusion model, but got {self.network_config.get('_target_')}."
            )

        if not isinstance(self.obs_dim, int):
            raise ValueError(
                f"Observation dimensionality must be an integer, indicating non-dictionary will be served. But got {type(self.obs_dim)}."
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
        num_inference_steps: int | None = None,
        clamp_range: tuple | None = None,
    ):
        """Runs the reverse diffusion process to predict an action sequence from the current
        observation.

        Shapes:
            obs_seq: [B, obs_horizon * obs_dim] (flattened conditioning)
            goal: [B, obs_dim] (flattened conditioning)
            returns: [B, act_horizon, act_dim] (denoised actions to execute)
        """

        if not isinstance(obs_seq, torch.Tensor):
            raise ValueError(
                f"Expected batch['obs_seq'] to be a torch.Tensor, but got {type(obs_seq)}."
            )

        if not isinstance(goal, torch.Tensor):
            raise ValueError(f"Expected batch['goal'] to be a torch.Tensor, but got {type(goal)}.")

        if self.normalize:
            obs_seq = self.normalizer.normalize(obs_seq)
            goal = self.normalizer.normalize(goal)

        # network_cond: B, horizon * (proprio_dim + embedding_dim) + embedding_dim
        network_cond = self._prepare_network_cond(obs_seq, goal)

        return self._run_diffusion_loop(network_cond, num_inference_steps, clamp_range)

    def _shared_step(self, batch: dict[str, Any], batch_idx: int, phase: str) -> torch.Tensor:
        """Main step logic, it doesn't differ between training and validation except for the
        logging.

        Shapes:
            batch["obs_seq"]: [B, obs_horizon, obs_dim]
            batch["goal"]: [B, obs_dim]
            batch["act_seq"]: [B, pred_horizon, act_dim]
            returns: scalar loss tensor []
        """

        obs_seq = batch["obs_seq"]
        goal = batch["goal"]

        if not isinstance(obs_seq, torch.Tensor):
            raise ValueError(
                f"Expected batch['obs_seq'] to be a torch.Tensor, but got {type(obs_seq)}."
            )

        if not isinstance(goal, torch.Tensor):
            raise ValueError(f"Expected batch['goal'] to be a torch.Tensor, but got {type(goal)}.")

        if self.normalize:
            obs_seq = self.normalizer.normalize(obs_seq)
            goal = self.normalizer.normalize(goal)

        action_seq = batch["act_seq"]
        network_cond = self._prepare_network_cond(obs_seq, goal)

        # network_cond: B, horizon * (proprio_dim + embedding_dim) + embedding_dim
        loss = self._compute_loss(network_cond, action_seq)

        self.log(f"{phase}/loss", loss, prog_bar=True, sync_dist=(phase == "val"))
        return loss

    def _prepare_network_cond(self, obs_seq: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Prepares the conditioning for the diffusion model by embedding the observations and
        goal."""

        B = get_batch_size(obs_seq)

        proprio_seq = obs_seq[:, :, : self.proprio_dim].reshape(
            B, self.obs_horizon * self.proprio_dim
        )  # B, horizon * proprio_dim

        flatten_obs_seq = obs_seq.reshape(B * self.obs_horizon, self.obs_dim)  # type: ignore
        flatten_obs_embedding = self.state_embedder(flatten_obs_seq[:, self.proprio_dim :])
        state_embeddings = flatten_obs_embedding.reshape(
            B, self.obs_horizon * self.state_embedding_dim
        )  # B, horizon * embedding_dim

        goal_embedding = self.state_embedder(goal[..., self.proprio_dim :])  # B, embedding_dim

        # Concatenate all together
        network_cond = torch.cat(
            [proprio_seq, state_embeddings, goal_embedding], dim=-1
        )  # B, horizon * (proprio_dim + embedding_dim) + embedding_dim

        return network_cond
