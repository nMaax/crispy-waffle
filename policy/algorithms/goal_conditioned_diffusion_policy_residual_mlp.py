from typing import Any

import torch

from policy.algorithms.diffusion_policy import DiffusionPolicy
from policy.algorithms.networks import ResidualMLP
from policy.utils import flatten_tensor_from_mapping, get_batch_size

# TODO:
# - Try to see if setting a goal that (a) is not StackCube final state, but (b) is a reasonable intermediate state
#   the DP listens to it and try to reproduce it (if it ignores it we can conclude that the goal is overall ignored, not just for OOD goal states)


class GoalConditionedDiffusionPolicyResidualMLP(DiffusionPolicy):
    def __init__(
        self,
        *args,
        # TODO: should manage more cleanly these hardcoded dimensions,
        # while it is true that I will only use canonicalized states
        # from now, they still should be managed via some parameters or
        # automatic inferenced from the dataset
        proprio_dim: int = 18,  # qpos, qvel, including fingers
        task_dim: int = 30,  # TCP, A, B, A-B, TCP-A, TCP-B
        hidden_dims: list[int] = [128, 128, 128],
        state_embedding_dim: int = 64,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.proprio_dim = proprio_dim
        self.task_dim = task_dim
        self.hidden_dims = hidden_dims
        self.state_embedding_dim = state_embedding_dim

        self.unet_cond_dim = (
            self.obs_horizon * (proprio_dim + state_embedding_dim) + state_embedding_dim
        )  # proprio + embedded observation for each timestep in the horizon + the embedded goal

        # TODO: should not hard-code a MLP here, it should be set via parameters / hydra
        self.state_embedder = ResidualMLP(
            input_dim=task_dim,
            output_dim=state_embedding_dim,
            hidden_dims=hidden_dims,
        )

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

        action_seq = batch["act_seq"]

        unet_cond = self._prepare_unet_cond(
            obs_seq, goal
        )  # B, horizon * (proprio_dim + embedding_dim) + embedding_dim

        loss = self._compute_loss(unet_cond, action_seq)

        self.log(f"{phase}/loss", loss, prog_bar=True, sync_dist=(phase == "val"))
        return loss

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
        if isinstance(obs_seq, dict):
            obs_seq = flatten_tensor_from_mapping(obs_seq)

        if isinstance(goal, dict):
            goal = flatten_tensor_from_mapping(goal)

        unet_cond = self._prepare_unet_cond(
            obs_seq, goal
        )  # B, horizon * (proprio_dim + embedding_dim) + embedding_dim

        return super().get_action(unet_cond, num_inference_steps, clamp_range)

    def _prepare_unet_cond(self, obs_seq: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:

        B = get_batch_size(obs_seq)

        # DIRECT INPUT FOR DIFFUSION POLICY
        proprio_seq = obs_seq[:, :, : self.proprio_dim].reshape(
            B, self.obs_horizon * self.proprio_dim
        )  # B, horizon * 18

        # OBSERVATION TO EMBED VIA MLP
        flatten_obs_seq = obs_seq.view(B * self.obs_horizon, self.obs_dim)
        flatten_obs_embedding = self.state_embedder(flatten_obs_seq[:, self.proprio_dim :])
        state_embeddings = flatten_obs_embedding.view(
            B, self.obs_horizon * self.state_embedding_dim
        )  # B, horizon * embedding_dim

        # GOAL TO EMBED VIA MLP
        goal_embedding = self.state_embedder(goal[..., self.proprio_dim :])

        # Concatenate all together
        unet_cond = torch.cat(
            [proprio_seq, state_embeddings, goal_embedding], dim=-1
        )  # B, horizon * (proprio_dim + embedding_dim) + embedding_dim

        return unet_cond
