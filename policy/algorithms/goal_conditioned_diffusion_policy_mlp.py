from typing import Any

import torch

from policy.algorithms.diffusion_policy import DiffusionPolicy
from policy.algorithms.networks import MLP
from policy.utils import flatten_tensor_from_mapping

# TODO:
# - Try to see if setting a goal that (a) is not StackCube final state, but (b) is a reasonable intermediate state
#   the DP listens to it and try to reproduce it (if it ignores it we can conclude that the goal is overall ignored, not just for OOD goal states)


class GoalConditionedDiffusionPolicyMLP(DiffusionPolicy):
    def __init__(
        self,
        *args,
        # TODO: should manage more cleanly these hardcoded dimensions,
        # while it is true that I will only use canonicalized states
        # from now, they still should be managed via some parameters or
        # automatic inferenced from the dataset
        proprio_dim: int = 18,
        planner_input_dim: int = 7 + 7 + 7,  # TCP, A, B Poses
        hidden_dims: list[int] = [128, 128, 128],
        state_embedding_dim: int = 64,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.unet_cond_dim = (proprio_dim + state_embedding_dim) * (
            self.obs_horizon + 1
        )  # +1 for the goal state

        # TODO: should not hard-code a MLP here, it should be set via parameters / hydra
        self.planner = MLP(
            input_dim=planner_input_dim,
            output_dim=state_embedding_dim,
            hidden_dims=hidden_dims,
        )

    def _shared_step(self, batch: dict[str, Any], batch_idx: int, phase: str) -> torch.Tensor:
        obs_seq = batch["obs_seq"]
        goal_state = batch["goal"]
        action_seq = batch["act_seq"]

        unet_cond = self._prepare_unet_cond(obs_seq, goal_state)
        flatten_unet_cond = flatten_tensor_from_mapping(unet_cond)
        loss = self._compute_loss(flatten_unet_cond, action_seq)

        self.log(f"{phase}/loss", loss, prog_bar=True, sync_dist=(phase == "val"))
        return loss

    def get_action(
        self,
        obs_seq: torch.Tensor | dict,
        goal: torch.Tensor | dict,
        num_inference_steps: int | None = None,
        clamp_range: tuple | None = None,
    ):
        unet_cond = self._prepare_unet_cond(obs_seq, goal)
        return super().get_action(unet_cond, num_inference_steps, clamp_range)

    def _prepare_unet_cond(
        self, obs_seq: torch.Tensor | dict, goal: torch.Tensor | dict
    ) -> torch.Tensor:

        obs_seq_tensor = obs_seq["state"] if isinstance(obs_seq, dict) else obs_seq
        goal_state_tensor = goal["state"] if isinstance(goal, dict) else goal

        # EMBEDDINGs from MLP
        flatten_obs_seq_tensor = obs_seq_tensor.view(
            obs_seq_tensor.shape[0] * self.obs_horizon, -1
        )
        flatten_obs_embedding = self.planner(flatten_obs_seq_tensor[:, 18:])
        obs_embedding = flatten_obs_embedding.view(obs_seq_tensor.shape[0], self.obs_horizon, -1)

        goal_embedding = self.planner(goal_state_tensor[..., 18:]).unsqueeze(1)

        embeddings_seq = torch.cat([obs_embedding, goal_embedding], dim=1)

        # DIRECT INFO FOR UNET
        proprio_seq = obs_seq_tensor[:, :, 0:18]

        # For the goal we just craft a zero vector
        proprio_seq = torch.cat([proprio_seq, torch.zeros_like(proprio_seq[:, 0:1, :])], dim=1)

        unet_cond = torch.cat([proprio_seq, embeddings_seq], dim=-1)

        return unet_cond
