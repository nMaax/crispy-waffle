from typing import Any

import torch

from policy.algorithms.diffusion_policy import DiffusionPolicy
from policy.algorithms.networks.mlp import MLP
from policy.utils import flatten_tensor_from_mapping


class GoalConditionedDiffusionPolicy(DiffusionPolicy):
    def __init__(
        self,
        *args,
        goal_dim: int = 6,
        plan_embedding_dim: int = 64,
        skip_connection: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.goal_dim = goal_dim
        self.plan_embedding_dim = plan_embedding_dim
        self.skip_connection = skip_connection

        if self.skip_connection:
            self.unet_cond_dim = plan_embedding_dim + self.obs_horizon * self.obs_dim
        else:
            self.unet_cond_dim = plan_embedding_dim

        self.trickster_input_dim = self.obs_horizon * self.obs_dim + self.goal_dim
        self.trickster_mlp = MLP(
            input_dim=self.trickster_input_dim,
            output_dim=self.plan_embedding_dim,
            hidden_dims=[256, 256, 256],
        )

    def _shared_step(self, batch: dict[str, Any], batch_idx: int, phase: str) -> torch.Tensor:
        obs_seq = batch["obs_seq"]
        act_seq = batch["act_seq"]
        goal_state = batch["goal_state"]
        if goal_state.dim() == 2:
            goal_state = goal_state.unsqueeze(1)

        flat_obs = flatten_tensor_from_mapping(obs_seq)
        flat_goal = flatten_tensor_from_mapping(goal_state)
        flat_obs_and_goal = torch.cat([flat_obs, flat_goal], dim=-1)

        plan_embedding = self.trickster_mlp(flat_obs_and_goal)

        if self.skip_connection:
            unet_cond = torch.cat([flat_obs, plan_embedding], dim=-1)
        else:
            unet_cond = plan_embedding

        loss = self._compute_loss(unet_cond, act_seq)

        self.log(f"{phase}/loss", loss, prog_bar=True, sync_dist=(phase == "val"))
        return loss

    def get_action(
        self,
        obs_seq: torch.Tensor,
        goal_state: torch.Tensor,
        num_inference_steps: int | None = None,
        clamp_range: tuple | None = None,
    ) -> torch.Tensor:
        flat_obs = flatten_tensor_from_mapping(obs_seq)
        flat_goal = flatten_tensor_from_mapping(goal_state)
        flat_obs_and_goal = torch.cat([flat_obs, flat_goal], dim=-1)

        plan_embedding = self.trickster_mlp(flat_obs_and_goal)

        if self.skip_connection:
            unet_cond = torch.cat([flat_obs, plan_embedding], dim=-1)
        else:
            unet_cond = plan_embedding

        return super().get_action(unet_cond, num_inference_steps, clamp_range)
