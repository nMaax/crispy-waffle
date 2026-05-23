from typing import Any

import torch

from policy.algorithms.diffusion_policy import DiffusionPolicy
from policy.algorithms.networks.mlp import MLP
from policy.utils import flatten_tensor_from_mapping


class GoalConditionedDiffusionPolicy(DiffusionPolicy):
    def __init__(
        self,
        *args,
        plan_embedding_dim: int = 64,
        abs_plan_cond: bool = True,
        history_dropout_prob: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.plan_embedding_dim = plan_embedding_dim
        self.unet_cond_dim = self.obs_horizon * self.obs_dim + self.plan_embedding_dim

        # If abs_plan: 6 (curr_A, curr_B) + 6 (goal_A, goal_B) = 12
        # If relative: 3 (curr_B - curr_A) + 3 (goal_B - goal_A) = 6
        self.planner_input_dim = 12 if abs_plan_cond else 6
        self.planner = MLP(
            input_dim=self.planner_input_dim,
            output_dim=self.plan_embedding_dim,
            hidden_dims=[256, 256, 256],
        )
        self.abs_plan_cond = abs_plan_cond
        self.history_dropout_prob = history_dropout_prob

    def _shared_step(self, batch: dict[str, Any], batch_idx: int, phase: str) -> torch.Tensor:
        obs_seq = batch["obs_seq"]
        goal_state = batch["goal_state"]
        if goal_state.dim() == 2:
            goal_state = goal_state.unsqueeze(1)
        act_seq = batch["act_seq"]

        unet_cond = self._prepare_unet_cond(obs_seq, goal_state)
        loss = self._compute_loss(unet_cond, act_seq)

        self.log(f"{phase}/loss", loss, prog_bar=True, sync_dist=(phase == "val"))
        return loss

    def get_action(
        self,
        obs_seq: torch.Tensor | dict,
        goal_state: torch.Tensor,
        num_inference_steps: int | None = None,
        clamp_range: tuple | None = None,
    ) -> torch.Tensor:

        unet_cond = self._prepare_unet_cond(obs_seq, goal_state)
        return super().get_action(unet_cond, num_inference_steps, clamp_range)

    def _prepare_unet_cond(self, obs_seq: torch.Tensor, goal_state: torch.Tensor) -> torch.Tensor:
        # Extract the current frame
        flat_last_obs = flatten_tensor_from_mapping(obs_seq[:, -1])
        flat_goal_state = flatten_tensor_from_mapping(goal_state)

        #  Extract ONLY the specific cube positions to feed the planner
        #  TODO: this assumes to use canonicalized observations, kinda dirty
        curr_cubeA = flat_last_obs[..., 25:28]
        curr_cubeB = flat_last_obs[..., 32:35]

        goal_cubeA = flat_goal_state[..., 25:28]
        goal_cubeB = flat_goal_state[..., 32:35]

        if self.abs_plan_cond:
            curr_state_for_planner = torch.cat([curr_cubeA, curr_cubeB], dim=-1)
            goal_state_for_planner = torch.cat([goal_cubeA, goal_cubeB], dim=-1)
        else:
            curr_state_for_planner = curr_cubeB - curr_cubeA
            goal_state_for_planner = goal_cubeB - goal_cubeA

        flat_planner_input = torch.cat([curr_state_for_planner, goal_state_for_planner], dim=-1)
        plan_embedding = self.planner(flat_planner_input)

        # History dropout
        flat_obs = flatten_tensor_from_mapping(obs_seq)
        flat_obs_for_unet = flat_obs.clone()

        # Only apply masking during training to force the U-Net to rely on the plan
        if self.training:
            if torch.rand(1).item() < self.history_dropout_prob:
                history_size = (self.obs_horizon - 1) * self.obs_dim

                # Zero out everything except the most recent observation frame
                if history_size > 0:
                    flat_obs_for_unet[:, :history_size] = 0.0

                # Also zero out the cubes positions in the final frame
                # so the U-Net only has proprioception and the plan embedding!
                flat_obs_for_unet[:, history_size + 25 : history_size + 28] = 0.0
                flat_obs_for_unet[:, history_size + 32 : history_size + 35] = 0.0

        unet_cond = torch.cat([flat_obs_for_unet, plan_embedding], dim=-1)
        return unet_cond
