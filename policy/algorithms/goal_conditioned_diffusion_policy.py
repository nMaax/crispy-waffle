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
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.proprio_dim = 25  # TODO: this should be defined in a more graceful way, maybe by looking at the dataset or something
        self.plan_embedding_dim = plan_embedding_dim
        self.unet_cond_dim = self.proprio_dim * self.obs_horizon + self.plan_embedding_dim

        # TODO: like down below, this should be defined in a more graceful way
        self.planner_input_dim = 20
        self.planner = MLP(
            input_dim=self.planner_input_dim,
            output_dim=self.plan_embedding_dim,
            hidden_dims=[256, 256, 256],
        )

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

    def _prepare_unet_cond(
        self, obs_seq: torch.Tensor | dict, goal_state: torch.Tensor
    ) -> torch.Tensor:

        # TODO: kind of dirty to use a slice directly on obs_seq, we should not assume it to be a tensor
        flat_proprio_and_TCP = flatten_tensor_from_mapping(obs_seq[:, :, :25])
        flat_last_obs = flatten_tensor_from_mapping(obs_seq[:, -1], device=self.device)
        flat_goal_state = flatten_tensor_from_mapping(goal_state, device=self.device)

        #  TODO: this assumes to use canonicalized observations, kinda dirty, should be handled more gracefully
        last_proprio_and_TCP = flat_last_obs[..., :25]
        curr_cubeA = flat_last_obs[..., 25:32]
        curr_cubeB = flat_last_obs[..., 32:39]

        goal_cubeA = flat_goal_state[..., 25:32]
        goal_cubeB = flat_goal_state[..., 32:39]

        last_tcp_pos = last_proprio_and_TCP[..., 0:3]

        curr_cubeA_rel_pos = curr_cubeA[..., 0:3] - last_tcp_pos
        curr_cubeA_quat = curr_cubeA[..., 3:7]

        curr_cubeB_rel_pos = curr_cubeB[..., 0:3] - last_tcp_pos
        curr_cubeB_quat = curr_cubeB[..., 3:7]

        delta_A = goal_cubeA[..., 0:3] - curr_cubeA[..., 0:3]
        delta_B = goal_cubeB[..., 0:3] - curr_cubeB[..., 0:3]

        flat_planner_input = torch.cat(
            [
                curr_cubeA_rel_pos,
                curr_cubeA_quat,
                curr_cubeB_rel_pos,
                curr_cubeB_quat,
                delta_A,
                delta_B,
            ],
            dim=-1,
        )
        plan_embedding = self.planner(flat_planner_input)

        unet_cond = torch.cat([flat_proprio_and_TCP, plan_embedding], dim=-1)
        return unet_cond
