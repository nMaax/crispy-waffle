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
        dropout_rate: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.proprio_dim = 25  # TODO: this should be defined in a more graceful way, maybe by looking at the dataset or something
        self.plan_embedding_dim = plan_embedding_dim
        self.dropout_rate = dropout_rate
        self.unet_cond_dim = (
            self.proprio_dim * self.obs_horizon + 14 * self.obs_horizon + self.plan_embedding_dim
        )

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
        act_seq = batch["act_seq"]

        unet_cond = self._prepare_unet_cond(obs_seq, goal_state)
        loss = self._compute_loss(unet_cond, act_seq)

        self.log(f"{phase}/loss", loss, prog_bar=True, sync_dist=(phase == "val"))
        return loss

    def get_action(
        self,
        obs_seq: torch.Tensor | dict,
        goal_state: torch.Tensor | dict,
        num_inference_steps: int | None = None,
        clamp_range: tuple | None = None,
    ) -> torch.Tensor:

        unet_cond = self._prepare_unet_cond(obs_seq, goal_state)
        return super().get_action(unet_cond, num_inference_steps, clamp_range)

    def _prepare_unet_cond(
        self, obs_seq: torch.Tensor | dict, goal_state: torch.Tensor | dict
    ) -> torch.Tensor:

        if isinstance(obs_seq, dict):
            obs_seq_tensor = obs_seq["state"]
        else:
            obs_seq_tensor = obs_seq

        if isinstance(goal_state, dict):
            goal_state_tensor = goal_state["state"]
        else:
            goal_state_tensor = goal_state

        proprio_seq = obs_seq_tensor[:, :, :25]
        cubeA_seq = obs_seq_tensor[:, :, 25:32]
        cubeB_seq = obs_seq_tensor[:, :, 32:39]

        last_proprio = proprio_seq[:, -1, :]
        last_cubeA = cubeA_seq[:, -1, :]
        last_cubeB = cubeB_seq[:, -1, :]

        last_tcp_pos = last_proprio[..., 0:3]

        last_cubeA_rel_pos = last_cubeA[..., 0:3] - last_tcp_pos
        last_cubeA_quat = last_cubeA[..., 3:7]
        last_cubeB_rel_pos = last_cubeB[..., 0:3] - last_tcp_pos
        last_cubeB_quat = last_cubeB[..., 3:7]

        goal_cubeA = goal_state_tensor[..., 0:7]
        goal_cubeB = goal_state_tensor[..., 7:14]

        delta_A = goal_cubeA[..., 0:3] - last_cubeA[..., 0:3]
        delta_B = goal_cubeB[..., 0:3] - last_cubeB[..., 0:3]

        flat_planner_input = torch.cat(
            [
                last_cubeA_rel_pos,
                last_cubeA_quat,
                last_cubeB_rel_pos,
                last_cubeB_quat,
                delta_A,
                delta_B,
            ],
            dim=-1,
        )

        plan_embedding = self.planner(flat_planner_input)

        flat_proprio = flatten_tensor_from_mapping(proprio_seq)

        cubes_seq = torch.cat([cubeA_seq, cubeB_seq], dim=-1)
        flat_cubes = flatten_tensor_from_mapping(cubes_seq)

        if self.training and torch.rand(1).item() < self.dropout_rate:
            flat_cubes = torch.zeros_like(flat_cubes)

        unet_cond = torch.cat([flat_proprio, flat_cubes, plan_embedding], dim=-1)

        return unet_cond
