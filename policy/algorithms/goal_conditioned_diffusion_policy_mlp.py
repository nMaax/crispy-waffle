from typing import Any

import torch

from policy.algorithms.diffusion_policy import DiffusionPolicy
from policy.algorithms.networks import MLP
from policy.utils import flatten_tensor_from_mapping


class GoalConditionedDiffusionPolicyMLP(DiffusionPolicy):
    def __init__(
        self,
        *args,
        plan_embedding_dim: int = 64,
        # dropout_rate: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # self.dropout_rate = dropout_rate

        # XXX: dirty, as well as the overall idea of hardcoding dimensions
        self.proprio_dim = 18
        self.tcp_dim = 7
        self.plan_embedding_dim = plan_embedding_dim

        self.unet_cond_dim = (
            self.proprio_dim + self.tcp_dim
        ) * self.obs_horizon + self.plan_embedding_dim

        # 3 Nodes, 7 Semantic Features (3 for One-Hot ID + 4 for Quaternion)
        self.planner = MLP(
            input_dim=37,
            output_dim=self.plan_embedding_dim,
            hidden_dims=[256, 256, 256],
        )

    def _shared_step(self, batch: dict[str, Any], batch_idx: int, phase: str) -> torch.Tensor:
        obs_seq = batch["obs_seq"]
        goal_state = batch["goal"]
        action_seq = batch["act_seq"]

        unet_cond = self._prepare_unet_cond(obs_seq, goal_state)
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
        unet_cond = self._prepare_unet_cond(obs_seq, goal)
        return super().get_action(unet_cond, num_inference_steps, clamp_range)

    def _prepare_unet_cond(
        self, obs_seq: torch.Tensor | dict, goal: torch.Tensor | dict
    ) -> torch.Tensor:

        obs_seq_tensor = obs_seq["state"] if isinstance(obs_seq, dict) else obs_seq
        goal_state_tensor = goal["state"] if isinstance(goal, dict) else goal

        proprio_seq = obs_seq_tensor[:, :, 0:18]
        tcp_seq = obs_seq_tensor[:, :, 18:25]
        cubeA_seq = obs_seq_tensor[:, :, 25:32]
        cubeB_seq = obs_seq_tensor[:, :, 32:39]

        # Extract last frame for the Planner
        last_tcp = tcp_seq[..., -1, :]
        last_cubeA = cubeA_seq[..., -1, :]
        last_cubeB = cubeB_seq[..., -1, :]

        # We ignore proprio as it is not reasonably craftable during simulations
        goal_tcp = goal_state_tensor[..., 18:25]
        goal_cubeA = goal_state_tensor[..., 25:32]
        goal_cubeB = goal_state_tensor[..., 32:39]

        last_cubeA_rel_pos = last_cubeA[..., 0:3] - last_tcp[..., 0:3]
        last_cubeA_quat = last_cubeA[..., 3:7]
        last_cubeB_rel_pos = last_cubeB[..., 0:3] - last_tcp[..., 0:3]
        last_cubeB_quat = last_cubeB[..., 3:7]

        goal_cubeA_rel_pos = goal_cubeA[..., 0:3] - goal_tcp[..., 0:3]
        goal_cubeA_quat = goal_cubeA[..., 3:7]
        goal_cubeB_rel_pos = goal_cubeB[..., 0:3] - goal_tcp[..., 0:3]
        goal_cubeB_quat = goal_cubeB[..., 3:7]

        delta_tcp = goal_tcp[..., 0:3] - last_tcp[..., 0:3]
        delta_A = goal_cubeA[..., 0:3] - last_cubeA[..., 0:3]
        delta_B = goal_cubeB[..., 0:3] - last_cubeB[..., 0:3]

        flat_planner_input = torch.cat(
            [
                last_cubeA_rel_pos,
                last_cubeA_quat,
                last_cubeB_rel_pos,
                last_cubeB_quat,
                goal_cubeA_rel_pos,
                goal_cubeA_quat,
                goal_cubeB_rel_pos,
                goal_cubeB_quat,
                delta_tcp,
                delta_A,
                delta_B,
            ],
            dim=-1,
        )

        # EMBEDDING from MLP
        plan_embedding = self.planner(flat_planner_input)

        # DIRECT INFO FOR UNET
        flat_proprio = flatten_tensor_from_mapping(proprio_seq)
        flat_tcp = flatten_tensor_from_mapping(tcp_seq)

        # XXX: skip absolute cubes positions, however we risk the unet ignores
        # them, so we need to dropout them sometimes
        # ignore for now...

        # cubes_seq = torch.cat([cubeA_seq, cubeB_seq], dim=-1)
        # flat_cubes = flatten_tensor_from_mapping(cubes_seq)

        # if self.training and torch.rand(1).item() < self.dropout_rate:
        #     flat_cubes = torch.zeros_like(flat_cubes)

        unet_cond = torch.cat([flat_proprio, flat_tcp, plan_embedding], dim=-1)
        return unet_cond
