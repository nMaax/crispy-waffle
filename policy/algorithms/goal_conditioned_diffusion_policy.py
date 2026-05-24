from typing import Any

import torch

from policy.algorithms.diffusion_policy import DiffusionPolicy
from policy.algorithms.networks.egnn import SiameseEGNNPlanner
from policy.utils import flatten_tensor_from_mapping, get_batch_size, get_device


class GoalConditionedDiffusionPolicy(DiffusionPolicy):
    def __init__(
        self,
        *args,
        graph_embedding_dim: int = 64,
        drpopout_rate: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.dropout_rate = drpopout_rate

        self.proprio_dim = 25  # XXX: dirty
        self.graph_embedding_dim = graph_embedding_dim

        self.unet_cond_dim = self.proprio_dim * self.obs_horizon + self.graph_embedding_dim * 2

        # 3 Nodes, 7 Semantic Features (3 for One-Hot ID + 4 for Quaternion)
        self.planner = SiameseEGNNPlanner(
            num_nodes=3, channels_h=7, out_dim=self.graph_embedding_dim
        )

    def _shared_step(self, batch: dict[str, Any], batch_idx: int, phase: str) -> torch.Tensor:
        obs_seq = batch["obs_seq"]
        goal_state = batch["goal_state"]
        action_seq = batch["act_seq"]

        unet_cond = self._prepare_unet_cond(obs_seq, goal_state)
        loss = self._compute_loss(unet_cond, action_seq)

        self.log(f"{phase}/loss", loss, prog_bar=True, sync_dist=(phase == "val"))
        return loss

    def get_action(
        self,
        obs_seq: torch.Tensor | dict,
        goal_state: torch.Tensor | dict,
        num_inference_steps: int | None = None,
        clamp_range: tuple | None = None,
    ):
        unet_cond = self._prepare_unet_cond(obs_seq, goal_state)
        return super().get_action(unet_cond, num_inference_steps, clamp_range)

    def _prepare_unet_cond(
        self, obs_seq: torch.Tensor | dict, goal_state: torch.Tensor | dict
    ) -> torch.Tensor:

        batch_size = get_batch_size(obs_seq)
        device = get_device(obs_seq)

        obs_seq_tensor = obs_seq["state"] if isinstance(obs_seq, dict) else obs_seq
        goal_state_tensor = goal_state["state"] if isinstance(goal_state, dict) else goal_state

        proprio_seq = obs_seq_tensor[:, :, :25]
        cubeA_seq = obs_seq_tensor[:, :, 25:32]
        cubeB_seq = obs_seq_tensor[:, :, 32:39]

        # Extract last frame for the Planner
        last_proprio = proprio_seq[:, -1, :]
        last_tcp = last_proprio[..., 18:25]
        last_cubeA = cubeA_seq[:, -1, :]
        last_cubeB = cubeB_seq[:, -1, :]
        goal_cubeA = goal_state_tensor[..., 0:7]
        goal_cubeB = goal_state_tensor[..., 7:14]

        # Coordinates (x)
        curr_coords = torch.stack(
            [last_tcp[..., 0:3], last_cubeA[..., 0:3], last_cubeB[..., 0:3]], dim=1
        )

        # Since we cannot have the goal TCP we use the current TCP position in the goal graph as well
        goal_coords = torch.stack(
            [last_tcp[..., 0:3], goal_cubeA[..., 0:3], goal_cubeB[..., 0:3]], dim=1
        )

        # Semantic Features (h)
        # ID Tags to tell the EGNN what it's looking at
        id_tcp = torch.tensor([1, 0, 0], device=device).expand(batch_size, -1)
        id_A = torch.tensor([0, 1, 0], device=device).expand(batch_size, -1)
        id_B = torch.tensor([0, 0, 1], device=device).expand(batch_size, -1)

        # Quaternions
        quat_tcp = last_tcp[..., 3:7]

        curr_feats = torch.stack(
            [
                torch.cat([id_tcp, quat_tcp], dim=-1),
                torch.cat([id_A, last_cubeA[..., 3:7]], dim=-1),
                torch.cat([id_B, last_cubeB[..., 3:7]], dim=-1),
            ],
            dim=1,
        )

        goal_feats = torch.stack(
            [
                torch.cat([id_tcp, quat_tcp], dim=-1),
                torch.cat([id_A, goal_cubeA[..., 3:7]], dim=-1),
                torch.cat([id_B, goal_cubeB[..., 3:7]], dim=-1),
            ],
            dim=1,
        )

        # --- EQUIVARIANT EMBEDDINGS ---
        curr_embedding = self.planner(curr_coords, curr_feats)
        goal_embedding = self.planner(goal_coords, goal_feats)

        plan_embedding = torch.cat([goal_embedding, curr_embedding], dim=-1)

        # --- U-NET COND ---
        flat_proprio = flatten_tensor_from_mapping(proprio_seq)

        # cubes_seq = torch.cat([cubeA_seq, cubeB_seq], dim=-1)
        # flat_cubes = flatten_tensor_from_mapping(cubes_seq)

        # if self.training and torch.rand(1).item() < self.dropout_rate:
        #     flat_cubes = torch.zeros_like(flat_cubes)

        unet_cond = torch.cat([flat_proprio, plan_embedding], dim=-1)
        return unet_cond
