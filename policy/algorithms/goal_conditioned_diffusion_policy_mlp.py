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
        proprio_dim: int = 18,  # joint qpos and qvels, including fingers
        state_embedder_input_dim: int = 30,  # TCP, A, B, A-B, TCP-A, TCP-B
        state_embedder_hidden_dims: list[int] = [128, 128, 128],
        state_embedding_dim: int = 64,
        planner_hidden_dims: list[int] = [256, 256],
        planner_output_dim: int = 128,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.proprio_dim = proprio_dim
        self.state_embedder_input_dim = state_embedder_input_dim
        self.state_embedding_dim = state_embedding_dim
        self.state_embedder_hidden_dims = state_embedder_hidden_dims

        self.state_embedder = MLP(
            input_dim=state_embedder_input_dim,
            output_dim=state_embedding_dim,
            hidden_dims=state_embedder_hidden_dims,
        )

        planner_input_dim = state_embedding_dim * (self.obs_horizon + 1)  # +1 for the goal state

        self.planner_input_dim = planner_input_dim
        self.planner_output_dim = planner_output_dim
        self.planner_hidden_dims = planner_hidden_dims

        self.planner = MLP(
            input_dim=planner_input_dim,
            output_dim=planner_output_dim,
            hidden_dims=planner_hidden_dims,
        )

        self.unet_cond_dim = proprio_dim * self.obs_horizon + planner_output_dim

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
        goal_tensor = goal["state"] if isinstance(goal, dict) else goal

        B = obs_seq_tensor.shape[0]

        # EMBEDDINGs from MLP
        flatten_obs_seq_tensor = obs_seq_tensor.view(B * self.obs_horizon, -1)
        flatten_obs_embedding = self.state_embedder(flatten_obs_seq_tensor[:, self.proprio_dim :])
        obs_seq_embeddings = flatten_obs_embedding.view(B, self.obs_horizon, -1)

        goal_embedding = self.state_embedder(goal_tensor[..., self.proprio_dim :]).unsqueeze(1)

        embeddings_seq = torch.cat([obs_seq_embeddings, goal_embedding], dim=1)

        plan_embedding = self.planner(embeddings_seq.reshape(B, -1))

        # DIRECT INFO FOR UNET
        proprio_seq = obs_seq_tensor[:, :, : self.proprio_dim].reshape(B, -1)

        unet_cond = torch.cat([proprio_seq, plan_embedding], dim=-1)

        return unet_cond
