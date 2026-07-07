from typing import Any

import torch

from policy.algorithms.diffusion_policy import DiffusionPolicy
from policy.utils import flatten_tensor_from_mapping

# TODO:
# - Try to see if setting a goal that (a) is not StackCube final state, but (b) is a reasonable intermediate state
#   the DP listens to it and try to reproduce it (if it ignores it we can conclude that the goal is overall ignored, not just for OOD goal states)


class GoalConditionedDiffusionPolicyMLP(DiffusionPolicy):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.proprio_dim = 18
        self.unet_cond_dim = self.unet_cond_dim + (self.obs_dim - self.proprio_dim)

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

        flatten_obs = flatten_tensor_from_mapping(obs_seq)
        flatten_goal = flatten_tensor_from_mapping(goal)

        unet_cond = torch.cat([flatten_obs, flatten_goal[:, self.proprio_dim :]], dim=-1)

        return unet_cond
