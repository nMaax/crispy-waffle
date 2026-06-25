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
        self.unet_cond_dim = self.unet_cond_dim + (49 - 18)

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

        flatten_obs = flatten_tensor_from_mapping(obs_seq_tensor)
        flatten_goal = flatten_tensor_from_mapping(goal_state_tensor)

        unet_cond = torch.cat([flatten_obs, flatten_goal], dim=-1)

        return unet_cond
