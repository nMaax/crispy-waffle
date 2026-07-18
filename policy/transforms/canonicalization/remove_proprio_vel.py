from collections.abc import Mapping

import torch

from policy.utils.typing_utils import TensorTree, get_tensor


class RemoveProprioVel:
    def __init__(self, qpos_dim: int = 9, qvel_dim: int = 9, fill_with_zeroes: bool = False):
        """Wrapper utility to mask or remove robot joint velocities (qvel) from ManiSkill state
        observations."""

        # State tensor structure (obs_mode="state" or obs["state"]):
        #    [ 0 : qpos_dim ]               -> qpos (Joint Positions)
        #    [ qvel_start : qvel_end ]      -> qvel (Joint Velocities)
        #    [ qvel_end : ]                 -> Extra task states (e.g., tcp_pose,
        #                                   object_pose, goal_pose)
        #
        #    e.g. for standard Franka Panda:
        #    - indices [0:9]   (9 elements) = qpos (joint 1 to 7, left finger, right finger)
        #    - indices [9:18]  (9 elements) = qvel (joint 1 to 7, left finger, right finger)
        #    - indices [18:]   (X elements) = Task-related data

        self.qpos_dim = qpos_dim
        self.qvel_dim = qvel_dim
        self.fill_with_zeroes = fill_with_zeroes

        # qvel starts right after qpos ends
        self.qvel_start = qpos_dim
        self.qvel_end = qpos_dim + qvel_dim

    def __call__(self, obs: TensorTree) -> TensorTree:
        if isinstance(obs, Mapping):
            return self._process_dict(obs)
        elif isinstance(obs, torch.Tensor):
            return self._process_tensor(obs)

    def _process_dict(self, state_dict: Mapping[str, TensorTree]) -> dict[str, TensorTree]:
        # Native state_dict (contains "agent" -> "qpos" / "qvel")
        if "agent" in state_dict and isinstance(state_dict["agent"], Mapping):
            agent_dict = dict(state_dict["agent"])
            if "qvel" in agent_dict:
                if self.fill_with_zeroes:
                    qvel = get_tensor(agent_dict, "qvel")
                    agent_dict["qvel"] = torch.zeros_like(qvel)
                else:
                    del agent_dict["qvel"]
            return {**state_dict, "agent": agent_dict}

        # Standardized PnP dict (contains "proprio")
        elif "proprio" in state_dict:
            proprio = get_tensor(state_dict, "proprio")
            return {**state_dict, "proprio": self._process_tensor(proprio)}

        # Legacy dict wrapping flat tensor (contains "state")
        elif "state" in state_dict:
            state = get_tensor(state_dict, "state")
            return {**state_dict, "state": self._process_tensor(state)}

        return dict(state_dict)

    def _process_tensor(self, state_tensor: torch.Tensor) -> torch.Tensor:
        if self.fill_with_zeroes:
            out = state_tensor.clone()
            out[..., self.qvel_start : self.qvel_end] = 0.0
            return out
        else:
            prefix = state_tensor[..., : self.qvel_start]
            suffix = state_tensor[..., self.qvel_end :]
            return torch.cat([prefix, suffix], dim=-1)
