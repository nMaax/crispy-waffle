import numpy as np
import torch


class RemoveProprioVel:
    def __init__(self, qpos_dim: int = 9, qvel_dim: int = 9, fill_with_zeroes: bool = True):
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

    def __call__(self, obs):
        if isinstance(obs, dict):
            return self._process_dict(obs)
        else:
            return self._process_tensor(obs)

    def _process_dict(self, state_dict):
        # 1. Native state_dict (contains "agent" -> "qpos" / "qvel")
        if "agent" in state_dict and isinstance(state_dict["agent"], dict):
            agent_dict = state_dict["agent"]
            if "qvel" in agent_dict:
                if self.fill_with_zeroes:
                    if isinstance(agent_dict["qvel"], torch.Tensor):
                        agent_dict["qvel"] = torch.zeros_like(agent_dict["qvel"])
                    else:
                        agent_dict["qvel"] = np.zeros_like(agent_dict["qvel"])
                else:
                    agent_dict.pop("qvel")
            return state_dict

        # 2. Standardized PnP dict (contains "proprio")
        elif "proprio" in state_dict:
            state_dict["proprio"] = self._process_tensor(state_dict["proprio"])
            return state_dict

        # 3. Legacy dict wrapping flat tensor (contains "state")
        elif "state" in state_dict:
            state_dict["state"] = self._process_tensor(state_dict["state"])
            return state_dict

        return state_dict

    def _process_tensor(self, state_tensor):
        if isinstance(state_tensor, torch.Tensor):
            if self.fill_with_zeroes:
                out = state_tensor.clone()
                out[..., self.qvel_start : self.qvel_end] = 0.0
                return out
            else:
                prefix = state_tensor[..., : self.qvel_start]
                suffix = state_tensor[..., self.qvel_end :]
                return torch.cat([prefix, suffix], dim=-1)

        elif isinstance(state_tensor, np.ndarray):
            if self.fill_with_zeroes:
                out = state_tensor.copy()
                out[..., self.qvel_start : self.qvel_end] = 0.0
                return out
            else:
                prefix = state_tensor[..., : self.qvel_start]
                suffix = state_tensor[..., self.qvel_end :]
                return np.concatenate([prefix, suffix], axis=-1)

        return state_tensor
