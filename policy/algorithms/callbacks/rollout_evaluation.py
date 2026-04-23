from collections import deque
from typing import cast

import gymnasium as gym
import lightning as L
import mani_skill.envs  # noqa: F401 (registers environments silently, ruff must ignore this)
import numpy as np
import torch

from policy.utils.typing_utils import PolicyProtocol

BASE_SEED_VAL = 42000
BASE_SEED_TEST = 67000


class RolloutEvaluationCallback(L.Callback):
    def __init__(
        self,
        env_id: str,
        num_val_episodes: int = 20,
        num_test_episodes: int = 100,
        obs_mode: str = "state",
        control_mode: str = "pd_ee_delta_pose",
    ):
        super().__init__()
        self.env_id = env_id
        self.num_val_episodes = num_val_episodes
        self.num_test_episodes = num_test_episodes
        self.obs_mode = obs_mode
        self.control_mode = control_mode

    def on_validation_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._run_rollouts(pl_module, self.num_val_episodes, "val")

    def on_test_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._run_rollouts(pl_module, self.num_test_episodes, "test")

    def _run_rollouts(self, pl_module: L.LightningModule, num_episodes: int, phase: str):
        env = gym.make(self.env_id, obs_mode=self.obs_mode, control_mode=self.control_mode)
        successes = 0

        # Put model in eval mode
        pl_module.eval()

        # Cast to PolicyProtocol so we can access get_action and cond_horizon without
        # depending on a specific implementation class.
        policy = cast(PolicyProtocol, pl_module)
        obs_horizon = policy.cond_horizon

        # Inject arbitrary base seeds to avoid using those at training
        base_seed = BASE_SEED_VAL if phase == "val" else BASE_SEED_TEST

        for i in range(num_episodes):
            # Pass the seed based on the episode index
            seed = base_seed + i
            obs, info = env.reset(seed=seed)
            done = False

            # Setup observation history
            obs_deque = deque([obs] * obs_horizon, maxlen=obs_horizon)

            while not done:
                stacked_obs = np.stack(obs_deque)
                # Note: Adjust the tensor device and structure to match your exact pipeline
                obs_tensor = torch.tensor(
                    stacked_obs, dtype=torch.float32, device=pl_module.device
                ).unsqueeze(0)
                obs_seq = {"state": obs_tensor}

                # Get action from the policy using the casted object
                action_seq = policy.get_action(obs_seq)
                action = action_seq[0, 0].cpu().numpy()

                obs, reward, terminated, truncated, info = env.step(action)
                obs_deque.append(obs)
                done = terminated or truncated

                if info.get("success", False):
                    successes += 1
                    break

        env.close()
        success_rate = successes / num_episodes

        # Log the metric directly to the module
        pl_module.log(f"{phase}/success_rate", float(success_rate), sync_dist=True)
