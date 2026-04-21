from collections import deque
from typing import cast

import gymnasium as gym
import lightning as L
import numpy as np
import torch

from policy.algorithms import DiffusionPolicy

BASE_SEED_VAL = 42000
BASE_SEED_TEST = 67000


class RolloutEvaluationCallback(L.Callback):
    def __init__(self, env_id: str, num_val_episodes: int = 20, num_test_episodes: int = 100):
        super().__init__()
        self.env_id = env_id
        self.num_val_episodes = num_val_episodes
        self.num_test_episodes = num_test_episodes

    def on_validation_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._run_rollouts(pl_module, self.num_val_episodes, "val")

    def on_test_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._run_rollouts(pl_module, self.num_test_episodes, "test")

    def _run_rollouts(self, pl_module: L.LightningModule, num_episodes: int, phase: str):
        # We assume the pl_module has the get_action method and obs_horizon defined
        # TODO: as in diffusion, how to generalize to handle either (or both) obs and env_states?
        env = gym.make(self.env_id, obs_mode="state", control_mode="pd_ee_delta_pose")
        successes = 0

        # Put model in eval mode
        pl_module.eval()

        # Explicitly tell Pyright to stop assuming custom attributes are Tensors/Modules, but the DiffusionPolicy itself.
        # TODO: Maybe I can find some more general Type that is not fixed to DiffusionPolicy, maybe a Protocol for all my IL approached
        policy = cast(DiffusionPolicy, pl_module)
        obs_horizon: int = policy.obs_horizon

        # Inject arbitrary base seeds to avoid using those at training
        base_seed = BASE_SEED_VAL if phase == "val" else BASE_SEED_TEST

        for i in range(num_episodes):
            # TODO: as in diffusion, how to generalize to handle either (or both) obs and env_states?
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
                obs_seq_dict = {"state": obs_tensor}

                # Get action from the policy using the casted object
                action_seq = policy.get_action(obs_seq_dict)
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
