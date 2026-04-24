from collections import deque
from typing import Any, Literal, cast

import gymnasium as gym
import lightning as L
import mani_skill.envs  # noqa: F401 (registers environments silently, ruff must ignore this)
import numpy as np
import torch
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.utilities import rank_zero_info
from rich.progress import Progress
from tqdm import tqdm

from policy.utils.typing_utils import PolicyProtocol

BASE_SEED_VAL = 42000
BASE_SEED_TEST = 67000


class RolloutEvaluationCallback(L.Callback):
    def __init__(
        self,
        env_id: str,
        control_mode: str,
        num_val_episodes: int = 20,
        num_test_episodes: int = 100,
        conditioning_source: Literal["obs", "env_states"] = "obs",
        obs_mode: str = "state",
    ):
        super().__init__()
        self.env_id = env_id
        self.num_val_episodes = num_val_episodes
        self.num_test_episodes = num_test_episodes
        self.conditioning_source = conditioning_source
        self.obs_mode = obs_mode
        self.control_mode = control_mode

    def _get_policy_input(self, env, step_obs):
        """Helper to extract the correct conditioning state."""
        if self.conditioning_source == "env_states":
            # Bypass obs_mode completely and fetch the raw physics state
            return env.unwrapped.get_state()

        # Default to whatever observation the env returned (e.g. obs_mode="state")
        return step_obs

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._run_rollouts(trainer, pl_module, self.num_val_episodes, "val")

    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._run_rollouts(trainer, pl_module, self.num_test_episodes, "test")

    def _run_rollouts(
        self, trainer: L.Trainer, pl_module: L.LightningModule, num_episodes: int, phase: str
    ):
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

        # Check if we are using RichProgressBar or TQDMProgressBar
        is_rich = isinstance(trainer.progress_bar_callback, RichProgressBar)
        pbar: Any = None
        task_id: Any = None

        if is_rich:
            pbar = Progress(
                transient=True
            )  # transient=True ensures the bar clears up after completion
            task_id = pbar.add_task(
                f"[cyan][{phase.capitalize()}] Rollout Eval...", total=num_episodes
            )
            pbar.start()
        else:
            pbar = tqdm(
                total=num_episodes,
                desc=f"[{phase.capitalize()}] Rollout Eval",
                leave=False,
                position=2,
            )  # leave=False ensures the bar clears up after completion, position=2 to avoid overwriting other bars

        for i in range(num_episodes):
            # Pass the seed based on the episode index
            seed = base_seed + i
            obs, info = env.reset(seed=seed)
            policy_input = self._get_policy_input(env, obs)
            done = False

            # Setup observation history
            policy_inputs_deque = deque([policy_input] * obs_horizon, maxlen=obs_horizon)

            while not done:
                stacked_obs = np.stack(policy_inputs_deque)
                # Note: Adjust the tensor device and structure to match your exact pipeline
                obs_tensor = torch.tensor(
                    stacked_obs, dtype=torch.float32, device=pl_module.device
                ).unsqueeze(0)
                obs_seq = {self.conditioning_source: obs_tensor}

                # Get action from the policy using the casted object
                action_seq = policy.get_action(obs_seq)
                action = action_seq[:, 0].cpu().numpy()

                obs, reward, terminated, truncated, info = env.step(action)

                policy_input = self._get_policy_input(env, obs)

                policy_inputs_deque.append(policy_input)
                done = terminated or truncated

                if info.get("success", False):
                    successes += 1
                    break

            if is_rich:
                pbar.update(task_id, advance=1)
            else:
                pbar.update(1)

        if is_rich:
            pbar.stop()
        else:
            pbar.close()

        env.close()
        success_rate = successes / num_episodes

        # Log the metric directly to the module
        pl_module.log(f"{phase}/success_rate", float(success_rate), sync_dist=True, prog_bar=True)

        # And as a separate line itself
        if trainer.is_global_zero:  # If multiple GPUs are running, only log the success rate message from the main process to avoid duplicates
            msg = (
                f"Epoch {trainer.current_epoch} - {phase} rollout success rate: {success_rate:.2%}"
            )
            if is_rich:
                rank_zero_info(msg)
            else:
                # tqdm.write pushes the message above the progress bars securely
                tqdm.write(msg)
