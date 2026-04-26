import random
from collections import deque
from typing import Any, Literal, cast

import gymnasium as gym
import lightning as L
import mani_skill.envs  # noqa: F401
import torch
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.utilities import rank_zero_info
from rich.progress import Progress
from tqdm import tqdm

from policy.utils.typing_utils import PolicyProtocol


class RolloutEvaluationCallback(L.Callback):
    # Offset validation and test
    BASE_SEED_VAL: int = 42000
    BASE_SEED_TEST: int = 67000

    def __init__(
        self,
        num_val_episodes: int = 20,
        num_test_episodes: int = 100,
        seed: int | None = None,
    ):
        """
        Deploys the policy in an environment (parallelized for CUDA, sequential for CPU) for testing.

        parameters:
            - num_val_episodes: int, total episodes to run during validation
            - num_test_episodes: int, total episodes to run during testing
            - seed: int or None, an optional main seed to derive the validation and test seeds from.
        """
        super().__init__()
        self.env_id = None
        self.obs_mode = None
        self.control_mode = None
        self.physx_backend = None
        self.use_phsyx_env_states = None

        self.num_val_episodes = num_val_episodes
        self.num_test_episodes = num_test_episodes

        if seed is None:
            raise ValueError("seed must be provided.")
        self.val_seed = seed + self.BASE_SEED_VAL
        self.test_seed = seed + self.BASE_SEED_TEST

        print(
            f"Seeds for Maniskill simulations fetched from main seed {seed} -> Validation seed: {self.val_seed}, Test seed: {self.test_seed}"
        )

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        """Called when fit, validate, test, predict, or tune begins."""
        datamodule = trainer.datamodule

        if datamodule is None:
            raise ValueError("A datamodule must be attached to the trainer to use this callback.")

        self.env_id = datamodule.env_id

        # Double check that the environment is actually registered and available to use
        if self.env_id not in gym.envs.registry:
            raise RuntimeError(f"Environment '{self.env_id}' is not registered in Gymnasium.")

        self.obs_mode = datamodule.obs_mode
        self.control_mode = datamodule.control_mode
        self.physx_backend = datamodule.physx_backend
        self.use_phsyx_env_states = datamodule.use_phsyx_env_states

        # Check for CUDA backend semantically
        is_cuda = self.physx_backend is not None and "cuda" in self.physx_backend.lower()

        if is_cuda and not torch.cuda.is_available():
            raise RuntimeError(
                f"Dataset specifies CUDA backend '{self.physx_backend}', "
                "but CUDA is not available on this machine. Cannot run parallel CUDA environments."
            )

        rank_zero_info(
            f"Rollout Config synced from Datamodule -> env_id: {self.env_id}, obs_mode: {self.obs_mode} ({self.use_phsyx_env_states=}), "
            f"control_mode: {self.control_mode}, backend: {self.physx_backend}"
        )

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._run_rollouts(trainer, pl_module, self.num_val_episodes, "val")

    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._run_rollouts(trainer, pl_module, self.num_test_episodes, "test")

    def _get_policy_input(self, env, step_obs):
        """Helper to extract the correct conditioning state."""
        if self.use_phsyx_env_states:
            # Bypass obs_mode completely and fetch the raw physics state
            return env.unwrapped.get_state()

        # Default to whatever observation the env returned (e.g. obs_mode="state")
        return step_obs

    def _run_rollouts(
        self, trainer: L.Trainer, pl_module: L.LightningModule, num_episodes: int, phase: str
    ):
        is_cuda = self.physx_backend is not None and "cuda" in self.physx_backend.lower()

        # In CUDA, batch environments together. In CPU, execute 1 at a time sequentially
        num_envs = num_episodes if is_cuda else 1
        num_iterations = 1 if is_cuda else num_episodes

        env = gym.make(
            self.env_id,
            obs_mode=self.obs_mode,
            control_mode=self.control_mode,
            num_envs=num_envs,
        )

        # Put model in eval mode
        pl_module.eval()

        # Cast to PolicyProtocol so we can access get_action and cond_horizon without
        # depending on a specific implementation class.
        policy = cast(PolicyProtocol, pl_module)
        obs_horizon = policy.cond_horizon

        # Check if we are using RichProgressBar or TQDMProgressBar
        is_rich = isinstance(trainer.progress_bar_callback, RichProgressBar)
        pbar: Any = None
        task_id: Any = None

        if is_rich:
            pbar = Progress(
                transient=True
            )  # transient=True ensures the bar clears up after completion
            task_id = pbar.add_task(f"  [{phase.capitalize()}] Rollout", total=num_episodes)
            pbar.start()
        else:
            pbar = tqdm(
                total=num_episodes,
                desc=f"  [{phase.capitalize()}] Rollout",
                leave=False,
                position=2,
            )  # leave=False ensures the bar clears up after completion, position=2 to avoid overwriting other bars

        total_successes = 0.0

        for iteration in range(num_iterations):
            # For CPU sequentially, offset the seed per iteration to ensure variety.
            # For CUDA parallel, seed once and rely on internal ManiSkill RNGs per sub-scene.
            current_seed = (self.val_seed if phase == "val" else self.test_seed) + iteration

            obs, info = env.reset(seed=current_seed)
            policy_input = self._get_policy_input(env, obs)

            obs_seq = policy_input.unsqueeze(1).repeat(1, obs_horizon, 1)

            dones = torch.zeros(num_envs, dtype=torch.bool, device=pl_module.device)
            successes = torch.zeros(num_envs, dtype=torch.bool, device=pl_module.device)
            envs_completed_this_iter = 0

            # Step until all environments within this batch have concluded
            while not dones.all():
                with torch.no_grad():
                    action_seq = policy.get_action(obs_seq)
                    action = action_seq[:, 0]

                action = action if is_cuda else action.cpu()

                obs, reward, terminated, truncated, info = env.step(action)

                policy_input = self._get_policy_input(env, obs)

                # Shift the buffer and insert the new observation
                # TODO: should rename obs_seq to cond_seq or policy_input_buffers or something
                obs_seq = torch.roll(obs_seq, shifts=-1, dims=1)
                obs_seq[:, -1] = policy_input

                env_is_done = torch.as_tensor(
                    terminated | truncated, dtype=torch.bool, device=pl_module.device
                )
                just_finished = env_is_done & ~dones

                if "success" in info:
                    success_tensor = info["success"].to(pl_module.device)
                    successes[just_finished] = success_tensor[just_finished]

                dones = dones | env_is_done

                newly_completed = dones.sum().item()
                advance = newly_completed - envs_completed_this_iter
                if advance > 0:
                    if is_rich:
                        pbar.update(task_id, advance=advance)
                    else:
                        pbar.update(advance)
                envs_completed_this_iter = newly_completed

            total_successes += successes.float().sum().item()

        if is_rich:
            pbar.stop()
        else:
            pbar.close()

        env.close()

        # Compute over the total number of tested episodes
        success_rate = total_successes / num_episodes

        # Log it to the main logger (e.g. WandB)
        pl_module.log(f"{phase}/success_rate", float(success_rate), sync_dist=True, prog_bar=True)

        # And as a separate line itself on the terminal
        # TODO: this causes the TQDM bars to dont clean from the terminal and persists
        # TODO: another thing is that after a few epichs there is too much info next to the bar
        msg = f"\n  [{phase.capitalize()} | Epoch {trainer.current_epoch}] Rollout Success Rate: {success_rate:.2%}\n"
        rank_zero_info(msg)
