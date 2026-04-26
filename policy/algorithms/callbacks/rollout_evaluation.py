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
        Deploys the policy in a parallelized environment for testing.

        parameters:
            - num_val_episodes: int, how many parallel episodes to run during validation
            - num_test_episodes: int, how many parallel episodes to run during testing
            - seed: int or None, an optional main seed to derive the validation and test seeds from. If None, random seeds will be generated.
        """
        super().__init__()
        self.env_id = None
        self.obs_mode = None
        self.control_mode = None
        self.physx_backend = None
        self.use_phsyx_env_states = None

        self.num_val_episodes = num_val_episodes
        self.num_test_episodes = num_test_episodes

        # Inject arbitrary base seeds to avoid using those at training
        main_seed = seed if seed is not None else random.randint(0, int(1e5))
        self.val_seed = main_seed + self.BASE_SEED_VAL
        self.test_seed = main_seed + self.BASE_SEED_TEST

        # Debug
        print(
            f"Seeds for Maniskill simulations fetched from main seed {main_seed} -> Validation seed: {self.val_seed}, Test seed: {self.test_seed}"
        )

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        """Called when fit, validate, test, predict, or tune begins."""
        datamodule = trainer.datamodule

        if datamodule is None:
            raise ValueError("A datamodule must be attached to the trainer to use this callback.")

        self.env_id = datamodule.env_id
        self.obs_mode = datamodule.obs_mode
        self.control_mode = datamodule.control_mode
        self.physx_backend = datamodule.physx_backend
        self.use_phsyx_env_states = datamodule.use_phsyx_env_states

        if self.physx_backend == "physix_cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"Dataset specifies GPU backend '{self.physx_backend}', "
                "but CUDA is not available on this machine. Cannot run parallel GPU environments."
            )

        rank_zero_info(
            f"Rollout Config synced from Datamodule -> obs_mode: {self.obs_mode}, "
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
        # Instantiate parallel GPU environments by specifying num_envs
        env = gym.make(
            self.env_id,
            obs_mode=self.obs_mode,
            control_mode=self.control_mode,
            num_envs=num_episodes,
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

        # Reset the environment, 'obs' is directly a batched PyTorch Tensor
        seed = self.val_seed if phase == "val" else self.test_seed
        obs, info = env.reset(seed=seed)
        policy_input = self._get_policy_input(env, obs)

        # Populate dequeue
        policy_inputs_deque = deque([policy_input] * obs_horizon, maxlen=obs_horizon)

        # Track completions
        dones = torch.zeros(num_episodes, dtype=torch.bool, device=pl_module.device)
        successes = torch.zeros(num_episodes, dtype=torch.bool, device=pl_module.device)
        episodes_completed = 0

        # Loop until all parallel environments have signaled done at least once
        while not dones.all():
            # Stack tensors along the time dimension (dim=1), dim=0 is the batch dimension
            stacked_obs = torch.stack(list(policy_inputs_deque), dim=1)
            obs_seq = {self.use_phsyx_env_states: stacked_obs}

            with torch.no_grad():
                action_seq = policy.get_action(obs_seq)
                action = action_seq[:, 0]

            # Step all environments simultaneously
            obs, reward, terminated, truncated, info = env.step(action)

            policy_input = self._get_policy_input(env, obs)
            policy_inputs_deque.append(policy_input)

            # Evaluate who just finished right now
            env_is_done = terminated | truncated
            just_finished = env_is_done & ~dones

            if "success" in info:
                # Move success dict to device
                success_tensor = info["success"].to(pl_module.device)
                # Update the success status
                successes[just_finished] = success_tensor[just_finished]

            # Update dones with the new completions from this step
            dones = dones | env_is_done

            # Update progress bar with how many new envs finished in this step
            newly_completed = dones.sum().item()
            advance = newly_completed - episodes_completed
            if advance > 0:
                if is_rich:
                    pbar.update(task_id, advance=advance)
                else:
                    pbar.update(advance)
            episodes_completed = newly_completed

        if is_rich:
            pbar.stop()
        else:
            pbar.close()

        env.close()
        success_rate = successes / num_episodes

        # Calculate success rate from the batched successes tensor
        success_rate = successes.float().mean().item()

        # Log it to the main logger (e.g. WandB)
        pl_module.log(f"{phase}/success_rate", float(success_rate), sync_dist=True, prog_bar=True)

        # And as a separate line itself on the terminal
        # TODO: this causes the TQDM bars to dont clean from the terminal and persists
        # TODO: another thing is that after a few epichs there is too much info next to the bar
        msg = f"\n  [{phase.capitalize()} | Epoch {trainer.current_epoch}] Rollout Success Rate: {success_rate:.2%}\n"
        rank_zero_info(msg)
