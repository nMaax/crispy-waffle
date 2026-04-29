import warnings
from typing import Any, cast

import gymnasium as gym
import lightning as L
import mani_skill.envs  # noqa: F401
import torch
from gymnasium.spaces import Box
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.utilities import rank_zero_info
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers import FrameStack, RecordEpisode
from rich.progress import Progress
from tqdm import tqdm

from policy.utils import flatten_tensor_dict, to_tensor
from policy.utils.typing_utils import PolicyProtocol

# WARN: Just a notification by Transformers, however we do not use a higher version (enforced via .toml), so we can ignore this
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.deepspeed")

# TODO: when on GPU it tries to use all GPUs however tensor appear on different devices! Overall scale code to work on double gpus


class RolloutEvaluationCallback(L.Callback):
    OFFSET_SEED_VAL: int = 42000
    OFFSET_SEED_TEST: int = 67000

    def __init__(
        self,
        num_val_episodes: int = 20,
        num_test_episodes: int = 100,
        max_episode_steps: int | None = None,
        clamp_action: bool = True,
        video_dir: str | None = None,
        render_mode: str | None = None,
        seed: int | None = None,
    ):
        """Deploys the policy in an environment (parallelized for CUDA, sequential for CPU) for
        testing."""
        super().__init__()
        self.env_id = None
        self.obs_mode = None
        self.control_mode = None
        self.physx_backend = None
        self.use_physx_env_states = None

        self.num_val_episodes = num_val_episodes
        self.num_test_episodes = num_test_episodes
        self.max_episode_steps = max_episode_steps
        self.clamp_action = clamp_action
        self.video_dir = video_dir

        if render_mode is None and self.video_dir is not None:
            self.render_mode = "rgb_array"
        else:
            self.render_mode = render_mode

        if seed is None:
            raise ValueError("seed must be provided.")
        self.val_seed = seed + self.OFFSET_SEED_VAL
        self.test_seed = seed + self.OFFSET_SEED_TEST

        rank_zero_info(
            f"Seeds for Maniskill simulations fetched from main seed {seed} -> Validation seed: {self.val_seed}, Test seed: {self.test_seed}"
        )

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        """Called when fit, validate, test, predict, or tune begins."""
        datamodule = getattr(trainer, "datamodule", None)

        if datamodule is None:
            raise ValueError("A datamodule must be attached to the trainer to use this callback.")

        if datamodule.env_id is None:
            raise ValueError("The datamodule must specify an env_id to use this callback.")

        self.env_id = datamodule.env_id

        # Double check that the environment is actually registered and available to use
        if self.env_id not in gym.envs.registry:
            raise RuntimeError(
                f"Environment '{self.env_id}' is not registered in Gymnasium + Maniskill."
            )

        self.obs_mode = datamodule.obs_mode
        self.control_mode = datamodule.control_mode
        self.physx_backend = datamodule.physx_backend
        self.use_physx_env_states = datamodule.use_physx_env_states

        if "cuda" in self.physx_backend.lower() and not torch.cuda.is_available():
            raise RuntimeError(
                f"Dataset specifies CUDA backend '{self.physx_backend}', "
                "but CUDA is not available on this machine. Cannot run parallel CUDA environments."
            )

        rank_zero_info(
            f"Rollout Config synced from Datamodule -> env_id: {self.env_id}, obs_mode: {self.obs_mode} ({self.use_physx_env_states=}), "
            f"control_mode: {self.control_mode}, backend: {self.physx_backend}"
        )

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._run_rollouts(trainer, pl_module, self.num_val_episodes, "val")

    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._run_rollouts(trainer, pl_module, self.num_test_episodes, "test")

    def _get_policy_conditioning(
        self,
        env: FrameStack,
        obs: torch.Tensor | dict[str, Any],
        device: torch.device | None = None,
    ) -> torch.Tensor | dict[str, Any]:
        """Helper to extract the correct conditioning state and ensure batched tensor format."""
        if self.use_physx_env_states:
            policy_conditioning = env.unwrapped.get_state()  # type: ignore
            policy_conditioning = to_tensor(policy_conditioning, device=device)
        else:
            policy_conditioning = obs

        return policy_conditioning

    def _run_rollouts(
        self, trainer: L.Trainer, pl_module: L.LightningModule, num_episodes: int, phase: str
    ) -> None:

        # TODO: should refactor to a more clean function later once it works: extract subfunctions, clean up rendundant lines, etc.

        assert isinstance(pl_module, PolicyProtocol), (
            f"Expected the LightningModule to implement PolicyProtocol, "
            f"but got {type(pl_module).__name__}."
        )

        if self.env_id is None:
            raise ValueError("env_id is not set. This should have been set during setup().")

        if num_episodes <= 0:
            return

        # TODO: I could rather use AsyncVectorEnv like in https://github.com/haosulab/ManiSkill/blob/main/examples/baselines/diffusion_policy/diffusion_policy/make_env.py
        if self.physx_backend == "physx_cuda":
            num_iterations = 1
            num_envs = num_episodes
        else:
            num_iterations = num_episodes
            num_envs = 1

        env = gym.make(
            id=self.env_id,
            obs_mode=self.obs_mode,
            control_mode=self.control_mode,
            render_mode=self.render_mode,
            num_envs=num_envs,
            max_episode_steps=self.max_episode_steps,
        )
        env = cast(BaseEnv, env)

        # Enable video recording if directory is defined
        if self.video_dir:
            max_episode_steps = gym_utils.find_max_episode_steps_value(env)
            env = RecordEpisode(
                env,
                output_dir=f"{self.video_dir}/{phase}",
                save_trajectory=False,
                save_video=True,
                max_steps_per_video=max_episode_steps,
                source_type="diffusion_policy",
                source_desc=f"Diffusion Policy rollout ({phase})",
            )

        env = FrameStack(env, num_stack=pl_module.cond_horizon)

        # Cache the action bounds for later clamping
        action_space = env.action_space
        if not isinstance(action_space, Box):
            raise ValueError(f"Expected Box action space, got {type(action_space)}")

        action_low = torch.as_tensor(
            action_space.low, device=pl_module.device, dtype=torch.float32
        )
        action_high = torch.as_tensor(
            action_space.high, device=pl_module.device, dtype=torch.float32
        )

        # Put model in eval mode
        pl_module.eval()

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

            dones = torch.zeros(num_envs, dtype=torch.bool, device=pl_module.device)
            successes = torch.zeros(num_envs, dtype=torch.bool, device=pl_module.device)
            envs_completed_this_iter = 0

            # Step until all environments within this batch have concluded
            while not dones.all():
                policy_conditioning = self._get_policy_conditioning(
                    env=env, obs=obs, device=pl_module.device
                )
                flatten_cond = flatten_tensor_dict(policy_conditioning, device=pl_module.device)

                with torch.no_grad():
                    action_seq = pl_module.get_action(flatten_cond)

                # Diffusion open-loop execution Chunking
                for i in range(action_seq.shape[1]):
                    action = action_seq[:, i]

                    if self.clamp_action:
                        action = torch.clamp(
                            action, action_low.to(action.dtype), action_high.to(action.dtype)
                        )

                    # For CPU seq-envs we need [dim] tensors. For CUDA, ManiSkill takes [batch, dim] tensors.
                    # env_action = action if self.is_cuda else action.squeeze(0).cpu()

                    obs, reward, terminated, truncated, info = env.step(action)

                    truncated = torch.as_tensor(
                        truncated, dtype=torch.bool, device=pl_module.device
                    )
                    terminated = torch.as_tensor(
                        terminated, dtype=torch.bool, device=pl_module.device
                    )

                    # Extract boolean flags correctly across backend CPU/GPU differences
                    # NOTE: In a batched GPU environment (e.g., 100 parallel envs), if one single environment truncates
                    # (e.g., drops the object or reaches the step limit) on step 2 of an 8-step action chunk, truncated.any() evaluates to True.
                    # This breaks the for loop, forcing the policy to immediately replan for the entire batch of 100 environments,
                    # even though 99 of them were perfectly happy executing the rest of their chunk.
                    # It just means that inference will run slightly slower toward the end of an evaluation epoch as environments start finishing at different times, causing more frequent replanning
                    # However, fixing this would require complex tensor masking etc., so I will keepm it this way for now for simplicity
                    env_is_done = terminated | truncated
                    just_finished = env_is_done & ~dones

                    if "success" in info:
                        succ = info["success"]
                        if not isinstance(succ, torch.Tensor):
                            succ = torch.tensor([succ], dtype=torch.bool, device=pl_module.device)
                        else:
                            succ = succ.to(pl_module.device)
                        successes[just_finished] = succ[just_finished]

                    dones = dones | env_is_done

                    newly_completed = dones.sum().item()
                    advance = newly_completed - envs_completed_this_iter
                    if advance > 0:
                        if is_rich:
                            pbar.update(task_id, advance=advance)
                        else:
                            pbar.update(advance)
                    envs_completed_this_iter = newly_completed

                    if truncated.any():
                        break

            total_successes += successes.float().sum().item()

        if is_rich:
            pbar.stop()
        else:
            pbar.close()

        env.close()

        # Compute over the total number of tested episodes
        success_rate = total_successes / num_episodes

        # TODO: what else should I log? e.g. truncated?
        # Log it to the main logger (e.g. WandB)
        pl_module.log(f"{phase}/success_rate", float(success_rate), sync_dist=True, prog_bar=True)

        # And as a separate line itself on the terminal
        # TODO: this causes the TQDM bars to dont clean from the terminal and persists
        # TODO: another thing is that after a few epichs there is too much info next to the bar
        msg = f"\n  [{phase.capitalize()} | Epoch {trainer.current_epoch}] Rollout Success Rate: {success_rate:.2%}\n"
        rank_zero_info(msg)
