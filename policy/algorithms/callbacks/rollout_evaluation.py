import warnings
from typing import Any, cast

import gymnasium as gym
import lightning as L
import mani_skill.envs  # noqa: F401
import torch
from gymnasium.spaces import Box
from lightning.pytorch.callbacks import RichProgressBar, TQDMProgressBar
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
    """A Lightning Callback for performing rollout evaluation of a policy in a ManiSkill
    environment."""

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
        super().__init__()

        if seed is None:
            raise ValueError("seed must be provided.")

        self.num_val_episodes = num_val_episodes
        self.num_test_episodes = num_test_episodes
        self.max_episode_steps = max_episode_steps
        self.clamp_action = clamp_action
        self.video_dir = video_dir

        if render_mode is None and self.video_dir is not None:
            self.render_mode = "rgb_array"
        else:
            self.render_mode = render_mode

        self.val_seed = seed + self.OFFSET_SEED_VAL
        self.test_seed = seed + self.OFFSET_SEED_TEST

        self.env_id = None
        self.obs_mode = None
        self.control_mode = None
        self.physx_backend = None
        self.use_physx_env_states = None

        rank_zero_info(
            f"Seeds for rollout simulation fetched from main seed: {seed}\n"
            f"\tValidation seed: {self.val_seed}\n"
            f"\tTest seed: {self.test_seed}"
        )

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        datamodule = getattr(trainer, "datamodule", None)

        if datamodule is None:
            raise ValueError("A datamodule must be attached to the trainer to use this callback.")

        if not getattr(datamodule, "env_id", None):
            raise ValueError("Datamodule must specify an env_id.")

        if datamodule.env_id not in gym.envs.registry:
            raise RuntimeError(
                f"Environment '{self.env_id}' is not registered in Gymnasium + Maniskill."
            )

        if "cuda" in datamodule.physx_backend.lower() and not torch.cuda.is_available():
            raise RuntimeError(
                f"Dataset specifies CUDA backend '{self.physx_backend}', "
                "but CUDA is not available on this machine. Cannot run parallel CUDA environments."
            )

        self.env_id = datamodule.env_id
        self.obs_mode = datamodule.obs_mode
        self.control_mode = datamodule.control_mode
        self.physx_backend = datamodule.physx_backend
        self.use_physx_env_states = datamodule.use_physx_env_states

        rank_zero_info(
            f"Rollout Config synced from Datamodule:\n"
            f"\tenv_id: {self.env_id},\n"
            f"\tobs_mode: {self.obs_mode} ({self.use_physx_env_states=}),\n"
            f"\tcontrol_mode: {self.control_mode},\n"
            f"\tbackend: {self.physx_backend}"
        )

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._run_rollouts(trainer, pl_module, self.num_val_episodes, "val")

    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._run_rollouts(trainer, pl_module, self.num_test_episodes, "test")

    def _run_rollouts(
        self, trainer: L.Trainer, pl_module: L.LightningModule, num_episodes: int, phase: str
    ) -> None:
        """Runs num_episodes episodes in the environment using the current policy and logs success
        rate.

        Shapes (internal):
            obs: [num_envs, cond_horizon, obs_dim]
            action_seq (from policy): [num_envs, act_horizon, act_dim]
            action (stepped in env): [num_envs, act_dim]
        """

        if not isinstance(pl_module, PolicyProtocol):
            raise AttributeError(
                f"Expected the LightningModule to implement PolicyProtocol, "
                f"but got {type(pl_module).__name__}."
            )

        self._validate_setup()
        assert self.env_id is not None

        if num_episodes <= 0:
            return

        # On CUDA we run all episodes in parallel, on CPU we run sequentially
        if self.physx_backend == "physx_cuda":
            num_iterations = 1
            num_envs = num_episodes
        elif self.physx_backend == "physx_cpu":
            # TODO: I could rather use AsyncVectorEnv
            # like in https://github.com/haosulab/ManiSkill/blob/main/examples/baselines/diffusion_policy/diffusion_policy/make_env.py
            num_iterations = num_episodes
            num_envs = 1
        else:
            raise ValueError(f"Unsupported physx_backend: {self.physx_backend}")

        env = gym.make(
            id=self.env_id,
            obs_mode=self.obs_mode,
            control_mode=self.control_mode,
            render_mode=self.render_mode,
            num_envs=num_envs,
            max_episode_steps=self.max_episode_steps,
        )

        if self.video_dir:
            env = cast(BaseEnv, env)
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

        action_space = env.action_space

        if not isinstance(action_space, Box):
            raise ValueError(f"Expected Box action space, got {type(action_space)}")

        # Fetch action limits for later clamping
        action_low = torch.as_tensor(
            action_space.low, device=pl_module.device, dtype=torch.float32
        )
        action_high = torch.as_tensor(
            action_space.high, device=pl_module.device, dtype=torch.float32
        )

        pl_module.eval()

        use_rich_bar = isinstance(trainer.progress_bar_callback, RichProgressBar)
        pbar, task_id = self._init_progress_bar(num_episodes, phase, use_rich_bar=use_rich_bar)

        total_successes = 0
        total_truncations = 0
        total_episode_lengths = 0
        for iteration in range(num_iterations):
            iteration_seed = self._get_iteration_seed(phase, iteration)
            obs, info = env.reset(seed=iteration_seed)

            dones = torch.zeros(num_envs, dtype=torch.bool, device=pl_module.device)
            successes = torch.zeros(num_envs, dtype=torch.bool, device=pl_module.device)
            truncations = torch.zeros(num_envs, dtype=torch.bool, device=pl_module.device)
            lengths = torch.zeros(num_envs, dtype=torch.float, device=pl_module.device)

            envs_completed_this_iter = 0

            # Step until all environments within this batch have concluded
            while not dones.all():
                policy_conditioning = self._get_policy_conditioning(
                    env=env, obs=obs, device=pl_module.device
                )
                flatten_cond = flatten_tensor_dict(policy_conditioning, device=pl_module.device)

                with torch.no_grad():
                    action_seq = pl_module.get_action(flatten_cond)

                # Execute each action in the action chunk
                for i in range(pl_module.act_horizon):
                    action = action_seq[:, i]

                    if self.clamp_action:
                        action = torch.clamp(
                            action, action_low.to(action.dtype), action_high.to(action.dtype)
                        )

                    obs, reward, terminated, truncated, info = env.step(action)

                    lengths[~dones] += 1

                    truncated = torch.as_tensor(
                        truncated, dtype=torch.bool, device=pl_module.device
                    )
                    terminated = torch.as_tensor(
                        terminated, dtype=torch.bool, device=pl_module.device
                    )

                    env_is_done = terminated | truncated
                    just_finished = env_is_done & ~dones

                    if "success" in info:
                        succ = info["success"]
                        succ_tensor = torch.as_tensor(
                            succ, device=pl_module.device, dtype=torch.bool
                        )
                        successes[just_finished] = succ_tensor.view(-1)[just_finished]

                    truncations[just_finished] = truncated.view(-1)[just_finished]

                    dones = dones | env_is_done

                    newly_completed = int(dones.sum().item())
                    advance = newly_completed - envs_completed_this_iter

                    if advance > 0:
                        self._update_progress_bar(pbar, task_id, advance)

                    envs_completed_this_iter = newly_completed

                    # NOTE: In a batched GPU environment (e.g., 100 parallel envs), if one single environment truncates
                    # (e.g., drops the object or reaches the step limit) on step 2 of an 8-step action chunk, truncated.any() evaluates to True.
                    # This breaks the for loop, forcing the policy to immediately replan for the entire batch of 100 environments,
                    # even though 99 of them were perfectly happy executing the rest of their chunk.
                    # It just means that inference will run slightly slower toward the end of an evaluation epoch as environments start finishing at different times, causing more frequent replanning
                    # However, fixing this would require complex tensor masking etc., so I will keepm it this way for now for simplicity

                    if truncated.any():
                        break

            total_successes += successes.float().sum().item()
            total_truncations += truncations.float().sum().item()
            total_episode_lengths += lengths.sum().item()

        self._close_progress_bar(pbar)

        env.close()

        success_rate = total_successes / num_episodes
        avg_truncation_rate = total_truncations / num_episodes
        avg_episode_length = total_episode_lengths / num_episodes

        pl_module.log(f"{phase}/success_rate", float(success_rate), sync_dist=True, prog_bar=True)
        pl_module.log(f"{phase}/truncation_rate", float(avg_truncation_rate), sync_dist=True)
        pl_module.log(f"{phase}/avg_episode_length", float(avg_episode_length), sync_dist=True)

        pl_module.print(
            f"  [{phase.capitalize()} | Step {trainer.global_step:06d}] Rollout Success Rate: {success_rate:.4%}"
        )

    def _get_policy_conditioning(
        self,
        env: FrameStack,
        obs: torch.Tensor | dict[str, Any],
        device: torch.device | None = None,
    ) -> torch.Tensor | dict[str, Any]:
        """Extracts the conditioning state from either raw observations or physics engine states.

        Shapes:
            obs: [num_envs, cond_horizon, obs_dim]
            returns: [num_envs, cond_horizon, target_dim]
        """
        if self.use_physx_env_states:
            # .unwrapped contains the raw data from the physics engine
            policy_conditioning = env.unwrapped.get_state()  # type: ignore
            policy_conditioning = to_tensor(policy_conditioning, device=device)
        else:
            policy_conditioning = obs

        return policy_conditioning

    def _get_iteration_seed(self, phase: str, iteration: int) -> int:
        """Computes the seed for a specific evaluation iteration."""
        base_seed = self.val_seed if phase == "val" else self.test_seed
        return base_seed + iteration

    def _init_progress_bar(self, total: int, phase: str, use_rich_bar: bool) -> tuple[Any, Any]:
        """Initialize a progress bar using either Rich or TQDM."""
        if use_rich_bar:
            pbar = Progress(
                transient=True
            )  # trinsient=True makes the progress bar disappear after completion
            task_id = pbar.add_task(f"  [{phase.capitalize()}] Rollout", total=total)
            pbar.start()
            return pbar, task_id
        else:
            pbar = tqdm(
                total=total, desc=f"  [{phase.capitalize()}] Rollout", leave=False, position=2
            )  # leave=False makes the progress bar disappear after completion, positioin=2 ensures it  doesn't conflict with the bar managed by Lightning
            return pbar, None

    def _update_progress_bar(
        self, pbar: RichProgressBar | TQDMProgressBar, task_id: Any, advance: int
    ) -> None:
        """Advance the progress bar by the specified amount."""
        if isinstance(pbar, Progress):
            pbar.update(task_id, advance=advance)
        elif isinstance(pbar, tqdm):
            pbar.update(advance)
        else:
            raise ValueError(f"Unsupported progress bar type: {type(pbar).__name__}")

    def _close_progress_bar(self, pbar: Any) -> None:
        """Close the progress bar."""
        if isinstance(pbar, Progress):
            pbar.stop()
        elif isinstance(pbar, tqdm):
            pbar.close()
        else:
            raise ValueError(f"Unsupported progress bar type: {type(pbar).__name__}")

    def _validate_setup(self) -> None:
        """Ensures all required attributes from setup() are initialized."""
        missing = []
        if self.env_id is None:
            missing.append("env_id")
        if self.obs_mode is None:
            missing.append("obs_mode")
        if self.control_mode is None:
            missing.append("control_mode")
        if self.physx_backend is None:
            missing.append("physx_backend")

        if missing:
            raise ValueError(
                f"Callback setup incomplete. Missing attributes: {', '.join(missing)}. "
                "Ensure trainer.datamodule has these fields and setup() was called."
            )
