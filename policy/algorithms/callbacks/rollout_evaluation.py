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
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from rich.progress import Progress
from tqdm import tqdm

from policy.utils import flatten_tensor_dict, to_tensor
from policy.utils.typing_utils import PolicyProtocol

# WARN: Just a notification by Transformers, however we do not use a higher version (enforced via .toml), so we can ignore this
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.deepspeed")


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
        num_envs: int | None = None,  # Default to the min(num_episodes_val, num_episodes_test)
        ignore_terminations: bool = True,  # Default to the strict ManiSkill way
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

        if num_envs is not None and num_envs > 0:
            self.num_envs = num_envs
        else:
            self.num_envs = min(num_val_episodes, num_test_episodes)
        self.ignore_terminations = ignore_terminations
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

        # Prevent secondary GPUs from spawning SAPIEN environments
        # From [Maniskill](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/quickstart.html#additional-gpu-simulation-rendering-customization):
        # "We currently do not properly support exposing multiple
        # visible CUDA devices to a single process as it has some rendering bugs at the moment."
        # The global rank 0 is not necessarely GPU 0, if e.g. you set CUDA_VISIBLE_DEVICES=1 env variable then (1->0, 2->1, etc.)
        if not trainer.is_global_zero:
            return

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
        # TODO: I could rather use AsyncVectorEnv
        # like in https://github.com/haosulab/ManiSkill/blob/main/examples/baselines/diffusion_policy/diffusion_policy/make_env.py

        env = gym.make(
            id=self.env_id,
            obs_mode=self.obs_mode,
            control_mode=self.control_mode,
            render_mode=self.render_mode,
            num_envs=self.num_envs,
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
        env = ManiSkillVectorEnv(
            env, ignore_terminations=self.ignore_terminations, record_metrics=True
        )

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

        # 1. Initialize variables globally for the whole phase
        total_successes = 0
        total_truncations = 0
        total_episode_lengths = 0
        episodes_completed = 0

        # 2. Reset the environment exactly ONCE.
        # The Auto-Reset wrapper will handle seeding subsequent episodes automatically based on this initial seed.
        seed = self.val_seed if phase == "val" else self.test_seed
        obs, info = env.reset(seed=seed)

        # 3. The Continuous Loop
        while episodes_completed < num_episodes:
            policy_conditioning = self._get_policy_conditioning(
                env=env, obs=obs, device=pl_module.device
            )
            flatten_cond = flatten_tensor_dict(policy_conditioning, device=pl_module.device)

            with torch.no_grad():
                action_seq = pl_module.get_action(flatten_cond)

            # Execute the action chunk
            for i in range(pl_module.act_horizon):
                action = action_seq[:, i]

                if self.clamp_action:
                    action = torch.clamp(
                        action, action_low.to(action.dtype), action_high.to(action.dtype)
                    )

                obs, reward, terminated, truncated, info = env.step(action)

                truncated = torch.as_tensor(truncated, dtype=torch.bool, device=pl_module.device)

                # --- METRICS & COUNTING ---

                # Consider that the info dictionary returned by the environment looks like this:
                # info
                # ├── elapsed_steps: shape=torch.Size([25]), dtype=torch.int32
                # ├── is_cubeA_grasped: shape=torch.Size([25]), dtype=torch.bool
                # ├── is_cubeA_on_cubeB: shape=torch.Size([25]), dtype=torch.bool
                # ├── is_cubeA_static: shape=torch.Size([25]), dtype=torch.bool
                # ├── success: shape=torch.Size([25]), dtype=torch.bool
                # ├── reconfigure: bool
                # ├── final_observation: shape=torch.Size([25, 2, 48]), dtype=torch.float32
                # ├── final_info
                # │   ├── elapsed_steps: shape=torch.Size([25]), dtype=torch.int32
                # │   ├── is_cubeA_grasped: shape=torch.Size([25]), dtype=torch.bool
                # │   ├── is_cubeA_on_cubeB: shape=torch.Size([25]), dtype=torch.bool
                # │   ├── is_cubeA_static: shape=torch.Size([25]), dtype=torch.bool
                # │   ├── success: shape=torch.Size([25]), dtype=torch.bool
                # │   └── episode
                # │       ├── success_once: shape=torch.Size([25]), dtype=torch.bool
                # │       ├── return: shape=torch.Size([25]), dtype=torch.float32
                # │       ├── episode_len: shape=torch.Size([25]), dtype=torch.int32
                # │       ├── reward: shape=torch.Size([25]), dtype=torch.float32
                # │       └── success_at_end: shape=torch.Size([25]), dtype=torch.bool
                # ├── _final_info: shape=torch.Size([25]), dtype=torch.bool
                # ├── _final_observation: shape=torch.Size([25]), dtype=torch.bool
                # └── _elapsed_steps: shape=torch.Size([25]), dtype=torch.bool

                if "_final_info" in info:
                    mask = info["_final_info"]

                    # Get the indices of the environments that just finished
                    finished_envs = mask.nonzero(as_tuple=True)[0]

                    for env_idx in finished_envs:
                        # 4. Safely stop if we overshot our target
                        if episodes_completed >= num_episodes:
                            continue

                        episodes_completed += 1

                        # GPU Backend: final_info is a dictionary of batched tensors
                        if isinstance(info["final_info"], dict):
                            # Grab success (sometimes it's at the top level, sometimes inside 'episode')
                            if "success" in info["final_info"]:
                                total_successes += int(
                                    info["final_info"]["success"][env_idx].item()
                                )
                            elif "success" in info["final_info"].get("episode", {}):
                                total_successes += int(
                                    info["final_info"]["episode"]["success"][env_idx].item()
                                )

                            # Grab length
                            if (
                                "episode" in info["final_info"]
                                and "episode_len" in info["final_info"]["episode"]
                            ):
                                total_episode_lengths += int(
                                    info["final_info"]["episode"]["episode_len"][env_idx].item()
                                )

                        # CPU Backend Fallback: final_info is a list/tuple of dicts
                        elif isinstance(info["final_info"], list | tuple):
                            env_info = info["final_info"][env_idx]

                            if "success" in env_info:
                                total_successes += int(env_info["success"])
                            elif "success" in env_info.get("episode", {}):
                                total_successes += int(env_info["episode"]["success"])

                            if "episode" in env_info and "episode_len" in env_info["episode"]:
                                total_episode_lengths += int(env_info["episode"]["episode_len"])

                        # Truncations can safely be pulled from the global truncated tensor
                        total_truncations += int(truncated[env_idx].item())

                        self._update_progress_bar(pbar, task_id, 1)

                # 5. Break out of the chunk if we hit our global target
                if episodes_completed >= num_episodes:
                    break

                # --- THE DESYNC TOGGLE ---
                if self.ignore_terminations:
                    # ManiSkill Way: Break chunk only if everyone timed out
                    if truncated.all():
                        break
                else:
                    # Dynamic Way: Break chunk if anyone timed out
                    if truncated.any():
                        break

        self._close_progress_bar(pbar)

        env.close()

        success_rate = total_successes / num_episodes
        avg_truncation_rate = total_truncations / num_episodes
        avg_episode_length = total_episode_lengths / num_episodes

        # We do not support multi GPUs in maniskill, so we set sync_dist=False
        pl_module.log(f"{phase}/success_rate", float(success_rate), sync_dist=False, prog_bar=True)
        pl_module.log(f"{phase}/truncation_rate", float(avg_truncation_rate), sync_dist=False)
        pl_module.log(f"{phase}/avg_episode_length", float(avg_episode_length), sync_dist=False)

        pl_module.print(
            f"  [{phase.capitalize()} | Step {trainer.global_step:06d}] Rollout Success Rate: {success_rate:.4%}"
        )

    def _get_iteration_seed(self, phase: str, iteration: int) -> int:
        """Computes the seed for a specific evaluation iteration."""
        base_seed = self.val_seed if phase == "val" else self.test_seed
        return base_seed + iteration

    def _get_policy_conditioning(
        self,
        env: FrameStack | ManiSkillVectorEnv,
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
