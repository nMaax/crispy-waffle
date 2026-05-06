import time
import warnings
from typing import Any, cast

import gymnasium as gym
import hydra_zen
import lightning as L
import mani_skill.envs  # noqa: F401
import torch
from gymnasium.spaces import Box
from lightning.fabric.utilities import rank_zero_warn
from lightning.pytorch.callbacks import RichProgressBar, TQDMProgressBar
from lightning.pytorch.utilities import rank_zero_info
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers import FrameStack, RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from rich.progress import Progress
from tqdm import tqdm

import policy.algorithms.environments  # noqa: F401
from policy.utils import flatten_tensor_dict, to_tensor
from policy.utils.adapters import NoOpAdapter
from policy.utils.typing_utils import AdapterProtocol, HydraConfigFor, PolicyProtocol

# WARN: Just a notification by Transformers, however we do not use a higher version (enforced via .toml), so we can ignore this
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.deepspeed")


class RolloutEvaluationCallback(L.Callback):
    """A Lightning Callback for performing rollout evaluation of a policy in a ManiSkill
    environment."""

    OFFSET_SEED_VAL: int = 42000
    OFFSET_SEED_TEST: int = 67000

    def __init__(
        self,
        adapter: HydraConfigFor[AdapterProtocol] | None = None,
        num_val_episodes: int = 20,
        num_test_episodes: int = 100,
        max_episode_steps: int | None = None,
        num_envs: int | None = None,
        ignore_terminations: bool = True,
        clamp_action: bool = True,
        video_dir: str | None = None,
        render_mode: str | None = None,
        seed: int | None = None,
        # Optional overrides to decouple from datamodule
        env_id: str | None = None,
        obs_mode: str | None = None,
        control_mode: str | None = None,
        physx_backend: str | None = None,
        use_physx_env_states: bool | None = None,
    ):
        super().__init__()

        if seed is None:
            raise ValueError("seed must be provided.")

        self.adapter_config = adapter
        self.adapter: AdapterProtocol | None = None

        self.num_val_episodes = num_val_episodes
        self.num_test_episodes = num_test_episodes

        self.num_envs = num_envs
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

        # Set environment properties directly (fallback to datamodule happens in setup)
        self.env_id = env_id
        self.obs_mode = obs_mode
        self.control_mode = control_mode
        self.physx_backend = physx_backend
        self.use_physx_env_states = use_physx_env_states

        rank_zero_info(
            f"Seeds for rollout simulation fetched from main seed: {seed}\n"
            f"\tValidation seed: {self.val_seed}\n"
            f"\tTest seed: {self.test_seed}"
        )

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:

        if self.adapter_config is not None:
            self.adapter = hydra_zen.instantiate(self.adapter_config)
        else:
            self.adapter = NoOpAdapter()

        rank_zero_info(f"Using adapter: {type(self.adapter).__name__}")

        datamodule = getattr(trainer, "datamodule", None)

        # Fallback logic: Use provided config, otherwise fetch from Datamodule
        def _resolve_param(param_value: Any, param_name: str) -> Any:
            if param_value is not None:
                return param_value
            if datamodule is not None and hasattr(datamodule, param_name):
                return getattr(datamodule, param_name)
            raise ValueError(
                f"`{param_name}` must be explicitly provided to RolloutEvaluationCallback "
                f"or attached to trainer.datamodule."
            )

        self.env_id = _resolve_param(self.env_id, "env_id")
        self.obs_mode = _resolve_param(self.obs_mode, "obs_mode")
        self.control_mode = _resolve_param(self.control_mode, "control_mode")
        self.physx_backend = _resolve_param(self.physx_backend, "physx_backend")
        self.use_physx_env_states = _resolve_param(
            self.use_physx_env_states, "use_physx_env_states"
        )

        if self.env_id not in gym.envs.registry:
            raise RuntimeError(
                f"Environment '{self.env_id}' is not registered in Gymnasium + Maniskill."
            )

        if "cuda" in self.physx_backend.lower() and not torch.cuda.is_available():
            raise RuntimeError(
                f"Rollout specifies CUDA backend '{self.physx_backend}', "
                "but CUDA is not available on this machine. Cannot run parallel CUDA environments."
            )

        if self.num_envs is None:
            if "cpu" in self.physx_backend.lower():
                self.num_envs = 1
            else:
                self.num_envs = min(self.num_val_episodes, self.num_test_episodes)

            rank_zero_info(
                f"In Rollout variable `num_envs` was not provided, automatically set to {self.num_envs} based on the physx_backend {self.physx_backend}."
            )

        if "cpu" in self.physx_backend.lower() and self.num_envs > 1:
            rank_zero_warn(
                f"Dataset specifies CPU backend but num_envs in Rollout was set to {self.num_envs}. "
                "num_envs > 1 implies the use of GPU backed for simulations, however the model was trained on a CPU backend. "
                "This is highly unadvised and may lead to failing simulations due to float precision mismatch. "
                "We will proceed with num_envs > 1, however consider setting num_envs=1 or switching to a GPU replayed data for training."
            )
        elif "cuda" in self.physx_backend.lower() and self.num_envs == 1:
            rank_zero_warn(
                "Dataset specifies CUDA backend but num_envs in Rollout was set to 1. "
                "num_envs=1 implies the use of CPU backend for simulations, however the model was trained on a GPU backend. "
                "This is highly unadvised and may lead to failing simulations due to float precision mismatch. "
                "We will proceed with num_envs=1, however consider setting num_envs > 1 or switching to a CPU replayed data for training."
            )

        rank_zero_info(
            f"Rollout Config setup complete:\n"
            f"\tenv_id: {self.env_id},\n"
            f"\tobs_mode: {self.obs_mode} ({self.use_physx_env_states=}),\n"
            f"\tcontrol_mode: {self.control_mode},\n"
            f"\tbackend: {self.physx_backend}\n"
            f"\tnum_envs: {self.num_envs}\n"
            f"\tnum_episodes (val/test): {self.num_val_episodes} / {self.num_test_episodes}"
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

        if not isinstance(self.adapter, AdapterProtocol):
            raise AttributeError(
                f"Expected the adapter to implement AdapterProtocol, "
                f"but got {type(self.adapter).__name__}."
            )

        self._validate_setup()
        assert self.env_id is not None

        if num_episodes <= 0:
            return

        # On CUDA we run all episodes in parallel, on CPU we run sequentially
        # TODO: I could rather use Sync/AsyncVectorEnv, or maybe CPUGym?
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

        total_success_once = 0
        total_success_at_end = 0
        total_truncations = 0
        total_episode_lengths = 0
        episodes_completed = 0

        # Reset will handle seeding subsequent episodes automatically based on this initial seed
        seed = self.val_seed if phase == "val" else self.test_seed
        obs, info = env.reset(seed=seed)

        if self.render_mode == "human":
            env.render()

        while episodes_completed < num_episodes:
            # BUG: When using raw physx tensors we get a mismatch of shapes since
            # we did not stack subsequent frames physx_states on the conditioning history
            # the most clean fix here is to make a custom environment wrapper that handles the history by itself
            # using the same interface as the maniskill FrameStack wrapper, but stacking the physx states instead of the observations
            policy_conditioning = self._get_policy_conditioning(
                env=env, obs=obs, device=pl_module.device
            )

            permuted_policy_conditioning = self.adapter.apply(policy_conditioning)

            flatten_cond = flatten_tensor_dict(
                permuted_policy_conditioning, device=pl_module.device
            )

            with torch.no_grad():
                action_seq = pl_module.get_action(flatten_cond)

            # Execute the action chunk
            for i in range(pl_module.act_horizon):
                action = action_seq[:, i]

                if self.clamp_action:
                    action = torch.clamp(
                        action, action_low.to(action.dtype), action_high.to(action.dtype)
                    )

                # We dont use terminated since "_final_info" in the info dict will already identify any early terminations
                obs, reward, terminated, truncated, info = env.step(action)

                if self.render_mode == "human":
                    time.sleep(0.05)
                    env.render()

                obs = to_tensor(obs, device=pl_module.device, dtype=torch.float32)
                truncated = torch.as_tensor(truncated, device=pl_module.device, dtype=torch.bool)

                # Consider that the info dictionary returned by the environment looks like this (for StackCube-v1):
                #
                # > *final_* entries will only appear once the env is effectively done,
                # > so in the first iteration do not assume they exist
                #
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

                # "_final_info" is a boolean mask indicating which envs finished THIS EXACT step,
                # it is NOT a state variable: once an environment finhishes, "_final_info" will be True for that environment for that single step only!
                # The only other time it will be True is when, given ignore_terminations=True, the environment reaches the time limit and truncates:
                # in that case "_final_info" will be True for all environments on the last step
                if "_final_info" in info:
                    mask = info["_final_info"]

                    # Get the indices of the environments that just finished
                    finished_envs = mask.nonzero(as_tuple=True)[0]

                    for env_idx in finished_envs:
                        # Safely stop if we overshot our target
                        if episodes_completed >= num_episodes:
                            continue

                        episodes_completed += 1

                        ep_dict = info["final_info"].get("episode", {})

                        # Track success_once
                        if "success_once" in ep_dict:
                            total_success_once += int(ep_dict["success_once"][env_idx].item())
                        elif "success" in info["final_info"]:
                            # Fallback if "episode" sub-dict can't be found
                            total_success_once += int(
                                info["final_info"]["success"][env_idx].item()
                            )

                        # Track success_at_end (may not be present if ignore_terminations=False)
                        if "success_at_end" in ep_dict:
                            total_success_at_end += int(ep_dict["success_at_end"][env_idx].item())

                        # Track length (ManiSkill uses 'episode_len', Gym uses 'l')
                        if "episode_len" in ep_dict:
                            total_episode_lengths += int(ep_dict["episode_len"][env_idx].item())

                        total_truncations += int(truncated[env_idx].item())
                        self._update_progress_bar(pbar, task_id, 1)

                if episodes_completed >= num_episodes:
                    break

                if self.ignore_terminations:
                    if truncated.all():
                        break
                else:
                    # NOTE: In a batched GPU environment, if one single environment truncates
                    # on step 2 of an 8-step action chunk, truncated.any() evaluates to True.
                    # This breaks the for loop, forcing the policy to immediately replan for the entire batch of the next environments,
                    # A truly pure asynchronous approach would require a complicated work of masks and action chunking, which is honestly useless.
                    #
                    # In fact, this is actually desired: what happens if you set ignore_terminations=False
                    # is that early successes cause that specific environment to quietly auto-reset in the background.
                    # This causes the robot to execute those actions that were not yet executed because the robot succceded midway
                    # in the new "ghost" episode.
                    #
                    # Breaking with truncated.any() forces such ghost episodes to being discarded once the other late environments finish.
                    #
                    # Conversely, when ignore_terminations=True, the environments are forced to wait out the entire
                    # time limit (holding their success), preventing early resets and keeping all environment
                    # stopwatches rigidly synchronized without any ghost actions or early replanning.
                    #
                    # In other words, in ignore_terminations = False we do not test our model capacity to "hold" the success state while waiting for the others.
                    # Generally, always prefer ignore_terminations = True

                    if truncated.any():
                        break

        self._close_progress_bar(pbar)

        env.close()

        success_once_rate = total_success_once / num_episodes
        success_at_end_rate = total_success_at_end / num_episodes
        avg_truncation_rate = total_truncations / num_episodes
        avg_episode_length = total_episode_lengths / num_episodes

        # We do not support multi GPUs in maniskill, so we set sync_dist=False
        pl_module.log(
            f"{phase}/success_once_rate", float(success_once_rate), sync_dist=False, prog_bar=True
        )
        pl_module.log(f"{phase}/success_at_end_rate", float(success_at_end_rate), sync_dist=False)
        pl_module.log(f"{phase}/truncation_rate", float(avg_truncation_rate), sync_dist=False)
        pl_module.log(f"{phase}/avg_episode_length", float(avg_episode_length), sync_dist=False)

        pl_module.print(
            f"  [{phase.capitalize()} | Step {trainer.global_step:06d} (E{trainer.current_epoch:04d})] "
            f"Success (once): {success_once_rate:.4%} | Success (at end): {success_at_end_rate:.4%}"
        )

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
