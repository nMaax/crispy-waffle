import types
from dataclasses import dataclass
from typing import Any, TypeVar
from unittest.mock import MagicMock, patch

import gymnasium as gym
import lightning as L
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from policy.algorithms.callbacks.rollout_evaluation import RolloutEvaluationCallback

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


@dataclass
class FakeRolloutDataModule(L.LightningDataModule):
    env_id: str = "FakeManiSkill-v0"
    obs_mode: str = "state"
    control_mode: str = "pd_joint_pos"
    physx_backend: str | None = None  # set to "cuda" to trigger batched mode
    use_physx_env_states: bool = False

    def __post_init__(self):
        super().__init__()

    # We don't actually use batches in the callback, but Lightning wants a loader.
    def val_dataloader(self):
        ds = TensorDataset(torch.zeros(2, 1))
        return DataLoader(ds, batch_size=1)

    def test_dataloader(self):
        ds = TensorDataset(torch.zeros(2, 1))
        return DataLoader(ds, batch_size=1)


class FakeRolloutPolicyModule(L.LightningModule):
    """Non-goal-conditioned policy that returns a fixed action sequence.

    ``action_scale`` controls the magnitude of returned actions (useful for
    clamp_action tests).  ``record_nit`` records the num_inference_steps
    value received, if any.
    """

    def __init__(
        self,
        obs_horizon: int = 2,
        act_horizon: int = 1,
        act_dim: int = 3,
        action_scale: float = 0.0,
        record_nit: bool = False,
    ):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.act_horizon = act_horizon
        self.act_dim = act_dim
        self.action_scale = action_scale
        self.record_nit = record_nit
        self.last_num_inference_steps = None
        # tiny parameter so `.to(device)` works and module has a device
        self.p = torch.nn.Parameter(torch.zeros(()))

    def get_action(
        self,
        obs_seq: torch.Tensor | Any,
        num_inference_steps: int | None = None,
        output_clip_range: tuple | None = None,
    ) -> torch.Tensor:
        assert isinstance(obs_seq, torch.Tensor)
        b = obs_seq.shape[0]
        if self.record_nit:
            self.last_num_inference_steps = num_inference_steps
        return (
            torch.ones((b, self.act_horizon, self.act_dim), device=obs_seq.device)
            * self.action_scale
        )

    # Lightning boilerplate
    def validation_step(self, batch, batch_idx):
        return None

    def test_step(self, batch, batch_idx):
        return None

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


class FakeGoalConditionedPolicyModule(FakeRolloutPolicyModule):
    """Goal-conditioned variant: has ``goal_conditioned=True`` and accepts a ``goal`` arg."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.goal_conditioned = True

    def reset(self):
        pass

    def get_action(
        self,
        obs_seq: torch.Tensor | Any,
        goal: torch.Tensor | Any = None,
        num_inference_steps: int | None = None,
        output_clip_range: tuple | None = None,
    ) -> torch.Tensor:
        assert isinstance(obs_seq, torch.Tensor)
        b = obs_seq.shape[0]
        if self.record_nit:
            self.last_num_inference_steps = num_inference_steps
        return (
            torch.ones((b, self.act_horizon, self.act_dim), device=obs_seq.device)
            * self.action_scale
        )


class FakeUnwrappedEnv(gym.Env):
    def __init__(
        self, obs: torch.Tensor, elapsed_steps: torch.Tensor, goal_conditioned: bool = False
    ):
        self._obs = obs
        self._init_raw_obs = obs
        self.elapsed_steps = elapsed_steps
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape[1:])
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,))
        self._num_envs = obs.shape[0]
        self.elapsed_steps = torch.zeros(self._num_envs, device=obs.device, dtype=torch.int32)
        # Only attach generate_heuristic_goal when goal-conditioned, so that
        # ``hasattr(env, "generate_heuristic_goal")`` is False otherwise.
        if goal_conditioned:
            self.generate_heuristic_goal = lambda: torch.randn_like(self._obs)

    @property
    def device(self) -> torch.device:
        return self._obs.device

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def single_observation_space(self):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=self._obs.shape[1:])

    def update_obs_space(self, *args, **kwargs):
        pass


class FakeVectorEnv(gym.Env):
    """A tiny Gym-like env that finishes in ``episode_len`` steps.

    Emits the ``_final_info`` / ``final_info`` / ``episode`` dict structure that
    the production callback reads for per-episode aggregation.
    """

    ACTION_DIM = 3

    def __init__(
        self,
        num_envs: int,
        obs_dim: int = 4,
        episode_len: int = 1,
        success: bool = True,
        goal_conditioned: bool = False,
    ):
        self.obs_dim = obs_dim
        self.episode_len = episode_len
        self._success = success
        self._goal_conditioned = goal_conditioned
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.ACTION_DIM,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self._num_envs = num_envs
        self.elapsed_steps = torch.zeros(self._num_envs, dtype=torch.int32)
        self._closed = False
        self._last_obs = torch.zeros((self.num_envs, self.obs_dim), dtype=torch.float32)
        self.last_action = None

    @property
    def unwrapped(self) -> gym.Env:
        return FakeUnwrappedEnv(
            self._last_obs, self.elapsed_steps, goal_conditioned=self._goal_conditioned
        )

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def single_observation_space(self):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,))

    @property
    def single_action_space(self):
        return self.action_space

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is None:
            seed = 0

        g = torch.Generator().manual_seed(int(seed))
        self._last_obs = torch.randn((self.num_envs, self.obs_dim), generator=g)
        self.elapsed_steps = torch.zeros(self._num_envs, dtype=torch.int32)
        info = {}
        return self._last_obs, info

    def step(self, action: torch.Tensor):
        assert action.shape[0] == self._num_envs
        self.last_action = action

        self.elapsed_steps += 1
        obs = self._last_obs
        reward = torch.zeros((self._num_envs,), dtype=torch.float32)

        finished = self.elapsed_steps >= self.episode_len
        terminated = torch.zeros((self._num_envs,), dtype=torch.bool)
        truncated = finished.clone()

        success_val = torch.ones((self._num_envs,), dtype=torch.bool) * self._success
        info: dict[str, Any] = {"success": success_val}

        if finished.any():
            mask = finished
            info["_final_info"] = mask
            info["final_info"] = {
                "success": success_val,
                "episode": {
                    "success_once": success_val,
                    "success_at_end": success_val,
                    "episode_len": self.elapsed_steps.to(torch.float32),
                },
            }
            # Auto-reset finished envs (simulating ManiSkill background reset)
            self.elapsed_steps[finished] = 0

        return obs, reward, terminated, truncated, info

    def close(self):
        self._closed = True


@pytest.fixture
def capture_log(monkeypatch: pytest.MonkeyPatch):
    """Capture calls to LightningModule.log without requiring a logger."""
    calls: list[tuple[str, Any, dict[str, Any]]] = []

    def _log(self: L.LightningModule, name: str, value: Any, *args, **kwargs):
        calls.append((name, value, kwargs))

    monkeypatch.setattr(L.LightningModule, "log", _log, raising=True)
    return calls


def _patch_gym(monkeypatch: pytest.MonkeyPatch, **env_kwargs):
    """Patches gym to use FakeVectorEnv.

    Returns a list that will hold the created env.
    """
    created: list[FakeVectorEnv] = []

    # Pretend env is registered
    monkeypatch.setattr(
        gym, "envs", types.SimpleNamespace(registry={"FakeManiSkill-v0": object()})
    )

    def _make(id: str, num_envs: int, **kwargs):
        assert isinstance(num_envs, int) and num_envs >= 1
        env = FakeVectorEnv(num_envs=num_envs, **env_kwargs)
        created.append(env)
        return env

    monkeypatch.setattr(gym, "make", _make, raising=True)
    return created


# ====================================================================== #
# Existing tests (fixed)
# ====================================================================== #


@pytest.mark.parametrize(
    "physx_backend",
    [
        pytest.param("physx_cpu"),
        pytest.param(
            "physx_cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA available."),
        ),
    ],
)
def test_rollout_evaluation_callback_cpu_mode_logs_success_rate(
    monkeypatch: pytest.MonkeyPatch,
    capture_log: list[tuple[str, Any, dict[str, Any]]],
    physx_backend: str,
):
    _patch_gym(monkeypatch)

    datamodule = FakeRolloutDataModule(physx_backend=physx_backend)
    model = FakeRolloutPolicyModule()

    rollout_cb = RolloutEvaluationCallback(num_episodes=5, seed=123)

    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=[rollout_cb],
        max_epochs=1,
    )

    trainer.validate(model=model, datamodule=datamodule, verbose=False)
    trainer.test(model=model, datamodule=datamodule, verbose=False)

    logged = {name: float(value) for (name, value, _kw) in capture_log}
    assert logged["val/success_once_rate"] == 1.0
    assert logged["test/success_once_rate"] == 1.0


def test_setup_parameter_resolution_and_validation(monkeypatch):
    _patch_gym(monkeypatch)
    datamodule = FakeRolloutDataModule(
        env_id="FakeManiSkill-v0",
        obs_mode="state",
        control_mode="pd_joint_pos",
        physx_backend="physx_cpu",
    )

    # Use a FakeRolloutPolicyModule so obs_horizon is a real int (FrameStack needs int)
    fake_module = FakeRolloutPolicyModule(obs_horizon=2)

    # Test successful fallback to Datamodule properties
    cb = RolloutEvaluationCallback(num_episodes=5, seed=123)
    mock_trainer = MagicMock(datamodule=datamodule)
    cb.setup(trainer=mock_trainer, pl_module=fake_module, stage="fit")

    assert cb.env_id == "FakeManiSkill-v0"
    assert cb.num_envs == 1  # CPU backend defaults to 1

    # Test explicit overrides take precedence over datamodule
    cb_override = RolloutEvaluationCallback(
        num_episodes=5, seed=123, env_id="ExplicitEnv-v0", physx_backend="physx_cuda"
    )
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch.dict(gym.envs.registry, {"ExplicitEnv-v0": object()}),
    ):
        cb_override.setup(trainer=mock_trainer, pl_module=fake_module, stage="fit")
    assert cb_override.env_id == "ExplicitEnv-v0"
    assert cb_override.num_envs == 5  # CUDA backend defaults to num_episodes

    # Test missing parameters raise ValueError
    cb_missing = RolloutEvaluationCallback(num_episodes=5, seed=123)
    with pytest.raises(ValueError, match="must be explicitly provided"):
        cb_missing.setup(trainer=MagicMock(datamodule=None), pl_module=fake_module, stage="fit")

    # Test unregistered env raises RuntimeError
    cb_unreg = RolloutEvaluationCallback(
        env_id="Unknown-v0",
        obs_mode="state",
        control_mode="pd",
        physx_backend="physx_cpu",
        num_episodes=5,
        seed=123,
    )
    with pytest.raises(RuntimeError, match="not registered in Gymnasium"):
        cb_unreg.setup(trainer=MagicMock(datamodule=None), pl_module=fake_module, stage="fit")

    # Test CUDA backend requirement raises error when CUDA is unavailable
    cb_cuda_fail = RolloutEvaluationCallback(
        env_id="FakeManiSkill-v0",
        obs_mode="state",
        control_mode="pd",
        physx_backend="physx_cuda",
        num_episodes=5,
        seed=123,
    )
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            cb_cuda_fail.setup(
                trainer=MagicMock(datamodule=None), pl_module=fake_module, stage="fit"
            )


@pytest.mark.parametrize("invalid_num_episodes", [0, -2])
def test_run_rollouts_early_return_on_invalid_episodes(monkeypatch, invalid_num_episodes):
    _patch_gym(monkeypatch)

    datamodule = FakeRolloutDataModule(physx_backend="physx_cpu")
    model = FakeRolloutPolicyModule()
    cb = RolloutEvaluationCallback(num_episodes=invalid_num_episodes, seed=123)

    mock_trainer = MagicMock(datamodule=datamodule, is_global_zero=True)
    cb.setup(trainer=mock_trainer, pl_module=model, stage="fit")

    with patch("gymnasium.make") as mock_make:
        cb._run_rollouts(mock_trainer, model, invalid_num_episodes, "val")
        mock_make.assert_not_called()


@patch("policy.algorithms.callbacks.rollout_evaluation.RecordEpisode")
@patch(
    "policy.algorithms.callbacks.rollout_evaluation.gym_utils.find_max_episode_steps_value",
    return_value=100,
)
def test_video_dir_applies_record_episode_wrapper(
    mock_find_steps, mock_record_episode, monkeypatch, capture_log
):
    _patch_gym(monkeypatch)
    datamodule = FakeRolloutDataModule(physx_backend="physx_cpu")
    model = FakeRolloutPolicyModule()

    video_dir = "/tmp/fake_video_dir"
    rollout_cb = RolloutEvaluationCallback(num_episodes=1, seed=123, video_dir=video_dir)

    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=[rollout_cb],
        max_epochs=1,
    )

    assert rollout_cb.render_mode == "rgb_array"
    mock_record_episode.return_value = FakeVectorEnv(num_envs=1)

    trainer.validate(model=model, datamodule=datamodule, verbose=False)

    assert mock_record_episode.called
    call_kwargs = mock_record_episode.call_args[1]

    assert call_kwargs["output_dir"] == f"{video_dir}/val"
    assert call_kwargs["save_video"] is True
    assert call_kwargs["max_steps_per_video"] == 100


# ====================================================================== #
# New tests
# ====================================================================== #


def test_seed_none_raises():
    """RolloutEvaluationCallback must reject seed=None."""
    with pytest.raises(ValueError, match="seed must be provided"):
        RolloutEvaluationCallback(seed=None)


def test_all_four_metrics_logged(monkeypatch, capture_log):
    """Ensures all 4 metrics (success_once, success_at_end, truncation, avg_length) are logged."""
    _patch_gym(monkeypatch, episode_len=1, success=True)
    datamodule = FakeRolloutDataModule(physx_backend="physx_cpu")
    model = FakeRolloutPolicyModule()

    rollout_cb = RolloutEvaluationCallback(num_episodes=3, seed=42)
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=[rollout_cb],
        max_epochs=1,
    )
    trainer.validate(model=model, datamodule=datamodule, verbose=False)

    logged = {name: float(value) for (name, value, _kw) in capture_log}
    assert "val/success_once_rate" in logged
    assert "val/success_at_end_rate" in logged
    assert "val/truncation_rate" in logged
    assert "val/avg_episode_length" in logged


def test_clamp_action_clamps_out_of_bounds(monkeypatch, capture_log):
    """With clamp_action=True, out-of-bounds policy actions are clamped to the action space."""
    created = _patch_gym(monkeypatch, episode_len=1)
    datamodule = FakeRolloutDataModule(physx_backend="physx_cpu")
    # Policy returns actions with magnitude 5.0 (well outside [-1, 1])
    model = FakeRolloutPolicyModule(action_scale=5.0)

    rollout_cb = RolloutEvaluationCallback(num_episodes=1, seed=42, clamp_action=True)
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=[rollout_cb],
        max_epochs=1,
    )
    trainer.validate(model=model, datamodule=datamodule, verbose=False)

    # The FakeVectorEnv recorded the last action it received
    assert len(created) > 0
    last_action = created[0].last_action
    assert last_action is not None
    assert last_action.min() >= -1.0
    assert last_action.max() <= 1.0


def test_clamp_action_disabled_passes_through(monkeypatch, capture_log):
    """With clamp_action=False, out-of-bounds actions are NOT clamped."""
    created = _patch_gym(monkeypatch, episode_len=1)
    datamodule = FakeRolloutDataModule(physx_backend="physx_cpu")
    model = FakeRolloutPolicyModule(action_scale=5.0)

    rollout_cb = RolloutEvaluationCallback(num_episodes=1, seed=42, clamp_action=False)
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=[rollout_cb],
        max_epochs=1,
    )
    trainer.validate(model=model, datamodule=datamodule, verbose=False)

    assert len(created) > 0
    last_action = created[0].last_action
    assert last_action is not None
    assert last_action.abs().max() > 1.0


def test_teardown_closes_env(monkeypatch):
    """Teardown() must close the rollout environment."""
    _patch_gym(monkeypatch)
    datamodule = FakeRolloutDataModule(physx_backend="physx_cpu")
    model = FakeRolloutPolicyModule()

    cb = RolloutEvaluationCallback(num_episodes=1, seed=42)
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=[cb],
        max_epochs=1,
    )
    trainer.validate(model=model, datamodule=datamodule, verbose=False)

    # After rollout, _gym_env should exist; teardown should close it
    assert hasattr(cb, "_gym_env")
    cb.teardown(trainer=MagicMock(), pl_module=model, stage="validate")
    assert cb._gym_env._closed is True


def test_num_inference_steps_propagation(monkeypatch, capture_log):
    """num_inference_steps from the callback config must reach get_action."""
    _patch_gym(monkeypatch, episode_len=1)
    datamodule = FakeRolloutDataModule(physx_backend="physx_cpu")
    model = FakeRolloutPolicyModule(record_nit=True)

    rollout_cb = RolloutEvaluationCallback(num_episodes=1, seed=42, num_inference_steps=7)
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=[rollout_cb],
        max_epochs=1,
    )
    trainer.validate(model=model, datamodule=datamodule, verbose=False)

    assert model.last_num_inference_steps == 7


def test_goal_conditioned_rollout(monkeypatch, capture_log):
    """Goal-conditioned policy + env with generate_heuristic_goal runs without error."""
    _patch_gym(monkeypatch, episode_len=1, goal_conditioned=True)
    datamodule = FakeRolloutDataModule(physx_backend="physx_cpu")
    model = FakeGoalConditionedPolicyModule(obs_horizon=2, act_horizon=1, act_dim=3)

    rollout_cb = RolloutEvaluationCallback(num_episodes=1, seed=42)
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=[rollout_cb],
        max_epochs=1,
    )
    trainer.validate(model=model, datamodule=datamodule, verbose=False)

    logged = {name: float(value) for (name, value, _kw) in capture_log}
    assert "val/success_once_rate" in logged


def test_goal_conditioned_env_without_generate_heuristic_goal_raises(monkeypatch, capture_log):
    """If the policy is goal-conditioned but the env lacks generate_heuristic_goal, raise."""
    # goal_conditioned=False on the env -> no generate_heuristic_goal on _inner_env
    _patch_gym(monkeypatch, episode_len=1, goal_conditioned=False)
    datamodule = FakeRolloutDataModule(physx_backend="physx_cpu")
    model = FakeGoalConditionedPolicyModule(obs_horizon=2, act_horizon=1, act_dim=3)

    rollout_cb = RolloutEvaluationCallback(num_episodes=1, seed=42)
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=[rollout_cb],
        max_epochs=1,
    )
    with pytest.raises(AttributeError, match="generate_heuristic_goal"):
        trainer.validate(model=model, datamodule=datamodule, verbose=False)


def test_rollout_evaluation_forwards_robot_uids(monkeypatch):
    make_kwargs_captured = {}

    monkeypatch.setattr(
        gym, "envs", types.SimpleNamespace(registry={"FakeManiSkill-v0": object()})
    )

    def _make(id: str, num_envs: int, **kwargs):
        make_kwargs_captured.update(kwargs)
        return FakeVectorEnv(num_envs=num_envs)

    monkeypatch.setattr(gym, "make", _make, raising=True)

    datamodule = FakeRolloutDataModule(physx_backend="physx_cpu")
    model = FakeRolloutPolicyModule(obs_horizon=2, act_horizon=1, act_dim=3)

    rollout_cb = RolloutEvaluationCallback(num_episodes=1, seed=42, robot_uids="panda_wristcam")
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=[rollout_cb],
        max_epochs=1,
    )
    trainer.validate(model=model, datamodule=datamodule, verbose=False)
    assert make_kwargs_captured.get("robot_uids") == "panda_wristcam"
