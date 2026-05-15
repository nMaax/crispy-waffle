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
    use_physx_env_states: bool = False  # keep default unless explicitly tested

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
    def __init__(self, obs_horizon: int = 2, act_horizon: int = 1, act_dim: int = 3):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.act_horizon = act_horizon
        self.act_dim = act_dim
        # tiny parameter so `.to(device)` works and module has a device
        self.p = torch.nn.Parameter(torch.zeros(()))

    # Return zeros action sequence on the same device
    def get_action(self, obs_seq: torch.Tensor | Any) -> torch.Tensor:
        assert isinstance(obs_seq, torch.Tensor)
        b = obs_seq.shape[0]
        return torch.zeros((b, self.act_horizon, self.act_dim), device=obs_seq.device)

    # Lightning boilerplate
    def validation_step(self, batch, batch_idx):
        return None

    def test_step(self, batch, batch_idx):
        return None

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


class FakeUnwrappedEnv(gym.Env):
    def __init__(self, obs: torch.Tensor, elapsed_steps: torch.Tensor):
        self._obs = obs
        self._init_raw_obs = obs
        self.elapsed_steps = elapsed_steps
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape[1:])
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,))
        self._num_envs = obs.shape[0]
        self.elapsed_steps = torch.zeros(self._num_envs, device=obs.device, dtype=torch.int32)

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
    """A tiny Gym-like env that finishes in 1 step.

    It supports both "CPU mode" (num_envs=1, sequential episodes) and "CUDA mode"
    (num_envs=num_episodes, batched episodes), purely based on what the callback passes in.
    """

    ACTION_DIM = 3

    def __init__(self, num_envs: int, obs_dim: int = 4):
        self.obs_dim = obs_dim
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

    @property
    def unwrapped(self) -> gym.Env:
        return FakeUnwrappedEnv(self._last_obs, self.elapsed_steps)

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
        info = {}
        return self._last_obs, info

    def step(self, action: torch.Tensor):
        assert action.shape[0] == self._num_envs

        self.elapsed_steps += 1

        obs = self._last_obs
        reward = torch.zeros((self._num_envs,), dtype=torch.float32)

        # Since the callback uses ignore_terminations=True, it is waiting for truncation. Set them to true
        terminated = torch.zeros((self._num_envs,), dtype=torch.bool)
        truncated = torch.ones((self._num_envs,), dtype=torch.bool)

        info = {"success": torch.ones((self._num_envs,), dtype=torch.bool)}
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


def _patch_gym(monkeypatch: pytest.MonkeyPatch):
    # Pretend env is registered
    monkeypatch.setattr(
        gym, "envs", types.SimpleNamespace(registry={"FakeManiSkill-v0": object()})
    )

    # Patch gym.make to return our fake env
    def _make(id: str, obs_mode: str, control_mode: str, num_envs: int, **kwargs):
        assert id == "FakeManiSkill-v0"
        assert isinstance(num_envs, int) and num_envs >= 1
        return FakeVectorEnv(num_envs=num_envs)

    monkeypatch.setattr(gym, "make", _make, raising=True)


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

    # Minimal trainer; disable progress bars/logging overhead
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

    # Validate triggers on_validation_epoch_end -> rollouts
    trainer.validate(model=model, datamodule=datamodule, verbose=False)

    # Test triggers on_test_epoch_end -> rollouts
    trainer.test(model=model, datamodule=datamodule, verbose=False)

    # Verify log calls exist and are exactly 1.0 (our fake env always succeeds)
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

    # Test successful fallback to Datamodule properties
    cb = RolloutEvaluationCallback(num_episodes=5, seed=123)
    mock_trainer = MagicMock(datamodule=datamodule)
    cb.setup(trainer=mock_trainer, pl_module=MagicMock(), stage="fit")

    assert cb.env_id == "FakeManiSkill-v0"
    assert cb.num_envs == 1  # CPU backend defaults to 1

    # Test explicit overrides take precedence over datamodule
    cb_override = RolloutEvaluationCallback(
        num_episodes=5, seed=123, env_id="ExplicitEnv-v0", physx_backend="physx_cuda"
    )
    # Patch cuda availability to bypass the hardware check for this assertion
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch.dict(gym.envs.registry, {"ExplicitEnv-v0": object()}),
    ):
        cb_override.setup(trainer=mock_trainer, pl_module=MagicMock(), stage="fit")
    assert cb_override.env_id == "ExplicitEnv-v0"  # Overridden
    assert cb_override.num_envs == 5  # CUDA backend defaults to num_episodes

    # Test missing parameters raise ValueError
    cb_missing = RolloutEvaluationCallback(num_episodes=5, seed=123)
    with pytest.raises(ValueError, match="must be explicitly provided"):
        # No datamodule provided to fallback to
        cb_missing.setup(trainer=MagicMock(datamodule=None), pl_module=MagicMock(), stage="fit")

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
        cb_unreg.setup(trainer=MagicMock(datamodule=None), pl_module=MagicMock(), stage="fit")

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
                trainer=MagicMock(datamodule=None), pl_module=MagicMock(), stage="fit"
            )


@pytest.mark.parametrize("invalid_num_episodes", [0, -2])
def test_run_rollouts_early_return_on_invalid_episodes(monkeypatch, invalid_num_episodes):
    """Ensures that passing 0 or negative episodes simply exits without instantiating
    environments."""
    _patch_gym(monkeypatch)

    datamodule = FakeRolloutDataModule(physx_backend="physx_cpu")
    model = FakeRolloutPolicyModule()
    cb = RolloutEvaluationCallback(num_episodes=invalid_num_episodes, seed=123)

    mock_trainer = MagicMock(datamodule=datamodule, is_global_zero=True)
    cb.setup(trainer=mock_trainer, pl_module=MagicMock(), stage="fit")

    # Mock gym.make to ensure it is NEVER called
    with patch("gymnasium.make") as mock_make:
        cb._run_rollouts(
            mock_trainer,
            model,
            invalid_num_episodes,
            "val",
        )
        mock_make.assert_not_called()


@patch("policy.algorithms.callbacks.rollout_evaluation.RecordEpisode")
@patch(
    "policy.algorithms.callbacks.rollout_evaluation.gym_utils.find_max_episode_steps_value",
    return_value=100,
)
def test_video_dir_applies_record_episode_wrapper(
    mock_find_steps, mock_record_episode, monkeypatch, capture_log
):
    """Ensures that providing a video_dir triggers the RecordEpisode wrapper with correct paths."""
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

    # By default, setup() sets render_mode to 'rgb_array' if video_dir is passed
    assert rollout_cb.render_mode == "rgb_array"

    # Ensure mock_record_episode returns our fake env so the rest of the loop succeeds
    mock_record_episode.return_value = FakeVectorEnv(num_envs=1)

    # Run validation which triggers _run_rollouts
    trainer.validate(model=model, datamodule=datamodule, verbose=False)

    # Assert RecordEpisode wrapper was applied correctly
    assert mock_record_episode.called
    call_kwargs = mock_record_episode.call_args[1]

    assert call_kwargs["output_dir"] == f"{video_dir}/val"
    assert call_kwargs["save_video"] is True
    assert call_kwargs["max_steps_per_video"] == 100  # Matches our mocked return_value
