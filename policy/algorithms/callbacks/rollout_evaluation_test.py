import types
from dataclasses import dataclass
from typing import Any, TypeVar

import gymnasium as gym
import lightning as L
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from policy.algorithms.callbacks.rollout_evaluation import RolloutEvaluationCallback

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

# Should test setup works
# Should test run_rollouts with num_envs=(-2, 0, 1, 5), with


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

    # Used only if use_phsyx_env_states=True
    def get_state(self):
        return self._obs

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
