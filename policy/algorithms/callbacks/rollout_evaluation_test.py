import types
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import lightning as L
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from policy.algorithms.callbacks.rollout_evaluation import RolloutEvaluationCallback


@dataclass
class FakeRolloutDataModule(L.LightningDataModule):
    """Just enough attributes for RolloutEvaluationCallback.setup()"""

    env_id: str = "FakeManiSkill-v0"
    obs_mode: str = "state"
    control_mode: str = "pd_joint_pos"
    physx_backend: str | None = None  # set to "cuda" to trigger batched mode
    use_phsyx_env_states: bool = False  # keep default unless explicitly tested

    def __post_init__(self):
        super().__init__()

    # We don't actually use batches in the callback, but Lightning wants a loader.
    def val_dataloader(self):
        ds = TensorDataset(torch.zeros(2, 1))
        return DataLoader(ds, batch_size=1)

    def test_dataloader(self):
        ds = TensorDataset(torch.zeros(2, 1))
        return DataLoader(ds, batch_size=1)


class FakePolicyModule(L.LightningModule):
    """A minimal LightningModule satisfying the PolicyProtocol used by the callback."""

    def __init__(self, cond_horizon: int = 2, act_horizon: int = 1, act_dim: int = 3):
        super().__init__()
        self.cond_horizon = cond_horizon
        self.act_horizon = act_horizon
        self.act_dim = act_dim
        # tiny parameter so `.to(device)` works and module has a device
        self.p = torch.nn.Parameter(torch.zeros(()))

    # Return zeros action sequence on the same device
    def get_action(self, cond_seq: torch.Tensor | Any) -> torch.Tensor:
        assert isinstance(cond_seq, torch.Tensor)
        b = cond_seq.shape[0]
        return torch.zeros((b, self.act_horizon, self.act_dim), device=cond_seq.device)

    # Lightning boilerplate
    def validation_step(self, batch, batch_idx):
        return None

    def test_step(self, batch, batch_idx):
        return None

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


class _FakeUnwrapped:
    def __init__(self, obs: torch.Tensor):
        self._obs = obs

    # Used only if use_phsyx_env_states=True
    def get_state(self):
        return self._obs


class FakeVectorEnv:
    """A tiny Gym-like env that finishes in 1 step.

    It supports both "CPU mode" (num_envs=1, sequential episodes) and "CUDA mode"
    (num_envs=num_episodes, batched episodes), purely based on what the callback passes in.
    """

    def __init__(self, num_envs: int, obs_dim: int = 4):
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self._closed = False
        self._last_obs = torch.zeros((self.num_envs, self.obs_dim), dtype=torch.float32)
        self.unwrapped = _FakeUnwrapped(self._last_obs)

    def reset(self, seed: int | None = None):
        if seed is None:
            seed = 0

        g = torch.Generator().manual_seed(int(seed))
        self._last_obs = torch.randn((self.num_envs, self.obs_dim), generator=g)
        self.unwrapped = _FakeUnwrapped(self._last_obs)
        info = {}
        return self._last_obs, info

    def step(self, action: torch.Tensor):
        # Expect shape (num_envs, act_dim) in cuda-mode, or (1, act_dim) in cpu-mode
        assert isinstance(action, torch.Tensor)
        assert action.shape[0] == self.num_envs

        obs = self._last_obs  # keep obs stable
        reward = torch.zeros((self.num_envs,), dtype=torch.float32)
        terminated = torch.ones((self.num_envs,), dtype=torch.bool)  # done in 1 step
        truncated = torch.zeros((self.num_envs,), dtype=torch.bool)

        # Mark all successful
        info = {"success": torch.ones((self.num_envs,), dtype=torch.bool)}
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
    def _make(env_id: str, obs_mode: str, control_mode: str, num_envs: int, **kwargs):
        assert env_id == "FakeManiSkill-v0"
        assert isinstance(num_envs, int) and num_envs >= 1
        return FakeVectorEnv(num_envs=num_envs)

    monkeypatch.setattr(gym, "make", _make, raising=True)


@pytest.mark.parametrize(
    "physx_backend,expected_num_envs",
    [
        (None, 1),  # CPU sequential mode -> num_envs=1
        ("cpu", 1),  # still CPU
    ],
)
def test_rollout_evaluation_callback_cpu_mode_logs_success_rate(
    monkeypatch: pytest.MonkeyPatch,
    capture_log,
    physx_backend: str | None,
    expected_num_envs: int,
):
    _patch_gym(monkeypatch)

    datamodule = FakeRolloutDataModule(physx_backend=physx_backend)
    model = FakePolicyModule()

    cb = RolloutEvaluationCallback(num_val_episodes=3, num_test_episodes=5, seed=123)

    # Minimal trainer; disable progress bars/logging overhead
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

    # Validate triggers on_validation_epoch_end -> rollouts
    trainer.validate(model=model, datamodule=datamodule, verbose=False)

    # Test triggers on_test_epoch_end -> rollouts
    trainer.test(model=model, datamodule=datamodule, verbose=False)

    # Verify log calls exist and are exactly 1.0 (our fake env always succeeds)
    logged = {name: float(value) for (name, value, _kw) in capture_log}
    assert logged["val/success_rate"] == 1.0
    assert logged["test/success_rate"] == 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_rollout_evaluation_callback_cuda_mode_logs_success_rate(
    monkeypatch: pytest.MonkeyPatch,
    capture_log,
):
    _patch_gym(monkeypatch)

    # Trigger the callback's "cuda" branch by including "cuda" in physx_backend.
    datamodule = FakeRolloutDataModule(physx_backend="cuda")
    model = FakePolicyModule().to("cuda")

    cb = RolloutEvaluationCallback(num_val_episodes=4, num_test_episodes=6, seed=123)

    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=[cb],
        max_epochs=1,
    )

    trainer.validate(model=model, datamodule=datamodule, verbose=False)
    trainer.test(model=model, datamodule=datamodule, verbose=False)

    logged = {name: float(value) for (name, value, _kw) in capture_log}
    assert logged["val/success_rate"] == 1.0
    assert logged["test/success_rate"] == 1.0


def test_rollout_evaluation_callback_get_policy_conditioning_env_state_path(monkeypatch):
    cb = RolloutEvaluationCallback(seed=0)
    cb.use_phsyx_env_states = True

    env = FakeVectorEnv(num_envs=1)
    obs, _ = env.reset(seed=0)
    out = cb._get_policy_conditioning(env, obs)

    # When use_phsyx_env_states is True, it should come from env.unwrapped.get_state()
    assert torch.allclose(out, env.unwrapped.get_state())
