from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import gymnasium as gym
import lightning
import pytest
from omegaconf import DictConfig

import policy.configs
import policy.eval
from policy.algorithms.callbacks.rollout_evaluation import RolloutEvaluationCallback
from policy.configs.config import Config
from policy.experiment import instantiate_trainer
from tests.algorithms.callbacks.test_rollout_evaluation import FakeVectorEnv
from tests.conftest import setup_with_overrides

CONFIG_DIR = Path(policy.configs.__file__).parent

eval_experiment_configs = [p.stem for p in (CONFIG_DIR / "experiment").glob("*__test.yaml")]
"""All test-phase experiment configs, the ones meant to be run through `eval.py`."""


@pytest.fixture
def mock_load_from_checkpoint(monkeypatch: pytest.MonkeyPatch):
    mock_fn = Mock(return_value=Mock(spec=lightning.LightningModule))
    monkeypatch.setattr(lightning.LightningModule, "load_from_checkpoint", mock_fn)
    return mock_fn


@pytest.fixture
def mock_trainer_test(monkeypatch: pytest.MonkeyPatch):
    mock_fn = Mock(return_value=None)
    monkeypatch.setattr(lightning.Trainer, "test", mock_fn)
    return mock_fn


@setup_with_overrides(
    [f"experiment={name} ckpt_path=dummy.ckpt" for name in eval_experiment_configs]
)
def test_eval_wires_up_experiment(
    dict_config: DictConfig,
    mock_load_from_checkpoint: Mock,
    mock_trainer_test: Mock,
):
    """Checks that running a `*__test.yaml` experiment through `eval.py` loads the config, resolves
    the algorithm's checkpoint class, and calls `trainer.test` with that model."""
    policy.eval.main(dict_config)

    mock_load_from_checkpoint.assert_called_once_with(Path("dummy.ckpt"), weights_only=False)
    mock_trainer_test.assert_called_once()
    assert mock_trainer_test.call_args.kwargs["model"] is mock_load_from_checkpoint.return_value


@setup_with_overrides("algorithm=no_op")
def test_eval_requires_ckpt_path(dict_config: DictConfig):
    with pytest.raises(ValueError, match="Checkpoint path must be specified"):
        policy.eval.main(dict_config)


@setup_with_overrides(
    "experiment=GCDP-Obj-Attn-MLPPool__SCLR__default__train ckpt_path=dummy.ckpt"
)
def test_eval_rejects_train_experiment(dict_config: DictConfig):
    """A `*__train.yaml` experiment run through `eval.py` should be rejected: `eval.py` never
    trains, so this is almost certainly the wrong entrypoint (use `main.py` instead)."""
    with pytest.raises(ValueError, match="not supported; use main.py instead"):
        policy.eval.main(dict_config)


@setup_with_overrides(
    [f"experiment={name} ckpt_path=dummy.ckpt" for name in eval_experiment_configs]
)
def test_eval_rollout_callback_is_self_contained(config: Config, monkeypatch: pytest.MonkeyPatch):
    """`eval.py` calls `trainer.test(dataloaders=...)`, never `datamodule=...`
    (`policy/eval.py:36`), so `RolloutEvaluationCallback.setup` has no `trainer.datamodule` to fall
    back on for `env_id`/`obs_mode`/`control_mode`/`physx_backend`
    (`policy/algorithms/callbacks/rollout_evaluation.py:111,118`).

    Each experiment's own
    `rollout_evaluation` callback config must therefore supply these directly.
    """
    trainer = instantiate_trainer(config.trainer)
    rollout_cb = next(cb for cb in trainer.callbacks if isinstance(cb, RolloutEvaluationCallback))

    def _fake_make(id: str, num_envs: int, **kwargs):
        return FakeVectorEnv(num_envs=num_envs)

    monkeypatch.setattr(gym, "make", _fake_make)

    # No datamodule attached, mirroring eval.py's real `trainer.test(dataloaders=...)` usage.
    rollout_cb.setup(trainer=Mock(datamodule=None), pl_module=Mock(obs_horizon=2), stage="test")
