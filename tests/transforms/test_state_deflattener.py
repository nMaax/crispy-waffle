from unittest.mock import patch

import pytest
import torch

from policy.transforms.schema.state_deflattener import ManiSkillStateDeFlattener


class TestManiSkillStateDeFlattener:
    def test_init_loads_schema(self):
        df = ManiSkillStateDeFlattener("StackCube-v1")
        assert isinstance(df.schema, dict)
        assert "agent" in df.schema
        assert "extra" in df.schema

    def test_unknown_env_raises(self):
        with pytest.raises(ValueError, match="No STATE_SCHEMA found"):
            ManiSkillStateDeFlattener("FakeEnv-v0")

    def test_env_without_schema_raises(self):
        """An env class registered but lacking STATE_SCHEMA must raise."""
        fake_cls = type("FakeEnv", (), {})  # no STATE_SCHEMA attribute

        class FakeRegistration:
            cls = fake_cls

        with patch(
            "mani_skill.utils.registration.REGISTERED_ENVS",
            {"FakeNoSchema-v0": FakeRegistration()},
        ):
            with pytest.raises(ValueError, match="No STATE_SCHEMA found"):
                ManiSkillStateDeFlattener("FakeNoSchema-v0")

    def test_call_tensor_slices(self):
        df = ManiSkillStateDeFlattener("StackCube-v1")
        flat = torch.arange(48, dtype=torch.float32)
        out = df(flat)
        assert isinstance(out, dict)
        assert set(out.keys()) == {"agent", "extra"}
        assert out["agent"]["qpos"].shape[-1] == 9
        assert out["agent"]["qvel"].shape[-1] == 9
        assert out["extra"]["tcp_pose"].shape[-1] == 7
        assert out["extra"]["cubeA_pose"].shape[-1] == 7
        # Verify values: qpos = flat[0:9], qvel = flat[9:18]
        assert torch.allclose(out["agent"]["qpos"], flat[0:9])
        assert torch.allclose(out["agent"]["qvel"], flat[9:18])
        assert torch.allclose(out["extra"]["tcp_pose"], flat[18:25])

    def test_call_mapping_passthrough(self):
        df = ManiSkillStateDeFlattener("StackCube-v1")
        obs = {"a": 1, "b": torch.randn(3)}
        out = df(obs)
        assert out is obs

    def test_call_batched_tensor(self):
        df = ManiSkillStateDeFlattener("StackCube-v1")
        flat = torch.randn(4, 48)
        out = df(flat)
        assert out["agent"]["qpos"].shape == (4, 9)
        assert out["extra"]["tcp_pose"].shape == (4, 7)
