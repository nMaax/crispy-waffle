import numpy as np
import pytest
import torch

from policy.transforms.schema.dict_flattener import DictFlattener
from policy.transforms.schema.state_deflattener import ManiSkillStateDeFlattener


class TestDictFlattener:
    def test_flatten_nested_dict_tensors(self):
        df = DictFlattener()
        obs = {"a": {"b": torch.randn(2, 3), "c": torch.randn(2, 4)}}
        out = df(obs)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 7)

    def test_flatten_flat_dict_tensors(self):
        df = DictFlattener()
        obs = {"a": torch.randn(2, 3), "b": torch.randn(2, 4)}
        out = df(obs)
        assert out.shape == (2, 7)

    def test_flatten_single_key(self):
        df = DictFlattener()
        out = df({"a": torch.randn(2, 3)})
        assert out.shape == (2, 3)

    def test_flatten_numpy_arrays(self):
        df = DictFlattener()
        obs = {"a": np.random.randn(2, 3), "b": np.random.randn(2, 4)}
        out = df(obs)
        assert isinstance(out, np.ndarray)
        assert out.shape == (2, 7)

    def test_passthrough_tensor(self):
        df = DictFlattener()
        t = torch.randn(2, 3)
        out = df(t)
        assert out is t

    def test_empty_dict_raises(self):
        df = DictFlattener()
        with pytest.raises(ValueError, match="at least one leaf"):
            df({})

    def test_unsupported_leaf_raises(self):
        df = DictFlattener()
        with pytest.raises(TypeError, match="Unsupported value type"):
            df({"a": "not_a_tensor"})

    def test_round_trip_with_deflattener(self):
        """DictFlattener(ManiSkillStateDeFlattener(flat)) == flat (identity)."""
        flat = torch.randn(48)
        deflattener = ManiSkillStateDeFlattener("StackCube-v1")
        df = DictFlattener()
        reconstructed = df(deflattener(flat))
        assert reconstructed.shape == flat.shape
        assert torch.allclose(reconstructed, flat)
