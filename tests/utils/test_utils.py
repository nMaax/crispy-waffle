import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from policy.utils.utils import (
    cat_dicts,
    concat_leaf_tensors,
    flatten_and_concat_leaf_tensors,
    get_batch_size,
    get_device,
    get_subtree,
    get_tensor,
    get_total_dim,
    pop_leaf_key,
    print_config,
    print_mapping_tree,
    recursive_index,
    stack_dicts,
    to_tensor,
)


def test_print_config(capsys):
    cfg = OmegaConf.create({"algorithm": {"lr": 0.001}, "trainer": "default_trainer", "extra": 42})
    print_config(cfg, print_order=["algorithm", "missing_field", "trainer"])
    captured = capsys.readouterr()
    assert "algorithm" in captured.out or "CONFIG" in captured.out
    assert "default_trainer" in captured.out

    # Test queue missing key branch
    class DynamicConfig(dict):
        def __contains__(self, item):
            if item == "disappearing_key":
                # First time True (to get into queue), second time False (to trigger line 56)
                if not hasattr(self, "_checked"):
                    self._checked = True
                    return True
                return False
            return super().__contains__(item)

    dyn_cfg = DynamicConfig({"disappearing_key": 123})
    print_config(dyn_cfg, print_order=["disappearing_key"])


def test_print_mapping_tree(capsys):
    data = {
        "level1": {
            "tensor_leaf": torch.zeros((2, 3), dtype=torch.float32),
            "array_leaf": np.ones((4, 5)),
            "basic_leaf": 42,
        }
    }
    print_mapping_tree(data, use_rank_zero_info=False)
    captured = capsys.readouterr()
    assert "level1" in captured.out
    assert "tensor_leaf: shape=torch.Size([2, 3])" in captured.out
    assert "array_leaf: shape=(4, 5)" in captured.out
    assert "int" in captured.out


def test_to_tensor():
    raw_data = {"a": [1, 2, 3], "b": {"c": np.array([4.0, 5.0])}}
    res = to_tensor(raw_data, dtype=torch.float32)
    assert isinstance(res["a"], torch.Tensor)
    assert isinstance(res["b"]["c"], torch.Tensor)
    assert res["a"].dtype == torch.float32
    assert torch.equal(res["a"], torch.tensor([1.0, 2.0, 3.0]))


def test_recursive_index():
    data = {"a": torch.arange(10), "b": [torch.arange(10), torch.arange(10)]}
    indexed = recursive_index(data, slice(0, 3))
    assert indexed["a"].shape == (3,)
    assert indexed["b"][0].shape == (3,)
    assert recursive_index(42, 0) == 42


def test_get_batch_size():
    data = {"a": {"b": torch.zeros((5, 10))}}
    assert get_batch_size(data) == 5
    assert get_batch_size(torch.zeros((8, 2))) == 8

    with pytest.raises(ValueError, match="must contain at least one tensor"):
        get_batch_size({})


def test_get_total_dim():
    # Tensor
    assert get_total_dim(torch.zeros((4, 8))) == 8
    # Int
    assert get_total_dim(16) == 16
    # Sequence / tuple
    assert get_total_dim((10, 20, 30)) == 30
    # Mapping with 'shape'
    assert get_total_dim({"shape": 5}) == 5
    assert get_total_dim({"shape": (2, 4)}) == 4
    assert get_total_dim({"shape": torch.tensor([2, 6])}) == 6
    # Nested mapping
    nested = {"a": torch.zeros((1, 4)), "b": {"c": (1, 8)}}
    assert get_total_dim(nested) == 12

    with pytest.raises(TypeError, match="Unexpected 'shape' value type"):
        get_total_dim({"shape": "invalid"})

    with pytest.raises(TypeError, match="Unexpected DimSpec type"):
        get_total_dim("invalid_string")


def test_get_device():
    t = torch.zeros(3)
    assert get_device({"a": t}) == t.device
    assert get_device(t) == t.device

    with pytest.raises(ValueError, match="must contain at least one tensor"):
        get_device({})


def test_cat_dicts():
    t1 = {"a": torch.tensor([[1.0]]), "b": {"c": torch.tensor([[2.0]])}}
    t2 = {"a": torch.tensor([[3.0]]), "b": {"c": torch.tensor([[4.0]])}}
    res = cat_dicts([t1, t2])
    assert torch.equal(res["a"], torch.tensor([[1.0], [3.0]]))
    assert torch.equal(res["b"]["c"], torch.tensor([[2.0], [4.0]]))

    with pytest.raises(AssertionError, match="Expected element to be a Mapping"):
        cat_dicts([t1, torch.tensor([1.0])])

    with pytest.raises(AssertionError, match="Expected element to be a Tensor"):
        cat_dicts([torch.tensor([[1.0]]), {"a": 1}])


def test_stack_dicts():
    t1 = {"a": torch.tensor([1.0]), "b": {"c": torch.tensor([2.0])}}
    t2 = {"a": torch.tensor([3.0]), "b": {"c": torch.tensor([4.0])}}
    res = stack_dicts([t1, t2])
    assert torch.equal(res["a"], torch.tensor([[1.0], [3.0]]))
    assert torch.equal(res["b"]["c"], torch.tensor([[2.0], [4.0]]))

    with pytest.raises(AssertionError, match="Expected element to be a Mapping"):
        stack_dicts([t1, torch.tensor([1.0])])

    with pytest.raises(AssertionError, match="Expected element to be a Tensor"):
        stack_dicts([torch.tensor([[1.0]]), {"a": 1}])


def test_concat_leaf_tensors():
    data = {"a": torch.tensor([[1.0, 2.0]]), "b": {"c": torch.tensor([[3.0, 4.0]])}}
    concat = concat_leaf_tensors(data, dim=-1, device=torch.device("cpu"))
    assert torch.equal(concat, torch.tensor([[1.0, 2.0, 3.0, 4.0]]))

    with pytest.raises(ValueError, match="must contain at least one tensor"):
        concat_leaf_tensors({})


def test_flatten_and_concat_leaf_tensors():
    d = {"a": torch.zeros((2, 3, 4)), "b": torch.ones((2, 5))}
    flat = flatten_and_concat_leaf_tensors(d)
    assert flat.shape == (2, 17)


def test_pop_leaf_key_mapping():
    tree = {"proprio": torch.ones((2, 3)), "tcp": torch.zeros((2, 4)), "extras": torch.zeros((2, 5))}
    popped, remainder = pop_leaf_key(tree, "proprio", size=3)
    assert torch.equal(popped, torch.ones((2, 3)))
    assert set(remainder.keys()) == {"tcp", "extras"}


def test_pop_leaf_key_mapping_missing_key():
    tree = {"tcp": torch.zeros((2, 4))}
    popped, remainder = pop_leaf_key(tree, "proprio", size=3)
    assert popped is None
    assert remainder is tree


def test_pop_leaf_key_mapping_non_tensor_leaf_raises():
    with pytest.raises(TypeError, match="Expected leaf at key 'proprio' to be a Tensor"):
        pop_leaf_key({"proprio": {"nested": torch.zeros(2)}}, "proprio", size=3)


def test_pop_leaf_key_flat_tensor():
    x = torch.arange(10, dtype=torch.float32).view(2, 5)
    popped, remainder = pop_leaf_key(x, "proprio", size=2)
    assert torch.equal(popped, x[..., :2])
    assert torch.equal(remainder, x[..., 2:])


def test_get_tensor():
    t = torch.tensor([1.0, 2.0])
    tree = {"leaf": t, "sub": {"nested": torch.tensor([3.0])}}
    assert torch.equal(get_tensor(tree, "leaf"), t)

    with pytest.raises(TypeError, match="Expected a Tensor at key 'sub', got dict."):
        get_tensor(tree, "sub")

    with pytest.raises(KeyError):
        get_tensor(tree, "missing")


def test_get_subtree():
    sub = {"nested": torch.tensor([3.0])}
    tree = {"leaf": torch.tensor([1.0, 2.0]), "sub": sub}
    assert get_subtree(tree, "sub") == sub

    with pytest.raises(TypeError, match="Expected a nested mapping at key 'leaf', got Tensor."):
        get_subtree(tree, "leaf")

    with pytest.raises(KeyError):
        get_subtree(tree, "missing")
