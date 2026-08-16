import torch

from policy.utils.typing_utils import is_mapping_of, is_sequence_of
from policy.utils.typing_utils.typeguards import (
    is_mapping_of as tg_is_mapping_of,
)
from policy.utils.typing_utils.typeguards import (
    is_sequence_of as tg_is_sequence_of,
)


def test_is_sequence_of():
    # Valid sequences
    assert is_sequence_of([1, 2, 3], int)
    assert is_sequence_of((1, 2, 3), int)
    assert is_sequence_of(["a", "b"], str)
    assert is_sequence_of([1, "a"], (int, str))
    assert is_sequence_of([], int)  # empty sequence vacuously true

    # Invalid sequences
    assert not is_sequence_of([1, "a"], int)
    assert not is_sequence_of(42, int)
    assert not is_sequence_of({"a": 1}, str)
    assert not is_sequence_of(None, int)

    # Submodule identity
    assert is_sequence_of is tg_is_sequence_of


def test_is_mapping_of():
    # Valid mappings
    assert is_mapping_of({"a": 1, "b": 2}, str, int)
    assert is_mapping_of({1: "a", 2: "b"}, int, str)
    assert is_mapping_of({}, str, int)  # empty mapping vacuously true
    assert is_mapping_of({"a": torch.zeros(1)}, str, torch.Tensor)

    # Invalid mappings
    assert not is_mapping_of({"a": 1, "b": "2"}, str, int)
    assert not is_mapping_of({1: 1, "b": 2}, str, int)
    assert not is_mapping_of([("a", 1)], str, int)
    assert not is_mapping_of("string", str, str)
    assert not is_mapping_of(None, str, int)

    # Submodule identity
    assert is_mapping_of is tg_is_mapping_of
