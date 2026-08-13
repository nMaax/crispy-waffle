import pytest
import torch

from policy.algorithms.networks.encoder import CondEntry, ConditioningSpec


def test_of_kind_filters_entries_by_kind():
    spec = ConditioningSpec(
        {
            "proprio": CondEntry(width=18, kind="per_timestep"),
            "task": CondEntry(width=8, kind="per_timestep"),
            "goal": CondEntry(width=5, kind="global"),
            "context": CondEntry(width=8, kind="sequence"),
        }
    )
    assert set(spec.of_kind("per_timestep")) == {"proprio", "task"}
    assert set(spec.of_kind("global")) == {"goal"}
    assert set(spec.of_kind("sequence")) == {"context"}
    assert set(spec.of_kind("per_timestep", "global")) == {"proprio", "task", "goal"}


def test_of_kind_preserves_declaration_order():
    spec = ConditioningSpec(
        {
            "obs": CondEntry(width=1, kind="per_timestep"),
            "goal": CondEntry(width=2, kind="global"),
            "task": CondEntry(width=3, kind="per_timestep"),
        }
    )
    assert list(spec.of_kind("per_timestep")) == ["obs", "task"]


def test_zero_width_is_rejected():
    with pytest.raises(ValueError, match="width must be positive"):
        CondEntry(width=0, kind="global")


def test_empty_spec_is_rejected():
    with pytest.raises(ValueError, match="at least one entry"):
        ConditioningSpec({})


class TestValidatePayload:
    def _spec(self) -> ConditioningSpec:
        return ConditioningSpec(
            {
                "obs": CondEntry(width=10, kind="per_timestep"),
                "goal": CondEntry(width=5, kind="global"),
                "context": CondEntry(width=8, kind="sequence"),
            }
        )

    def test_accepts_a_matching_payload(self):
        self._spec().validate_payload(
            {
                "obs": torch.randn(2, 4, 10),
                "goal": torch.randn(2, 5),
                "context": torch.randn(2, 3, 8),
            }
        )

    def test_rejects_a_missing_entry(self):
        with pytest.raises(ValueError, match="missing.*'goal'"):
            self._spec().validate_payload(
                {"obs": torch.randn(2, 4, 10), "context": torch.randn(2, 3, 8)}
            )

    def test_rejects_an_unexpected_entry(self):
        with pytest.raises(ValueError, match="unexpected.*'extra'"):
            self._spec().validate_payload(
                {
                    "obs": torch.randn(2, 4, 10),
                    "goal": torch.randn(2, 5),
                    "context": torch.randn(2, 3, 8),
                    "extra": torch.randn(2, 1),
                }
            )

    def test_rejects_a_width_mismatch(self):
        """The exact failure this spec exists to catch early, with a real error message instead of
        an opaque matmul shape error inside the network."""
        with pytest.raises(ValueError, match="declared as per_timestep of width 10"):
            self._spec().validate_payload(
                {
                    "obs": torch.randn(2, 4, 11),  # wrong width
                    "goal": torch.randn(2, 5),
                    "context": torch.randn(2, 3, 8),
                }
            )

    def test_rejects_a_global_entry_with_a_time_axis(self):
        with pytest.raises(ValueError, match="declared as global"):
            self._spec().validate_payload(
                {
                    "obs": torch.randn(2, 4, 10),
                    "goal": torch.randn(2, 1, 5),  # global entries have no time axis
                    "context": torch.randn(2, 3, 8),
                }
            )

    def test_accepts_a_nested_tree_leaf(self):
        """An entry's payload can be a nested tree of tensors (e.g. proprio + task under 'obs'), as
        long as their combined last-dim width and shared rank match what was declared."""
        self._spec().validate_payload(
            {
                "obs": {"proprio": torch.randn(2, 4, 6), "task": torch.randn(2, 4, 4)},
                "goal": torch.randn(2, 5),
                "context": torch.randn(2, 3, 8),
            }
        )
