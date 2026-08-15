import pytest
import torch

from policy.algorithms.networks.encoder import ConditioningContract


def test_conditioning_contract_properties_and_film_width():
    contract = ConditioningContract(
        step_dim=26,
        global_dim=10,
        context_dim=8,
    )
    assert contract.step_dim == 26
    assert contract.global_dim == 10
    assert contract.context_dim == 8
    assert contract.get_film_width(obs_horizon=2) == 26 * 2 + 10


def test_contract_mapping_interface():
    contract = ConditioningContract(
        step_dim=18,
        global_dim=10,
        context_dim=8,
    )
    assert contract["obs"] == 18
    assert contract["goal"] == 10
    assert contract["context"] == 8
    assert set(contract.keys()) == {"obs", "goal", "context"}
    assert len(contract) == 3


def test_empty_contract_raises_error():
    with pytest.raises(ValueError, match="at least one non-zero dimension"):
        ConditioningContract(step_dim=0, global_dim=0, context_dim=None)


def test_negative_dimensions_raise_error():
    with pytest.raises(ValueError, match="non-negative"):
        ConditioningContract(step_dim=-1)
    with pytest.raises(ValueError, match="non-negative"):
        ConditioningContract(global_dim=-1)
    with pytest.raises(ValueError, match="positive"):
        ConditioningContract(context_dim=0)


class TestValidatePayload:
    def _contract(self) -> ConditioningContract:
        return ConditioningContract(
            step_dim=10,
            global_dim=5,
            context_dim=8,
        )

    def test_accepts_a_matching_payload(self):
        self._contract().validate_payload(
            {
                "obs": torch.randn(2, 4, 10),
                "goal": torch.randn(2, 5),
                "context": torch.randn(2, 3, 8),
            }
        )

    def test_rejects_missing_context_key(self):
        with pytest.raises(ValueError, match="missing cross-attention context key 'context'"):
            self._contract().validate_payload(
                {"obs": torch.randn(2, 4, 10), "goal": torch.randn(2, 5)}
            )

    def test_rejects_context_rank_mismatch(self):
        with pytest.raises(ValueError, match="must be a 3D tensor"):
            self._contract().validate_payload(
                {
                    "obs": torch.randn(2, 4, 10),
                    "goal": torch.randn(2, 5),
                    "context": torch.randn(2, 8),
                }
            )

    def test_rejects_context_width_mismatch(self):
        with pytest.raises(ValueError, match="feature dimension .* does not match"):
            self._contract().validate_payload(
                {
                    "obs": torch.randn(2, 4, 10),
                    "goal": torch.randn(2, 5),
                    "context": torch.randn(2, 3, 16),
                }
            )
