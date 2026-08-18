from __future__ import annotations

from collections.abc import Mapping

from policy.utils import get_ndim, get_total_dim
from policy.utils.typing_utils import TensorTree


class ConditioningContract:
    """Explicit dimension contract provided by ``ConditioningEncoder`` to downstream decoders.

    Categorizes conditioning into the exact dimensional buckets required by decoders:
    - ``step_dim``: Channel dimension of per-step streams scaled by ``obs_horizon`` (T) in FiLM.
    - ``global_dim``: Channel dimension of time-invariant streams scaled by 1 in FiLM (e.g. goals).
    - ``context_dim``: Channel dimension for cross-attention key-value tokens (None if FiLM-only).
    - ``context_key``: Key name in external_cond payload holding the cross-attention tokens.
    """

    def __init__(
        self,
        step_dim: int = 0,
        global_dim: int = 0,
        context_dim: int | None = None,
        context_key: str = "context",
    ) -> None:
        if step_dim < 0:
            raise ValueError(f"step_dim must be non-negative, got {step_dim}.")
        if global_dim < 0:
            raise ValueError(f"global_dim must be non-negative, got {global_dim}.")
        if context_dim is not None and context_dim <= 0:
            raise ValueError(f"context_dim must be positive when set, got {context_dim}.")
        if step_dim == 0 and global_dim == 0 and context_dim is None:
            raise ValueError("ConditioningContract must declare at least one non-zero dimension.")

        self.step_dim = step_dim
        self.global_dim = global_dim
        self.context_dim = context_dim
        self.context_key = context_key

    @property
    def context_mask_key(self) -> str:
        """Payload key holding the cross-attention key-padding mask, when one is supplied."""
        return f"{self.context_key}_mask"

    def get_film_width(self, obs_horizon: int) -> int:
        """Calculates total 1D FiLM vector dimension for a given observation horizon."""
        return self.step_dim * obs_horizon + self.global_dim

    def __getitem__(self, key: str) -> int:
        if key == "obs":
            return self.step_dim
        if key == "goal":
            if self.global_dim > 0:
                return self.global_dim
            raise KeyError("goal")
        if key == self.context_key:
            if self.context_dim is not None:
                return self.context_dim
            raise KeyError(self.context_key)
        if key == "task":
            if self.global_dim > 0:
                return self.global_dim
            raise KeyError("task")
        raise KeyError(key)

    def keys(self) -> list[str]:
        result = []
        if self.step_dim > 0:
            result.append("obs")
        if self.global_dim > 0:
            result.append("goal")
        if self.context_dim is not None:
            result.append(self.context_key)
        return result

    def values(self) -> list[int]:
        return [self[k] for k in self.keys()]

    def items(self) -> list[tuple[str, int]]:
        return [(k, self[k]) for k in self.keys()]

    def __contains__(self, key: object) -> bool:
        return key in self.keys()

    def get(self, key: str, default: int | None = None) -> int | None:
        try:
            return self[key]
        except KeyError:
            return default

    def __iter__(self):
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ConditioningContract):
            return False
        return (
            self.step_dim == other.step_dim
            and self.global_dim == other.global_dim
            and self.context_dim == other.context_dim
            and self.context_key == other.context_key
        )

    def __repr__(self) -> str:
        return (
            f"ConditioningContract(step_dim={self.step_dim}, "
            f"global_dim={self.global_dim}, context_dim={self.context_dim}, "
            f"context_key={self.context_key!r})"
        )

    def validate_payload(self, payload: Mapping[str, TensorTree]) -> None:
        """Validates that a forward conditioning payload matches this contract."""
        if not payload:
            raise ValueError("Empty conditioning payload.")

        if self.context_dim is not None:
            if self.context_key not in payload:
                raise ValueError(
                    f"Payload missing cross-attention context key {self.context_key!r}. "
                    f"Keys provided: {sorted(payload.keys())}"
                )
            context_tensor = payload[self.context_key]
            if get_ndim(context_tensor) != 3:
                raise ValueError(
                    f"Cross-attention context must be a 3D tensor [B, L, D], got {get_ndim(context_tensor)}D."
                )
            if get_total_dim(context_tensor) != self.context_dim:
                raise ValueError(
                    f"Cross-attention context feature dimension ({get_total_dim(context_tensor)}) "
                    f"does not match declared context_dim ({self.context_dim})."
                )
