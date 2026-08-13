from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Literal

from policy.utils import get_ndim, get_total_dim
from policy.utils.typing_utils import TensorTree

CondKind = Literal["per_timestep", "global", "sequence"]


@dataclass(frozen=True)
class CondEntry:
    """One conditioning entry's declared shape: how wide it is, and what kind of thing it is."""

    width: int
    kind: CondKind

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError(f"CondEntry width must be positive, got {self.width}.")


class ConditioningSpec(Mapping[str, CondEntry]):
    """The full set of conditioning entries ``ConditioningEncoder`` declares to its network."""

    def __init__(self, entries: Mapping[str, CondEntry]):
        self._entries = dict(entries)
        if not self._entries:
            raise ValueError("A ConditioningSpec must declare at least one entry.")

    def __getitem__(self, key: str) -> CondEntry:
        return self._entries[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._entries!r})"

    def of_kind(self, *kinds: CondKind) -> dict[str, CondEntry]:
        return {key: entry for key, entry in self._entries.items() if entry.kind in kinds}

    def validate_payload(self, payload: Mapping[str, TensorTree]) -> None:
        declared, provided = set(self._entries), set(payload)
        if declared != provided:
            raise ValueError(
                f"Conditioning payload keys {sorted(provided)} do not match the declared spec "
                f"{sorted(declared)} (missing: {sorted(declared - provided)}, unexpected: "
                f"{sorted(provided - declared)})."
            )

        for key, entry in self._entries.items():
            leaf = payload[key]
            width, ndim = get_total_dim(leaf), get_ndim(leaf)
            expected_ndim = 2 if entry.kind == "global" else 3
            if ndim == expected_ndim and width == entry.width:
                continue
            else:
                raise ValueError(
                    f"Conditioning entry {key!r} is declared as {entry.kind} of width {entry.width} "
                    f"(so {expected_ndim}D payload), but got a {ndim}D payload of total width {width}."
                )
