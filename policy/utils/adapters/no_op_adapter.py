from typing import Any

import torch


class NoOpAdapter:
    """A dummy adapter that does nothing."""

    def apply(self, obs: torch.Tensor | dict[str, Any]) -> torch.Tensor | dict[str, Any]:
        return obs
