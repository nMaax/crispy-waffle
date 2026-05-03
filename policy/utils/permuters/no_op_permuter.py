from typing import Any

import torch


class NoOpPermuter:
    """A dummy permuter that does nothing."""

    def apply(self, obs: torch.Tensor | dict[str, Any]) -> torch.Tensor | dict[str, Any]:
        return obs
