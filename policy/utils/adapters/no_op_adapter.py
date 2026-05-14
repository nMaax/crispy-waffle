from typing import Any

import torch

from policy.utils.typing_utils import AdapterProtocol


class NoOpAdapter(AdapterProtocol):
    """A dummy adapter that does nothing."""

    def apply(self, obs: torch.Tensor | dict[str, Any]) -> torch.Tensor | dict[str, Any]:
        return obs
