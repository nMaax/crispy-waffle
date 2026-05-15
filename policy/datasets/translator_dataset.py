from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import Dataset

from policy.utils.typing_utils import AdapterProtocol

from .trajectory_dataset import TrajectoryDataset


class TranslatorDataset(Dataset):
    """Generates training pairs by converting raw environment observations into a target domain
    using a specified adapter."""

    def __init__(
        self, base_dataset: TrajectoryDataset, adapter: Callable[[torch.Tensor], torch.Tensor]
    ):
        if not isinstance(adapter, AdapterProtocol):
            raise ValueError(
                "Adapter must be a callable that takes a tensor and returns a tensor."
            )

        self.base_dataset = base_dataset
        self.adapter = adapter

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(
        self, idx: int
    ) -> tuple[dict[str, Any] | torch.Tensor, dict[str, Any] | torch.Tensor]:
        batch = self.base_dataset[idx]

        x = batch["obs_seq"]

        with torch.no_grad():
            y = self.adapter.apply(x)

        return x, y
