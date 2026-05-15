from typing import Any

import torch
from torch.utils.data import Dataset

from policy.transforms import PnPCanonicalizer

from .translator_dataset import TranslatorDataset


class PnPDataset(Dataset):
    """Wraps an existing TranslatorDataset to enforce a canonical input shape for pick-and-place
    tasks and attach an integer task ID."""

    def __init__(self, base_translator_dataset: TranslatorDataset, env_id: str, task_idx: int):
        self.base_translator_dataset = base_translator_dataset
        self.task_idx = task_idx
        self.pnp_canonicalizer = PnPCanonicalizer(env_id)

    def __len__(self) -> int:
        return len(self.base_translator_dataset)

    def __getitem__(
        self, idx: int
    ) -> tuple[dict[str, Any] | torch.Tensor, dict[str, Any] | torch.Tensor, int]:
        x, y = self.base_translator_dataset[idx]

        with torch.no_grad():
            canonical_x = self.pnp_canonicalizer.apply(x)

        return canonical_x, y, self.task_idx
