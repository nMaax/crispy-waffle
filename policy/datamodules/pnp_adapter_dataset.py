from typing import Any

import torch
from torch.utils.data import Dataset

from policy.datamodules.adapter_dataset import AdapterDataset
from policy.utils.adapters.canonical_pnp_adapter import CanonicalPnPAdapter


class PnpAdapterDataset(Dataset):
    """Wraps an existing AdapterDataset to enforce a canonical input shape (X) and attach an
    integer task ID."""

    def __init__(self, base_adapter_dataset: AdapterDataset, env_id: str, task_idx: int):
        self.base_adapter_dataset = base_adapter_dataset
        self.task_idx = task_idx
        self.canonical_adapter = CanonicalPnPAdapter(env_id)

    def __len__(self) -> int:
        return len(self.base_adapter_dataset)

    def __getitem__(
        self, idx: int
    ) -> tuple[dict[str, Any] | torch.Tensor, dict[str, Any] | torch.Tensor, int]:
        x, y = self.base_adapter_dataset[idx]

        with torch.no_grad():
            canonical_x = self.canonical_adapter.apply(x)

        return canonical_x, y, self.task_idx
