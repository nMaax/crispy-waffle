import torch
from torch.utils.data import Dataset

from policy.transforms import observation_pipeline
from policy.utils.typing_utils import TensorTree

from .aligned_states_dataset import AlignedStatesDataset


class TaskConditionedAlignedStatesDataset(Dataset):
    """Wraps a aligned states dataset to apply canonical pick-and-place formatting for the input
    and inject an integer task ID for multi-task networks."""

    def __init__(self, base_translator_dataset: AlignedStatesDataset, env_id: str, task_idx: int):
        self.base_translator_dataset = base_translator_dataset
        self.env_id = env_id
        self.task_idx = task_idx

    def __len__(self) -> int:
        return len(self.base_translator_dataset)

    def __getitem__(self, idx: int) -> tuple[TensorTree, TensorTree, int]:
        x, y = self.base_translator_dataset[idx]

        canonicalize = observation_pipeline(
            self.env_id, is_flat=isinstance(x, torch.Tensor), canonicalize=True, as_dict=False
        )
        with torch.no_grad():
            canonical_x = canonicalize(x)

        return canonical_x, y, self.task_idx
