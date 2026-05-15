from torch.utils.data import Dataset


class DummyDataset(Dataset):
    """A minimal dataset to trigger Lightning loops for simulation-only phases."""

    def __len__(self):
        return 1

    def __getitem__(self, idx: int):
        return {}
