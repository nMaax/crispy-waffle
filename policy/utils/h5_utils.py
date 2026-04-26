import h5py
import numpy as np


def load_h5_data(data: h5py.Group | h5py.File) -> dict[str, np.ndarray | dict]:
    """Recursively loads h5py data into memory as numpy arrays."""
    out: dict[str, np.ndarray | dict] = dict()
    for k in data.keys():
        item = data[k]
        if isinstance(item, h5py.Dataset):
            out[k] = item[:]
        elif isinstance(item, h5py.Group):
            out[k] = load_h5_data(item)
    return out


def extract_h5_shapes(data: h5py.Group | h5py.Dataset | h5py.Datatype | None):
    """Recursively extracts shapes from h5py objects without loading into RAM."""
    if isinstance(data, h5py.Group):
        return {k: extract_h5_shapes(data[k]) for k in data.keys()}
    elif isinstance(data, h5py.Dataset):
        return {"shape": tuple(data.shape), "dtype": str(data.dtype)}
    else:
        return None
