from typing import Any

import h5py


def load_h5_data(data):
    """Recursively loads h5py data into memory as numpy arrays."""
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out


def extract_h5_shapes(data: Any):
    """Recursively extracts shapes from h5py objects without loading into RAM."""
    if isinstance(data, h5py.Group):
        # h5py.Group behaves like a dict, but isn't one!
        # We must iterate explicitly here so isinstance(dict) doesn't fail.
        return {k: extract_h5_shapes(data[k]) for k in data.keys()}
    elif isinstance(data, h5py.Dataset):
        return {"shape": tuple(data.shape), "dtype": str(data.dtype)}
    else:
        return None
