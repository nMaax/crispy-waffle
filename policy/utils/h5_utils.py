from pathlib import Path
from typing import Any

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


def extract_h5_shapes(data: h5py.Group | h5py.Dataset) -> dict["str", tuple] | None:
    """Recursively extracts shapes from h5py objects without loading into RAM."""
    if isinstance(data, h5py.Group):
        result = {}
        for k in data.keys():
            h5entry = data[k]
            if not isinstance(h5entry, h5py.Group | h5py.Dataset):
                raise ValueError(f"Unexpected h5 entry type: {type(h5entry)}")
            result[k] = extract_h5_shapes(h5entry)
        return result
    else:
        return data.shape[-1]


def peek_trajectory_dimension(
    dataset_file: str | Path, episode_key: str, dimension_key: str
) -> Any:
    """Extracts the shape/dimension of a specified key from an HDF5 trajectory group."""
    with h5py.File(dataset_file, "r") as data:
        traj = data[episode_key]
        if not isinstance(traj, h5py.Group):
            raise TypeError(f"Expected trajectory to be an HDF5 group or dict, got {type(traj)}")

        if dimension_key not in traj:
            raise KeyError(f"Key '{dimension_key}' not found in trajectory group.")

        item = traj[dimension_key]
        if not isinstance(item, h5py.Group | h5py.Dataset):
            raise TypeError(f"Expected data to be an HDF5 group or dataset, got {type(data)}")

        return extract_h5_shapes(item)
