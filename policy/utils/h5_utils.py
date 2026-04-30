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


def extract_h5_shapes(data: h5py.Group | h5py.Dataset | None):
    """Recursively extracts shapes from h5py objects without loading into RAM."""
    if isinstance(data, h5py.Group):
        result = {}
        for k in data.keys():
            h5entry = data[k]
            if not isinstance(h5entry, h5py.Group | h5py.Dataset):
                raise ValueError(f"Unexpected h5 entry type: {type(h5entry)}")
            result[k] = extract_h5_shapes(h5entry)
        return result
    elif isinstance(data, h5py.Dataset):
        return {"shape": tuple(data.shape), "dtype": str(data.dtype)}
    else:
        return None


def peek_trajectory_dimensions(dataset_file: str | Path, episode_id: int) -> tuple[int, Any, Any]:
    """Helper to extract dimensions from an HDF5 trajectory group without loading full data."""
    with h5py.File(dataset_file, "r") as data:
        traj = data[f"traj_{episode_id}"]
        if not isinstance(traj, h5py.Group | dict):
            raise TypeError(f"Expected trajectory to be an HDF5 group or dict, got {type(traj)}")

        actions = traj["actions"]
        if not isinstance(actions, h5py.Dataset | np.ndarray):
            raise TypeError(
                f"Expected actions to be an HDF5 dataset or np.ndarray, got {type(actions)}"
            )

        env_states = traj["env_states"]
        if not isinstance(env_states, h5py.Group | h5py.Dataset | dict | np.ndarray):
            raise TypeError(
                f"Expected env_states to be an HDF5 group/dataset, dict, or np.ndarray, got {type(env_states)}"
            )

        obs = traj["obs"]
        if not isinstance(obs, h5py.Group | h5py.Dataset | dict | np.ndarray):
            raise TypeError(
                f"Expected obs to be an HDF5 group/dataset, dict, or np.ndarray, got {type(obs)}"
            )

        act_dim = actions.shape[-1]
        env_state_dim = extract_h5_shapes(env_states)
        obs_dim = extract_h5_shapes(obs)

        if act_dim is None or env_state_dim is None or obs_dim is None:
            raise ValueError(f"Dimensionalities of {dataset_file}'s data could not be fetched.")

        return act_dim, env_state_dim, obs_dim
