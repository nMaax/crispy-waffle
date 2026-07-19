from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest
import torch

from policy.utils.h5_utils import (
    extract_h5_shapes,
    h5_group_to_dict_of_tensors,
    load_h5_data,
    peek_trajectory_dimension,
    peek_trajectory_is_dataset,
)


@pytest.fixture
def temp_h5_file(tmp_path: Path):
    h5_path = tmp_path / "sample.h5"
    with h5py.File(h5_path, "w") as f:
        traj0 = f.create_group("traj_0")
        traj0.create_dataset("obs", data=np.ones((10, 5), dtype=np.float32))
        traj0.create_dataset("actions", data=np.zeros((10, 2), dtype=np.float32))

        subgroup = traj0.create_group("nested_group")
        subgroup.create_dataset("sub_obs", data=np.ones((10, 3), dtype=np.float32))

        f.create_dataset("flat_dataset", data=np.arange(5))
    return h5_path


def test_load_h5_data(temp_h5_file: Path):
    with h5py.File(temp_h5_file, "r") as f:
        data = load_h5_data(f["traj_0"])
        assert isinstance(data["obs"], np.ndarray)
        assert data["obs"].shape == (10, 5)
        assert isinstance(data["nested_group"], dict)
        assert data["nested_group"]["sub_obs"].shape == (10, 3)


def test_extract_h5_shapes(temp_h5_file: Path):
    with h5py.File(temp_h5_file, "r") as f:
        shapes = extract_h5_shapes(f["traj_0"])
        assert shapes["obs"] == 5
        assert shapes["actions"] == 2
        assert shapes["nested_group"]["sub_obs"] == 3


def test_h5_group_to_dict_of_tensors(temp_h5_file: Path):
    with h5py.File(temp_h5_file, "r") as f:
        res = h5_group_to_dict_of_tensors(f["traj_0"])
        assert isinstance(res, dict)
        assert isinstance(res["obs"], torch.Tensor)
        assert res["obs"].shape == (10, 5)


def test_peek_trajectory_dimension(temp_h5_file: Path):
    obs_dim = peek_trajectory_dimension(temp_h5_file, "traj_0", "obs")
    assert obs_dim == 5

    nested_dim = peek_trajectory_dimension(temp_h5_file, "traj_0", "nested_group")
    assert nested_dim == {"sub_obs": 3}

    with pytest.raises(TypeError, match="Expected trajectory to be an HDF5 group"):
        peek_trajectory_dimension(temp_h5_file, "flat_dataset", "obs")

    with pytest.raises(KeyError, match="Key 'non_existent' not found"):
        peek_trajectory_dimension(temp_h5_file, "traj_0", "non_existent")


def test_peek_trajectory_is_dataset(tmp_path: Path):
    h5_path = tmp_path / "only_groups.h5"
    with h5py.File(h5_path, "w") as f:
        g = f.create_group("traj_0")
        g.create_dataset("obs", data=np.ones((5, 2)))
        g.create_group("nested")

    assert peek_trajectory_is_dataset(h5_path, dimension_key="obs") is True
    assert peek_trajectory_is_dataset(h5_path, dimension_key="nested", episode_key="traj_0") is False

    with pytest.raises(TypeError, match="Expected an h5py.Group"):
        with h5py.File(h5_path, "a") as f:
            f.create_dataset("flat_ds", data=np.zeros(2))
        peek_trajectory_is_dataset(h5_path, dimension_key="obs", episode_key="flat_ds")

    with pytest.raises(KeyError, match="Key 'missing' not found"):
        peek_trajectory_is_dataset(h5_path, dimension_key="missing", episode_key="traj_0")


def test_extract_h5_shapes_invalid_entry():
    mock_group = MagicMock(spec=h5py.Group)
    mock_group.keys.return_value = ["invalid"]
    mock_group.__getitem__.return_value = 123  # Not Group or Dataset

    with pytest.raises(ValueError, match="Unexpected h5 entry type"):
        extract_h5_shapes(mock_group)


def test_peek_trajectory_dimension_invalid_item(tmp_path: Path):
    h5_path = tmp_path / "dummy.h5"
    with h5py.File(h5_path, "w") as f:
        g = f.create_group("traj_0")
        g.create_dataset("obs", data=np.ones(2))

    mock_group = MagicMock(spec=h5py.Group)
    mock_group.__contains__.return_value = True
    mock_group.__getitem__.return_value = 123  # Not Group or Dataset

    with patch("h5py.File.__getitem__", return_value=mock_group):
        with pytest.raises(TypeError, match="Expected data to be an HDF5 group or dataset"):
            peek_trajectory_dimension(h5_path, "traj_0", "obs")
