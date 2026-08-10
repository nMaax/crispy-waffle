from __future__ import annotations

import json
from pathlib import Path

import h5py
import torch

from policy.utils.typing_utils import TensorTree

DEMO_ROOT = Path.home() / ".maniskill" / "demos"
DEFAULT_TRAJECTORY_NAME = "trajectory.state.pd_ee_delta_pos.physx_cuda.h5"
DEFAULT_FRAME_SPEC = "0%,25%,50%,75%,100%"


def default_demo_path(env_id: str, trajectory_name: str = DEFAULT_TRAJECTORY_NAME) -> Path:
    """The conventional location of an env's motion-planning demonstrations."""
    return DEMO_ROOT / env_id / "motionplanning" / trajectory_name


def env_id_from_sidecar(dataset_path: Path) -> str | None:
    """Reads the env id from the `.json` metadata file recorded next to a demo `.h5`."""
    sidecar = dataset_path.with_suffix(".json")
    if not sidecar.exists():
        return None
    try:
        metadata = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    env_id = metadata.get("env_info", {}).get("env_id")
    return str(env_id) if env_id else None


def resolve_env_id(dataset_path: Path, override: str | None = None) -> str:
    """Determines which env a demo file belongs to."""
    if override:
        return override
    from_sidecar = env_id_from_sidecar(dataset_path)
    if from_sidecar:
        return from_sidecar
    return dataset_path.parent.parent.name


def trajectory_keys(h5_file: h5py.File) -> list[str]:
    """All `traj_*` keys in recorded order."""
    return sorted(
        (key for key in h5_file.keys() if key.startswith("traj_")),
        key=lambda key: int(key.split("_")[1]),
    )


def resolve_trajectory_key(h5_file: h5py.File, episode_idx: int) -> str:
    """Resolves an episode index to a `traj_*` key.

    Accepts either the recorded trajectory number (`traj_7`) or a positional index into the sorted
    list, including negative indices, so that `-1` means "the last episode".
    """
    keys = trajectory_keys(h5_file)
    if not keys:
        raise ValueError("No `traj_*` groups found in the dataset.")

    named = f"traj_{episode_idx}"
    if named in keys:
        return named

    if episode_idx >= len(keys) or episode_idx < -len(keys):
        raise IndexError(
            f"Episode {episode_idx} is out of range: the dataset has {len(keys)} episodes "
            f"({keys[0]}..{keys[-1]})."
        )
    return keys[episode_idx]


def parse_frame_spec(spec: str, seq_len: int) -> list[tuple[int, str]]:
    """Turns a frame specification into `(index, label)` pairs.

    Entries are either percentages of the episode (`50%`) or absolute indices (`12`); negative
    indices count from the end.
    """
    frames: list[tuple[int, str]] = []
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            continue
        try:
            if token.endswith("%"):
                fraction = float(token[:-1]) / 100.0
                index = int(round(fraction * (seq_len - 1)))
            else:
                index = int(token)
                if index < 0:
                    index += seq_len
        except ValueError:
            raise ValueError(
                f"Could not parse {token!r} in frame spec {spec!r}. Expected a percentage "
                "like '50%' or an integer index."
            ) from None
        frames.append((max(0, min(index, seq_len - 1)), token))

    if not frames:
        raise ValueError(f"Frame spec {spec!r} selected no frames.")
    return frames


def window_indices(end_index: int, obs_horizon: int) -> list[int]:
    """The `obs_horizon` frame indices ending at `end_index`."""
    return [max(0, i) for i in range(end_index - obs_horizon + 1, end_index + 1)]


def build_obs_batch(obs_tree, frame_indices: list[int], obs_horizon: int, device) -> TensorTree:
    """Stacks one observation window per sampled frame into a batch of `[N, obs_horizon, ...]`."""
    from policy.utils import cat_dicts, map_leaves, recursive_index, to_tensor

    windows = []
    for end_index in frame_indices:
        window = recursive_index(obs_tree, window_indices(end_index, obs_horizon))
        tensor_window = to_tensor(window, device=device, dtype=torch.float32)
        windows.append(map_leaves(lambda t: t.unsqueeze(0), tensor_window))
    return cat_dicts(windows)


def broadcast_goal(goal, batch_size: int) -> TensorTree:
    """Expands one goal instance across a batch, so every frame is scored against the same goal."""
    from policy.utils import map_leaves

    return map_leaves(lambda t: t.unsqueeze(0).expand(batch_size, *t.shape), goal)
