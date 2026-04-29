from __future__ import annotations

from collections.abc import Mapping, Sequence
from logging import getLogger as get_logger
from typing import Any

import numpy as np
import rich
import rich.syntax
import rich.tree
import torch
from lightning_utilities.core.rank_zero import rank_zero_info
from omegaconf import DictConfig, OmegaConf

logger = get_logger(__name__)


# @rank_zero_only
def print_config(
    config: DictConfig,
    print_order: Sequence[str] = (
        "algorithm",
        "datamodule",
        "trainer",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    TAKEN FROM https://github.com/ashleve/lightning-hydra-template/blob/6a92395ed6afd573fa44dd3a054a603acbdcac06/src/utils/__init__.py#L56

    Args:
        config: Configuration composed by Hydra.
        print_order: Determines in what order config components are printed.
        resolve: Whether to resolve reference fields of DictConfig.
    """

    style: str = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue: list[Any] = []

    for f in print_order:
        if f in config:
            queue.append(f)
        else:
            logger.info(f"Field '{f}' not found in config")

    for f in config:
        if f not in queue:
            queue.append(f)

    for f in queue:
        if f not in config:
            logger.info(f"Field '{f}' not found in config")
            continue
        branch = tree.add(f, style=style, guide_style=style)

        config_group = config[f]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    # with open("config_tree.log", "w") as file:
    #     rich.print(tree, file=file)


def print_dict_tree(
    data: Mapping[str, Any], indent: str = "", use_rank_zero_info: bool = False
) -> None:
    """Recursively prints a dictionary as a tree.

    Prints .shape and .dtype for atomic elements that possess them.
    """
    items: list[tuple[str, Any]] = list(data.items())
    for i, (key, value) in enumerate(items):
        is_last = i == len(items) - 1
        branch = "└── " if is_last else "├── "

        print_wrapper = rank_zero_info if use_rank_zero_info else print

        # Branch: If the value is another dictionary, recurse
        if isinstance(value, dict):
            print_wrapper(f"{indent}{branch}{key}")
            new_indent = indent + ("    " if is_last else "│   ")
            print_dict_tree(value, new_indent)

            # Leaf: If the value has a shape (and potentially a dtype)
        elif hasattr(value, "shape"):
            # Get dtype if it exists, otherwise leave empty
            dtype_str = f", dtype={value.dtype}" if hasattr(value, "dtype") else ""
            print_wrapper(f"{indent}{branch}{key}: shape={value.shape}{dtype_str}")

            # Leaf: Basic types
        else:
            print_wrapper(f"{indent}{branch}{key}: {type(value).__name__}")


def to_tensor(
    data: np.ndarray | dict[str, Any], device: torch.device | None = None
) -> torch.Tensor | dict[str, Any]:
    """Recursively converts a nested dictionary of numpy arrays to a nested dictionary of
    tensors."""
    if isinstance(data, dict):
        return {k: to_tensor(v, device) for k, v in data.items()}
    tensor = torch.from_numpy(data).float()
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def get_batch_size(data: Mapping[str, Any] | torch.Tensor) -> int:
    """Recursively finds the batch size from a nested dictionary of tensors."""
    if isinstance(data, torch.Tensor):
        return data.shape[0]
    for value in data.values():
        return get_batch_size(value)
    raise ValueError("data must contain at least one tensor")


def flatten_tensor_dict(
    data: Mapping[str, Any] | torch.Tensor, device: torch.device | None = None
) -> torch.Tensor:
    """Recursively flattens a dictionary of tensors and concatenates them."""
    if isinstance(data, torch.Tensor):
        if device is not None:
            data = data.to(device)
        return data.flatten(start_dim=1)

    tensors: list[torch.Tensor] = []
    for value in data.values():
        tensors.append(flatten_tensor_dict(value))

    if not tensors:
        raise ValueError("data must contain at least one tensor")

    return torch.cat(tensors, dim=1)


def sum_shapes(d) -> int:
    """Recursively sums the last dimension of all leaf dicts that contain a 'shape' key, or returns
    the value if it's an int or a tuple."""
    if isinstance(d, dict):
        if "shape" in d:
            return d["shape"][-1]
        return sum(sum_shapes(v) for v in d.values() if isinstance(v, dict | int | tuple))
    elif isinstance(d, int):
        return d
    elif isinstance(d, tuple):
        return d[-1]
    else:
        raise TypeError(f"Unsupported type for sum_shapes: {type(d)}")
