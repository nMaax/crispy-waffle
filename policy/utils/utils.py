from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from logging import getLogger as get_logger
from typing import Any, overload

import numpy as np
import rich
import rich.syntax
import rich.tree
import torch
from lightning_utilities.core.rank_zero import rank_zero_info
from omegaconf import DictConfig, OmegaConf

from policy.utils.typing_utils import DimSpec, RawTree, StateSchema, TensorTree

logger = get_logger(__name__)


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


def print_mapping_tree(
    data: Mapping[str, Any], indent: str = "", use_rank_zero_info: bool = False
) -> None:
    """Recursively prints a Mapping as a tree.

    Prints .shape and .dtype for atomic elements that possess them.
    """
    print_wrapper = rank_zero_info if use_rank_zero_info else print

    items: list[tuple[str, Any]] = list(data.items())
    for i, (key, value) in enumerate(items):
        is_last = i == len(items) - 1
        branch = "└── " if is_last else "├── "

        # Branch: If the value is another mapping, recurse
        if isinstance(value, Mapping):
            print_wrapper(f"{indent}{branch}{key}")
            new_indent = indent + ("    " if is_last else "│   ")
            print_mapping_tree(value, new_indent)

        # Leaf: If the value has a shape (and potentially a dtype)
        elif hasattr(value, "shape"):
            # Get dtype if it exists, otherwise leave empty
            dtype_str = f", dtype={value.dtype}" if hasattr(value, "dtype") else ""
            print_wrapper(f"{indent}{branch}{key}: shape={value.shape}{dtype_str}")

        # Leaf: Basic types
        else:
            print_wrapper(f"{indent}{branch}{key}: {type(value).__name__}")


def to_tensor(
    data: RawTree,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> TensorTree:
    """Recursively converts a nested raw data tree to a nested dictionary of tensors."""
    if isinstance(data, Mapping):
        return {k: to_tensor(v, device=device, dtype=dtype) for k, v in data.items()}
    else:
        return torch.as_tensor(data, device=device, dtype=dtype)


def recursive_index(data: Any, idx: Any) -> Any:
    """Recursively indexes/slices leaf tensors inside nested dictionaries or lists."""
    if isinstance(data, Mapping):
        return {k: recursive_index(v, idx) for k, v in data.items()}
    elif isinstance(data, list):
        return [recursive_index(v, idx) for v in data]
    elif hasattr(data, "__getitem__") or isinstance(data, torch.Tensor):
        return data[idx]
    return data


@overload
def slice_by_schema(state: torch.Tensor, schema: StateSchema) -> TensorTree: ...


@overload
def slice_by_schema(state: np.ndarray, schema: StateSchema) -> RawTree: ...


def slice_by_schema(
    state: torch.Tensor | np.ndarray,
    schema: StateSchema,
) -> TensorTree | RawTree:
    """Recursively slices a state array or tensor according to a nested schema of index tuples."""
    result: dict[str, Any] = {}
    for key, val in schema.items():
        if isinstance(val, Mapping):
            result[key] = slice_by_schema(state, val)
        elif isinstance(val, tuple) and len(val) == 2:
            start, end = val
            result[key] = state[..., start:end]
        else:
            raise ValueError(f"Invalid schema entry for key '{key}': {val}")
    return result


def get_batch_size(data: TensorTree) -> int:
    """Recursively finds the batch size from a nested mapping of tensors."""
    if isinstance(data, torch.Tensor):
        return data.shape[0]
    else:
        for value in data.values():
            return get_batch_size(value)

    raise ValueError("data must contain at least one tensor")


def get_total_dim(data: DimSpec) -> int:
    """Recursively sums the last dimension of leaf structures.

    Accepts PyTorch tensors, configuration Mappings containing a 'shape' key, or raw shape
    descriptors (tuples, integers).
    """
    # Handle actual Tensors
    if isinstance(data, torch.Tensor):
        return int(data.shape[-1])

    # Handle Mappings
    if isinstance(data, Mapping):
        if "shape" in data:
            shape_val = data["shape"]
            if isinstance(shape_val, int):
                return shape_val
            elif isinstance(shape_val, Sequence | torch.Tensor) and not isinstance(shape_val, str):
                return int(shape_val[-1])
            else:
                raise TypeError(f"Unexpected 'shape' value type: {type(shape_val)}")
        else:
            # It's a nested container, recurse into values
            return sum(get_total_dim(v) for v in data.values())

    # Handle explicit shape tuples, e.g., (32, 10, 64)
    if isinstance(data, Sequence) and not isinstance(data, str):
        return int(data[-1])

    # Handle raw integer dimension sizes, e.g., 256
    if isinstance(data, int) and not isinstance(data, bool):
        return data

    raise TypeError(f"Unexpected DimSpec type: {type(data)}")


def get_device(data: TensorTree) -> torch.device:
    """Recursively finds the PyTorch device from a leaf tensor or nested tree of tensors."""
    if isinstance(data, torch.Tensor):
        return data.device
    for value in data.values():
        return get_device(value)
    raise ValueError("data must contain at least one tensor")


def cat_dicts(trees: Sequence[TensorTree]) -> TensorTree:
    """Recursively concatenates a sequence of nested dictionaries of tensors into a single nested
    dictionary of tensors."""
    first = trees[0]
    if isinstance(first, Mapping):
        sub_trees_by_key: dict[str, list[TensorTree]] = {k: [] for k in first}
        for t in trees:
            assert isinstance(t, Mapping), f"Expected element to be a Mapping, got {type(t)}"
            for k in first:
                sub_trees_by_key[k].append(t[k])
        return {k: cat_dicts(v) for k, v in sub_trees_by_key.items()}
    else:
        tensor_list: list[torch.Tensor] = []
        for t in trees:
            assert isinstance(t, torch.Tensor), f"Expected element to be a Tensor, got {type(t)}"
            tensor_list.append(t)
        return torch.cat(tensor_list, dim=0)


def concat_leaf_tensors(
    data: TensorTree,
    dim: int = -1,
    device: torch.device | None = None,
    preprocess: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Recursively concatenates all leaf tensors in a mapping along a specified dimension."""
    if isinstance(data, torch.Tensor):
        if device is not None:
            data = data.to(device)
        if preprocess is not None:
            data = preprocess(data)
        return data

    tensors: list[torch.Tensor] = []
    for value in data.values():
        tensors.append(concat_leaf_tensors(value, dim=dim, device=device, preprocess=preprocess))

    if not tensors:
        raise ValueError("data must contain at least one tensor")

    return torch.cat(tensors, dim=dim)


def flatten_and_concat_leaf_tensors(
    data: TensorTree, device: torch.device | None = None
) -> torch.Tensor:
    """Recursively flattens all leaf tensors starting from dimension 1, then concatenates them."""
    return concat_leaf_tensors(
        data, dim=1, device=device, preprocess=lambda x: x.flatten(start_dim=1)
    )


def merge_dicts(mappings: Sequence[Mapping[str, TensorTree]]) -> dict[str, TensorTree]:
    """Union of mapping over keys."""
    merged: dict[str, TensorTree] = {}
    for mapping in mappings:
        collisions = merged.keys() & mapping.keys()
        if collisions:
            raise ValueError(f"Duplicate key(s) {collisions} found while merging mappings.")
        merged.update(mapping)
    return merged


def map_leaves(
    fn: Callable[[torch.Tensor], torch.Tensor],
    tree: TensorTree,
) -> TensorTree:
    """Recursively applies `fn` to every leaf tensor in a tree, preserving the Mapping
    structure."""
    if isinstance(tree, torch.Tensor):
        return fn(tree)
    return {k: map_leaves(fn, v) for k, v in tree.items()}
