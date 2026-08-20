import torch

from policy.utils.typing_utils import TensorTree

# (group, key, width), in `flatten_state_dict`'s insertion order for StackCubeLockedRotation-v1's
# `_get_obs_state_dict()` (agent.qpos, agent.qvel, then `_get_obs_extra()`'s own field order).
_STACK_CUBE_FIELDS: list[tuple[str, str, int]] = [
    ("agent", "qpos", 9),
    ("agent", "qvel", 9),
    ("extra", "tcp_pose", 7),
    ("extra", "cubeA_pose", 7),
    ("extra", "cubeB_pose", 7),
    ("extra", "tcp_to_cubeA_pos", 3),
    ("extra", "tcp_to_cubeB_pos", 3),
    ("extra", "cubeA_to_cubeB_pos", 3),
]

_TOTAL_WIDTH = sum(width for _, _, width in _STACK_CUBE_FIELDS)


def unflatten_stack_cube_state(flat: torch.Tensor) -> TensorTree:
    """Reconstructs the ``{"agent": {...}, "extra": {...}}`` tree :class:`Canonicalizer` expects
    from a flat ``obs_mode="state"`` observation of ``StackCubeLockedRotation-v1``.

    ``obs_mode="state"`` observations never pass through ``FrameStack``'s dict-recursion path
    (``FrameStack.use_dict`` is only true for dict-shaped obs modes), so unflattening here lets a
    live rollout be driven from flat ``state`` while still feeding the tokenizer the same
    canonical tree a ``state_dict`` rollout would.
    """
    if flat.shape[-1] != _TOTAL_WIDTH:
        raise ValueError(f"Expected a flat state of width {_TOTAL_WIDTH}, got {flat.shape[-1]}.")

    tree: dict[str, dict[str, torch.Tensor]] = {"agent": {}, "extra": {}}
    offset = 0
    for group, key, width in _STACK_CUBE_FIELDS:
        tree[group][key] = flat[..., offset : offset + width]
        offset += width
    return tree
