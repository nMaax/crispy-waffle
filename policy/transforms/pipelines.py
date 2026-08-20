from collections.abc import Callable
from typing import cast

import torch

from policy.transforms.canonicalization import Canonicalizer
from policy.transforms.canonicalization.state_unflatten import unflatten_stack_cube_state
from policy.utils.functional_utils import compose
from policy.utils.typing_utils import TensorTree


def observation_pipeline(
    env_id: str,
    canonicalize: bool = True,
    obs_mode: str = "state_dict",
) -> Callable[[TensorTree], TensorTree]:
    """Builds and composes a sequence of observation transforms based on environment ID.

    Args:
        env_id: The ManiSkill environment ID for canonicalization.
        canonicalize: Whether to apply Canonicalizer.
        obs_mode: The raw observation's shape convention. ``"state"`` observations are flat
            tensors -- unsupported by ``Canonicalizer`` (see its docstring) -- so they're
            unflattened into the same ``{"agent": ..., "extra": ...}`` tree a ``"state_dict"``
            observation already is, before canonicalization.

    Returns:
        Composed transform callable accepting and returning a TensorTree.
    """
    if obs_mode == "state" and env_id != "StackCubeLockedRotation-v1":
        raise NotImplementedError(
            f"obs_mode='state' is only supported for StackCubeLockedRotation-v1 "
            f"(unflatten_stack_cube_state's field layout is hardcoded to it), got env_id={env_id!r}."
        )

    transforms: list[Callable[[TensorTree], TensorTree]] = []
    if obs_mode == "state":
        transforms.append(lambda obs: unflatten_stack_cube_state(cast(torch.Tensor, obs)))
    if canonicalize:
        transforms.append(Canonicalizer(env_id))
    return compose(transforms)
