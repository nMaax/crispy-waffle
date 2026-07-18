from collections.abc import Callable

from policy.transforms.canonicalization import PnPCanonicalizer, RemoveProprioVel
from policy.transforms.schema import DictFlattener, ManiSkillStateDeFlattener
from policy.utils.functional_utils import compose
from policy.utils.typing_utils import TensorTree


def observation_pipeline(
    env_id: str,
    is_flat: bool,
    canonicalize: bool,
    as_dict: bool,
    no_proprio_vel: bool = False,
) -> Callable[[TensorTree], TensorTree]:
    """Builds and composes a sequence of observation transforms based on format flags and
    environment ID.

    Args:
        env_id: The ManiSkill environment ID for schema de-flattening / canonicalization.
        is_flat: Whether the incoming observation is a flat tensor/array.
        canonicalize: Whether to apply PnPCanonicalizer.
        as_dict: Whether the output should remain a dictionary (True) or be flattened (False).
        no_proprio_vel: Whether to remove proprioceptive velocity via RemoveProprioVel.

    Returns:
        Composed transform callable accepting and returning a TensorTree.
    """
    transforms: list[Callable[[TensorTree], TensorTree]] = []
    if is_flat and (canonicalize or no_proprio_vel or as_dict):
        transforms.append(ManiSkillStateDeFlattener(env_id))
    if canonicalize:
        transforms.append(PnPCanonicalizer(env_id))
    if no_proprio_vel:
        transforms.append(RemoveProprioVel())
    if not as_dict and (not is_flat or transforms):
        transforms.append(DictFlattener())
    return compose(transforms)


build_canonicalization_transforms = observation_pipeline
