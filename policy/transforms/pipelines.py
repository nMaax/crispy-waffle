from collections.abc import Callable

from policy.transforms.canonicalization import Canonicalizer
from policy.utils.functional_utils import compose
from policy.utils.typing_utils import TensorTree


def observation_pipeline(
    env_id: str,
    canonicalize: bool = True,
) -> Callable[[TensorTree], TensorTree]:
    """Builds and composes a sequence of observation transforms based on environment ID.

    Args:
        env_id: The ManiSkill environment ID for canonicalization.
        canonicalize: Whether to apply Canonicalizer.

    Returns:
        Composed transform callable accepting and returning a TensorTree.
    """
    transforms: list[Callable[[TensorTree], TensorTree]] = []
    if canonicalize:
        transforms.append(Canonicalizer(env_id))
    return compose(transforms)
