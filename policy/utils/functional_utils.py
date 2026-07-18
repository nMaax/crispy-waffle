from collections.abc import Callable, Sequence

from policy.utils.typing_utils import TensorTree


def compose(
    transforms: Sequence[Callable[[TensorTree], TensorTree]],
) -> Callable[[TensorTree], TensorTree]:
    """Composes a sequence of transforms into a single callable transform."""

    def apply(obs: TensorTree) -> TensorTree:
        for t in transforms:
            obs = t(obs)
        return obs

    return apply
