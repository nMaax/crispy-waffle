import torch

from policy.utils.functional_utils import compose


def test_compose():
    f1 = lambda x: x + 1.0  # # noqa: E731
    f2 = lambda x: x * 2.0  # # noqa: E731
    composed = compose([f1, f2])

    inp = torch.tensor(3.0)
    out = composed(inp)
    assert out == 8.0
