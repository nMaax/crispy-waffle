import torch

from policy.transforms.canonicalization.spec import ROLE_DIM, Role


def role_tensor(role: Role, like: torch.Tensor) -> torch.Tensor:
    return torch.tensor(role, dtype=like.dtype, device=like.device).expand(
        *like.shape[:-1], ROLE_DIM
    )


def as_flag_channel(flag: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Shapes a per-object scalar flag into a 1-wide channel, concatenable against `like`."""
    flag = flag.to(dtype=like.dtype, device=like.device)
    while flag.ndim < like.ndim:
        flag = flag.unsqueeze(-1)
    if flag.shape[:-1] != like.shape[:-1]:
        flag = flag.expand(*like.shape[:-1], 1)
    return flag
