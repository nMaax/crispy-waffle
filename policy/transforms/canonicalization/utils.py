import torch

from policy.transforms.canonicalization.spec import ROLE_DIM, Role


def role_tensor(role: Role, like: torch.Tensor) -> torch.Tensor:
    return torch.tensor(role, dtype=like.dtype, device=like.device).expand(
        *like.shape[:-1], ROLE_DIM
    )


def match_shape(t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Broadcasts a per-object scalar flag (e.g. is_pick/is_active) to like's batch shape, adding a
    trailing singleton dim so it concatenates against role/pose tensors."""
    t = t.to(dtype=like.dtype, device=like.device)
    while t.ndim < like.ndim:
        t = t.unsqueeze(-1)
    if t.shape[:-1] != like.shape[:-1]:
        t = t.expand(*like.shape[:-1], 1)
    return t
