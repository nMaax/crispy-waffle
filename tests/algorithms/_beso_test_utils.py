from unittest.mock import MagicMock

import torch


def mock_loop_internals(policy, network_return=None):
    """Mocks the Karras/sigma helpers so _run_diffusion_loop runs deterministically.

    Shared between `test_beso_policy.py` (BesoPolicy) and `test_beso_pp_policy.py`
    (BesoPlusPlusPolicy) -- the mocked internals are identical regardless of which class is being
    exercised.
    """
    if network_return is None:
        B_expanded = getattr(policy, "num_parallel_samples", 1)
        network_return = torch.ones((B_expanded, 1, policy.act_dim))
    policy.network = MagicMock(return_value=network_return)
    policy.ema = MagicMock()
    policy._get_karras_scalings = MagicMock(
        return_value=(torch.tensor(0.0), torch.tensor(1.0), torch.tensor(1.0))
    )
    policy._get_sigmas_exponential = MagicMock(return_value=torch.tensor([1.0, 0.0]))
    # Shape-aware mocks so .view(B_expanded, 1, 1) works for any batch size.
    policy._t_fn = MagicMock(side_effect=lambda sigma: torch.zeros_like(sigma))
    policy._sigma_fn = MagicMock(side_effect=lambda t: torch.ones_like(t))
