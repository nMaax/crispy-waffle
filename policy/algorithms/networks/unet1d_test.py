import hydra
import pytest
import torch
from omegaconf import DictConfig


@pytest.mark.parametrize("algorithm_network_config", ["unet1d"], indirect=True)
def test_unet1d_instantiates_and_runs(dict_config: DictConfig):
    """Test specifically reserved for the Unet1D network configuration."""
    network_config = dict_config.algorithm.network
    network = hydra.utils.instantiate(network_config)

    assert isinstance(network, torch.nn.Module)
    assert network.__class__.__name__ == "ConditionalUnet1D"

    B, pred_horizon, act_dim = 2, 16, network_config.input_dim

    sample = torch.randn(B, pred_horizon, act_dim)
    timestep = torch.tensor([10, 50])
    global_cond = torch.randn(B, network_config.global_cond_dim)

    out = network(sample, timestep, global_cond=global_cond)
    assert out.shape == sample.shape
