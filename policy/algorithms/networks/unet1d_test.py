import hydra
import hydra_zen
import torch


def test_unet1d_instantiates_and_runs():
    with hydra.initialize(version_base="1.2", config_path="../../configs/algorithm/network"):
        cfg = hydra.compose(config_name="unet1d")

    # Dummy inputs based on the overrides we just provided
    batch_size = 128
    horizon = 16  # Number of timesteps to predict/diffuse upon
    input_dim = 8  # Dimensionality of each element in the sequence being diffused
    external_cond_dim = 67  # Usually cond_horizon * cond_dim

    # Inject the dimensions directly into instantiate
    network = hydra_zen.instantiate(cfg, input_dim=input_dim, external_cond_dim=external_cond_dim)

    # Sample to de-noise
    sample = torch.randn(batch_size, horizon, input_dim)

    # Timestep integer (will be embedded by Sinusoidal positioning)
    timestep = torch.randint(0, 100, (batch_size,))

    # This and the timestep embedding (which is generated inside the network) are what enters FiLM
    flatten_cond = torch.randn(batch_size, external_cond_dim)

    # Run a forward pass to ensure the architecture doesn't crash
    output = network(sample, timestep, external_cond=flatten_cond)

    # For a 1D Unet in diffusion, output shape should match the sample shape
    assert output.shape == sample.shape
