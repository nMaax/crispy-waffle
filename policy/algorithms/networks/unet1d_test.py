import hydra
import hydra_zen
import torch


def test_unet1d_instantiates_and_runs():
    with hydra.initialize(version_base="1.2", config_path="../../configs/algorithm/network"):
        cfg = hydra.compose(config_name="unet1d")

    # Inject the dimensions directly into instantiate
    network = hydra_zen.instantiate(cfg, input_dim=8, global_cond_dim=104)

    # Dummy inputs based on the overrides we just provided
    batch_size = 2
    horizon = 16
    input_dim = 8
    global_cond_dim = 104
    sample = torch.randn(batch_size, horizon, input_dim)

    timestep = torch.randint(0, 100, (batch_size,))

    global_cond = torch.randn(batch_size, global_cond_dim)

    # Run a forward pass to ensure the architecture doesn't crash
    output = network(sample, timestep, global_cond=global_cond)

    # For a 1D Unet in diffusion, output shape should match the sample shape
    assert output.shape == sample.shape
