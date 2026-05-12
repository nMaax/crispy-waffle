import hydra
import hydra_zen
import torch


def test_mlp_instantiates_and_runs():
    with hydra.initialize(version_base="1.2", config_path="../../configs/algorithm/network"):
        cfg = hydra.compose(config_name="mlp")

    batch_size = 32
    input_dim = 16
    output_dim = 4
    hidden_dims = [64, 128]

    network = hydra_zen.instantiate(
        cfg, input_dim=input_dim, output_dim=output_dim, hidden_dims=hidden_dims
    )

    sample = torch.randn(batch_size, input_dim)

    output = network(sample)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, output_dim), (
        f"Expected shape {(batch_size, output_dim)}, got {output.shape}"
    )

    expected_layer_count = (len(hidden_dims) * 2) + 1
    assert len(network.net) == expected_layer_count
