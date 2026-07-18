from collections.abc import Mapping

import torch

from policy.transforms import MinMaxNormalizer, ZScoreNormalizer


class CustomMapping(Mapping):
    """Custom mapping implementation for testing generic Mapping compatibility."""

    def __init__(self, data: dict):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)


def test_z_score_normalizer_flat_tensor():
    torch.manual_seed(42)
    dim = 10
    normalizer = ZScoreNormalizer(spec=dim)
    assert not normalizer.is_fit

    data = torch.randn(100, dim) * 5.0 + 3.0
    normalizer.fit(data)

    assert normalizer.is_fit
    assert torch.allclose(normalizer.mean, data.mean(dim=0), atol=1e-5)
    assert torch.allclose(normalizer.std, data.std(dim=0), atol=1e-5)

    normed = normalizer(data)
    assert torch.allclose(normed.mean(dim=0), torch.zeros(dim), atol=1e-4)
    assert torch.allclose(normed.std(dim=0), torch.ones(dim), atol=1e-4)

    reconstructed = normalizer.unnormalize(normed)
    assert torch.allclose(reconstructed, data, atol=1e-5)


def test_min_max_normalizer_flat_tensor():
    torch.manual_seed(42)
    dim = 8
    normalizer = MinMaxNormalizer(spec=dim, min_val=-1.0, max_val=1.0)
    assert not normalizer.is_fit

    data = torch.randn(100, dim) * 10.0
    normalizer.fit(data)

    assert normalizer.is_fit
    assert torch.allclose(normalizer.min, data.min(dim=0).values, atol=1e-5)
    assert torch.allclose(normalizer.max, data.max(dim=0).values, atol=1e-5)

    normed = normalizer(data)
    assert (normed >= -1.0 - 1e-5).all() and (normed <= 1.0 + 1e-5).all()

    reconstructed = normalizer.unnormalize(normed)
    assert torch.allclose(reconstructed, data, atol=1e-5)


def test_nested_mapping_normalizer():
    torch.manual_seed(42)
    spec = {
        "proprio": 6,
        "extra": {"pose": 7, "vel": 3},
    }

    z_norm = ZScoreNormalizer(spec=spec)
    mm_norm = MinMaxNormalizer(spec=spec)

    batch = {
        "proprio": torch.randn(50, 6) * 2.0 + 1.0,
        "extra": {
            "pose": torch.randn(50, 7) * 0.5 - 2.0,
            "vel": torch.randn(50, 3) * 10.0,
        },
    }

    z_norm.fit(batch)
    mm_norm.fit(batch)

    assert z_norm.is_fit
    assert mm_norm.is_fit

    # Test with custom Mapping input
    custom_batch = CustomMapping(batch)
    normed_z = z_norm(custom_batch)
    normed_mm = mm_norm(custom_batch)

    assert isinstance(normed_z, dict)
    assert isinstance(normed_mm, dict)

    reconstructed_z = z_norm.unnormalize(normed_z)
    reconstructed_mm = mm_norm.unnormalize(normed_mm)

    assert torch.allclose(reconstructed_z["proprio"], batch["proprio"], atol=1e-5)
    assert torch.allclose(
        reconstructed_z["extra"]["pose"], batch["extra"]["pose"], atol=1e-5
    )
    assert torch.allclose(reconstructed_mm["proprio"], batch["proprio"], atol=1e-5)


def test_incremental_vs_direct_fit():
    torch.manual_seed(42)
    dim = 12
    full_data = torch.randn(200, dim) * 4.0 + 7.0
    batches = [full_data[i : i + 20] for i in range(0, 200, 20)]

    # ZScoreNormalizer comparison (flat)
    norm_direct = ZScoreNormalizer(spec=dim)
    norm_direct.fit(full_data)

    norm_inc = ZScoreNormalizer(spec=dim)
    norm_inc.fit_incremental(batches)

    assert torch.allclose(norm_direct.mean, norm_inc.mean, atol=1e-4)
    assert torch.allclose(norm_direct.std, norm_inc.std, atol=1e-4)

    # MinMaxNormalizer comparison (flat)
    mm_direct = MinMaxNormalizer(spec=dim)
    mm_direct.fit(full_data)

    mm_inc = MinMaxNormalizer(spec=dim)
    mm_inc.fit_incremental(batches)

    assert torch.allclose(mm_direct.min, mm_inc.min, atol=1e-5)
    assert torch.allclose(mm_direct.max, mm_inc.max, atol=1e-5)

    # Nested dictionary equivalence comparison
    spec_nested = {"a": 6, "b": 6}
    full_nested = {
        "a": full_data[:, :6],
        "b": full_data[:, 6:],
    }
    batches_nested = [
        {"a": b[:, :6], "b": b[:, 6:]} for b in batches
    ]

    z_nest_direct = ZScoreNormalizer(spec=spec_nested)
    z_nest_direct.fit(full_nested)

    z_nest_inc = ZScoreNormalizer(spec=spec_nested)
    z_nest_inc.fit_incremental(batches_nested)

    assert torch.allclose(
        z_nest_direct.norms["a"].mean, z_nest_inc.norms["a"].mean, atol=1e-4
    )
    assert torch.allclose(
        z_nest_direct.norms["b"].std, z_nest_inc.norms["b"].std, atol=1e-4
    )


def test_single_sample_n1_fit_incremental():
    dim = 5
    z_norm = ZScoreNormalizer(spec=dim)
    single_sample = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])

    z_norm.fit_incremental([single_sample])

    assert z_norm.is_fit
    assert torch.allclose(z_norm.mean, single_sample[0])
    assert torch.allclose(z_norm.std, torch.ones(dim))


def test_empty_fit_incremental():
    dim = 5
    z_norm = ZScoreNormalizer(spec=dim)
    z_norm.fit_incremental([])

    assert not z_norm.is_fit
