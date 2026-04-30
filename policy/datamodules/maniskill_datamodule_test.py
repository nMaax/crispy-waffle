import numpy as np
import pytest
import torch

from policy.datamodules.datamodule_tests import DataModuleTests
from policy.datamodules.maniskill_datamodule import ManiSkillDataModule
from policy.datamodules.maniskill_dataset import ManiSkillDataset


@pytest.mark.parametrize("datamodule_config", ["maniskill_datamodule"], indirect=True)
class TestManiSkillDataModule(DataModuleTests[ManiSkillDataModule]):
    """Test suite for the ManiSkillDataModule."""

    def test_padding_is_correct(self, datamodule: ManiSkillDataModule):
        # Ensure the train_set is built
        datamodule.setup("fit")
        dataset = datamodule.train_set

        if dataset is None:
            raise ValueError("Expected train_set to be initialized after setup('fit').")

        if not isinstance(dataset, ManiSkillDataset):
            raise TypeError("Expected train_set to be an instance of ManiSkillDataset.")

        # Test left padding
        # The first slice of any episode has t=0
        # cond_start = 0 - cond_horizon + 1. So if cond_horizon >= 2, cond_start < 0.
        assert datamodule.cond_horizon >= 2, (
            "This test assumes cond_horizon >= 2 to trigger left padding."
        )

        first_idx = 0
        traj_idx, cond_start, cond_end, act_start, act_end, L = dataset.slices[first_idx]
        assert cond_start < 0, "Expected negative start index to test left padding."

        sample = dataset[first_idx]
        pad_before = -cond_start

        def check_left_edge_padding(padded_tensor, pad_len):
            """Recursively check edge padding for tensors or dicts of tensors (like 'obs')."""
            if isinstance(padded_tensor, dict):
                for v in padded_tensor.values():
                    check_left_edge_padding(v, pad_len)
            else:
                for i in range(pad_len):
                    # Elements in the padded region should be identical to the first valid element
                    assert torch.allclose(padded_tensor[i], padded_tensor[pad_len]), (
                        "Left edge padding is incorrect."
                    )

        check_left_edge_padding(sample["cond_seq"], pad_before)
        check_left_edge_padding(sample["act_seq"], pad_before)

        # Test left padding
        # Grab the last slice of the first trajectory
        last_idx = L - 1
        traj_idx_last, _, _, a_s, a_e, L_last = dataset.slices[last_idx]
        assert traj_idx_last == traj_idx, "Expected to still be on the first trajectory."
        assert a_e > L_last, "Expected act_end to exceed sequence length to test right padding."

        sample_last = dataset[last_idx]
        act_seq_last = sample_last["act_seq"]
        pad_after = a_e - L_last
        valid_end_idx = datamodule.pred_horizon - pad_after - 1

        if dataset.delta_action_mask is not None:
            mask = torch.tensor(dataset.delta_action_mask, dtype=torch.bool)
            # Deltas are zero-padded (mask=True), absolutes are edge-padded (mask=False)
            for i in range(1, pad_after + 1):
                padded_step = act_seq_last[-i]
                edge_step = act_seq_last[valid_end_idx]

                assert torch.allclose(padded_step[~mask], edge_step[~mask]), (
                    "Right edge padding for absolute actions is incorrect."
                )
                assert torch.allclose(padded_step[mask], torch.zeros_like(padded_step[mask])), (
                    "Right zero padding for delta actions is incorrect."
                )
        else:
            # Everything is edge padded
            for i in range(1, pad_after + 1):
                assert torch.allclose(act_seq_last[-i], act_seq_last[valid_end_idx]), (
                    "Right edge padding for act_seq is incorrect."
                )

    def test_slice_and_pad_mask_variations(self, datamodule: ManiSkillDataModule):
        """Tests the `_slice_and_pad` method directly against different padding mask
        configurations."""
        datamodule.setup("fit")
        dataset = datamodule.train_set

        if dataset is None:
            raise ValueError("Expected train_set to be initialized after setup('fit').")

        # Create a controlled dummy sequence: 5 timesteps, variable action dimension
        act_dim = datamodule.act_dim
        seq_length = 5
        # Array like: [[0,1,2], [3,4,5], [6,7,8], [9,10,11], [12,13,14]]
        dummy_data = np.arange(seq_length * act_dim).reshape(seq_length, act_dim)

        # Setup slice boundaries that exceed the array length to trigger RIGHT padding
        start = 3
        end = 8  # 3 timesteps beyond L
        L = seq_length
        pad_after = end - L  # 3 frames of padding expected

        # Expected edge value is the last valid row in dummy_data
        edge_value = dummy_data[-1]

        # --- CASE A: pad_mask is None (Default Edge Padding) ---
        padded_none = dataset._slice_and_pad(dummy_data, start, end, L, pad_mask=None)
        for i in range(1, pad_after + 1):
            assert np.array_equal(padded_none[-i], edge_value), (
                "Failed Case A: `None` mask did not result in standard edge padding."
            )

        # --- CASE B: All Zeros Mask (All True) ---
        mask_all_zeros = np.ones(act_dim, dtype=bool)
        padded_zeros = dataset._slice_and_pad(dummy_data, start, end, L, pad_mask=mask_all_zeros)
        for i in range(1, pad_after + 1):
            assert np.array_equal(padded_zeros[-i], np.zeros(act_dim)), (
                "Failed Case B: All-True mask did not result in pure zero padding."
            )

        # --- CASE C: All Edge Mask (All False) ---
        mask_all_edges = np.zeros(act_dim, dtype=bool)
        padded_edges = dataset._slice_and_pad(dummy_data, start, end, L, pad_mask=mask_all_edges)
        for i in range(1, pad_after + 1):
            assert np.array_equal(padded_edges[-i], edge_value), (
                "Failed Case C: All-False mask did not result in pure edge padding."
            )

        # --- CASE D: Mixed Mask (Last dimension is Edge, others are Zeros) ---
        if act_dim > 1:
            mask_mixed = np.ones(act_dim, dtype=bool)
            mask_mixed[-1] = False  # Last dim gets edge padding

            padded_mixed = dataset._slice_and_pad(dummy_data, start, end, L, pad_mask=mask_mixed)
            for i in range(1, pad_after + 1):
                # Check that all elements EXCEPT the last one are zero
                assert np.array_equal(padded_mixed[-i, :-1], np.zeros(act_dim - 1)), (
                    "Failed Case D (Mixed): Elements designated for zero padding were not zero."
                )

                # Check that the LAST element matches the edge value
                assert padded_mixed[-i, -1] == edge_value[-1], (
                    "Failed Case D (Mixed): Element designated for edge padding did not match edge value."
                )
