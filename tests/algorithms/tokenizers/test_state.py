import pytest
import torch
from mani_skill.utils.geometry.rotation_conversions import axis_angle_to_quaternion

from policy.algorithms.tokenizers.state import StateTokenizer


def _pose(pos: tuple[float, float, float], axis_angle: tuple[float, float, float] = (0, 0, 0)):
    p = torch.tensor(pos, dtype=torch.float32)
    aa = torch.tensor(axis_angle, dtype=torch.float32)
    q = axis_angle_to_quaternion(aa)
    return torch.cat([p, q], dim=-1)


class TestStateTokenizer:
    def test_output_dim_matches_task_dim(self):
        tokenizer = StateTokenizer(task_dim={"task": 42}, relative_goal=False)
        assert tokenizer.output_dim == 42
        assert tokenizer.tokens_per_step == 1

        # A canonical pool: poses collapse 7D -> 6D SE(3) deltas, categoricals keep full width,
        # and every entry's slot axis is folded into the single per-timestep vector.
        task_spec = {"obj_pose": (3, 7), "obj_role": (3, 4), "obj_valid": (3,)}
        tokenizer_spec = StateTokenizer(task_dim=task_spec, relative_goal=True)
        assert tokenizer_spec.output_dim == 3 * 6 + 3 * 4 + 3  # 33

    def test_a_key_with_no_goal_relative_reduction_is_rejected_at_construction(self):
        """Neither a pose nor categorical: there is no defined delta, so refuse rather than
        invent one. Caught while sizing, so it cannot surface first as a shape error."""
        with pytest.raises(ValueError, match="neither a pose nor categorical"):
            StateTokenizer(task_dim={"obj_pose": (3, 7), "other": 3}, relative_goal=True)

    def test_single_side_tokenization_flattens_mapping(self):
        tokenizer = StateTokenizer(task_dim={"a": 3, "b": 7}, relative_goal=False)
        obs_dict = {"a": torch.randn(2, 4, 3), "b": torch.randn(2, 4, 7)}
        expected = torch.cat([obs_dict["a"], obs_dict["b"]], dim=-1)
        assert torch.equal(tokenizer.tokenize(obs_dict, None), expected)

    def test_two_sided_tokenization_computes_se3_quaternion_deltas(self):
        task_spec = {"obj_0_pose": 7, "obj_1_pose": 7, "tcp_pose": 7}
        tokenizer = StateTokenizer(task_dim=task_spec, relative_goal=True)
        assert tokenizer.output_dim == 18

        obs = {
            "obj_0_pose": _pose((0, 0, 0)).unsqueeze(0).unsqueeze(0),
            "obj_1_pose": _pose((1, 0, 0)).unsqueeze(0).unsqueeze(0),
            "tcp_pose": _pose((0, 1, 0)).unsqueeze(0).unsqueeze(0),
        }
        goal = {
            "obj_0_pose": _pose((1, 2, 3), axis_angle=(0, 0, torch.pi / 2)).unsqueeze(0),
            "obj_1_pose": _pose((1, 0, 0)).unsqueeze(0),
            "tcp_pose": _pose((0, 1, 0)).unsqueeze(0),
        }
        tokens = tokenizer.tokenize(obs, goal)
        assert tokens.shape == (1, 1, 18)

        # obj_0_pose (first 6 dims): position delta (1,2,3), rotation delta pi/2 about z
        r_0 = tokens[0, 0, :6]
        assert torch.allclose(r_0[:3], torch.tensor([1.0, 2.0, 3.0]))
        assert torch.allclose(r_0[3:6].norm(), torch.tensor(torch.pi / 2), atol=1e-5)

    def test_missing_key_in_goal_raises_keyerror(self):
        task_spec = {"obj_0_pose": 7, "obj_1_pose": 7}
        tokenizer = StateTokenizer(task_dim=task_spec, relative_goal=True)
        obs = {"obj_0_pose": _pose((0, 0, 0)), "obj_1_pose": _pose((1, 0, 0))}
        goal = {"obj_0_pose": _pose((1, 1, 1))}
        with pytest.raises(KeyError, match="obj_1_pose"):
            tokenizer.tokenize(obs, goal)

    def test_tokenize_raises_if_both_none(self):
        tokenizer = StateTokenizer(task_dim={"obj_pose": (2, 7)})
        with pytest.raises(ValueError, match="at least one"):
            tokenizer.tokenize(None, None)

    def test_single_side_support(self):
        assert StateTokenizer.supports_single_side is True
