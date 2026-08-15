import pytest
import torch
from mani_skill.utils.geometry.rotation_conversions import axis_angle_to_quaternion

from policy.algorithms.networks.encoder.tokenizers.state import StateTokenizer


def _pose(pos: tuple[float, float, float], axis_angle: tuple[float, float, float] = (0, 0, 0)):
    p = torch.tensor(pos, dtype=torch.float32)
    aa = torch.tensor(axis_angle, dtype=torch.float32)
    q = axis_angle_to_quaternion(aa)
    return torch.cat([p, q], dim=-1)


class TestStateTokenizer:
    def test_output_dim_matches_task_dim(self):
        tokenizer = StateTokenizer(task_dim=42, relative_goal=False)
        assert tokenizer.output_dim == 42
        assert tokenizer.tokens_per_step == 1

        tokenizer_rel = StateTokenizer(task_dim=42, relative_goal=True)
        assert tokenizer_rel.output_dim == 36  # (42 // 7) * 6 = 36

    def test_single_side_tokenization_preserves_dict_or_tensor(self):
        tokenizer = StateTokenizer(task_dim=10)
        t = torch.randn(2, 4, 10)
        assert torch.equal(tokenizer.tokenize(t, None), t)
        assert torch.equal(tokenizer.tokenize(None, t), t)

        obs_dict = {"a": torch.randn(2, 4, 3), "b": torch.randn(2, 4, 7)}
        expected = torch.cat([obs_dict["a"], obs_dict["b"]], dim=-1)
        assert torch.equal(tokenizer.tokenize(obs_dict, None), expected)

    def test_two_sided_tokenization_differences_goal_and_obs(self):
        tokenizer = StateTokenizer(task_dim=5)
        obs = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        goal = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])
        # obs gets time axis [1, 1, 5]; goal is [1, 5]
        delta = tokenizer.tokenize(obs.unsqueeze(0).unsqueeze(0), goal.unsqueeze(0))
        assert torch.equal(delta, (goal - obs).unsqueeze(0).unsqueeze(0))

    def test_two_sided_tokenization_computes_se3_quaternion_deltas(self):
        tokenizer = StateTokenizer(task_dim=21, relative_goal=True)
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


    def test_tokenize_raises_if_both_none(self):
        tokenizer = StateTokenizer(task_dim=5)
        with pytest.raises(ValueError, match="at least one"):
            tokenizer.tokenize(None, None)

    def test_single_side_support(self):
        assert StateTokenizer.supports_single_side is True
