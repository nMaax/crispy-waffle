import pytest
import torch
from mani_skill.utils.geometry.rotation_conversions import axis_angle_to_quaternion

from policy.algorithms.networks.encoder.tokenizers.object import ObjectTokenizer


def _pose(pos: tuple[float, float, float], axis_angle: tuple[float, float, float] = (0, 0, 0)):
    p = torch.tensor(pos, dtype=torch.float32)
    aa = torch.tensor(axis_angle, dtype=torch.float32)
    q = axis_angle_to_quaternion(aa)
    return torch.cat([p, q], dim=-1)


class TestObjectTokenizer:
    def test_output_dim_relative_and_absolute_modes(self):
        assert ObjectTokenizer(relative_goal=True).output_dim == 6
        assert ObjectTokenizer(relative_goal=False).output_dim == 7

    def test_tokens_per_step_matches_object_keys_length(self):
        assert ObjectTokenizer().tokens_per_step == 3
        assert ObjectTokenizer(object_keys=("a_pose", "tcp_pose")).tokens_per_step == 2

    def _obs_goal(self):
        obs = {
            "a_pose": _pose((0, 0, 0)).unsqueeze(0).unsqueeze(0),
            "b_pose": _pose((1, 0, 0)).unsqueeze(0).unsqueeze(0),
            "tcp_pose": _pose((0, 1, 0)).unsqueeze(0).unsqueeze(0),
        }
        goal = {
            "a_pose": _pose((1, 2, 3), axis_angle=(0, 0, torch.pi / 2)).unsqueeze(0),
            "b_pose": _pose((1, 0, 0)).unsqueeze(0),
            "tcp_pose": _pose((0, 1, 0)).unsqueeze(0),
        }
        return obs, goal

    def test_r_is_the_goal_minus_obs_delta_with_a_proper_rotation_operation(self):
        tokenizer = ObjectTokenizer(relative_goal=True)
        obs, goal = self._obs_goal()

        tokens = tokenizer.tokenize(obs, goal)
        assert tokens.shape == (1, 1, 3, 6)

        # objA (index 0): position delta (1,2,3), rotation delta of pi/2 about z.
        r_a = tokens[0, 0, 0]
        assert torch.allclose(r_a[:3], torch.tensor([1.0, 2.0, 3.0]))
        assert torch.allclose(r_a[3:6].norm(), torch.tensor(torch.pi / 2), atol=1e-5)

        # objB and TCP (indices 1, 2): goal == obs, so r must be exactly zero.
        assert torch.allclose(tokens[0, 0, 1], torch.zeros(6), atol=1e-6)
        assert torch.allclose(tokens[0, 0, 2], torch.zeros(6), atol=1e-6)

    def test_single_side_tokenization_absolute_mode(self):
        tokenizer = ObjectTokenizer(relative_goal=False)
        obs, _ = self._obs_goal()

        tokens = tokenizer.tokenize(obs, None)
        assert tokens.shape == (1, 1, 3, 7)
        assert torch.equal(tokens[0, 0, 0], obs["a_pose"][0, 0])
        assert torch.equal(tokens[0, 0, 1], obs["b_pose"][0, 0])
        assert torch.equal(tokens[0, 0, 2], obs["tcp_pose"][0, 0])

    def test_tokenize_requires_dict_shaped_task_trees(self):
        tokenizer = ObjectTokenizer()
        with pytest.raises(TypeError):
            tokenizer.tokenize(torch.randn(1, 1, 30), torch.randn(1, 30))

    def test_tokenize_raises_on_missing_pose_keys(self):
        tokenizer = ObjectTokenizer()
        obs, goal = self._obs_goal()
        del obs["b_pose"]
        with pytest.raises(KeyError):
            tokenizer.tokenize(obs, goal)

    def test_single_side_support(self):
        assert ObjectTokenizer.supports_single_side is True
