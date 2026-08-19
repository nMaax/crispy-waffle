import pytest
import torch
from mani_skill.utils.geometry.rotation_conversions import axis_angle_to_quaternion

from policy.algorithms.tokenizers.object import ObjectTokenizer

TCP = torch.tensor([1.0, 0.0, 0.0, 0.0])
PICK = torch.tensor([0.0, 1.0, 0.0, 0.0])
TARGET = torch.tensor([0.0, 0.0, 1.0, 0.0])
CLUTTER = torch.tensor([0.0, 0.0, 0.0, 1.0])


def _pose(pos: tuple[float, float, float], axis_angle: tuple[float, float, float] = (0, 0, 0)):
    p = torch.tensor(pos, dtype=torch.float32)
    aa = torch.tensor(axis_angle, dtype=torch.float32)
    q = axis_angle_to_quaternion(aa)
    return torch.cat([p, q], dim=-1)


class TestObjectTokenizer:
    def test_output_dim_relative_and_absolute_modes(self):
        assert ObjectTokenizer(relative_goal=True).output_dim == 10
        assert ObjectTokenizer(relative_goal=False).output_dim == 11

    def test_include_role_false_drops_the_trailing_one_hot(self):
        assert ObjectTokenizer(relative_goal=True, include_role=False).output_dim == 6
        assert ObjectTokenizer(relative_goal=False, include_role=False).output_dim == 7

    def test_tokens_per_step_matches_object_keys_length(self):
        assert ObjectTokenizer().tokens_per_step is None
        assert ObjectTokenizer(object_keys=("obj_0_pose", "obj_1_pose")).tokens_per_step == 2

    def _obs_goal(self):
        obs = {
            "tcp_pose": _pose((0, 1, 0)).unsqueeze(0).unsqueeze(0),
            "tcp_role": TCP.unsqueeze(0).unsqueeze(0),
            "obj_0_pose": _pose((0, 0, 0)).unsqueeze(0).unsqueeze(0),
            "obj_0_role": PICK.unsqueeze(0).unsqueeze(0),
            "obj_1_pose": _pose((1, 0, 0)).unsqueeze(0).unsqueeze(0),
            "obj_1_role": TARGET.unsqueeze(0).unsqueeze(0),
        }
        goal = {
            "tcp_pose": _pose((0, 1, 0)).unsqueeze(0),
            "obj_0_pose": _pose((1, 2, 3), axis_angle=(0, 0, torch.pi / 2)).unsqueeze(0),
            "obj_1_pose": _pose((1, 0, 0)).unsqueeze(0),
        }
        return obs, goal

    def test_tcp_is_tokenized_with_its_own_goal_delta(self):
        tokenizer = ObjectTokenizer(relative_goal=True)
        obs, goal = self._obs_goal()
        goal["tcp_pose"] = _pose((2, 1, 0.5)).unsqueeze(0)

        tokens = tokenizer.tokenize(obs, goal)
        tok_tcp = tokens[0, 0, 0]
        # The gripper's own displacement to the goal.
        assert torch.allclose(tok_tcp[:3], torch.tensor([2.0, 0.0, 0.5]))
        assert torch.allclose(tok_tcp[6:10], TCP)

    def test_enriched_token_structure(self):
        tokenizer = ObjectTokenizer(relative_goal=True)
        obs, goal = self._obs_goal()

        tokens = tokenizer.tokenize(obs, goal)
        # tcp_pose, obj_0_pose, obj_1_pose
        assert tokens.shape == (1, 1, 3, 10)

        # Token 1 (obj_0, pick):
        # goal_delta: goal_0 (1,2,3) - obj_0 (0,0,0) = (1, 2, 3), rotvec pi/2 about z
        tok_0 = tokens[0, 0, 1]
        assert torch.allclose(tok_0[:3], torch.tensor([1.0, 2.0, 3.0]))
        assert torch.allclose(tok_0[3:6].norm(), torch.tensor(torch.pi / 2), atol=1e-5)
        assert torch.allclose(tok_0[6:10], PICK)

        # Token 2 (obj_1, target):
        # goal_delta: goal_1 == obj_1 -> 0
        tok_1 = tokens[0, 0, 2]
        assert torch.allclose(tok_1[:6], torch.zeros(6), atol=1e-6)
        assert torch.allclose(tok_1[6:10], TARGET)

    def test_role_free_tokens_keep_only_the_goal_delta(self):
        tokenizer = ObjectTokenizer(relative_goal=True, include_role=False)
        obs, goal = self._obs_goal()

        tokens = tokenizer.tokenize(obs, goal)
        assert tokens.shape == (1, 1, 3, 6)
        assert tokenizer.categorical_mask.all()

        with_role = ObjectTokenizer(relative_goal=True).tokenize(obs, goal)
        assert torch.allclose(tokens, with_role[..., :6])

    def test_dynamic_clutter_tokens(self):
        tokenizer = ObjectTokenizer(relative_goal=True)
        obs, goal = self._obs_goal()
        obs["obj_2_pose"] = _pose((0.5, 0.5, 0)).unsqueeze(0).unsqueeze(0)
        obs["obj_2_role"] = CLUTTER.unsqueeze(0).unsqueeze(0)
        obs["obj_3_pose"] = _pose((-0.5, 0.5, 0)).unsqueeze(0).unsqueeze(0)
        obs["obj_3_role"] = CLUTTER.unsqueeze(0).unsqueeze(0)
        goal["obj_2_pose"] = _pose((0.5, 0.5, 0)).unsqueeze(0)
        goal["obj_3_pose"] = _pose((-0.5, 0.5, 0)).unsqueeze(0)

        tokens = tokenizer.tokenize(obs, goal)
        assert tokens.shape == (1, 1, 5, 10)  # tcp, obj_0, obj_1, obj_2, obj_3

        assert torch.allclose(tokens[0, 0, 3, 6:10], CLUTTER)
        assert torch.allclose(tokens[0, 0, 4, 6:10], CLUTTER)

    def test_single_side_tokenization_absolute_mode(self):
        tokenizer = ObjectTokenizer(relative_goal=False)
        obs, _ = self._obs_goal()

        tokens = tokenizer.tokenize(obs, None)
        assert tokens.shape == (1, 1, 3, 11)
        # Check raw pose in token[0:7] matches the source pose
        assert torch.equal(tokens[0, 0, 0, 0:7], obs["tcp_pose"][0, 0])
        assert torch.equal(tokens[0, 0, 1, 0:7], obs["obj_0_pose"][0, 0])
        assert torch.equal(tokens[0, 0, 2, 0:7], obs["obj_1_pose"][0, 0])

    def test_natural_obj_slots_with_role_tensors(self):
        """Tests that natural object slot order is preserved with role tensors."""
        tokenizer = ObjectTokenizer(relative_goal=True)
        obs = {
            "tcp_pose": _pose((0, 1, 0)).unsqueeze(0).unsqueeze(0),
            "tcp_role": TCP.unsqueeze(0).unsqueeze(0),
            "obj_0_pose": _pose((0, 0, 0)).unsqueeze(0).unsqueeze(0),
            "obj_0_role": CLUTTER.unsqueeze(0).unsqueeze(0),
            "obj_1_pose": _pose((1, 0, 0)).unsqueeze(0).unsqueeze(0),
            "obj_1_role": CLUTTER.unsqueeze(0).unsqueeze(0),
            "obj_2_pose": _pose((2, 0, 0)).unsqueeze(0).unsqueeze(0),
            "obj_2_role": PICK.unsqueeze(0).unsqueeze(0),  # pick in slot 2!
            "obj_3_pose": _pose((3, 0, 0)).unsqueeze(0).unsqueeze(0),
            "obj_3_role": TARGET.unsqueeze(0).unsqueeze(0),  # target in slot 3!
        }
        goal = {
            "tcp_pose": _pose((0, 1, 0)).unsqueeze(0),
            "obj_0_pose": _pose((0, 0, 0)).unsqueeze(0),
            "obj_1_pose": _pose((1, 0, 0)).unsqueeze(0),
            "obj_2_pose": _pose((3, 0, 0.05)).unsqueeze(0),  # goal: pick placed on target
            "obj_3_pose": _pose((3, 0, 0)).unsqueeze(0),
        }

        tokens = tokenizer.tokenize(obs, goal)
        assert tokens.shape == (1, 1, 5, 10)

        assert torch.allclose(tokens[0, 0, 0, 6:10], TCP)
        assert torch.allclose(tokens[0, 0, 1, 6:10], CLUTTER)
        assert torch.allclose(tokens[0, 0, 2, 6:10], CLUTTER)
        # Slot 2 (pick): goal delta to target
        assert torch.allclose(tokens[0, 0, 3, 6:10], PICK)
        assert torch.allclose(tokens[0, 0, 3, :3], torch.tensor([1.0, 0.0, 0.05]))
        # Slot 3 (target): goal delta 0
        assert torch.allclose(tokens[0, 0, 4, 6:10], TARGET)
        assert torch.allclose(tokens[0, 0, 4, :6], torch.zeros(6), atol=1e-6)

    def test_missing_tcp_pose_raises_keyerror(self):
        tokenizer = ObjectTokenizer()
        obs, goal = self._obs_goal()
        del obs["tcp_pose"]
        del goal["tcp_pose"]
        with pytest.raises(KeyError, match="tcp_pose"):
            tokenizer.tokenize(obs, goal)

    def test_missing_role_tensor_raises_keyerror(self):
        tokenizer = ObjectTokenizer()
        obs, goal = self._obs_goal()
        del obs["obj_0_role"]
        with pytest.raises(KeyError, match="obj_0_role"):
            tokenizer.tokenize(obs, goal)

    def test_missing_canonical_keys_raises_keyerror(self):
        tokenizer = ObjectTokenizer(relative_goal=False)
        with pytest.raises(KeyError, match="No canonical object pose keys"):
            tokenizer.tokenize({"random_key": torch.randn(1, 7)}, None)

    def test_missing_goal_key_raises_keyerror(self):
        tokenizer = ObjectTokenizer(relative_goal=True)
        obs, goal = self._obs_goal()
        del goal["obj_0_pose"]
        with pytest.raises(KeyError):
            tokenizer.tokenize(obs, goal)

    def test_single_side_support(self):
        assert ObjectTokenizer.supports_single_side is True
