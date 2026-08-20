import pytest
import torch
from mani_skill.utils.geometry.rotation_conversions import axis_angle_to_quaternion

from policy.algorithms.tokenizers.object import ObjectTokenizer

ROLE_TCP = [1.0, 0.0, 0.0, 0.0]
ROLE_PICK = [0.0, 1.0, 0.0, 0.0]
ROLE_TARGET = [0.0, 0.0, 1.0, 0.0]
ROLE_CLUTTER = [0.0, 0.0, 0.0, 1.0]


def _pose(pos: tuple[float, float, float], axis_angle: tuple[float, float, float] = (0, 0, 0)):
    p = torch.tensor(pos, dtype=torch.float32)
    aa = torch.tensor(axis_angle, dtype=torch.float32)
    q = axis_angle_to_quaternion(aa)
    return torch.cat([p, q], dim=-1)


def _task_dim(num_slots: int) -> dict[str, tuple[int, ...]]:
    return {
        "obj_pose": (num_slots, 7),
        "obj_role": (num_slots, 4),
        "obj_valid": (num_slots,),
    }


def _pool(poses: list[torch.Tensor], roles: list[list[float]], time_axis: bool):
    """Stacks a pool into canonical [B, (T,) K, F] tensors."""
    pose = torch.stack(poses, dim=0).unsqueeze(0)
    role = torch.tensor(roles).unsqueeze(0)
    valid = torch.ones(1, len(poses))
    if time_axis:
        pose, role, valid = pose.unsqueeze(1), role.unsqueeze(1), valid.unsqueeze(1)
    return {"obj_pose": pose, "obj_role": role, "obj_valid": valid}


class TestObjectTokenizer:
    def test_output_dim_relative_and_absolute_modes(self):
        assert ObjectTokenizer(_task_dim(3), relative_goal=True).output_dim == 12
        assert ObjectTokenizer(_task_dim(3), relative_goal=False).output_dim == 13

    def test_tokens_per_step_matches_num_slots(self):
        assert ObjectTokenizer(_task_dim(3)).tokens_per_step == 3
        assert ObjectTokenizer(_task_dim(9)).tokens_per_step == 9

    def _obs_goal(self):
        # Slot 0 is the TCP, slot 1 the pick object, slot 2 the target.
        obs = _pool(
            [_pose((0, 1, 0)), _pose((0, 0, 0)), _pose((1, 0, 0))],
            [ROLE_TCP, ROLE_PICK, ROLE_TARGET],
            time_axis=True,
        )
        goal = _pool(
            [
                _pose((0, 1, 0)),
                _pose((1, 2, 3), axis_angle=(0, 0, torch.pi / 2)),
                _pose((1, 0, 0)),
            ],
            [ROLE_TCP, ROLE_PICK, ROLE_TARGET],
            time_axis=False,
        )
        return obs, goal

    def test_enriched_token_structure(self):
        tokenizer = ObjectTokenizer(_task_dim(3), relative_goal=True)
        obs, goal = self._obs_goal()

        out = tokenizer.tokenize(obs, goal)
        assert set(out) == {"tokens", "role"}
        assert out["tokens"].shape == (1, 1, 3, 12)
        assert out["role"].shape == (1, 1, 3, 4)

        # Token 0 is the TCP itself, so its pose relative to the TCP is identically zero.
        assert torch.allclose(out["tokens"][0, 0, 0, :6], torch.zeros(6), atol=1e-6)
        assert torch.allclose(out["role"][0, 0, 0], torch.tensor(ROLE_TCP))

        # Token 1 (pick):
        # 1. rel_to_tcp: (0,0,0) - tcp (0,1,0) = (0, -1, 0)
        tok_1 = out["tokens"][0, 0, 1]
        assert torch.allclose(tok_1[:3], torch.tensor([0.0, -1.0, 0.0]))
        # 2. goal_delta: goal (1,2,3) - obs (0,0,0) = (1, 2, 3), rotvec pi/2 about z
        assert torch.allclose(tok_1[6:9], torch.tensor([1.0, 2.0, 3.0]))
        assert torch.allclose(tok_1[9:12].norm(), torch.tensor(torch.pi / 2), atol=1e-5)
        assert torch.allclose(out["role"][0, 0, 1], torch.tensor(ROLE_PICK))

        # Token 2 (target): rel_to_tcp (1,0,0) - (0,1,0) = (1,-1,0), goal == obs so delta is 0
        tok_2 = out["tokens"][0, 0, 2]
        assert torch.allclose(tok_2[:3], torch.tensor([1.0, -1.0, 0.0]))
        assert torch.allclose(tok_2[6:12], torch.zeros(6), atol=1e-6)
        assert torch.allclose(out["role"][0, 0, 2], torch.tensor(ROLE_TARGET))

    def test_clutter_slots_extend_the_token_sequence(self):
        tokenizer = ObjectTokenizer(_task_dim(5), relative_goal=True)
        poses = [
            _pose((0, 1, 0)),
            _pose((0, 0, 0)),
            _pose((1, 0, 0)),
            _pose((0.5, 0.5, 0)),
            _pose((-0.5, 0.5, 0)),
        ]
        roles = [ROLE_TCP, ROLE_PICK, ROLE_TARGET, ROLE_CLUTTER, ROLE_CLUTTER]
        obs = _pool(poses, roles, time_axis=True)
        goal = _pool(poses, roles, time_axis=False)

        out = tokenizer.tokenize(obs, goal)
        assert out["tokens"].shape == (1, 1, 5, 12)
        assert out["role"].shape == (1, 1, 5, 4)
        assert torch.allclose(out["role"][0, 0, 3], torch.tensor(ROLE_CLUTTER))
        assert torch.allclose(out["role"][0, 0, 4], torch.tensor(ROLE_CLUTTER))

    def test_single_side_tokenization_absolute_mode(self):
        tokenizer = ObjectTokenizer(_task_dim(3), relative_goal=False)
        obs, _ = self._obs_goal()

        out = tokenizer.tokenize(obs, None)
        assert out["tokens"].shape == (1, 1, 3, 13)
        # The raw pose occupies channels 6:13 in absolute mode.
        assert torch.equal(out["tokens"][0, 0, 1, 6:13], obs["obj_pose"][0, 0, 1])
        assert torch.equal(out["tokens"][0, 0, 2, 6:13], obs["obj_pose"][0, 0, 2])

    def test_roles_are_read_per_slot_not_inferred_from_position(self):
        """A pick/target sitting in an arbitrary slot keeps its role (the random-pick case)."""
        tokenizer = ObjectTokenizer(_task_dim(5), relative_goal=True)
        obs = _pool(
            [
                _pose((0, 1, 0)),
                _pose((0, 0, 0)),
                _pose((1, 0, 0)),
                _pose((2, 0, 0)),
                _pose((3, 0, 0)),
            ],
            [ROLE_TCP, ROLE_CLUTTER, ROLE_CLUTTER, ROLE_PICK, ROLE_TARGET],
            time_axis=True,
        )
        goal = _pool(
            [
                _pose((0, 1, 0)),
                _pose((0, 0, 0)),
                _pose((1, 0, 0)),
                _pose((3, 0, 0.05)),  # pick placed on top of the target
                _pose((3, 0, 0)),
            ],
            [ROLE_TCP, ROLE_CLUTTER, ROLE_CLUTTER, ROLE_PICK, ROLE_TARGET],
            time_axis=False,
        )

        out = tokenizer.tokenize(obs, goal)
        assert out["tokens"].shape == (1, 1, 5, 12)

        assert torch.allclose(out["role"][0, 0, 3], torch.tensor(ROLE_PICK))
        assert torch.allclose(out["tokens"][0, 0, 3, 6:9], torch.tensor([1.0, 0.0, 0.05]))
        assert torch.allclose(out["role"][0, 0, 4], torch.tensor(ROLE_TARGET))
        assert torch.allclose(out["tokens"][0, 0, 4, 6:12], torch.zeros(6), atol=1e-6)

    def test_missing_pool_key_raises_keyerror(self):
        tokenizer = ObjectTokenizer(_task_dim(3))
        obs, goal = self._obs_goal()
        del obs["obj_role"]
        with pytest.raises(KeyError, match="obj_role"):
            tokenizer.tokenize(obs, goal)

    def test_missing_goal_key_raises_keyerror(self):
        tokenizer = ObjectTokenizer(_task_dim(3), relative_goal=True)
        obs, goal = self._obs_goal()
        del goal["obj_pose"]
        with pytest.raises(KeyError):
            tokenizer.tokenize(obs, goal)

    def test_non_canonical_task_dim_raises_typeerror(self):
        with pytest.raises(TypeError, match="canonical dict task_dim"):
            ObjectTokenizer(task_dim=42)

    def test_single_side_support(self):
        assert ObjectTokenizer.supports_single_side is True

    def test_token_spec_and_normalization_mask_mirror_each_other(self):
        tokenizer = ObjectTokenizer(_task_dim(3), relative_goal=True)
        spec, mask = tokenizer.token_spec, tokenizer.normalization_mask

        assert spec == {"tokens": 12, "role": 4}
        assert set(spec) == set(mask)
        for key, width in spec.items():
            assert mask[key].shape == (width,)
        assert mask["tokens"].all()
        assert not mask["role"].any()
