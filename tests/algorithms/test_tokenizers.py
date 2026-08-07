import pytest
import torch
from mani_skill.utils.geometry.rotation_conversions import axis_angle_to_quaternion

from policy.algorithms.tokenizers import FlattenStateTokenizer, PerObjectStateTokenizer


def _pose(pos: tuple[float, float, float], axis_angle: tuple[float, float, float] = (0, 0, 0)):
    """Builds a single [1, 7] pose tensor (position + quaternion, real part first)."""
    quat = axis_angle_to_quaternion(torch.tensor([axis_angle], dtype=torch.float32))
    return torch.cat([torch.tensor([list(pos)], dtype=torch.float32), quat], dim=-1)


class TestFlattenStateTokenizer:
    def test_output_dim_and_tokens_per_step(self):
        tokenizer = FlattenStateTokenizer(task_dim=30)
        assert tokenizer.output_dim == 30
        assert tokenizer.tokens_per_step == 1

    def test_tokenize_computes_the_goal_minus_obs_delta_for_flat_tensors(self):
        tokenizer = FlattenStateTokenizer(task_dim=5)
        obs = torch.randn(2, 3, 5)
        goal = torch.randn(2, 5)

        delta = tokenizer.tokenize(obs, goal)

        assert delta.shape == (2, 3, 5)
        assert torch.equal(delta, goal.unsqueeze(1) - obs)

    def test_tokenize_flattens_dict_task_trees_before_differencing(self):
        tokenizer = FlattenStateTokenizer(task_dim=30)
        obs = {"task_a": torch.randn(2, 3, 10), "task_b": torch.randn(2, 3, 20)}
        goal = {"task_a": torch.randn(2, 10), "task_b": torch.randn(2, 20)}

        delta = tokenizer.tokenize(obs, goal)

        assert delta.shape == (2, 3, 30)
        expected_a = goal["task_a"].unsqueeze(1) - obs["task_a"]
        expected_b = goal["task_b"].unsqueeze(1) - obs["task_b"]
        assert torch.equal(delta[..., :10], expected_a)
        assert torch.equal(delta[..., 10:], expected_b)

    def test_tokenize_single_side_passes_through_unchanged(self):
        tokenizer = FlattenStateTokenizer(task_dim=5)
        obs = torch.randn(2, 3, 5)
        goal = torch.randn(2, 5)

        assert torch.equal(tokenizer.tokenize(obs, None), obs)
        assert torch.equal(tokenizer.tokenize(None, goal), goal)

    def test_tokenize_requires_at_least_one_side(self):
        tokenizer = FlattenStateTokenizer(task_dim=5)
        with pytest.raises(ValueError, match="at least one"):
            tokenizer.tokenize(None, None)


class TestPerObjectStateTokenizer:
    def test_output_dim_scales_with_optional_components(self):
        assert PerObjectStateTokenizer().output_dim == 6
        assert PerObjectStateTokenizer(include_tcp_relative=True).output_dim == 12
        assert PerObjectStateTokenizer(include_position_norm=True).output_dim == 7
        assert (
            PerObjectStateTokenizer(
                include_tcp_relative=True, include_position_norm=True
            ).output_dim
            == 13
        )

    def test_tokens_per_step_matches_object_keys_length(self):
        assert PerObjectStateTokenizer().tokens_per_step == 3
        assert PerObjectStateTokenizer(object_keys=("a_pose", "tcp_pose")).tokens_per_step == 2

    def test_tcp_key_must_be_one_of_object_keys(self):
        with pytest.raises(ValueError, match="tcp_key"):
            PerObjectStateTokenizer(object_keys=("a_pose", "b_pose"), tcp_key="tcp_pose")

    def _obs_goal(self):
        # obs at t=0: objA at origin (identity orientation), objB offset by (1,0,0), TCP offset by
        # (0,1,0). Goal: objA moved by (1,2,3) with a 90-degree rotation about z; objB and TCP
        # unchanged from their own obs poses (rotated poses only matter for objA's own token).
        obs = {
            "a_pose": _pose((0, 0, 0)),
            "b_pose": _pose((1, 0, 0)),
            "tcp_pose": _pose((0, 1, 0)),
        }
        goal = {
            "a_pose": _pose((1, 2, 3), axis_angle=(0, 0, torch.pi / 2)),
            "b_pose": _pose((1, 0, 0)),
            "tcp_pose": _pose((0, 1, 0)),
        }
        # Add a time axis of size 1 to obs (tokenize expects [B, T, 7] for obs).
        return {k: v.unsqueeze(1) for k, v in obs.items()}, goal

    def test_r_is_the_goal_minus_obs_delta_with_a_proper_rotation_operation(self):
        tokenizer = PerObjectStateTokenizer()
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

    def test_c_is_tcp_relative_and_exactly_zero_for_the_tcp_token(self):
        tokenizer = PerObjectStateTokenizer(include_tcp_relative=True)
        obs, goal = self._obs_goal()

        tokens = tokenizer.tokenize(obs, goal)
        assert tokens.shape == (1, 1, 3, 12)

        # c_TCP = TCP_pose - TCP_pose, exactly zero by construction -- no special-casing needed.
        c_tcp = tokens[0, 0, 2, 6:12]
        assert torch.equal(c_tcp, torch.zeros(6))

        # c_objB = TCP_pose - objB_pose = (0,1,0) - (1,0,0) = (-1,1,0), no rotation delta.
        c_b = tokens[0, 0, 1, 6:12]
        assert torch.allclose(c_b[:3], torch.tensor([-1.0, 1.0, 0.0]))
        assert torch.allclose(c_b[3:6], torch.zeros(3), atol=1e-6)

    def test_norm_is_the_position_only_norm_of_r_not_the_rotation_part(self):
        tokenizer = PerObjectStateTokenizer(include_position_norm=True)
        obs, goal = self._obs_goal()

        tokens = tokenizer.tokenize(obs, goal)
        assert tokens.shape == (1, 1, 3, 7)

        r_a = tokens[0, 0, 0, :6]
        norm_a = tokens[0, 0, 0, 6]
        assert torch.allclose(norm_a, r_a[:3].norm())
        assert not torch.allclose(norm_a, r_a.norm())  # would differ if rotation were included

    def test_tokenize_requires_both_sides(self):
        tokenizer = PerObjectStateTokenizer()
        obs, goal = self._obs_goal()
        with pytest.raises(ValueError, match="supports_single_side"):
            tokenizer.tokenize(obs, None)
        with pytest.raises(ValueError, match="supports_single_side"):
            tokenizer.tokenize(None, goal)

    def test_tokenize_requires_dict_shaped_task_trees(self):
        tokenizer = PerObjectStateTokenizer()
        with pytest.raises(TypeError):
            tokenizer.tokenize(torch.randn(1, 1, 30), torch.randn(1, 30))

    def test_tokenize_raises_on_missing_pose_keys(self):
        tokenizer = PerObjectStateTokenizer()
        obs, goal = self._obs_goal()
        del obs["b_pose"]
        with pytest.raises(KeyError):
            tokenizer.tokenize(obs, goal)

    def test_compatible_goal_deltas_and_single_side_support(self):
        assert PerObjectStateTokenizer.compatible_goal_deltas == frozenset({"input"})
        assert PerObjectStateTokenizer.supports_single_side is False
        assert FlattenStateTokenizer.compatible_goal_deltas == frozenset(
            {None, "input", "embedding"}
        )
        assert FlattenStateTokenizer.supports_single_side is True
