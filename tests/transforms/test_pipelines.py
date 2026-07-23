import torch

from policy.transforms.pipelines import observation_pipeline


class TestObservationPipeline:
    def test_noop_all_flags_false(self):
        """is_flat=True, all flags False → empty compose → identity."""
        pipeline = observation_pipeline(
            "StackCube-v1", is_flat=True, canonicalize=False, as_dict=False, no_proprio_vel=False
        )
        t = torch.randn(48)
        out = pipeline(t)
        assert out is t

    def test_flat_canonicalize_to_flat(self):
        """Flat tensor → deflatten → canonicalize → flatten → tensor."""
        pipeline = observation_pipeline(
            "StackCube-v1", is_flat=True, canonicalize=True, as_dict=False
        )
        out = pipeline(torch.randn(48))
        assert isinstance(out, torch.Tensor)
        # Canonical dim: proprio(18)+tcp(7)+a(7)+b(7)+a_to_b(3)+tcp_to_a(3)+tcp_to_b(3) = 48
        assert out.shape[-1] == 48

    def test_flat_no_proprio_vel_to_flat(self):
        """Flat tensor → deflatten → remove qvel → flatten → tensor (39 dims)."""
        pipeline = observation_pipeline(
            "StackCube-v1",
            is_flat=True,
            canonicalize=False,
            as_dict=False,
            no_proprio_vel=True,
        )
        out = pipeline(torch.randn(48))
        assert isinstance(out, torch.Tensor)
        # 48 - 9 (qvel) = 39
        assert out.shape[-1] == 39

    def test_flat_as_dict(self):
        """Flat tensor → deflatten → dict output."""
        pipeline = observation_pipeline(
            "StackCube-v1", is_flat=True, canonicalize=False, as_dict=True
        )
        out = pipeline(torch.randn(48))
        assert isinstance(out, dict)

    def test_dict_input_flattened(self):
        """Dict input with as_dict=False → DictFlattener → tensor output."""
        pipeline = observation_pipeline(
            "StackCube-v1", is_flat=False, canonicalize=False, as_dict=False
        )
        obs = {"a": torch.randn(3), "b": torch.randn(4)}
        out = pipeline(obs)
        assert isinstance(out, torch.Tensor)
        assert out.shape[-1] == 7
