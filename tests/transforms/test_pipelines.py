import torch

from policy.transforms.pipelines import observation_pipeline


class TestObservationPipeline:
    def test_canonicalize_false_identity(self):
        """Canonicalize=False → empty compose → identity."""
        pipeline = observation_pipeline("StackCube-v1", canonicalize=False)
        obs = {"agent": {"qpos": torch.randn(9)}, "extra": {"tcp_pose": torch.randn(7)}}
        out = pipeline(obs)
        assert out is obs

    def test_canonicalize_true_applies_canonicalizer(self):
        """Canonicalize=True → applies Canonicalizer(env_id)."""
        pipeline = observation_pipeline("StackCube-v1", canonicalize=True)
        obs = {
            "agent": {
                "qpos": torch.randn(9),
                "qvel": torch.randn(9),
            },
            "extra": {
                "tcp_pose": torch.randn(7),
                "cubeA_pose": torch.randn(7),
                "cubeB_pose": torch.randn(7),
            },
        }
        out = pipeline(obs)
        assert isinstance(out, dict)
        assert set(out.keys()) == {
            "proprio",
            "tcp_pose",
            "obj_0_pose",
            "obj_0_role",
            "obj_1_pose",
            "obj_1_role",
        }
        assert out["proprio"].shape[-1] == 18
        assert out["tcp_pose"].shape[-1] == 7
        assert out["obj_0_pose"].shape[-1] == 7
        assert out["obj_1_pose"].shape[-1] == 7
        assert torch.equal(out["obj_0_role"], torch.tensor([1.0, 0.0, 0.0]))
        assert torch.equal(out["obj_1_role"], torch.tensor([0.0, 1.0, 0.0]))
