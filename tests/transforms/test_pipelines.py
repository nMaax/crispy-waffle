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
        assert set(out.keys()) == {"proprio", "obj_pose", "obj_role", "obj_valid"}
        assert out["proprio"].shape[-1] == 18
        # tcp + cubeA + cubeB
        assert out["obj_pose"].shape == (3, 7)
        assert out["obj_role"].shape == (3, 4)
        assert out["obj_valid"].shape == (3,)
        assert torch.equal(out["obj_role"][0], torch.tensor([1.0, 0.0, 0.0, 0.0]))  # tcp
        assert torch.equal(out["obj_role"][1], torch.tensor([0.0, 1.0, 0.0, 0.0]))  # pick
        assert torch.equal(out["obj_role"][2], torch.tensor([0.0, 0.0, 1.0, 0.0]))  # target
