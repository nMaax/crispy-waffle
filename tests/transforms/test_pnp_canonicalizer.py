import pytest
import torch

from policy.transforms.canonicalization.pnp_canonicalizer import PnPCanonicalizer


def _shape(d, batch=False):
    return (2, d) if batch else (d,)


def _stack_cube_obs(batch=False):
    tcp_pose = torch.randn(*_shape(7, batch))
    cube_a_pose = torch.randn(*_shape(7, batch))
    cube_b_pose = torch.randn(*_shape(7, batch))

    return {
        "agent": {
            "qpos": torch.randn(*_shape(9, batch)),
            "qvel": torch.randn(*_shape(9, batch)),
        },
        "extra": {
            "tcp_pose": tcp_pose,
            "cubeA_pose": cube_a_pose,
            "cubeB_pose": cube_b_pose,
            "tcp_to_cubeA_pos": cube_a_pose[..., :3] - tcp_pose[..., :3],
            "tcp_to_cubeB_pos": cube_b_pose[..., :3] - tcp_pose[..., :3],
            "cubeA_to_cubeB_pos": cube_b_pose[..., :3] - cube_a_pose[..., :3],
        },
    }


def _place_sphere_obs(batch=False):
    tcp_pose = torch.randn(*_shape(7, batch))
    obj_pose = torch.randn(*_shape(7, batch))
    bin_pos = torch.randn(*_shape(3, batch))

    return {
        "agent": {
            "qpos": torch.randn(*_shape(9, batch)),
            "qvel": torch.randn(*_shape(9, batch)),
        },
        "extra": {
            "is_grasped": torch.zeros(*_shape(1, batch)),
            "tcp_pose": tcp_pose,
            "bin_pos": bin_pos,
            "obj_pose": obj_pose,
            "tcp_to_obj_pos": obj_pose[..., :3] - tcp_pose[..., :3],
        },
    }


EXPECTED_KEYS = {"proprio", "tcp_pose", "a_pose", "b_pose", "a_to_b", "tcp_to_a", "tcp_to_b"}


class TestPnPCanonicalizer:
    def test_call_non_mapping_raises(self):
        canon = PnPCanonicalizer("StackCube-v1")
        with pytest.raises(TypeError, match="expects a mapping"):
            canon(torch.randn(48))

    def test_unsupported_env_raises_keyerror(self):
        canon = PnPCanonicalizer("UnknownEnv-v0")
        with pytest.raises(KeyError):
            canon({"agent": {}})

    def test_parse_stack_cube(self):
        canon = PnPCanonicalizer("StackCube-v1")
        out = canon(_stack_cube_obs())
        assert set(out.keys()) == EXPECTED_KEYS
        assert out["proprio"].shape[-1] == 18  # qpos(9) + qvel(9)
        assert out["tcp_pose"].shape[-1] == 7
        assert out["a_pose"].shape[-1] == 7
        assert out["b_pose"].shape[-1] == 7
        assert out["a_to_b"].shape[-1] == 3
        assert out["tcp_to_a"].shape[-1] == 3
        assert out["tcp_to_b"].shape[-1] == 3

    @pytest.mark.parametrize(
        "env_id",
        [
            "StackCubeSwapped-v1",
            "StackCubeRestrictedSpawn-v1",
            "StackCubeLockedRotation-v1",
            "PlaceCubeLeft-v1",
        ],
    )
    def test_stack_cube_delegates(self, env_id):
        obs = _stack_cube_obs()
        out_base = PnPCanonicalizer("StackCube-v1")(obs)
        out_delegate = PnPCanonicalizer(env_id)(obs)
        for key in EXPECTED_KEYS:
            assert torch.allclose(out_base[key], out_delegate[key])

    def test_parse_place_sphere(self):
        canon = PnPCanonicalizer("PlaceSphere-v1")
        out = canon(_place_sphere_obs())
        assert set(out.keys()) == EXPECTED_KEYS
        # b_pose should have a fake quaternion [1,0,0,0] appended to bin_pos
        assert out["b_pose"].shape[-1] == 7
        assert torch.all(out["b_pose"][..., 3:] == torch.tensor([1.0, 0.0, 0.0, 0.0]))
        # a_pose is the sphere pose
        assert out["a_pose"].shape[-1] == 7

    def test_relative_positions_correct(self):
        canon = PnPCanonicalizer("StackCube-v1")
        obs = _stack_cube_obs()
        out = canon(obs)
        tcp = obs["extra"]["tcp_pose"][..., :3]
        a = obs["extra"]["cubeA_pose"][..., :3]
        b = obs["extra"]["cubeB_pose"][..., :3]
        assert torch.allclose(out["a_to_b"], b - a)
        assert torch.allclose(out["tcp_to_a"], a - tcp)
        assert torch.allclose(out["tcp_to_b"], b - tcp)

    def test_batched_input(self):
        canon = PnPCanonicalizer("StackCube-v1")
        out = canon(_stack_cube_obs(batch=True))
        for key in EXPECTED_KEYS:
            assert out[key].shape[0] == 2
