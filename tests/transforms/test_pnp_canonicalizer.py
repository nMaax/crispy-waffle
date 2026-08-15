import pytest
import torch

from policy.transforms.canonicalization.canonicalizer import Canonicalizer


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


EXPECTED_KEYS = {
    "proprio",
    "tcp_pose",
    "obj_0_pose",
    "obj_0_role",
    "obj_1_pose",
    "obj_1_role",
}


class TestCanonicalizerDimSpec:
    def test_dim_spec_default_covers_two_objects(self):
        spec = Canonicalizer.dim_spec(2)
        assert spec["proprio"] == 18
        assert spec["tcp_pose"] == 7
        assert spec["obj_0_pose"] == 7
        assert spec["obj_0_role"] == 3
        assert spec["obj_1_pose"] == 7
        assert spec["obj_1_role"] == 3
        assert "obj_2_pose" not in spec

    def test_dim_spec_generalizes_to_object_count(self):
        spec = Canonicalizer.dim_spec(8)
        assert spec["proprio"] == 18
        assert spec["tcp_pose"] == 7
        for i in range(8):
            assert spec[f"obj_{i}_pose"] == 7
            assert spec[f"obj_{i}_role"] == 3
        assert "obj_8_pose" not in spec


class TestCanonicalizer:

    def test_call_non_mapping_raises(self):
        canon = Canonicalizer("StackCube-v1")
        with pytest.raises(TypeError, match="expects a mapping"):
            canon(torch.randn(48))

    def test_unsupported_env_raises_keyerror(self):
        canon = Canonicalizer("UnknownEnv-v0")
        with pytest.raises(KeyError):
            canon({"agent": {}})

    def test_parse_stack_cube(self):
        canon = Canonicalizer("StackCube-v1")
        out = canon(_stack_cube_obs())
        assert set(out.keys()) == EXPECTED_KEYS
        assert out["proprio"].shape[-1] == 18  # qpos(9) + qvel(9)
        assert out["tcp_pose"].shape[-1] == 7
        assert out["obj_0_pose"].shape[-1] == 7
        assert out["obj_1_pose"].shape[-1] == 7
        assert torch.equal(out["obj_0_role"], torch.tensor([1.0, 0.0, 0.0]))
        assert torch.equal(out["obj_1_role"], torch.tensor([0.0, 1.0, 0.0]))

    @pytest.mark.parametrize(
        "env_id",
        [
            "StackCubeRestrictedSpawn-v1",
            "StackCubeLockedRotation-v1",
            "PlaceCubeLeft-v1",
            "PlaceCubeLeftLockedRotation-v1",
            "PlaceCubeRight-v1",
            "PlaceCubeRightLockedRotation-v1",
        ],
    )
    def test_stack_cube_delegates(self, env_id):
        obs = _stack_cube_obs()
        out_base = Canonicalizer("StackCube-v1")(obs)
        out_delegate = Canonicalizer(env_id)(obs)
        for key in EXPECTED_KEYS:
            assert torch.allclose(out_base[key], out_delegate[key])

    @pytest.mark.parametrize(
        "env_id", ["StackCubeSwapped-v1", "StackCubeSwappedLockedRotation-v1"]
    )
    def test_stack_cube_swapped_reverses_pick_and_target(self, env_id):
        # StackCubeSwapped picks cubeB and targets cubeA -- the opposite of every other
        # stack_cube-family task, so its roles (not just its poses) must be swapped too.
        obs = _stack_cube_obs()
        out_base = Canonicalizer("StackCube-v1")(obs)
        out_swapped = Canonicalizer(env_id)(obs)

        assert torch.equal(out_swapped["obj_0_pose"], out_base["obj_1_pose"])
        assert torch.equal(out_swapped["obj_1_pose"], out_base["obj_0_pose"])
        assert torch.equal(out_swapped["obj_0_role"], torch.tensor([1.0, 0.0, 0.0]))
        assert torch.equal(out_swapped["obj_1_role"], torch.tensor([0.0, 1.0, 0.0]))

    def test_batched_input(self):
        canon = Canonicalizer("StackCube-v1")
        out = canon(_stack_cube_obs(batch=True))
        for key in EXPECTED_KEYS:
            assert out[key].shape[0] == 2

    def test_parse_stack_cube_clutter(self):
        obs = _stack_cube_obs()
        obs["extra"]["obj_0_pose"] = torch.randn(7)
        obs["extra"]["obj_0_active"] = torch.tensor(True)
        obs["extra"]["obj_1_pose"] = torch.randn(7)
        obs["extra"]["obj_1_active"] = torch.tensor(False)
        obs["extra"]["obj_2_pose"] = torch.randn(7)
        obs["extra"]["obj_2_active"] = torch.tensor(True)

        canon = Canonicalizer("StackCubeClutter-v1")
        out = canon(obs)
        assert EXPECTED_KEYS.issubset(out.keys())
        assert "obj_0_pose" in out
        assert "obj_1_pose" in out
        assert "obj_2_pose" in out
        assert "obj_3_pose" in out
        assert "obj_4_pose" not in out
        assert torch.equal(out["obj_2_pose"], obs["extra"]["obj_0_pose"])
        assert torch.equal(out["obj_3_pose"], obs["extra"]["obj_2_pose"])
        assert torch.equal(out["obj_0_role"], torch.tensor([1.0, 0.0, 0.0]))
        assert torch.equal(out["obj_1_role"], torch.tensor([0.0, 1.0, 0.0]))
        assert torch.equal(out["obj_2_role"], torch.tensor([0.0, 0.0, 1.0]))
        assert torch.equal(out["obj_3_role"], torch.tensor([0.0, 0.0, 1.0]))

    def test_parse_stack_cube_clutter_random_pick(self):
        obs = {
            "agent": {
                "qpos": torch.randn(9),
                "qvel": torch.randn(9),
            },
            "extra": {
                "tcp_pose": torch.randn(7),
            },
        }
        for i in range(8):
            obs["extra"][f"obj_{i}_pose"] = torch.randn(7)
            obs["extra"][f"obj_{i}_is_pick"] = torch.tensor(i == 3)
            obs["extra"][f"obj_{i}_is_target"] = torch.tensor(i == 6)
            obs["extra"][f"obj_{i}_active"] = torch.tensor(i != 7)  # slot 7 inactive

        canon = Canonicalizer("StackCubeClutterRandomPick-v1")
        out = canon(obs)
        assert "proprio" in out
        assert "tcp_pose" in out
        # 7 active objects: obj_0 .. obj_6
        for i in range(7):
            assert f"obj_{i}_pose" in out
            assert f"obj_{i}_role" in out
            assert torch.equal(out[f"obj_{i}_pose"], obs["extra"][f"obj_{i}_pose"])
        assert "obj_7_pose" not in out

        # Check role vectors: obj_3 is pick [1,0,0], obj_6 is target [0,1,0], others are [0,0,1]
        assert torch.equal(out["obj_3_role"], torch.tensor([1.0, 0.0, 0.0]))
        assert torch.equal(out["obj_6_role"], torch.tensor([0.0, 1.0, 0.0]))
        assert torch.equal(out["obj_0_role"], torch.tensor([0.0, 0.0, 1.0]))
        assert torch.equal(out["obj_1_role"], torch.tensor([0.0, 0.0, 1.0]))
