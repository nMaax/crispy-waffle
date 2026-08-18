import pytest
import torch

from policy.transforms.canonicalization.canonicalizer import Canonicalizer

ROLE_TCP = torch.tensor([1.0, 0.0, 0.0, 0.0])
ROLE_PICK = torch.tensor([0.0, 1.0, 0.0, 0.0])
ROLE_TARGET = torch.tensor([0.0, 0.0, 1.0, 0.0])
ROLE_CLUTTER = torch.tensor([0.0, 0.0, 0.0, 1.0])

EXPECTED_KEYS = {"proprio", "obj_pose", "obj_role", "obj_valid"}


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


class TestCanonicalizerDimSpec:
    def test_dim_spec_default_covers_two_objects_plus_the_tcp(self):
        spec = Canonicalizer.dim_spec(2)
        assert spec["proprio"] == 18
        assert spec["obj_pose"] == (3, 7)
        assert spec["obj_role"] == (3, 4)
        assert spec["obj_valid"] == (3,)

    def test_dim_spec_generalizes_to_object_count(self):
        spec = Canonicalizer.dim_spec(8)
        assert spec["proprio"] == 18
        assert spec["obj_pose"] == (9, 7)
        assert spec["obj_role"] == (9, 4)
        assert spec["obj_valid"] == (9,)


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
        obs = _stack_cube_obs()
        out = Canonicalizer("StackCube-v1")(obs)

        assert set(out.keys()) == EXPECTED_KEYS
        assert out["proprio"].shape[-1] == 18  # qpos(9) + qvel(9)
        # Slot 0 is the TCP, then pick, then target.
        assert out["obj_pose"].shape == (3, 7)
        assert torch.equal(out["obj_pose"][0], obs["extra"]["tcp_pose"])
        assert torch.equal(out["obj_pose"][1], obs["extra"]["cubeA_pose"])
        assert torch.equal(out["obj_pose"][2], obs["extra"]["cubeB_pose"])
        assert torch.equal(out["obj_role"][0], ROLE_TCP)
        assert torch.equal(out["obj_role"][1], ROLE_PICK)
        assert torch.equal(out["obj_role"][2], ROLE_TARGET)
        assert torch.equal(out["obj_valid"], torch.ones(3))

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

        assert torch.equal(out_swapped["obj_pose"][1], out_base["obj_pose"][2])
        assert torch.equal(out_swapped["obj_pose"][2], out_base["obj_pose"][1])
        assert torch.equal(out_swapped["obj_role"][1], ROLE_PICK)
        assert torch.equal(out_swapped["obj_role"][2], ROLE_TARGET)

    def test_batched_input(self):
        out = Canonicalizer("StackCube-v1")(_stack_cube_obs(batch=True))
        for key in EXPECTED_KEYS:
            assert out[key].shape[0] == 2

    def test_parse_stack_cube_clutter_keeps_every_slot(self):
        obs = _stack_cube_obs()
        for i, active in enumerate([True, False, True]):
            obs["extra"][f"obj_{i}_pose"] = torch.randn(7)
            obs["extra"][f"obj_{i}_active"] = torch.tensor(active)

        out = Canonicalizer("StackCubeClutter-v1")(obs)

        # tcp + pick + target + 3 clutter slots: inactive slots are kept, not dropped.
        assert out["obj_pose"].shape == (6, 7)
        assert torch.equal(out["obj_valid"], torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 1.0]))
        # Slot index still names the same physical object, so no renumbering happened.
        for i in range(3):
            assert torch.equal(out["obj_pose"][3 + i], obs["extra"][f"obj_{i}_pose"])
            assert torch.equal(out["obj_role"][3 + i], ROLE_CLUTTER)

    def test_clutter_activity_is_per_sample_not_collapsed_across_the_batch(self):
        # A batched tree carries one activity flag per env; folding them together (e.g. with
        # torch.any) would leak a parked, off-table object into the envs where it is inactive.
        obs = _stack_cube_obs(batch=True)
        active = torch.tensor([[True, False], [False, True]])
        for i in range(2):
            obs["extra"][f"obj_{i}_pose"] = torch.randn(2, 7)
            obs["extra"][f"obj_{i}_active"] = active[:, i]

        out = Canonicalizer("StackCubeClutter-v1")(obs)

        assert out["obj_valid"].shape == (2, 5)
        assert torch.equal(out["obj_valid"][:, 3:], active.float())

    def test_parse_stack_cube_clutter_random_pick(self):
        obs = {
            "agent": {"qpos": torch.randn(9), "qvel": torch.randn(9)},
            "extra": {"tcp_pose": torch.randn(7)},
        }
        for i in range(8):
            obs["extra"][f"obj_{i}_pose"] = torch.randn(7)
            obs["extra"][f"obj_{i}_is_pick"] = torch.tensor(i == 3)
            obs["extra"][f"obj_{i}_is_target"] = torch.tensor(i == 6)
            obs["extra"][f"obj_{i}_active"] = torch.tensor(i != 7)  # slot 7 inactive

        out = Canonicalizer("StackCubeClutterRandomPick-v1")(obs)

        # tcp + the whole 8-slot pool; no fixed pick/target pair is prepended here.
        assert out["obj_pose"].shape == (9, 7)
        assert torch.equal(out["obj_role"][0], ROLE_TCP)
        for i in range(8):
            assert torch.equal(out["obj_pose"][1 + i], obs["extra"][f"obj_{i}_pose"])
        assert torch.equal(out["obj_role"][1 + 3], ROLE_PICK)
        assert torch.equal(out["obj_role"][1 + 6], ROLE_TARGET)
        assert torch.equal(out["obj_role"][1 + 0], ROLE_CLUTTER)
        assert torch.equal(out["obj_valid"], torch.tensor([1.0] * 8 + [0.0]))

    def test_random_pick_without_any_pool_slot_raises(self):
        obs = {
            "agent": {"qpos": torch.randn(9), "qvel": torch.randn(9)},
            "extra": {"tcp_pose": torch.randn(7)},
        }
        with pytest.raises(KeyError, match="No object pose keys"):
            Canonicalizer("StackCubeClutterRandomPick-v1")(obs)
