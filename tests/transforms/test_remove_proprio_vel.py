import torch

from policy.transforms.canonicalization.remove_proprio_vel import RemoveProprioVel


class TestRemoveProprioVel:
    # ------------------------------------------------------------------ #
    # Tensor branch
    # ------------------------------------------------------------------ #
    def test_tensor_remove_slice(self):
        rp = RemoveProprioVel(qpos_dim=9, qvel_dim=9, fill_with_zeroes=False)
        state = torch.randn(2, 30)
        out = rp(state)
        assert out.shape == (2, 21)  # 30 - 9 (qvel removed)
        # Prefix and suffix preserved
        assert torch.allclose(out[..., :9], state[..., :9])
        assert torch.allclose(out[..., 9:], state[..., 18:])

    def test_tensor_zero_slice(self):
        rp = RemoveProprioVel(qpos_dim=9, qvel_dim=9, fill_with_zeroes=True)
        state = torch.randn(2, 30)
        out = rp(state)
        assert out.shape == (2, 30)  # dim unchanged
        assert torch.allclose(out[..., :9], state[..., :9])  # qpos preserved
        assert torch.all(out[..., 9:18] == 0.0)  # qvel zeroed
        assert torch.allclose(out[..., 18:], state[..., 18:])  # suffix preserved

    # ------------------------------------------------------------------ #
    # Dict branch: agent.qvel
    # ------------------------------------------------------------------ #
    def test_dict_agent_qvel_remove(self):
        rp = RemoveProprioVel(fill_with_zeroes=False)
        obs = {"agent": {"qpos": torch.randn(9), "qvel": torch.randn(9)}, "extra": torch.randn(3)}
        out = rp(obs)
        assert "qvel" not in out["agent"]
        assert "qpos" in out["agent"]

    def test_dict_agent_qvel_zeroed(self):
        rp = RemoveProprioVel(fill_with_zeroes=True)
        obs = {"agent": {"qpos": torch.randn(9), "qvel": torch.randn(9)}, "extra": torch.randn(3)}
        out = rp(obs)
        assert "qvel" in out["agent"]
        assert torch.all(out["agent"]["qvel"] == 0.0)

    # ------------------------------------------------------------------ #
    # Dict branch: proprio
    # ------------------------------------------------------------------ #
    def test_dict_proprio(self):
        rp = RemoveProprioVel(fill_with_zeroes=False)
        proprio = torch.randn(30)
        obs = {"proprio": proprio}
        out = rp(obs)
        assert out["proprio"].shape[-1] == 21  # qvel removed

    def test_dict_state(self):
        rp = RemoveProprioVel(fill_with_zeroes=True)
        state = torch.randn(30)
        obs = {"state": state}
        out = rp(obs)
        assert out["state"].shape[-1] == 30  # dim unchanged
        assert torch.all(out["state"][..., 9:18] == 0.0)

    # ------------------------------------------------------------------ #
    # Passthrough
    # ------------------------------------------------------------------ #
    def test_dict_passthrough(self):
        rp = RemoveProprioVel()
        obs = {"foo": torch.randn(5)}
        out = rp(obs)
        assert torch.allclose(out["foo"], obs["foo"])

    # ------------------------------------------------------------------ #
    # Custom dims
    # ------------------------------------------------------------------ #
    def test_custom_dims(self):
        rp = RemoveProprioVel(qpos_dim=7, qvel_dim=7, fill_with_zeroes=False)
        state = torch.randn(21)
        out = rp(state)
        assert out.shape[-1] == 14  # 21 - 7
        assert torch.allclose(out[..., :7], state[..., :7])
        assert torch.allclose(out[..., 7:], state[..., 14:])
