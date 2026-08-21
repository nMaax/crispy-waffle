import pytest
import torch

from policy.algorithms.networks.encoder.embedders.graph_transformer import (
    EDGE_GOAL,
    EDGE_NONE,
    EDGE_SPATIAL,
    EDGE_TEMPORAL,
    GraphTransformer,
)

B, T, G, K, IN, OUT = 2, 2, 1, 4, 10, 8
SEQ = (T + G) * K


def _embedder(**overrides):
    kwargs = dict(
        input_dim=IN, output_dim=OUT, obs_horizon=T, goal_horizon=G, num_heads=2, num_layers=2
    )
    kwargs.update(overrides)
    return GraphTransformer(**kwargs)


def _role():
    """One-hot role per slot, cycling through the 4 role classes."""
    role = torch.zeros(B, T + G, K, 4)
    for k in range(K):
        role[:, :, k, k % 4] = 1.0
    return role


def _task(valid=None):
    return {
        "nodes": torch.randn(B, T + G, K, IN),
        "role": _role(),
        "valid": torch.ones(B, T + G, K) if valid is None else valid,
        "edge_feat": torch.randn(B, SEQ, SEQ, 6),
    }


class TestEdgeTopology:
    """The three relation kinds, and their directionality."""

    @pytest.fixture
    def kinds(self):
        return _embedder()._edge_kinds(T + G, K, torch.device("cpu"))

    def test_every_pair_in_a_timestep_is_connected_both_ways(self, kinds):
        for step in range(T + G):
            block = kinds[step * K : (step + 1) * K, step * K : (step + 1) * K]
            assert (block == EDGE_SPATIAL).all()

    def test_temporal_edges_are_symmetric_between_adjacent_historical_steps_on_the_same_slot(
        self, kinds
    ):
        for slot in range(K):
            assert kinds[1 * K + slot, 0 * K + slot] == EDGE_TEMPORAL
            # The reverse direction is connected too, as long as both steps are historical.
            assert kinds[0 * K + slot, 1 * K + slot] == EDGE_TEMPORAL
            # A different object at a different timestep is never connected.
            other = (slot + 1) % K
            assert kinds[1 * K + slot, 0 * K + other] == EDGE_NONE

    def test_temporal_edges_never_reach_the_goal_step(self, kinds):
        goal_step = T  # goal frames are the trailing timesteps
        for slot in range(K):
            # The pair between the last historical step and the goal is EDGE_GOAL, not
            # EDGE_TEMPORAL, even though they are "adjacent" in step index.
            assert kinds[goal_step * K + slot, (T - 1) * K + slot] != EDGE_TEMPORAL
            assert kinds[(T - 1) * K + slot, goal_step * K + slot] != EDGE_TEMPORAL

    def test_every_historical_observation_attends_to_its_own_goal(self, kinds):
        goal_step = T  # goal frames are the trailing timesteps
        for slot in range(K):
            for step in range(T):
                assert kinds[step * K + slot, goal_step * K + slot] == EDGE_GOAL
            # ... and a goal node never queries the observations.
            assert kinds[goal_step * K + slot, (T - 1) * K + slot] == EDGE_NONE

    def test_goal_nodes_are_keys_but_never_queries_of_the_past(self, kinds):
        goal_rows = kinds[T * K :, : T * K]
        assert (goal_rows == EDGE_NONE).all()


class TestGraphTransformer:
    def test_preserves_the_node_grid_and_widens_the_features(self):
        out = _embedder()(_task())
        assert out.shape == (B, T + G, K, OUT)

    def test_rejects_a_window_that_does_not_match_its_horizons(self):
        task = _task()
        task["nodes"] = torch.randn(B, T + G + 1, K, IN)
        with pytest.raises(ValueError, match="obs_horizon"):
            _embedder()(task)

    def test_all_inactive_slots_do_not_produce_nans(self):
        """A row with no permitted key would make softmax emit NaN; self-loops prevent that."""
        valid = torch.ones(B, T + G, K)
        valid[0, :, 1:] = 0.0  # only the TCP survives in this sample
        out = _embedder()(_task(valid))
        assert torch.isfinite(out).all()

    def test_inactive_slots_cannot_influence_active_ones(self):
        """The whole point of the validity mask: an absent object is parked off-table, and its
        stale pose must not leak into any real node's representation."""
        torch.manual_seed(0)
        embedder = _embedder().eval()

        valid = torch.ones(B, T + G, K)
        valid[:, :, 3:] = 0.0
        task = _task(valid)

        with torch.no_grad():
            before = embedder(task)
            # Corrupt everything about the inactive slots: features and their edges alike.
            task["nodes"][:, :, 3:] = torch.randn_like(task["nodes"][:, :, 3:]) * 100
            slot_index = torch.arange(SEQ) % K
            inactive = slot_index >= 3
            task["edge_feat"][:, inactive, :] = torch.randn_like(task["edge_feat"][:, inactive, :])
            task["edge_feat"][:, :, inactive] = torch.randn_like(task["edge_feat"][:, :, inactive])
            after = embedder(task)

        assert torch.allclose(before[:, :, :3], after[:, :, :3], atol=1e-6)

    def test_goal_edges_carry_no_geometric_payload(self):
        """EDGE_GOAL pairs must be expressed purely through the learned edge-kind embedding: the
        SE(3) delta is redundant there (it already reaches the model through each node's own
        goal-relative content), so ``edge_feat`` should be ignored for those pairs."""
        embedder = _embedder().eval()
        task = _task()

        task_corrupted = dict(task)
        task_corrupted["edge_feat"] = task["edge_feat"].clone()
        kinds = embedder._edge_kinds(T + G, K, torch.device("cpu"))
        goal_pairs = kinds == EDGE_GOAL
        task_corrupted["edge_feat"][:, goal_pairs] = torch.randn_like(
            task_corrupted["edge_feat"][:, goal_pairs]
        )

        with torch.no_grad():
            out = embedder(task)
            out_corrupted = embedder(task_corrupted)

        assert torch.allclose(out, out_corrupted, atol=1e-6)

    def test_edge_none_pairs_still_contribute_a_learned_bias(self):
        """Topology is now a soft prior: an EDGE_NONE pair is not hard-masked to -inf, so its
        learned bias (via ``edge_kind_emb[EDGE_NONE]``) should influence attention just like any
        other kind, as long as the key is a valid node."""
        embedder = _embedder().eval()
        task = _task()

        attn_mask = embedder._attention_mask(
            task["edge_feat"], task["valid"], T + G, K
        )
        kinds = embedder._edge_kinds(T + G, K, torch.device("cpu"))
        none_pairs = kinds == EDGE_NONE
        # Valid-key EDGE_NONE pairs must be finite (attended), not -inf.
        finite_mask = attn_mask[0][none_pairs]
        assert torch.isfinite(finite_mask).all()

    def test_gradients_reach_the_edge_encoder(self):
        embedder = _embedder()
        embedder(_task()).sum().backward()

        for name in ("edge_kind_emb.weight", "edge_bias.0.weight", "pos_emb", "role_emb.weight"):
            param = dict(embedder.named_parameters())[name]
            assert param.grad is not None, name
            assert torch.isfinite(param.grad).all(), name

    def test_role_is_added_additively(self):
        """Permuting which slot holds which role changes that slot's output -- role is actually
        consumed, not silently dropped."""
        embedder = _embedder().eval()
        task = _task()

        task_permuted = dict(task)
        task_permuted["role"] = task["role"].clone()
        task_permuted["role"][:, :, [0, 1]] = task_permuted["role"][:, :, [1, 0]]

        with torch.no_grad():
            out = embedder(task)
            out_permuted = embedder(task_permuted)

        assert not torch.allclose(out[:, :, 0, :], out_permuted[:, :, 0, :])
