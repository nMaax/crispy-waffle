import pytest
import torch

from policy.algorithms.tokenizers.graph import GraphTokenizer
from policy.transforms.canonicalization.spec import ROLE_CLUTTER, ROLE_PICK, ROLE_TARGET, ROLE_TCP

B, T, G, K = 2, 2, 1, 5
ROLES = [ROLE_TCP, ROLE_PICK, ROLE_TARGET, ROLE_CLUTTER, ROLE_CLUTTER]


def _task_dim(num_slots=K):
    return {
        "obj_pose": (num_slots, 7),
        "obj_role": (num_slots, 4),
        "obj_valid": (num_slots,),
    }


def _tree(time, valid=None, num_slots=K):
    quat = torch.randn(B, time, num_slots, 4)
    quat = quat / quat.norm(dim=-1, keepdim=True)
    return {
        "obj_pose": torch.cat([torch.randn(B, time, num_slots, 3), quat], dim=-1),
        "obj_role": torch.tensor(ROLES[:num_slots]).expand(B, time, num_slots, 4).clone(),
        "obj_valid": torch.ones(B, time, num_slots) if valid is None else valid,
    }


class TestGraphTokenizer:
    def test_emits_nodes_role_validity_and_pairwise_edges(self):
        tokenizer = GraphTokenizer(_task_dim())
        out = tokenizer.tokenize(_tree(T), _tree(G))

        seq = (T + G) * K
        assert set(out) == {"nodes", "role", "valid", "edge_feat"}
        assert out["nodes"].shape == (B, T + G, K, tokenizer.output_dim)
        assert out["role"].shape == (B, T + G, K, 4)
        assert out["valid"].shape == (B, T + G, K)
        assert out["edge_feat"].shape == (B, seq, seq, tokenizer.edge_dim)

    def test_goal_frames_are_appended_as_trailing_timesteps(self):
        tokenizer = GraphTokenizer(_task_dim())
        obs = _tree(T, valid=torch.ones(B, T, K))
        goal = _tree(G, valid=torch.zeros(B, G, K))

        out = tokenizer.tokenize(obs, goal)

        assert torch.equal(out["valid"][:, :T], torch.ones(B, T, K))
        assert torch.equal(out["valid"][:, T:], torch.zeros(B, G, K))

    def test_goal_timestep_node_delta_to_itself_is_zero(self):
        """Every slot's node content is its delta to its own goal-timestep pose, so at the goal
        timestep itself that delta is identically zero."""
        tokenizer = GraphTokenizer(_task_dim())
        out = tokenizer.tokenize(_tree(T), _tree(G))

        assert torch.allclose(out["nodes"][:, -1, :, :], torch.zeros(1), atol=1e-6)
        assert torch.allclose(out["role"][:, :, 0, :], torch.tensor(ROLE_TCP).float())

    def test_nodes_carry_the_delta_to_own_slots_goal_pose(self):
        """Node content at (t, k) is the SE(3) delta from pose[t, k] to pose[-1, k] (that slot's
        goal-timestep pose), analogous to ObjectTokenizer's goal_delta."""
        tokenizer = GraphTokenizer(_task_dim())
        obs, goal = _tree(T), _tree(G)
        out = tokenizer.tokenize(obs, goal)

        pose = torch.cat([obs["obj_pose"], goal["obj_pose"]], dim=1)
        t_i, k_i = 0, 2
        expected_pos = pose[:, -1, k_i, :3] - pose[:, t_i, k_i, :3]
        assert torch.allclose(out["nodes"][:, t_i, k_i, :3], expected_pos, atol=1e-5)

    def test_edges_carry_the_se3_delta_between_endpoints(self):
        tokenizer = GraphTokenizer(_task_dim())
        out = tokenizer.tokenize(_tree(T), _tree(G))
        edges = out["edge_feat"]
        seq = edges.shape[1]

        # A node's delta to itself is zero, and the positional half is antisymmetric.
        assert torch.allclose(edges[:, range(seq), range(seq)], torch.zeros(1), atol=1e-6)
        assert torch.allclose(edges[..., :3], -edges.transpose(1, 2)[..., :3], atol=1e-5)

    def test_edge_index_order_is_t_major_k_minor(self):
        """Edge [i, j] must line up with the flattening the embedder attends over."""
        tokenizer = GraphTokenizer(_task_dim())
        obs, goal = _tree(T), _tree(G)
        out = tokenizer.tokenize(obs, goal)

        pose = torch.cat([obs["obj_pose"], goal["obj_pose"]], dim=1)
        t_i, k_i, t_j, k_j = 1, 3, 0, 2
        expected = pose[:, t_j, k_j, :3] - pose[:, t_i, k_i, :3]
        assert torch.allclose(
            out["edge_feat"][:, t_i * K + k_i, t_j * K + k_j, :3], expected, atol=1e-5
        )

    def test_token_spec_and_normalization_mask_mirror_each_other(self):
        tokenizer = GraphTokenizer(_task_dim())
        spec, mask = tokenizer.token_spec, tokenizer.normalization_mask

        assert set(spec) == set(mask)
        for key, width in spec.items():
            assert mask[key].shape == (width,)
        # Roles would z-score to zero and validity is a flag; the geometry is normalizable.
        assert mask["nodes"].all()
        assert not mask["role"].any()
        assert not mask["valid"].any()
        assert mask["edge_feat"].all()

    def test_num_slots_drives_tokens_per_step(self):
        assert GraphTokenizer(_task_dim(5)).tokens_per_step == 5
        assert GraphTokenizer(_task_dim(9)).tokens_per_step == 9

    def test_single_side_tokenization_is_rejected(self):
        """One graph spans observations and goal together, so neither side stands alone."""
        assert GraphTokenizer.supports_single_side is False
        with pytest.raises(NotImplementedError, match="cannot tokenize a single side"):
            GraphTokenizer(_task_dim()).tokenize(_tree(T), None)

    def test_non_canonical_task_dim_raises_typeerror(self):
        with pytest.raises(TypeError, match="canonical dict task_dim"):
            GraphTokenizer(task_dim=42)
