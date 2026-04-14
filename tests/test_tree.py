"""Tests for DDTree tree building and walking."""

import numpy as np
import mlx.core as mx
from ddtree_mlx.compile import compile_tree
from ddtree_mlx.runtime import _build_tree_from_mlx_logits, _walk_dfs_exact_prefix
from ddtree_mlx.tree import build_ddtree_tree, follow_verified_tree, compute_dfs_order


def test_empty_budget():
    logits = np.random.randn(3, 100).astype(np.float32)
    tree = build_ddtree_tree(logits, budget=0)
    assert tree.node_count == 0
    assert tree.visibility.shape == (1, 1)
    assert tree.visibility[0, 0] == True


def test_empty_logits():
    logits = np.empty((0, 100), dtype=np.float32)
    tree = build_ddtree_tree(logits, budget=16)
    assert tree.node_count == 0


def test_basic_tree_structure():
    """Build a tree with budget=4 from 3-position logits."""
    np.random.seed(42)
    logits = np.random.randn(3, 50).astype(np.float32)
    tree = build_ddtree_tree(logits, budget=4)

    assert tree.node_count == 4
    assert len(tree.node_token_ids) == 4
    assert len(tree.node_depths) == 4
    assert len(tree.parents) == 5  # root + 4 nodes
    assert len(tree.child_maps) == 5
    assert tree.visibility.shape == (5, 5)

    # Root sees only itself
    assert tree.visibility[0, 0] == True
    assert not np.any(tree.visibility[0, 1:])

    # Every node sees itself
    for i in range(5):
        assert tree.visibility[i, i] == True

    # Every non-root node sees root
    for i in range(1, 5):
        assert tree.visibility[i, 0] == True

    # Parent is always -1 for root
    assert tree.parents[0] == -1

    # All depths are >= 1
    assert np.all(tree.node_depths >= 1)
    assert np.all(tree.node_depths <= 3)


def test_visibility_is_ancestor_only():
    """Verify that visibility matrix only allows ancestor attention."""
    np.random.seed(0)
    logits = np.random.randn(4, 100).astype(np.float32)
    tree = build_ddtree_tree(logits, budget=8)

    # For each node, check it can only see its ancestors and itself
    for i in range(1, 1 + tree.node_count):
        # Collect ancestors
        ancestors = set()
        node = i
        while node >= 0:
            ancestors.add(node)
            node = tree.parents[node]

        for j in range(1 + tree.node_count):
            if j in ancestors:
                assert tree.visibility[i, j], f"Node {i} should see ancestor {j}"
            else:
                assert not tree.visibility[i, j], f"Node {i} should NOT see non-ancestor {j}"


def test_follow_verified_tree_full_accept():
    """Test tree walk when all tokens on primary path match."""
    np.random.seed(42)
    logits = np.random.randn(3, 100).astype(np.float32)
    tree = build_ddtree_tree(logits, budget=8)

    # Build posterior that exactly matches the primary path (depth-first, rank-0 tokens)
    # Find the primary path: root -> highest prob child -> ...
    primary_path = [0]
    current = 0
    posterior = [0] * (1 + tree.node_count)
    for depth_idx in range(tree.node_depths.max()):
        children = tree.child_maps[current]
        if not children:
            break
        # Pick the first child (should be highest prob from heap)
        token_id = next(iter(children))
        child_idx = children[token_id]
        posterior[current] = token_id
        primary_path.append(child_idx)
        current = child_idx

    # Set a dummy posterior for the last node
    posterior[current] = 99999

    accepted, bonus = follow_verified_tree(tree.child_maps, posterior)
    assert accepted == primary_path
    assert bonus == 99999


def test_follow_verified_tree_immediate_reject():
    """Test tree walk when first token doesn't match any child."""
    np.random.seed(42)
    logits = np.random.randn(3, 100).astype(np.float32)
    tree = build_ddtree_tree(logits, budget=4)

    # Posterior with a token that matches no child
    posterior = [-1] * (1 + tree.node_count)
    accepted, bonus = follow_verified_tree(tree.child_maps, posterior)
    assert accepted == [0]  # only root accepted
    assert bonus == -1


def test_dfs_order():
    """Test DFS ordering produces valid traversal."""
    np.random.seed(42)
    logits = np.random.randn(4, 100).astype(np.float32)
    tree = build_ddtree_tree(logits, budget=16)

    dfs_order, inv_dfs_order = compute_dfs_order(tree)

    # DFS order should contain all nodes exactly once
    assert sorted(dfs_order) == list(range(1 + tree.node_count))

    # Root should be first
    assert dfs_order[0] == 0

    # Inverse should round-trip
    for pos, idx in enumerate(dfs_order):
        assert inv_dfs_order[idx] == pos

    # Parent should appear before child in DFS order
    for idx in range(1, 1 + tree.node_count):
        parent = tree.parents[idx]
        assert inv_dfs_order[parent] < inv_dfs_order[idx], \
            f"Parent {parent} (pos {inv_dfs_order[parent]}) should precede child {idx} (pos {inv_dfs_order[idx]})"


def test_budget_respected():
    """Tree should never have more nodes than budget."""
    for budget in [1, 2, 4, 8, 16, 32]:
        logits = np.random.randn(8, 200).astype(np.float32)
        tree = build_ddtree_tree(logits, budget=budget)
        assert tree.node_count <= budget


def test_budget_equal_vocab():
    """Tree build should support top-k equal to vocab size."""
    logits = np.random.randn(3, 16).astype(np.float32)
    tree = build_ddtree_tree(logits, budget=16)
    assert tree.node_count == 16


def test_compile_tree_mask_is_tree_only():
    logits = np.random.randn(3, 50).astype(np.float32)
    tree = build_ddtree_tree(logits, budget=4)
    compiled = compile_tree(tree, root_token_id=1, prefix_len=128)
    assert compiled.attention_mask.shape == (1, 1, 5, 5)


def test_mlx_tree_build_matches_numpy_tree_build():
    np.random.seed(7)
    logits = np.random.randn(4, 64).astype(np.float32)
    numpy_tree = build_ddtree_tree(logits, budget=8)
    mlx_tree = _build_tree_from_mlx_logits(mx.array(logits), budget=8)
    assert mlx_tree.node_token_ids.tolist() == numpy_tree.node_token_ids.tolist()
    assert mlx_tree.node_depths.tolist() == numpy_tree.node_depths.tolist()
    assert mlx_tree.parents == numpy_tree.parents


def test_walk_dfs_exact_prefix_fast_path():
    np.random.seed(42)
    logits = np.random.randn(3, 100).astype(np.float32)
    tree = build_ddtree_tree(logits, budget=8)
    dfs_order, _ = compute_dfs_order(tree)

    posterior = [-1] * (1 + tree.node_count)
    first_child = dfs_order[1]
    posterior[0] = int(tree.node_token_ids[first_child - 1])
    posterior[first_child] = 99999

    accepted, bonus, exact_prefix_len = _walk_dfs_exact_prefix(
        tree.child_maps, posterior, dfs_order
    )
    assert accepted == [0, first_child]
    assert bonus == 99999
    assert exact_prefix_len == len(accepted)


def test_walk_dfs_exact_prefix_divergence():
    logits = np.array(
        [
            [10, 9, 8, 7, 6],
            [10, 9, 8, 7, 6],
            [10, 9, 8, 7, 6],
        ],
        dtype=np.float32,
    )
    tree = build_ddtree_tree(logits, budget=4)
    dfs_order, _ = compute_dfs_order(tree)

    root_children = list(tree.child_maps[0].values())
    assert len(root_children) >= 2
    divergent_child = root_children[1]
    posterior = [-1] * (1 + tree.node_count)
    posterior[0] = int(tree.node_token_ids[divergent_child - 1])

    accepted, bonus, exact_prefix_len = _walk_dfs_exact_prefix(
        tree.child_maps, posterior, dfs_order
    )
    assert accepted == [0, divergent_child]
    assert bonus is None
    assert exact_prefix_len == 1


if __name__ == "__main__":
    test_empty_budget()
    test_empty_logits()
    test_basic_tree_structure()
    test_visibility_is_ancestor_only()
    test_follow_verified_tree_full_accept()
    test_follow_verified_tree_immediate_reject()
    test_dfs_order()
    test_budget_respected()
    test_budget_equal_vocab()
    test_compile_tree_mask_is_tree_only()
    test_mlx_tree_build_matches_numpy_tree_build()
    test_walk_dfs_exact_prefix_fast_path()
    test_walk_dfs_exact_prefix_divergence()
    print("All tree tests passed!")
