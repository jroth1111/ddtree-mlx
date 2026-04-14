"""
Cache management for DDTree: snapshot, restore, and commit.

Two commit strategies:
- FAST PATH: accepted path is a DFS prefix → use tape rollback for linear layers,
  index-select for attention KV cache. No re-forward needed.
- SLOW PATH: accepted path is NOT a DFS prefix → restore all caches from snapshot,
  re-forward accepted tokens through standard model path.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx


def snapshot_caches(cache_entries: list[Any]) -> list[Any]:
    """Take a deep snapshot of all cache states before tree verification.

    Returns a list of snapshots that can be restored via restore_caches.
    """
    snapshots = []
    for cache_entry in cache_entries:
        if hasattr(cache_entry, "state"):
            state = cache_entry.state
            if isinstance(state, list):
                # ArraysCache: list of arrays
                snapshots.append([mx.array(s) if s is not None else None for s in state])
            elif isinstance(state, tuple):
                # KVCache: (keys, values) tuple
                k, v = state
                snapshots.append((mx.array(k), mx.array(v)) if k is not None else None)
            else:
                snapshots.append(None)
        elif hasattr(cache_entry, "offset"):
            # Just save offset for basic caches
            snapshots.append(cache_entry.offset)
        else:
            snapshots.append(None)
    return snapshots


def restore_caches(cache_entries: list[Any], snapshots: list[Any]) -> None:
    """Restore all cache states from a snapshot."""
    for cache_entry, snap in zip(cache_entries, snapshots):
        if snap is None:
            continue
        if hasattr(cache_entry, "state"):
            if isinstance(snap, list):
                # ArraysCache
                cache_entry.state = snap
            elif isinstance(snap, tuple):
                # KVCache
                cache_entry.state = snap
        elif hasattr(cache_entry, "offset") and isinstance(snap, int):
            cache_entry.offset = snap


def fast_path_commit(
    cache_entries: list[Any],
    prefix_len: int,
    n_accepted: int,
) -> None:
    """Fast-path commit: accepted path is a DFS prefix.

    For attention layers (KVCache): trim cache to prefix + accepted tokens.
    For linear layers (ArraysCache with rollback): use tape rollback.
    """
    target_len = prefix_len + n_accepted
    for cache_entry in cache_entries:
        if hasattr(cache_entry, "rollback"):
            # RecurrentRollbackCache: replay first n_accepted steps from tape
            cache_entry.rollback(n_accepted - 1)  # rollback expects 0-based
        elif hasattr(cache_entry, "offset"):
            # KVCache: trim to keep only prefix + accepted tokens
            offset = int(getattr(cache_entry, "offset", 0) or 0)
            if offset > target_len:
                cache_entry.offset = target_len


def slow_path_commit(
    target_model: Any,
    cache_entries: list[Any],
    snapshots: list[Any],
    accepted_token_ids: mx.array,
    capture_layer_ids: set[int] | None = None,
) -> tuple[mx.array, dict[int, mx.array]]:
    """Slow-path commit: re-forward accepted tokens from snapshot.

    Restores caches, then runs standard sequential forward on accepted tokens.
    This guarantees lossless cache state identical to greedy AR.

    Args:
        target_model: The MLX target model.
        cache_entries: Cache list (will be modified in-place).
        snapshots: Snapshots from before tree verify.
        accepted_token_ids: (1, n_accepted) token IDs to commit.
        capture_layer_ids: Layer indices to capture hidden states.

    Returns:
        (logits, captured_hidden_states) from the sequential forward pass.
    """
    # Restore caches from snapshot
    restore_caches(cache_entries, snapshots)

    # Import here to avoid circular dependency
    from dflash_mlx.runtime import target_forward_with_hidden_states

    # Standard sequential forward (correct for both attention and linear layers)
    logits, captured = target_forward_with_hidden_states(
        target_model,
        input_ids=accepted_token_ids,
        cache=cache_entries,
        capture_layer_ids=capture_layer_ids,
    )

    return logits, captured
