"""
DDTree generate loop for MLX.

Orchestrates: draft → tree_build → tree_compile → tree_verify →
tree_walk → commit (fast/slow path) → update.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

import mlx.core as mx
import numpy as np

from .tree import build_ddtree_tree, follow_verified_tree, compute_dfs_order
from .compile import compile_tree, is_dfs_prefix
from .verify import tree_verify_forward
from .cache import snapshot_caches, restore_caches, fast_path_commit, slow_path_commit


# Default tree budget (configurable via env var or parameter)
DEFAULT_TREE_BUDGET = int(os.environ.get("DDTREE_BUDGET", "16"))


def generate_ddtree_once(
    *,
    target_model: Any,
    draft_model: Any,
    tokenizer: Any,
    prompt_tokens: list[int],
    max_new_tokens: int = 2048,
    tree_budget: int = DEFAULT_TREE_BUDGET,
    stop_token_ids: list[int] | None = None,
    suppress_token_ids: list[int] | None = None,
) -> dict:
    """Generate tokens using DDTree speculative decoding.

    Args:
        target_model: Loaded MLX target model.
        draft_model: Loaded DFlash draft model.
        tokenizer: HuggingFace tokenizer.
        prompt_tokens: Tokenized prompt (list of ints).
        max_new_tokens: Maximum tokens to generate.
        tree_budget: Number of tree nodes (excluding root).
        stop_token_ids: Tokens that signal end of generation.
        suppress_token_ids: Tokens to suppress during generation.

    Returns:
        Dict with generation results and timing statistics.
    """
    from dflash_mlx.runtime import (
        _target_embed_tokens,
        _lm_head_logits,
        target_forward_with_hidden_states,
        extract_context_feature_from_dict,
        make_target_cache,
        greedy_tokens_with_mask,
        build_suppress_token_mask,
        _eval_logits_and_captured,
        _arm_target_rollback_with_prefix,
        _restore_target_cache_after_acceptance,
    )
    from dflash_mlx.model import ContextOnlyDraftKVCache

    prompt_array = mx.array(prompt_tokens, dtype=mx.uint32)[None]
    prompt_len = len(prompt_tokens)
    stop_token_array = (
        mx.array(stop_token_ids, dtype=mx.uint32) if stop_token_ids else None
    )

    # Create caches
    target_cache = make_target_cache(
        target_model,
        enable_speculative_linear_cache=True,
    )
    draft_sink = int(os.environ.get("DFLASH_DRAFT_SINK", "64"))
    draft_window = int(os.environ.get("DFLASH_DRAFT_WINDOW", "1024"))
    draft_cache = [
        ContextOnlyDraftKVCache(sink_size=draft_sink, window_size=draft_window)
        for _ in range(len(draft_model.layers))
    ]
    capture_layer_ids = {int(lid) + 1 for lid in draft_model.target_layer_ids}

    # --- PREFILL ---
    start_ns = time.perf_counter_ns()
    prefill_start_ns = time.perf_counter_ns()
    prefill_logits, prefill_hidden = target_forward_with_hidden_states(
        target_model,
        input_ids=prompt_array,
        cache=target_cache,
        capture_layer_ids=capture_layer_ids,
    )
    _eval_logits_and_captured(prefill_logits, prefill_hidden)
    prefill_ns = time.perf_counter_ns() - prefill_start_ns

    suppress_mask = build_suppress_token_mask(
        int(prefill_logits.shape[-1]), suppress_token_ids
    )
    staged_first = greedy_tokens_with_mask(
        prefill_logits[:, -1, :], suppress_mask
    ).reshape(-1)
    target_hidden = extract_context_feature_from_dict(
        prefill_hidden, list(draft_model.target_layer_ids)
    )

    block_size = max(1, int(draft_model.block_size))
    generated_tokens: list[int] = []
    start = prompt_len
    cycles_completed = 0
    acceptance_history: list[int] = []
    fast_path_count = 0
    slow_path_count = 0

    # Timing accumulators
    draft_ns = 0
    tree_build_ns = 0
    tree_verify_ns = 0
    commit_ns = 0

    while len(generated_tokens) < max_new_tokens:
        remaining = max_new_tokens - len(generated_tokens)
        block_len = max(1, min(block_size, remaining))

        # --- DRAFT ---
        draft_start = time.perf_counter_ns()
        block_token_ids = mx.full((block_len,), draft_model.mask_token_id, dtype=mx.uint32)
        block_token_ids[0] = staged_first[0] if staged_first.ndim > 0 else staged_first

        if block_len > 1:
            noise_embedding = _target_embed_tokens(target_model)(block_token_ids[None])
            draft_hidden = draft_model(
                noise_embedding=noise_embedding,
                target_hidden=target_hidden,
                cache=draft_cache,
            )
            draft_logits = _lm_head_logits(target_model, draft_hidden[:, 1:, :])
            mx.eval(draft_logits)
        else:
            draft_logits = None
        draft_ns += time.perf_counter_ns() - draft_start

        if draft_logits is None or block_len <= 1:
            # No speculation possible — just commit the staged first token
            generated_tokens.append(int(staged_first.item()))
            # Forward staged_first through target for next round
            commit_start = time.perf_counter_ns()
            fwd_logits, fwd_hidden = target_forward_with_hidden_states(
                target_model,
                input_ids=staged_first[None] if staged_first.ndim == 1 else staged_first[None, None],
                cache=target_cache,
                capture_layer_ids=capture_layer_ids,
            )
            mx.eval(fwd_logits)
            target_hidden = extract_context_feature_from_dict(
                fwd_hidden, list(draft_model.target_layer_ids)
            )
            staged_first = greedy_tokens_with_mask(
                fwd_logits[:, -1, :], suppress_mask
            ).reshape(-1)
            start += 1
            commit_ns += time.perf_counter_ns() - commit_start
            if stop_token_array is not None and generated_tokens[-1] in stop_token_ids:
                break
            continue

        # --- TREE BUILD ---
        build_start = time.perf_counter_ns()
        draft_logits_np = np.array(draft_logits[0].astype(mx.float32), copy=False)
        tree = build_ddtree_tree(draft_logits_np, budget=tree_budget)
        root_token = int(staged_first.item())
        compiled = compile_tree(tree, root_token, prefix_len=start)
        dfs_order_list = compiled.dfs_order.tolist()
        tree_build_ns += time.perf_counter_ns() - build_start

        # --- SNAPSHOT + ARM ROLLBACK ---
        snapshots = snapshot_caches(target_cache)
        _arm_target_rollback_with_prefix(target_cache, prefix_len=start)

        # --- TREE VERIFY ---
        verify_start = time.perf_counter_ns()
        verify_logits, verify_hidden = tree_verify_forward(
            target_model,
            compiled_tree=compiled,
            cache=target_cache,
            capture_layer_ids=capture_layer_ids,
        )
        mx.eval(verify_logits)
        tree_verify_ns += time.perf_counter_ns() - verify_start

        # --- TREE WALK ---
        posterior = mx.argmax(verify_logits[0], axis=-1)
        posterior_list = posterior.tolist()
        accepted_indices, bonus_token = follow_verified_tree(
            tree.child_maps, posterior_list
        )
        n_accepted = len(accepted_indices)  # includes root
        acceptance_history.append(n_accepted)

        # Collect accepted token IDs (root + accepted nodes)
        accepted_token_ids_list = [root_token]
        for idx in accepted_indices[1:]:
            accepted_token_ids_list.append(int(tree.node_token_ids[idx - 1]))

        # --- COMMIT ---
        commit_start = time.perf_counter_ns()
        use_fast_path = is_dfs_prefix(accepted_indices, dfs_order_list)

        if use_fast_path:
            fast_path_count += 1
            fast_path_commit(target_cache, prefix_len=start, n_accepted=n_accepted)
            # Use hidden states captured during tree verify
            committed_hidden = extract_context_feature_from_dict(
                verify_hidden, list(draft_model.target_layer_ids)
            )
            # Select only accepted indices
            accepted_idx_array = mx.array(accepted_indices, dtype=mx.int32)
            committed_hidden = committed_hidden[:, accepted_idx_array, :]
        else:
            slow_path_count += 1
            accepted_ids_mx = mx.array(
                accepted_token_ids_list, dtype=mx.uint32
            )[None]
            _, committed_hidden_raw = slow_path_commit(
                target_model,
                target_cache,
                snapshots,
                accepted_ids_mx,
                capture_layer_ids=capture_layer_ids,
            )
            committed_hidden = extract_context_feature_from_dict(
                committed_hidden_raw, list(draft_model.target_layer_ids)
            )

        mx.eval(committed_hidden)
        commit_ns += time.perf_counter_ns() - commit_start

        # --- UPDATE ---
        # Add accepted tokens + bonus to generated output
        generated_tokens.extend(accepted_token_ids_list)
        generated_tokens.append(bonus_token)
        start += n_accepted + 1  # +1 for bonus token
        target_hidden = committed_hidden
        staged_first = mx.array([bonus_token], dtype=mx.uint32)
        cycles_completed += 1

        # Check stop tokens
        if stop_token_array is not None:
            for t in accepted_token_ids_list + [bonus_token]:
                if t in stop_token_ids:
                    break

    # Trim to max_new_tokens
    generated_tokens = generated_tokens[:max_new_tokens]

    # Remove stop tokens from end
    if stop_token_ids:
        while generated_tokens and generated_tokens[-1] in stop_token_ids:
            generated_tokens.pop()

    elapsed_us = (time.perf_counter_ns() - start_ns) / 1_000.0
    gen_count = len(generated_tokens)

    return {
        "generated_token_ids": generated_tokens,
        "generation_tokens": gen_count,
        "elapsed_us": elapsed_us,
        "prefill_us": prefill_ns / 1_000.0,
        "tokens_per_second": gen_count / (elapsed_us / 1e6) if elapsed_us > 0 else 0,
        "cycles_completed": cycles_completed,
        "acceptance_history": acceptance_history,
        "avg_acceptance": (
            sum(acceptance_history) / len(acceptance_history)
            if acceptance_history
            else 0
        ),
        "fast_path_count": fast_path_count,
        "slow_path_count": slow_path_count,
        "fast_path_ratio": (
            fast_path_count / (fast_path_count + slow_path_count)
            if (fast_path_count + slow_path_count) > 0
            else 0
        ),
        "phase_timings_us": {
            "prefill": prefill_ns / 1_000.0,
            "draft": draft_ns / 1_000.0,
            "tree_build": tree_build_ns / 1_000.0,
            "tree_verify": tree_verify_ns / 1_000.0,
            "commit": commit_ns / 1_000.0,
        },
        "tree_budget": tree_budget,
    }
