"""
DDTree generate loop for MLX.

Orchestrates: draft → tree_build → tree_compile → tree_verify →
tree_walk → commit (tree-aware exact path or legacy prefix/suffix) → update.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import mlx.core as mx
import numpy as np

from .tree import (
    DDTree,
    build_chain_tree_from_topk,
    build_ddtree_tree_from_topk,
    build_hybrid_tree_from_topk,
    build_root_wide_tree_from_topk,
    follow_verified_tree,
)
from .compile import compile_tree
from .verify import tree_verify_forward
from .cache import fast_path_commit, tree_aware_path_commit


# Default tree budget (configurable via env var or parameter)
DEFAULT_TREE_BUDGET = int(os.environ.get("DDTREE_BUDGET", "4"))


@dataclass
class _BudgetStats:
    cycles: int = 0
    accepted: int = 0
    elapsed_ns: int = 0

    def score(self) -> float:
        if self.cycles == 0 or self.elapsed_ns <= 0:
            return 0.0
        return self.accepted / (self.elapsed_ns / 1e9)


@dataclass
class _BudgetController:
    budgets: list[int]
    warmup_cycles: int
    min_cycles: int
    stats: dict[int, _BudgetStats] = field(default_factory=dict)
    cycle: int = 0

    def __post_init__(self) -> None:
        self.budgets = sorted(set(int(b) for b in self.budgets if int(b) > 0))
        self.stats = {budget: _BudgetStats() for budget in self.budgets}

    def choose(self, prefix_len: int) -> int:
        del prefix_len
        if not self.budgets:
            return 0
        for budget in self.budgets:
            if self.stats[budget].cycles < self.warmup_cycles:
                return budget
        eligible = [
            (stats.score(), -budget, budget)
            for budget, stats in self.stats.items()
            if stats.cycles >= self.min_cycles
        ]
        if not eligible:
            return self.budgets[self.cycle % len(self.budgets)]
        return max(eligible)[2]

    def record(self, budget: int, accepted: int, elapsed_ns: int) -> None:
        if budget not in self.stats:
            return
        stats = self.stats[budget]
        stats.cycles += 1
        stats.accepted += int(accepted)
        stats.elapsed_ns += int(elapsed_ns)
        self.cycle += 1

    def summary(self) -> dict[str, Any]:
        return {
            str(budget): {
                "cycles": stats.cycles,
                "accepted": stats.accepted,
                "tokens_per_second": stats.score(),
            }
            for budget, stats in self.stats.items()
        }


def _tree_token_id(tree: DDTree, root_token: int, tree_index: int) -> int:
    if tree_index == 0:
        return int(root_token)
    return int(tree.node_token_ids[tree_index - 1])


def _tree_token_ids(tree: DDTree, root_token: int, indices: list[int]) -> list[int]:
    return [_tree_token_id(tree, root_token, idx) for idx in indices]


def _parse_budget_candidates(max_budget: int) -> list[int]:
    raw = os.environ.get("DDTREE_ADAPTIVE_BUDGETS")
    if raw:
        values = [int(part) for part in raw.split(",") if part.strip()]
    else:
        values = [4, 8, 16, 24]
    candidates = sorted({value for value in values if 0 < value <= max_budget})
    if max_budget > 0 and max_budget not in candidates:
        candidates.append(max_budget)
    return sorted(candidates)


def _build_tree_from_mlx_logits(
    draft_logits: mx.array,
    *,
    budget: int,
    shape: str = "heap",
) -> DDTree:
    """Build a DDTree while transferring only top-k draft data to CPU."""
    if budget <= 0 or int(draft_logits.shape[0]) == 0:
        return build_ddtree_tree_from_topk(
            np.empty((0, 0), dtype=np.int64),
            np.empty((0, 0), dtype=np.float32),
            budget,
        )

    topk = min(int(budget), int(draft_logits.shape[-1]))
    logits = draft_logits.astype(mx.float32)
    top_indices = mx.argpartition(-logits, kth=topk - 1, axis=-1)[:, :topk]
    top_logits = mx.take_along_axis(logits, top_indices, axis=-1)
    sort_order = mx.argsort(-top_logits, axis=-1)
    top_token_ids = mx.take_along_axis(top_indices, sort_order, axis=-1)
    top_logits = mx.take_along_axis(top_logits, sort_order, axis=-1)
    top_log_probs = top_logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    mx.eval(top_token_ids, top_log_probs)

    top_token_ids_np = np.array(top_token_ids, copy=False)
    top_log_probs_np = np.array(top_log_probs, copy=False)
    if shape == "chain":
        return build_chain_tree_from_topk(top_token_ids_np, top_log_probs_np, budget)
    if shape in ("root", "root_wide", "root-wide"):
        return build_root_wide_tree_from_topk(
            top_token_ids_np, top_log_probs_np, budget
        )
    if shape == "hybrid":
        return build_hybrid_tree_from_topk(top_token_ids_np, top_log_probs_np, budget)
    return build_ddtree_tree_from_topk(
        top_token_ids_np,
        top_log_probs_np,
        budget=budget,
        depth_penalty=float(os.environ.get("DDTREE_DEPTH_PENALTY", "0")),
    )


def _walk_dfs_exact_prefix(
    child_maps: list[dict[int, int]],
    posterior_tokens: list[int],
    dfs_order: list[int],
) -> tuple[list[int], int | None, int]:
    """Walk while logits are exact for the DFS prefix.

    Returns (accepted_indices, bonus_token, exact_prefix_len). When bonus_token
    is None, the final accepted node is the first divergent child and must be
    committed by standard forward before walking deeper.
    """
    accepted_indices = [0]
    current_index = 0

    while True:
        next_token = int(posterior_tokens[current_index])
        child_index = child_maps[current_index].get(next_token)
        if child_index is None:
            return accepted_indices, next_token, len(accepted_indices)

        next_pos = len(accepted_indices)
        if next_pos < len(dfs_order) and int(dfs_order[next_pos]) == child_index:
            accepted_indices.append(child_index)
            current_index = child_index
            continue

        accepted_indices.append(child_index)
        return accepted_indices, None, next_pos


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
        generate_dflash_once,
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
    verify_linear_ns = 0
    verify_attention_ns = 0
    profile_verify = os.environ.get("DDTREE_PROFILE_VERIFY", "").lower() not in (
        "",
        "0",
        "false",
    )
    tree_aware_linear = os.environ.get("DDTREE_TREE_AWARE_LINEAR", "1").lower() not in (
        "",
        "0",
        "false",
    )
    tree_aware_commit_count = 0
    tree_shape = os.environ.get("DDTREE_TREE_SHAPE", "heap").lower()
    adaptive_budget = os.environ.get("DDTREE_ADAPTIVE_BUDGET", "0").lower() not in (
        "",
        "0",
        "false",
    )
    budget_controller = (
        _BudgetController(
            budgets=_parse_budget_candidates(int(tree_budget)),
            warmup_cycles=int(os.environ.get("DDTREE_ADAPTIVE_WARMUP", "2")),
            min_cycles=int(os.environ.get("DDTREE_ADAPTIVE_MIN_CYCLES", "2")),
        )
        if adaptive_budget
        else None
    )
    budget_history: list[int] = []
    draft_rank_histogram: dict[int, int] = {}
    dflash_fallback_prefix = int(os.environ.get("DDTREE_DFLASH_FALLBACK_PREFIX", "0"))
    dflash_fallback_used = False
    dflash_fallback_tokens = 0
    dflash_fallback_ns = 0

    while len(generated_tokens) < max_new_tokens:
        remaining = max_new_tokens - len(generated_tokens)
        block_len = max(1, min(block_size, remaining))
        if (
            dflash_fallback_prefix > 0
            and len(generated_tokens) >= dflash_fallback_prefix
        ):
            fallback_start = time.perf_counter_ns()
            fallback_prompt = list(prompt_tokens) + generated_tokens
            fallback_result = generate_dflash_once(
                target_model=target_model,
                tokenizer=tokenizer,
                draft_model=draft_model,
                prompt="",
                max_new_tokens=remaining,
                use_chat_template=False,
                stop_token_ids=stop_token_ids,
                suppress_token_ids=suppress_token_ids,
                prompt_tokens_override=fallback_prompt,
            )
            dflash_fallback_ns += time.perf_counter_ns() - fallback_start
            fallback_tokens = list(fallback_result.get("generated_token_ids", []))
            generated_tokens.extend(fallback_tokens)
            dflash_fallback_tokens += len(fallback_tokens)
            dflash_fallback_used = True
            break

        cycle_start_ns = time.perf_counter_ns()
        current_budget = (
            budget_controller.choose(start) if budget_controller is not None else tree_budget
        )
        budget_history.append(int(current_budget))

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
            _eval_logits_and_captured(fwd_logits, fwd_hidden)
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
        draft_logits_2d = draft_logits[0].astype(mx.float32)
        if suppress_mask is not None:
            floor = mx.array(-1e9, dtype=draft_logits_2d.dtype)
            draft_logits_2d = mx.where(suppress_mask, floor, draft_logits_2d)
        tree = _build_tree_from_mlx_logits(
            draft_logits_2d,
            budget=current_budget,
            shape=tree_shape,
        )
        root_token = int(staged_first.item())
        compiled = compile_tree(tree, root_token, prefix_len=start)
        dfs_order_list = compiled.dfs_order.tolist()
        tree_build_ns += time.perf_counter_ns() - build_start

        # --- ARM ROLLBACK ---
        if not tree_aware_linear:
            _arm_target_rollback_with_prefix(target_cache, prefix_len=start)

        # --- TREE VERIFY ---
        verify_start = time.perf_counter_ns()
        verify_profile = {} if profile_verify else None
        tree_cache_state: dict[str, Any] | None = {} if tree_aware_linear else None
        verify_logits, verify_hidden = tree_verify_forward(
            target_model,
            compiled_tree=compiled,
            cache=target_cache,
            capture_layer_ids=capture_layer_ids,
            profile_timings=verify_profile,
            tree_aware_linear=tree_aware_linear,
            tree_cache_state=tree_cache_state,
        )
        mx.eval(verify_logits)
        tree_verify_ns += time.perf_counter_ns() - verify_start
        if verify_profile is not None:
            verify_linear_ns += verify_profile.get("linear_ns", 0)
            verify_attention_ns += verify_profile.get("attention_ns", 0)

        # --- TREE WALK ---
        posterior = greedy_tokens_with_mask(verify_logits[0], suppress_mask)
        posterior_list = posterior.tolist()
        if tree_aware_linear:
            accepted_indices, bonus_token = follow_verified_tree(
                tree.child_maps, posterior_list
            )
            exact_prefix_len = len(accepted_indices)
        else:
            accepted_indices, bonus_token, exact_prefix_len = _walk_dfs_exact_prefix(
                tree.child_maps, posterior_list, dfs_order_list
            )

        # --- COMMIT ---
        commit_start = time.perf_counter_ns()
        all_hidden = extract_context_feature_from_dict(
            verify_hidden, list(draft_model.target_layer_ids)
        )
        use_fast_path = (
            accepted_indices == dfs_order_list[: len(accepted_indices)]
            if tree_aware_linear
            else exact_prefix_len == len(accepted_indices)
        )

        if tree_aware_linear:
            tree_aware_commit_count += 1
            tree_aware_path_commit(
                target_cache,
                prefix_len=start,
                accepted_indices=accepted_indices,
                tree_cache_state=tree_cache_state or {},
            )
            accepted_idx_array = mx.array(accepted_indices, dtype=mx.int32)
            committed_hidden = all_hidden[:, accepted_idx_array, :]
            if use_fast_path:
                fast_path_count += 1
            else:
                slow_path_count += 1
        elif use_fast_path:
            fast_path_count += 1
            fast_path_commit(
                target_cache,
                prefix_len=start,
                n_accepted=len(accepted_indices),
            )
            # Use hidden states captured during tree verify
            accepted_idx_array = mx.array(accepted_indices, dtype=mx.int32)
            committed_hidden = all_hidden[:, accepted_idx_array, :]
        else:
            slow_path_count += 1
            fast_path_commit(
                target_cache,
                prefix_len=start,
                n_accepted=exact_prefix_len,
            )
            prefix_indices = accepted_indices[:exact_prefix_len]
            prefix_idx_array = mx.array(prefix_indices, dtype=mx.int32)
            hidden_chunks = [all_hidden[:, prefix_idx_array, :]]

            suffix_indices = accepted_indices[exact_prefix_len:]
            suffix_ids = _tree_token_ids(tree, root_token, suffix_indices)
            suffix_ids_mx = mx.array(suffix_ids, dtype=mx.uint32)[None]
            suffix_logits, suffix_hidden_raw = target_forward_with_hidden_states(
                target_model,
                input_ids=suffix_ids_mx,
                cache=target_cache,
                capture_layer_ids=capture_layer_ids,
            )
            _eval_logits_and_captured(suffix_logits, suffix_hidden_raw)
            hidden_chunks.append(
                extract_context_feature_from_dict(
                    suffix_hidden_raw, list(draft_model.target_layer_ids)
                )
            )
            current_index = accepted_indices[-1]
            next_token = int(
                greedy_tokens_with_mask(suffix_logits[:, -1, :], suppress_mask).item()
            )

            while (
                next_token in tree.child_maps[current_index]
                and len(accepted_indices) < block_len
            ):
                current_index = tree.child_maps[current_index][next_token]
                accepted_indices.append(current_index)
                token_ids_mx = mx.array([[next_token]], dtype=mx.uint32)
                suffix_logits, suffix_hidden_raw = target_forward_with_hidden_states(
                    target_model,
                    input_ids=token_ids_mx,
                    cache=target_cache,
                    capture_layer_ids=capture_layer_ids,
                )
                _eval_logits_and_captured(suffix_logits, suffix_hidden_raw)
                hidden_chunks.append(
                    extract_context_feature_from_dict(
                        suffix_hidden_raw, list(draft_model.target_layer_ids)
                    )
                )
                next_token = int(
                    greedy_tokens_with_mask(
                        suffix_logits[:, -1, :], suppress_mask
                    ).item()
                )

            bonus_token = next_token
            committed_hidden = (
                mx.concatenate(hidden_chunks, axis=1)
                if len(hidden_chunks) > 1
                else hidden_chunks[0]
            )

        mx.eval(committed_hidden)
        commit_ns += time.perf_counter_ns() - commit_start

        # --- UPDATE ---
        # Add accepted tokens to generated output. The bonus is staged for the
        # next cycle and is not in target cache yet.
        accepted_token_ids_list = _tree_token_ids(tree, root_token, accepted_indices)
        n_accepted = len(accepted_indices)  # includes root
        acceptance_history.append(n_accepted)
        for accepted_index in accepted_indices[1:]:
            rank = int(tree.node_ranks[accepted_index - 1])
            draft_rank_histogram[rank] = draft_rank_histogram.get(rank, 0) + 1
        emitted_token_ids = accepted_token_ids_list
        stop_hit = False
        if stop_token_array is not None:
            for pos, token_id in enumerate(accepted_token_ids_list):
                if token_id in stop_token_ids:
                    emitted_token_ids = accepted_token_ids_list[: pos + 1]
                    stop_hit = True
                    break
        generated_tokens.extend(emitted_token_ids)
        start += n_accepted
        target_hidden = committed_hidden
        staged_first = mx.array([bonus_token], dtype=mx.uint32)
        cycles_completed += 1
        if budget_controller is not None:
            budget_controller.record(
                int(current_budget),
                n_accepted,
                time.perf_counter_ns() - cycle_start_ns,
            )

        if stop_hit:
            break

    # Trim to max_new_tokens
    generated_tokens = generated_tokens[:max_new_tokens]

    # Remove stop tokens from end
    if stop_token_ids:
        while generated_tokens and generated_tokens[-1] in stop_token_ids:
            generated_tokens.pop()

    elapsed_us = (time.perf_counter_ns() - start_ns) / 1_000.0
    gen_count = len(generated_tokens)
    phase_timings = {
        "prefill": prefill_ns / 1_000.0,
        "draft": draft_ns / 1_000.0,
        "tree_build": tree_build_ns / 1_000.0,
        "tree_verify": tree_verify_ns / 1_000.0,
        "commit": commit_ns / 1_000.0,
    }
    if profile_verify:
        phase_timings["tree_verify_linear"] = verify_linear_ns / 1_000.0
        phase_timings["tree_verify_attention"] = verify_attention_ns / 1_000.0

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
        "tree_aware_commit_count": tree_aware_commit_count,
        "tree_aware_linear": tree_aware_linear,
        "tree_shape": tree_shape,
        "adaptive_budget": adaptive_budget,
        "budget_history": budget_history,
        "budget_stats": budget_controller.summary() if budget_controller else {},
        "draft_rank_histogram": {
            str(rank): count for rank, count in sorted(draft_rank_histogram.items())
        },
        "dflash_fallback_used": dflash_fallback_used,
        "dflash_fallback_tokens": dflash_fallback_tokens,
        "dflash_fallback_us": dflash_fallback_ns / 1_000.0,
        "fast_path_ratio": (
            fast_path_count / (fast_path_count + slow_path_count)
            if (fast_path_count + slow_path_count) > 0
            else 0
        ),
        "phase_timings_us": phase_timings,
        "tree_budget": tree_budget,
    }
