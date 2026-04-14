"""
Tree verification forward pass through the hybrid Qwen 3.5 model.

Processes all tree nodes in DFS order through the model:
- Attention layers: per-token RoPE (batch-reshape trick) + tree attention mask
- Linear layers: sequential processing in DFS order (recurrent)

The DFS ordering ensures the most-probable path is processed first,
maximizing fast-path commit opportunities (tape rollback).
"""

from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .compile import CompiledTree

# Use dflash-mlx's model navigation helpers (handles VL models, nested wrappers)
from dflash_mlx.runtime import (
    _target_text_model,
    _lm_head_logits,
)


def _rope_with_positions(
    x: mx.array,
    position_ids: mx.array,
    rope_fn: Any,
) -> mx.array:
    """Apply RoPE with per-token positions via batch-reshape trick.

    Reshapes [1, H, T, D] → [T, H, 1, D], applies per-batch offsets,
    then reshapes back. Verified to produce zero diff vs individual application.

    Args:
        x: (1, n_heads, T, head_dim) query or key tensor.
        position_ids: (T,) int32 absolute positions per token.
        rope_fn: The model's rope module (nn.RoPE or variant).
    """
    _, H, T, D = x.shape
    # [1, H, T, D] → [T, H, 1, D]
    x_reshaped = x.transpose(0, 2, 1, 3).reshape(T, H, 1, D)
    # Apply rope with per-batch offsets
    x_roped = rope_fn(x_reshaped, offset=position_ids)
    # [T, H, 1, D] → [1, H, T, D]
    return x_roped.reshape(1, T, H, D).transpose(0, 2, 1, 3)


def _attention_forward_with_tree(
    attn: Any,
    x: mx.array,
    position_ids: mx.array,
    mask: mx.array,
    cache: Any,
) -> mx.array:
    """Replicate Qwen3NextAttention.__call__ with per-token RoPE and tree mask.

    Based on mlx_lm/models/qwen3_next.py:120-158.
    """
    B, L, D = x.shape

    # Q projection + split into queries and gate
    q_proj_output = attn.q_proj(x)
    queries, gate = mx.split(
        q_proj_output.reshape(B, L, attn.num_attention_heads, -1), 2, axis=-1
    )
    gate = gate.reshape(B, L, -1)

    # K, V projections
    keys, values = attn.k_proj(x), attn.v_proj(x)

    # Reshape and normalize
    queries = attn.q_norm(queries).transpose(0, 2, 1, 3)  # (B, H, L, head_dim)
    keys = attn.k_norm(
        keys.reshape(B, L, attn.num_key_value_heads, -1)
    ).transpose(0, 2, 1, 3)
    values = values.reshape(B, L, attn.num_key_value_heads, -1).transpose(0, 2, 1, 3)

    # Per-token RoPE (batch-reshape trick instead of scalar offset)
    queries = _rope_with_positions(queries, position_ids, attn.rope)
    keys = _rope_with_positions(keys, position_ids, attn.rope)

    # Update KV cache (appends tree nodes)
    if cache is not None:
        keys, values = cache.update_and_fetch(keys, values)

    # SDPA with tree attention mask — mask must cover full KV length
    kv_len = keys.shape[2]
    mask_kv_len = mask.shape[-1]
    if mask_kv_len != kv_len:
        # Pad or trim mask to match actual KV cache length
        if mask_kv_len < kv_len:
            # KV cache has more entries than expected — extend mask with zeros (attend)
            pad = mx.zeros((mask.shape[0], mask.shape[1], mask.shape[2], kv_len - mask_kv_len), dtype=mask.dtype)
            mask = mx.concatenate([pad, mask], axis=-1)
        else:
            # Mask is too wide — trim from the left (prefix side)
            mask = mask[:, :, :, mask_kv_len - kv_len:]

    output = mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=attn.scale, mask=mask
    )
    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

    # Gate and output projection
    return attn.o_proj(output * mx.sigmoid(gate))


def tree_verify_forward(
    target_model: Any,
    *,
    compiled_tree: CompiledTree,
    cache: list[Any],
    capture_layer_ids: Optional[set[int]] = None,
) -> tuple[mx.array, dict[int, mx.array]]:
    """Run the target model on all tree nodes with tree attention.

    Processes tokens in DFS order for optimal linear layer behavior.
    Attention layers use tree mask + per-token RoPE for correct scoring.

    Args:
        target_model: The loaded MLX target model (TextModel).
        compiled_tree: CompiledTree from compile_tree().
        cache: List of per-layer caches (KVCache for attention, ArraysCache for linear).
        capture_layer_ids: Set of layer indices to capture hidden states (for draft conditioning).

    Returns:
        (logits, captured_hidden_states):
        - logits: (1, tree_size, vocab_size) in tree-index order
        - captured_hidden_states: {layer_id: (1, tree_size, hidden_dim)} in tree-index order
    """
    ct = compiled_tree

    # Get model internals (handles VL models, nested wrappers)
    inner = _target_text_model(target_model)

    # Find actual KV cache offset from an attention layer cache
    fa_idx = getattr(inner, "fa_idx", None)
    if fa_idx is not None and cache[fa_idx] is not None:
        actual_prefix = int(getattr(cache[fa_idx], "offset", 0) or 0)
    else:
        # Fallback: find first KVCache
        actual_prefix = 0
        for c in cache:
            if hasattr(c, "offset") and not hasattr(c, "cache"):
                actual_prefix = int(c.offset or 0)
                break

    # Reorder tokens and mask to DFS order
    dfs = ct.dfs_order
    inv_dfs = ct.inv_dfs_order

    # Embed tokens in tree-index order, then reorder to DFS
    h = inner.embed_tokens(ct.input_ids)  # (1, tree_size, hidden_dim)
    h = h[:, dfs, :]  # reorder to DFS

    # Reorder position_ids to DFS order
    position_ids_dfs = ct.position_ids[dfs]

    # Build attention mask from tree visibility, using actual cache prefix length
    # Tree visibility (tree-index order): (N+1, N+1) bool
    tree_vis = ct.attention_mask[:, :, :, -ct.tree_size:]  # extract tree-to-tree part

    # Reorder to DFS order
    tree_vis_dfs = tree_vis[:, :, dfs, :][:, :, :, dfs]

    # All tree nodes attend to entire prefix → zeros (attend)
    # Use actual_prefix from cache, not compiled prefix_len
    prefix_mask_dfs = mx.zeros((1, 1, ct.tree_size, actual_prefix), dtype=mx.float32)

    mask_dfs = mx.concatenate([prefix_mask_dfs, tree_vis_dfs], axis=-1)
    mask_dfs = mask_dfs.astype(h.dtype)

    # SSM mask for linear layers (None for standard ArraysCache)
    ssm_cache_idx = getattr(inner, "ssm_idx", 0)
    from mlx_lm.models.base import create_ssm_mask
    ssm_mask = create_ssm_mask(h, cache[ssm_cache_idx])

    # Track captured hidden states
    captured: dict[int, mx.array] = {}
    if capture_layer_ids and 0 in capture_layer_ids:
        captured[0] = h[:, inv_dfs, :]  # store in tree-index order

    # Process through each layer
    for layer_idx, (layer, layer_cache) in enumerate(zip(inner.layers, cache)):
        if layer.is_linear:
            # Linear layer: process in DFS order (sequential recurrent)
            r = layer.linear_attn(layer.input_layernorm(h), ssm_mask, layer_cache)
            h = h + r
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        else:
            # Attention layer: custom forward with per-token RoPE + tree mask
            r = _attention_forward_with_tree(
                layer.self_attn,
                layer.input_layernorm(h),
                position_ids_dfs,
                mask_dfs,
                layer_cache,
            )
            h = h + r
            h = h + layer.mlp(layer.post_attention_layernorm(h))

        if capture_layer_ids and (layer_idx + 1) in capture_layer_ids:
            captured[layer_idx + 1] = h[:, inv_dfs, :]  # tree-index order

    # Final norm and LM head
    normalized = inner.norm(h)

    # Reorder back to tree-index order for logits
    normalized = normalized[:, inv_dfs, :]

    logits = _lm_head_logits(target_model, normalized)

    return logits, captured
