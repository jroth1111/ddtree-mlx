"""Custom MLX kernels for DDTree verification."""

from __future__ import annotations

from typing import Optional

import mlx.core as mx


def _make_tree_gated_delta_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        for (int t = 0; t < T; ++t) {
          auto parent_idx = parents[t];

          const device StT* parent_state;
          if (parent_idx < 0) {
            parent_state = state_in + (n * Dv + dv_idx) * Dk;
          } else {
            parent_state = states
              + (((b_idx * T + parent_idx) * Hv + hv_idx) * Dv + dv_idx) * Dk;
          }

          float state[n_per_t];
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = static_cast<float>(parent_state[s_idx]);
          }

          auto q_t = q + ((b_idx * T + t) * Hk + hk_idx) * Dk;
          auto k_t = k + ((b_idx * T + t) * Hk + hk_idx) * Dk;
          auto v_t = v + ((b_idx * T + t) * Hv + hv_idx) * Dv;
          auto g_t = g + (b_idx * T + t) * Hv;
          auto beta_t = beta + (b_idx * T + t) * Hv;

          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * g_t[hv_idx];
            kv_mem += state[i] * k_t[s_idx];
          }
          kv_mem = simd_sum(kv_mem);

          auto delta = (v_t[dv_idx] - kv_mem) * beta_t[hv_idx];

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + k_t[s_idx] * delta;
            out += state[i] * q_t[s_idx];
          }
          out = simd_sum(out);

          auto y_t = y + ((b_idx * T + t) * Hv + hv_idx) * Dv;
          if (thread_index_in_simdgroup == 0) {
            y_t[dv_idx] = static_cast<InT>(out);
          }

          auto state_t = states
            + (((b_idx * T + t) * Hv + hv_idx) * Dv + dv_idx) * Dk;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state_t[s_idx] = static_cast<StT>(state[i]);
          }
        }
    """

    return mx.fast.metal_kernel(
        name="ddtree_gated_delta_tree",
        input_names=["q", "k", "v", "g", "beta", "state_in", "parents", "T"],
        output_names=["y", "states"],
        source=source,
    )


_tree_gated_delta_kernel = _make_tree_gated_delta_kernel()


def tree_gated_delta_kernel(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    parents: mx.array,
) -> Optional[tuple[mx.array, mx.array]]:
    """Run scalar-gated GatedDelta recurrence over a parent-indexed tree."""
    if _tree_gated_delta_kernel is None:
        return None
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    if Dk % 32 != 0:
        return None

    input_type = q.dtype
    state_type = state.dtype
    return _tree_gated_delta_kernel(
        inputs=[q, k, v, g, beta, state, parents, T],
        template=[
            ("InT", input_type),
            ("StT", state_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), (B, T, Hv, Dv, Dk)],
        output_dtypes=[input_type, state_type],
    )
