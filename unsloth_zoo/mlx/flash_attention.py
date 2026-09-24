# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""FlashAttention-2 style causal attention backward for MLX training.

MLX's `mx.fast.scaled_dot_product_attention` has a fused forward, but its
backward is not implemented on Metal (`ScaledDotProductAttentionVJP::
use_fallback` returns true unconditionally, v0.32.1 through main as of
2026-09): every training step differentiates an unfused decomposition that
materializes the full S x S attention matrix, including the causally masked
half. That backward is most of the attention cost of a training step.

This module replaces it, for causal self-attention on GPUs with neural
accelerators (NAX), with three Metal kernels built on MLX's own NAX tile
primitives (`kernels/nax_attention.metal`):

- forward: MLX's NAX forward algorithm, additionally saving the per-row
  logsumexp (log2 domain);
- dQ: per 64-query block, recomputing P tile by tile from the logsumexp;
- dK/dV: per 64-key block, iterating query blocks from the diagonal on.

Nothing S x S is materialized, and blocks above the causal diagonal are
skipped. GQA is handled by indexing; dK/dV are written per query head and
summed over each KV group afterwards, which avoids atomics and keeps the
gradients deterministic.

`install_flash_attention_backward()` wraps `mx.fast.scaled_dot_product_attention`
once per process. The wrapper only routes to these kernels while a trainer
run holds the training patches (`mlx_training_patches_active()`) and the call
is a supported shape; everything else, including inference and evaluation,
goes to the original. Kill switch: UNSLOTH_MLX_FLASH_ATTENTION=0.
"""

from __future__ import annotations

import os
import platform
import re
import threading
from functools import lru_cache
from pathlib import Path

import mlx.core as mx

__all__ = [
    "flash_attention_available",
    "flash_attention_supports",
    "flash_causal_attention",
    "install_flash_attention_backward",
]

# Head dims the kernels are validated for. TD = D / 16 must be even for the
# P @ V tiling; 256 is excluded because it is slower than MLX's path there.
_SUPPORTED_HEAD_DIMS = (64, 96, 128)
_SUPPORTED_DTYPES = (mx.bfloat16, mx.float16)
# Below this the three kernel launches cost more than the unfused path.
_MIN_SEQUENCE_LENGTH = 128
_BLOCK = 64

_INSTALL_LOCK = threading.Lock()

_COMMON = """
using namespace mlx::steel;
constexpr int kU = 16;
constexpr int TD = BD / kU;
constexpr float kNegInf = -3.0e38f;
const uint3 tid = threadgroup_position_in_grid;
const ushort sg = simdgroup_index_in_threadgroup;
const float scale2 = scale[0] * 1.44269504089f;
const int S = seq_len[0];
using frag_t = NAXTile<float, 1, 1>::NAXFrag_t;
const short2 sc = frag_t::get_coord();
const short sm = sc.y;
const short sn = sc.x;
"""

# Query-major kernels: each simdgroup owns 16 query rows of a 64-row block.
_Q_MAJOR_PRELUDE = """
constexpr int BQ = 64, BK = 32, TK = BK / kU;
const int qb = tid.x, h = tid.y, b = tid.z;
const int kvh = h / (H / KV);
const int row0 = qb * BQ + sg * kU;
const size_t qoff = ((size_t)(b * H + h) * S + row0) * BD;
const device T* Qp = Q + qoff;
const device T* Kp = K + (size_t)(b * KV + kvh) * S * BD;
const device T* Vp = V + (size_t)(b * KV + kvh) * S * BD;
const int kb_lim = ((qb + 1) * BQ + BK - 1) / BK;
"""

# {out}[16, BK] = {a}[16, BD] @ {b}[BK, BD]^T
_A_BT_TILE = """
    NAXTile<float, 1, TK> {out};
    {out}.clear();
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ik += 2) {{
#pragma clang loop unroll_count(4)
      for (short id = 0; id < TD; id++) {{
        NAXTile<T, 1, 1> At;
        NAXTile<T, 2, 1> Bt;
        At.load({a} + id * kU, BD);
        Bt.load({b} + (ik * kU) * BD + id * kU, BD);
        frag_t::mma({out}.frag_at(0, ik), {out}.frag_at(0, ik + 1),
                    At.frag_at(0, 0), metal::false_type{{}},
                    Bt.frag_at(0, 0), Bt.frag_at(1, 0), metal::true_type{{}});
      }}
    }}
"""

# {acc}[16, BD] += {p}[16, BK] @ {b}[BK, BD]
_P_B_ACCUMULATE = """
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TD; id += 2) {{
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {{
        NAXTile<T, 1, 2> Bt;
        Bt.load({b} + (ik * kU) * BD + id * kU, BD);
        frag_t::mma({acc}.frag_at(0, id), {acc}.frag_at(0, id + 1),
                    {p}.frag_at(0, ik), metal::false_type{{}},
                    Bt.frag_at(0, 0), Bt.frag_at(0, 1), metal::false_type{{}});
      }}
    }}
"""

# Scale S into the log2 domain and apply the causal mask on diagonal blocks.
# Fragment element ii * 4 + jj sits at row sm + 8 * ii, column sn + jj.
_SCALE_AND_MASK_S = """
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < NAXTile<float, 1, TK>::kElemsPerTile; i++) St.elems()[i] *= scale2;
  if ((kb + 1) * BK > row0) {
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ik++) {
      thread auto& fg = St.frag_at(0, ik);
      STEEL_PRAGMA_UNROLL
      for (short ii = 0; ii < 2; ii++) {
        STEEL_PRAGMA_UNROLL
        for (short jj = 0; jj < 4; jj++) {
          const int r = row0 + ii * 8 + sm;
          const int c = kb * BK + ik * kU + jj + sn;
          fg[ii * 4 + jj] = (c <= r) ? fg[ii * 4 + jj] : kNegInf;
        }
      }
    }
  }
"""

_OPS = """
struct MaxOp { template <typename U> METAL_FUNC static constexpr U apply(U x, U y) { return metal::max(x, y); } };
struct SumOp { template <typename U> METAL_FUNC static constexpr U apply(U x, U y) { return x + y; } };
struct MulOp { template <typename U> METAL_FUNC static constexpr U apply(U x, U y) { return x * y; } };
struct ExpSubOp { template <typename U> METAL_FUNC static constexpr U apply(U x, U y) { return fast::exp2(x - y); } };
"""

_FORWARD_SRC = (
    _COMMON
    + _Q_MAJOR_PRELUDE
    + """
NAXTile<float, 1, TD> Otile;
Otile.clear();
metal::vec<float, 2> max_score = {kNegInf, kNegInf};
metal::vec<float, 2> sum_score = {0.0f, 0.0f};
for (int kb = 0; kb < kb_lim; kb++) {
  const device T* Kb = Kp + (size_t)kb * BK * BD;
  const device T* Vb = Vp + (size_t)kb * BK * BD;
"""
    + _A_BT_TILE.format(out = "St", a = "Qp", b = "Kb")
    + _SCALE_AND_MASK_S
    + """
  metal::vec<float, 2> new_max = max_score;
  St.template row_reduce<MaxOp>(new_max);
  St.template row_bin_op<ExpSubOp>(new_max);
  metal::vec<float, 2> factor;
  for (short i = 0; i < 2; i++) {
    factor[i] = fast::exp2(max_score[i] - new_max[i]);
    max_score[i] = new_max[i];
    sum_score[i] *= factor[i];
  }
  St.template row_reduce<SumOp>(sum_score);
  Otile.template row_bin_op<MulOp>(factor);
"""
    + _P_B_ACCUMULATE.format(acc = "Otile", p = "St", b = "Vb")
    + """
}
metal::vec<float, 2> rcp = 1.0f / sum_score;
Otile.template row_bin_op<MulOp>(rcp);
Otile.store(O + qoff, BD);
if (sn == 0) {
  const size_t lbase = (size_t)(b * H + h) * S + row0 + sm;
  L[lbase] = max_score[0] + metal::log2(sum_score[0]);
  L[lbase + 8] = max_score[1] + metal::log2(sum_score[1]);
}
"""
)

_DQ_SRC = (
    _COMMON
    + _Q_MAJOR_PRELUDE
    + """
const device T* dOp = dO + qoff;
const size_t lbase = (size_t)(b * H + h) * S + row0 + sm;
metal::vec<float, 2> Lr = {L[lbase], L[lbase + 8]};
metal::vec<float, 2> Dr = {Delta[lbase], Delta[lbase + 8]};
NAXTile<float, 1, TD> dQt;
dQt.clear();
for (int kb = 0; kb < kb_lim; kb++) {
  const device T* Kb = Kp + (size_t)kb * BK * BD;
  const device T* Vb = Vp + (size_t)kb * BK * BD;
"""
    + _A_BT_TILE.format(out = "St", a = "Qp", b = "Kb")
    + _SCALE_AND_MASK_S
    + """
  St.template row_bin_op<ExpSubOp>(Lr);
"""
    + _A_BT_TILE.format(out = "dPt", a = "dOp", b = "Vb")
    + """
  STEEL_PRAGMA_UNROLL
  for (short ik = 0; ik < TK; ik++) {
    thread auto& p = St.frag_at(0, ik);
    thread auto& dp = dPt.frag_at(0, ik);
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < 2; ii++)
      STEEL_PRAGMA_UNROLL
      for (short jj = 0; jj < 4; jj++)
        p[ii * 4 + jj] = p[ii * 4 + jj] * (dp[ii * 4 + jj] - Dr[ii]);
  }
"""
    + _P_B_ACCUMULATE.format(acc = "dQt", p = "St", b = "Kb")
    + """
}
STEEL_PRAGMA_UNROLL
for (short i = 0; i < NAXTile<float, 1, TD>::kElemsPerTile; i++) dQt.elems()[i] *= scale[0];
dQt.store(dQ + qoff, BD);
"""
)

# Key-major: each simdgroup owns 16 key rows and walks query blocks of 32
# from the diagonal on, working on the transposed problem (P^T = K Q^T).
_DKV_SRC = (
    _COMMON
    + """
constexpr int BK = 32, TK = BK / kU;
const int kb = tid.x, h = tid.y, b = tid.z;
const int kvh = h / (H / KV);
const int key0 = kb * 64 + sg * kU;
const device T* Kp = K + ((size_t)(b * KV + kvh) * S + key0) * BD;
const device T* Vp = V + ((size_t)(b * KV + kvh) * S + key0) * BD;
const size_t hbase = (size_t)(b * H + h) * S;
const device T* Qh = Q + hbase * BD;
const device T* dOh = dO + hbase * BD;
NAXTile<float, 1, TD> dKt, dVt;
dKt.clear();
dVt.clear();
for (int qb = key0 / BK; qb < S / BK; qb++) {
  const device T* Qb = Qh + (size_t)qb * BK * BD;
  const device T* dOb = dOh + (size_t)qb * BK * BD;
  float Lc[TK][4], Dc[TK][4];
  STEEL_PRAGMA_UNROLL
  for (short ik = 0; ik < TK; ik++)
    STEEL_PRAGMA_UNROLL
    for (short jj = 0; jj < 4; jj++) {
      const size_t c = hbase + qb * BK + ik * kU + sn + jj;
      Lc[ik][jj] = L[c];
      Dc[ik][jj] = Delta[c];
    }
"""
    + _A_BT_TILE.format(out = "Pt", a = "Kp", b = "Qb")
    + """
  STEEL_PRAGMA_UNROLL
  for (short ik = 0; ik < TK; ik++) {
    thread auto& fg = Pt.frag_at(0, ik);
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < 2; ii++)
      STEEL_PRAGMA_UNROLL
      for (short jj = 0; jj < 4; jj++) {
        const int r = key0 + ii * 8 + sm;
        const int c = qb * BK + ik * kU + jj + sn;
        const float s = fg[ii * 4 + jj] * scale2;
        fg[ii * 4 + jj] = (c >= r) ? fast::exp2(s - Lc[ik][jj]) : 0.0f;
      }
  }
"""
    + _P_B_ACCUMULATE.format(acc = "dVt", p = "Pt", b = "dOb")
    + _A_BT_TILE.format(out = "dPt", a = "Vp", b = "dOb")
    + """
  STEEL_PRAGMA_UNROLL
  for (short ik = 0; ik < TK; ik++) {
    thread auto& p = Pt.frag_at(0, ik);
    thread auto& dp = dPt.frag_at(0, ik);
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < 2; ii++)
      STEEL_PRAGMA_UNROLL
      for (short jj = 0; jj < 4; jj++)
        p[ii * 4 + jj] = p[ii * 4 + jj] * (dp[ii * 4 + jj] - Dc[ik][jj]);
  }
"""
    + _P_B_ACCUMULATE.format(acc = "dKt", p = "Pt", b = "Qb")
    + """
}
STEEL_PRAGMA_UNROLL
for (short i = 0; i < NAXTile<float, 1, TD>::kElemsPerTile; i++) dKt.elems()[i] *= scale[0];
const size_t obase = (hbase + key0) * BD;
dKt.store(dK + obase, BD);
dVt.store(dV + obase, BD);
"""
)


@lru_cache(maxsize = None)
def _header() -> str:
    path = Path(__file__).with_name("kernels") / "nax_attention.metal"
    return path.read_text() + _OPS


@lru_cache(maxsize = None)
def _kernels():
    header = _header()
    forward = mx.fast.metal_kernel(
        name = "unsloth_flash_attn_fwd",
        input_names = ["Q", "K", "V", "scale", "seq_len"],
        output_names = ["O", "L"],
        source = _FORWARD_SRC,
        header = header,
    )
    backward_dq = mx.fast.metal_kernel(
        name = "unsloth_flash_attn_bwd_dq",
        input_names = ["Q", "K", "V", "dO", "L", "Delta", "scale", "seq_len"],
        output_names = ["dQ"],
        source = _DQ_SRC,
        header = header,
    )
    backward_dkv = mx.fast.metal_kernel(
        name = "unsloth_flash_attn_bwd_dkv",
        input_names = ["Q", "K", "V", "dO", "L", "Delta", "scale", "seq_len"],
        output_names = ["dK", "dV"],
        source = _DKV_SRC,
        header = header,
    )
    return forward, backward_dq, backward_dkv


def _template(q, k):
    B, H, S, D = q.shape
    # The sequence length is a runtime input, not a template argument:
    # variable-length batches would otherwise compile a kernel per length.
    return [("T", q.dtype), ("BD", D), ("H", H), ("KV", k.shape[1])]


def _launch_geometry(q):
    B, H, S, _ = q.shape
    return dict(grid = (S // _BLOCK * 128, H, B), threadgroup = (128, 1, 1))


def _forward(q, k, v, scale):
    forward, _, _ = _kernels()
    B, H, S, _ = q.shape
    return forward(
        inputs = [q, k, v, mx.array([scale], mx.float32), mx.array([S], mx.int32)],
        template = _template(q, k),
        output_shapes = [q.shape, (B, H, S)],
        output_dtypes = [q.dtype, mx.float32],
        **_launch_geometry(q),
    )


def _backward(q, k, v, o, lse, do, scale):
    _, backward_dq, backward_dkv = _kernels()
    B, H, S, D = q.shape
    KV = k.shape[1]
    delta = (do.astype(mx.float32) * o.astype(mx.float32)).sum(-1)
    inputs = [
        q, k, v, do, lse, delta,
        mx.array([scale], mx.float32), mx.array([S], mx.int32),
    ]
    (dq,) = backward_dq(
        inputs = inputs,
        template = _template(q, k),
        output_shapes = [q.shape],
        output_dtypes = [q.dtype],
        **_launch_geometry(q),
    )
    dk_per_head, dv_per_head = backward_dkv(
        inputs = inputs,
        template = _template(q, k),
        output_shapes = [q.shape, q.shape],
        output_dtypes = [mx.float32, mx.float32],
        **_launch_geometry(q),
    )
    groups = H // KV
    dk = dk_per_head.reshape(B, KV, groups, S, D).sum(2).astype(k.dtype)
    dv = dv_per_head.reshape(B, KV, groups, S, D).sum(2).astype(v.dtype)
    return dq, dk, dv


@lru_cache(maxsize = None)
def _custom_attention(scale: float):
    @mx.custom_function
    def attention(q, k, v):
        return _forward(q, k, v, scale)

    @attention.vjp
    def attention_vjp(primals, cotangents, outputs):
        q, k, v = primals
        o, lse = outputs
        # The logsumexp output is internal; only O's cotangent flows back.
        return _backward(q, k, v, o, lse, cotangents[0], scale)

    return attention


def flash_causal_attention(q, k, v, scale: float):
    """Causal self-attention with the flash backward. `q` is [B, H, S, D],
    `k`/`v` are [B, KV, S, D]. Sequence lengths that aren't a multiple of 64
    are zero-padded, which is exact: padded keys lie in every real query's
    future, and padded query rows receive zero cotangent."""
    S = q.shape[2]
    pad = (-S) % _BLOCK
    if pad:
        widths = [(0, 0), (0, 0), (0, pad), (0, 0)]
        q, k, v = (mx.pad(x, widths) for x in (q, k, v))
    out = _custom_attention(float(scale))(q, k, v)[0]
    return out[:, :, :S] if pad else out


@lru_cache(maxsize = None)
def flash_attention_available() -> bool:
    """Same rule as MLX's `metal::is_nax_available()`: macOS 26.2+ and an
    Apple GPU of generation 17+ (18+ for the 'p' variants)."""
    if os.environ.get("UNSLOTH_MLX_FLASH_ATTENTION", "1") == "0":
        return False
    if platform.system() != "Darwin" or not mx.metal.is_available():
        return False
    try:
        release = tuple(int(part) for part in platform.mac_ver()[0].split(".")[:2])
    except ValueError:
        return False
    if release < (26, 2):
        return False
    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    match = re.fullmatch(r"applegpu_g(\d+)([a-z])", str(info.get("architecture", "")))
    if match is None:
        return False
    generation, variant = int(match.group(1)), match.group(2)
    return generation >= (18 if variant == "p" else 17)


def flash_attention_supports(q, k, v, mask, kwargs) -> bool:
    """Whether a `scaled_dot_product_attention` call can take the flash path."""
    if not isinstance(mask, str) or mask != "causal":
        return False
    if any(value is not None for value in kwargs.values()):  # sinks, ...
        return False
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return False
    B, H, S, D = q.shape
    return (
        D in _SUPPORTED_HEAD_DIMS
        and S >= _MIN_SEQUENCE_LENGTH
        and q.dtype in _SUPPORTED_DTYPES
        and k.dtype == q.dtype
        and v.dtype == q.dtype
        and k.shape[0] == B
        and k.shape[2] == S  # self-attention only: no KV cache offset
        and k.shape[3] == D
        and v.shape == k.shape
        and H % k.shape[1] == 0
    )


def install_flash_attention_backward() -> bool:
    """Wrap `mx.fast.scaled_dot_product_attention` once per process. Returns
    True if this call installed the wrapper."""
    if not flash_attention_available():
        return False
    from .utils import mlx_training_patches_active

    with _INSTALL_LOCK:
        current = mx.fast.scaled_dot_product_attention
        if getattr(current, "_unsloth_flash_attention", False):
            return False
        original = current

        def scaled_dot_product_attention(q, k, v, *args, scale = 1.0, mask = None, **kwargs):
            if (
                not args
                and mlx_training_patches_active()
                and flash_attention_supports(q, k, v, mask, kwargs)
            ):
                return flash_causal_attention(q, k, v, scale)
            return original(q, k, v, *args, scale = scale, mask = mask, **kwargs)

        scaled_dot_product_attention._unsloth_flash_attention = True
        scaled_dot_product_attention._unsloth_original = original
        mx.fast.scaled_dot_product_attention = scaled_dot_product_attention
        return True
