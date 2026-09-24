# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""The flash causal-attention kernels against an fp32 reference, on real Metal."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

# See test_mlx_attention_metal.py: importorskip also succeeds against the shim.
from mlx_simulation import mlx_is_simulated  # noqa: E402

if mlx_is_simulated():
    pytest.skip("needs real MLX: these run custom Metal kernels",
                allow_module_level = True)

from unsloth_zoo.mlx.flash_attention import (  # noqa: E402
    flash_attention_available,
    flash_attention_supports,
    flash_causal_attention,
    install_flash_attention_backward,
)
from unsloth_zoo.mlx.utils import (  # noqa: E402
    acquire_mlx_training_patches,
    release_mlx_training_patches,
)

needs_nax = pytest.mark.skipif(
    not flash_attention_available(),
    reason = "needs a GPU with neural accelerators (M5+) on macOS 26.2+",
)


def _inputs(B, H, KV, S, D, seed = 0):
    mx.random.seed(seed)
    q = mx.random.normal((B, H, S, D))
    k = mx.random.normal((B, KV, S, D))
    v = mx.random.normal((B, KV, S, D))
    cotangent = mx.random.normal((B, H, S, D))
    return q, k, v, cotangent


def _reference_fp32(q, k, v, scale):
    H, KV, S = q.shape[1], k.shape[1], q.shape[2]
    k = mx.repeat(k, H // KV, axis = 1)
    v = mx.repeat(v, H // KV, axis = 1)
    scores = (q @ k.swapaxes(-1, -2)) * scale
    causal = mx.tril(mx.ones((S, S), dtype = mx.bool_))
    return mx.softmax(mx.where(causal, scores, -mx.inf), axis = -1) @ v


def _mlx_fused(q, k, v, scale):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale = scale, mask = "causal")


def _output_and_grads(attention, q, k, v, cotangent, scale):
    def loss(q, k, v):
        out = attention(q, k, v, scale)
        return (out.astype(mx.float32) * cotangent).sum(), out

    (_, out), grads = mx.value_and_grad(loss, argnums = (0, 1, 2))(q, k, v)
    mx.eval(out, grads)
    return out, grads


def _relative_l2(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return mx.sqrt(((a - b) ** 2).sum() / (b ** 2).sum()).item()


# (B, H, KV, S, D): GQA and MHA, every supported head dim, and sequence
# lengths that are and aren't a multiple of the 64-row block.
SHAPES = [
    (2, 14, 2, 511, 64),
    (1, 12, 2, 1024, 128),
    (1, 32, 32, 384, 96),
    (2, 8, 4, 130, 64),
    (1, 16, 16, 1000, 128),
]


@needs_nax
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("B,H,KV,S,D", SHAPES)
def test_gradients_are_at_least_as_accurate_as_mlx_own_path(B, H, KV, S, D, dtype):
    q, k, v, cotangent = _inputs(B, H, KV, S, D)
    scale = D ** -0.5
    _, exact = _output_and_grads(_reference_fp32, q, k, v, cotangent, scale)
    low = [x.astype(dtype) for x in (q, k, v)]
    _, flash = _output_and_grads(flash_causal_attention, *low, cotangent, scale)
    _, fused = _output_and_grads(_mlx_fused, *low, cotangent, scale)
    for name, ours, theirs, truth in zip("qkv", flash, fused, exact):
        ours_error = _relative_l2(ours, truth)
        theirs_error = _relative_l2(theirs, truth)
        # Measured: flash sits at 0.6-0.9x of MLX's own error in bf16/fp16.
        assert ours_error <= 1.1 * theirs_error, (name, ours_error, theirs_error)


@needs_nax
@pytest.mark.parametrize("B,H,KV,S,D", SHAPES)
def test_forward_matches_mlx_fused_forward(B, H, KV, S, D):
    q, k, v, _ = _inputs(B, H, KV, S, D)
    q, k, v = (x.astype(mx.bfloat16) for x in (q, k, v))
    scale = D ** -0.5
    ours = flash_causal_attention(q, k, v, scale)
    theirs = _mlx_fused(q, k, v, scale)
    # Same NAX algorithm; padded lengths reorder nothing within a real row.
    assert mx.max(mx.abs(ours.astype(mx.float32) - theirs.astype(mx.float32))).item() < 1e-2


@needs_nax
def test_gradients_are_deterministic():
    q, k, v, cotangent = _inputs(1, 12, 2, 700, 128)
    q, k, v = (x.astype(mx.bfloat16) for x in (q, k, v))
    _, first = _output_and_grads(flash_causal_attention, q, k, v, cotangent, 0.088)
    _, second = _output_and_grads(flash_causal_attention, q, k, v, cotangent, 0.088)
    for a, b in zip(first, second):
        assert mx.array_equal(a, b).item()


@needs_nax
def test_backward_does_not_materialize_the_attention_matrix():
    # S=4096, H=12: one fp32 S x S matrix per head is 805 MB.
    q, k, v, cotangent = _inputs(1, 12, 2, 4096, 128)
    q, k, v = (x.astype(mx.bfloat16) for x in (q, k, v))
    mx.synchronize()
    mx.clear_cache()
    base = mx.get_active_memory()
    mx.reset_peak_memory()
    _output_and_grads(flash_causal_attention, q, k, v, cotangent, 0.088)
    peak = mx.get_peak_memory() - base
    assert peak < 400 * 2**20, peak


def _array(shape, dtype = mx.bfloat16):
    return mx.zeros(shape, dtype = dtype)


@pytest.mark.parametrize("q_shape,k_shape,dtype,mask,kwargs,supported", [
    ((1, 8, 256, 128), (1, 2, 256, 128), mx.bfloat16, "causal", {}, True),
    ((1, 8, 256, 128), (1, 2, 256, 128), mx.bfloat16, "causal", {"sinks": None}, True),
    ((1, 8, 256, 128), (1, 2, 256, 128), mx.bfloat16, None, {}, False),         # not causal
    ((1, 8, 256, 128), (1, 2, 256, 128), mx.bfloat16, "causal", {"sinks": 1}, False),
    ((1, 8, 1, 128), (1, 2, 256, 128), mx.bfloat16, "causal", {}, False),       # decode step
    ((1, 8, 64, 128), (1, 2, 64, 128), mx.bfloat16, "causal", {}, False),       # too short
    ((1, 8, 256, 256), (1, 2, 256, 256), mx.bfloat16, "causal", {}, False),     # head dim
    ((1, 8, 256, 128), (1, 2, 256, 128), mx.float32, "causal", {}, False),      # fp32
    ((1, 8, 256, 128), (1, 3, 256, 128), mx.bfloat16, "causal", {}, False),     # 8 % 3
])
def test_supports_only_causal_self_attention_it_is_validated_for(q_shape, k_shape, dtype,
                                                                  mask, kwargs, supported):
    q, k = _array(q_shape, dtype), _array(k_shape, dtype)
    if isinstance(kwargs.get("sinks"), int):
        kwargs = {"sinks": _array((q_shape[1],))}
    assert flash_attention_supports(q, k, k, mask, kwargs) is supported


@needs_nax
def test_wrapper_routes_to_flash_only_inside_a_training_run(monkeypatch):
    import unsloth_zoo.mlx.flash_attention as flash_attention

    install_flash_attention_backward()
    calls = []
    real = flash_attention.flash_causal_attention

    def counting(*args):
        calls.append(args[0].shape)
        return real(*args)

    monkeypatch.setattr(flash_attention, "flash_causal_attention", counting)
    q, k, v, _ = _inputs(1, 8, 2, 256, 128)
    q, k, v = (x.astype(mx.bfloat16) for x in (q, k, v))

    mx.eval(_mlx_fused(q, k, v, 0.088))
    assert calls == []
    acquire_mlx_training_patches()
    try:
        mx.eval(_mlx_fused(q, k, v, 0.088))
    finally:
        release_mlx_training_patches()
    assert calls == [q.shape]
