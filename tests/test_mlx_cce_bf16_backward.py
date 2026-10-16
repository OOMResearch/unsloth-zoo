"""BF16 frozen heads retain native CE derivatives with bounded token blocks."""

import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(
    not hasattr(mx, "metal") or not mx.metal.is_available(), reason="requires Metal"
)


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("precompute", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("rows", [0, 31, 129, 257])
def test_bf16_frozen_backward_matches_native_and_bounds_blocks(
    monkeypatch, quantized, precompute, compiled, rows
):
    from mlx import nn
    from unsloth_zoo.mlx.cce import runtime_cce as rt

    mx.random.seed(731)
    hidden = (mx.random.normal((rows, 64)) * 0.2).astype(mx.bfloat16)
    dense_weight = (mx.random.normal((4099, 64)) * 0.1).astype(mx.bfloat16)
    if quantized:
        weight, scales, biases = mx.quantize(dense_weight, group_size=64, bits=4)
    else:
        weight, scales, biases = dense_weight, None, None
    targets = (mx.arange(rows) * 19) % 4099
    targets = mx.where(mx.arange(rows) == 5, -100, targets)
    # This invalid wide label must not narrow to the ignored label.
    targets = mx.where(mx.arange(rows) == 7, (1 << 32) - 100, targets.astype(mx.int64))
    cotangent = mx.linspace(-0.7, 0.9, rows)
    calls = []
    matmul = rt._chunk_matmul

    def record(h, w, **kwargs):
        calls.append((h.shape[0], w.shape[0]))
        return matmul(h, w, **kwargs)

    monkeypatch.setattr(rt, "_chunk_matmul", record)
    runtime, _ = rt.make_chunked_cross_entropy_loss(
        chunk_size=1024,
        quantized=quantized,
        weight_is_frozen=not quantized,
        group_size=64 if quantized else None,
        bits=4 if quantized else None,
        precompute_hidden_gradient=precompute,
    )

    args = (weight, scales, biases, targets) if quantized else (weight, targets)
    native_calls = []
    cross_entropy = nn.losses.cross_entropy

    def native_ce(*args, **kwargs):
        native_calls.append(1)
        return cross_entropy(*args, **kwargs)

    monkeypatch.setattr(nn.losses, "cross_entropy", native_ce)
    mx.eval(runtime(hidden, *args))
    assert not native_calls  # Native CE is used only by the backward.

    def candidate(h):
        return (runtime(h, *args) * cotangent).sum()

    valid = (targets >= 0) & (targets < 4099)
    safe = mx.where(valid, targets, 0).astype(mx.int32)

    def reference(h):
        logits = (
            mx.quantized_matmul(h, weight, scales, biases, group_size=64, bits=4)
            if quantized
            else h @ weight.T
        )
        loss = nn.losses.cross_entropy(logits, safe).astype(mx.float32)
        return (loss * mx.where(valid, cotangent, 0.0)).sum()

    candidate_grad = mx.grad(candidate)
    reference_grad = mx.grad(reference)
    if compiled:
        candidate_grad = mx.compile(candidate_grad)
        reference_grad = mx.compile(reference_grad)
    actual = candidate_grad(hidden)
    if rows == 0:
        mx.eval(actual)
        assert actual.shape == hidden.shape
        assert not [n for n, vocab in calls if vocab == 4099]
        return
    expected = reference_grad(hidden)
    expected = mx.where((mx.arange(rows) == 7)[:, None], float("nan"), expected)
    mx.eval(actual, expected)
    assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-2, equal_nan=True).item()
    assert mx.all(actual[5] == 0).item()
    blocks = [n for n, vocab in calls if vocab == 4099]
    assert blocks
    assert all(n <= 128 for n in blocks)
    if rows > 128:
        assert set(blocks) == {128}
