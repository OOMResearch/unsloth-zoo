"""Selective gradient checkpointing: `ratio` checkpoints an evenly spaced
fraction of the language layers and leaves gradients unchanged."""

import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="MLX is only available on Apple Silicon"
)


def _model(num_layers=4, dims=8):
    import mlx.core as mx
    import mlx.nn as nn

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(dims, dims)

        def __call__(self, x):
            return x + mx.tanh(self.proj(x))

    class LM(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [Block() for _ in range(num_layers)]

        def __call__(self, x):
            for layer in self.layers:
                x = layer(x)
            return x.sum()

    mx.random.seed(0)
    return LM(), mx.random.normal((2, dims))


def _grads(model, x):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    _, grads = nn.value_and_grad(model, lambda m, x: m(x))(model, x)
    mx.eval(grads)
    return dict(tree_flatten(grads))


@pytest.mark.parametrize(
    "num_layers, ratio, expected",
    [
        (4, 1.0, {0, 1, 2, 3}),
        (4, 0.5, {1, 3}),
        (28, 0.5, set(range(1, 28, 2))),
        (28, 0.25, {2, 6, 10, 14, 18, 22, 26}),
        (28, 0.01, {14}),
    ],
)
def test_checkpointed_indices_are_evenly_spaced(num_layers, ratio, expected):
    from unsloth_zoo.mlx.utils import _checkpointed_layer_indices

    assert _checkpointed_layer_indices(num_layers, ratio) == expected


@pytest.mark.parametrize("ratio", [0.0, -0.5, 1.5])
def test_ratio_outside_unit_interval_is_rejected(ratio):
    from unsloth_zoo.mlx.utils import _checkpointed_layer_indices

    with pytest.raises(ValueError, match="gradient_checkpointing_ratio"):
        _checkpointed_layer_indices(4, ratio)


@pytest.mark.parametrize("ratio, checkpointed_calls", [(1.0, 4), (0.5, 2), (0.25, 1)])
def test_ratio_checkpoints_that_many_layers_with_identical_gradients(
    monkeypatch, ratio, checkpointed_calls
):
    import mlx.core as mx
    from unsloth_zoo.mlx import utils

    model, x = _model()
    block_cls = type(model.layers[0])
    expected = _grads(model, x)

    calls = []
    real_checkpoint = mx.checkpoint
    monkeypatch.setattr(
        utils.mx, "checkpoint", lambda fn: calls.append(fn) or real_checkpoint(fn)
    )
    utils.apply_gradient_checkpointing(model, ratio=ratio)
    try:
        actual = _grads(model, x)
    finally:
        utils.remove_gradient_checkpointing(model)

    assert len(calls) == checkpointed_calls
    assert actual.keys() == expected.keys()
    for name, grad in expected.items():
        assert mx.allclose(actual[name], grad, rtol=1e-5, atol=1e-6).item(), name
    # Fully undone: no class patch, no per-layer flags, no stray parameters.
    assert not hasattr(block_cls, "_orig_call")
    assert all(utils._GC_SKIP_ATTR not in layer.__dict__ for layer in model.layers)
    assert all(utils._GC_SKIP_ATTR not in layer for layer in model.layers)


def test_reapplying_with_a_higher_ratio_clears_earlier_skips(monkeypatch):
    import mlx.core as mx
    from unsloth_zoo.mlx import utils

    model, x = _model()
    calls = []
    real_checkpoint = mx.checkpoint
    monkeypatch.setattr(
        utils.mx, "checkpoint", lambda fn: calls.append(fn) or real_checkpoint(fn)
    )
    utils.apply_gradient_checkpointing(model, ratio=0.25)
    utils.apply_gradient_checkpointing(model)
    try:
        _grads(model, x)
    finally:
        utils.remove_gradient_checkpointing(model)
    assert len(calls) == 4
