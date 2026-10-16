"""Real save/reload regression for selectively quantized QLoRA, without downloads."""

import copy

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.llama import Model, ModelArgs
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.utils import load_model

from unsloth_zoo.mlx.loader import _apply_mlx_quantization, _MLXQuantizationSpec
from unsloth_zoo.mlx.utils import save_pretrained_merged


class _Tokenizer:
    def save_pretrained(self, path):
        pass


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("mixed", [False, True])
def test_selective_qlora_merged_save_reloads(tmp_path, bits, mixed):
    mx.random.seed(0)
    config = {
        "model_type": "llama",
        "hidden_size": 64,
        "num_hidden_layers": 1,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 64,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
    }
    model = Model(ModelArgs.from_dict(config))
    q_path = "model.layers.0.self_attn.q_proj"
    spec = _MLXQuantizationSpec(
        enabled=True,
        bits=bits,
        group_size=64,
        mode="affine",
        quantize_modules=[q_path],
    )
    model, _ = _apply_mlx_quantization(model, config, spec, is_vlm=False)
    layer = model.model.layers[0].self_attn
    if mixed:
        from mlx.nn import QuantizedLinear

        layer.v_proj = QuantizedLinear.from_linear(layer.v_proj, group_size=32, bits=4)
        model._config["quantization"]["model.layers.0.self_attn.v_proj"] = {
            "bits": 4,
            "group_size": 32,
            "mode": "affine",
        }
    layer.q_proj = LoRALinear.from_base(layer.q_proj, r=4, scale=1.0)
    layer.q_proj.lora_b = mx.random.normal(layer.q_proj.lora_b.shape) * 0.01
    original_config = copy.deepcopy(model._config)
    for method in ("merged_4bit", "merged_4bit", "merged_16bit"):
        directory = tmp_path / f"{method}-{len(list(tmp_path.iterdir()))}"
        save_pretrained_merged(model, _Tokenizer(), directory, save_method=method)
        # Compare against the exact fused artifact, including requantization.
        expected = model(mx.array([[1, 2, 3]]))
        reloaded, saved_config = load_model(directory)
        actual = reloaded(mx.array([[1, 2, 3]]))
        mx.eval(actual, expected)
        assert bool(mx.all(mx.isfinite(actual)))
        assert bool(mx.allclose(actual, expected, atol=1e-5))
        if method == "merged_4bit":
            quant = saved_config["quantization"]
            assert quant["bits"] == bits
            assert quant["group_size"] == 64
            assert reloaded.model.layers[0].self_attn.q_proj.bits == bits
            if mixed:
                assert reloaded.model.layers[0].self_attn.v_proj.group_size == 32
            assert model._config == original_config
        else:
            assert "quantization" not in saved_config
