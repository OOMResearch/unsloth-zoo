"""HF Mistral3 FP8 is sanitized to BF16, then optionally quantized by MLX."""

import pytest

pytest.importorskip("mlx.core")
from unsloth_zoo.mlx.loader import (
    _MLXQuantizationSpec,
    _ensure_quantization_compatible,
    _get_existing_mlx_quantization,
)


def test_mistral3_fp8_can_be_quantized_after_sanitizing():
    config = {
        "model_type": "mistral3",
        "quantization_config": {
            "activation_scheme": "static",
            "dequantize": False,
            "modules_to_not_convert": ["model.vision_tower", "lm_head"],
            "quant_method": "fp8",
            "weight_block_size": None,
        },
    }
    spec = _MLXQuantizationSpec(enabled=True, bits=4, group_size=64)
    assert _get_existing_mlx_quantization(config) is None
    assert _ensure_quantization_compatible(config, spec, "Ministral") == "none"
    assert config["quantization_config"]["quant_method"] == "fp8"


@pytest.mark.parametrize("key", ["quantization", "quantization_config"])
def test_mistral3_mlx_quantization_is_preserved(key):
    quantization = {
        "language_model.model.layers.0.self_attn.q_proj": {"bits": 4, "group_size": 64}
    }
    config = {"model_type": "mistral3", key: quantization}
    assert _get_existing_mlx_quantization(config) == quantization
    with pytest.raises(ValueError, match="per-module MLX"):
        _ensure_quantization_compatible(
            config,
            _MLXQuantizationSpec(enabled=True, bits=4, group_size=64),
            "Ministral",
        )


def test_native_mlx_metadata_wins_over_hf_fp8():
    quantization = {"bits": 4, "group_size": 64}
    config = {
        "model_type": "mistral3",
        "quantization": quantization,
        "quantization_config": {"quant_method": "fp8"},
    }
    assert _get_existing_mlx_quantization(config) == quantization


@pytest.mark.parametrize("method", ["bitsandbytes", "compressed-tensors"])
def test_other_hf_formats_still_reach_existing_guards(method):
    metadata = {"quant_method": method}
    assert (
        _get_existing_mlx_quantization(
            {"model_type": "mistral3", "quantization_config": metadata}
        )
        == metadata
    )


def test_other_architectures_keep_fp8_metadata():
    metadata = {"quant_method": "fp8"}
    assert (
        _get_existing_mlx_quantization(
            {"model_type": "deepseek_v4", "quantization_config": metadata}
        )
        == metadata
    )
