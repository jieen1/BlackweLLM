"""``compressed-tensors`` is a container; its formats are not interchangeable.

Selecting a loader on ``quant_method`` alone let an asymmetric group-wise INT4
checkpoint resolve to the same adapter as an FP8/NVFP4 one. Measured across the
six local Qwen3.6 checkpoints (2026-08-02), the per-layer tensors differ:

    mixed-precision   weight_packed · weight_scale · weight_global_scale
    pack-quantized    weight_packed · weight_scale · weight_zero_point · weight_shape

``pack-quantized`` carries ``num_bits=4, type=int, group_size=32`` and a
**zero point**. Nothing in this runtime models one.

The failure mode this guards is not a crash. Every tensor the loader looks for
is present, so ``assert_all_params_loaded`` would pass and the weights would
come back dequantized as if symmetric -- the shape of defect this repo has hit
repeatedly, where something resolves, loads, runs, and is quietly wrong.

Tests run off synthetic configs rather than the local checkpoints, so they hold
on a machine with an empty HuggingFace cache.
"""

from __future__ import annotations

import pytest

from runtime.model_registry import (
    LOADER_FOR_QUANT_METHOD,
    SUPPORTED_QUANT_FORMATS,
    UnsupportedArchitectureError,
    resolve_config,
)


def _config(quant_method: str, quant_format: str | None) -> dict:
    """A minimal Qwen3.6-shaped config carrying the quantization under test."""
    quantization: dict = {"quant_method": quant_method}
    if quant_format is not None:
        quantization["format"] = quant_format
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "hidden_size": 5120,
        "num_hidden_layers": 4,
        "layer_types": ["linear_attention", "full_attention"] * 2,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "intermediate_size": 17408,
        "vocab_size": 248320,
        "rope_theta": 10_000_000,
        "partial_rotary_factor": 0.25,
        "quantization_config": quantization,
    }


class TestFormatIsPartOfTheGate:
    def test_pack_quantized_is_refused(self):
        """Asymmetric INT4 must not reach a loader that cannot see its zero point."""
        with pytest.raises(UnsupportedArchitectureError) as excinfo:
            resolve_config(_config("compressed-tensors", "pack-quantized"))
        message = str(excinfo.value)
        assert "pack-quantized" in message
        assert "zero point" in message, (
            "the refusal should say why, not just that it was refused -- the "
            "next person needs to know this is about asymmetric quantization"
        )

    def test_supported_format_still_resolves(self):
        resolution = resolve_config(_config("compressed-tensors", "mixed-precision"))
        assert resolution.loader == "compressed_tensors"

    def test_method_without_subformat_still_resolves(self):
        resolution = resolve_config(_config("modelopt", None))
        assert resolution.loader == "modelopt"


class TestGateIsNotVacuous:
    def test_every_supported_method_declares_its_formats(self):
        """A method reachable through the loader map must appear in the format map.

        Without this, adding a loader for a new method silently reopens the
        original hole: the format check falls back to a default and stops
        discriminating.
        """
        missing = sorted(set(LOADER_FOR_QUANT_METHOD) - set(SUPPORTED_QUANT_FORMATS))
        assert not missing, (
            f"{missing} can select a loader but declare no allowed formats, so "
            "any format would be accepted for them"
        )

    def test_an_unknown_format_is_refused_not_defaulted(self):
        """An invented format must be rejected, not waved through."""
        with pytest.raises(UnsupportedArchitectureError):
            resolve_config(_config("compressed-tensors", "some-future-format"))
