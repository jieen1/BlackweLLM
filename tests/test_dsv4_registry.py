"""DSV4-Flash GGUF resolution: architecture producer + registry wiring.

Hermetic synthetic tests run everywhere; real-file tests skip unless the
82 GiB download is present.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from runtime.architecture import (
    UnsupportedArchitectureError,
    parse_dsv4_gguf_architecture,
)
from runtime.model_registry import (
    IMPLEMENTED_BACKENDS,
    SUPPORTED_QUANT_FORMATS,
    _loader_for,
    resolve_checkpoint,
)

REAL_GGUF = Path(
    "/home/bot/models/DeepSeek-V4-Flash-0731-GGUF/DeepSeek-V4-Flash-0731-IQ2_XS-Experts-Q8_0.gguf"
)

REAL_RATIOS = [0, 0] + [4, 128] * 20 + [4] + [0, 0, 0]  # 43 main + 3 MTP


def synthetic_kv(block_count: int = 43, ratios: list[int] | None = None) -> dict:
    return {
        "general.architecture": "deepseek4",
        "deepseek4.block_count": block_count,
        "deepseek4.context_length": 1048576,
        "deepseek4.embedding_length": 4096,
        "deepseek4.attention.head_count": 64,
        "deepseek4.attention.head_count_kv": 1,
        "deepseek4.attention.key_length": 512,
        "deepseek4.attention.sliding_window": 128,
        "deepseek4.attention.q_lora_rank": 1024,
        "deepseek4.attention.compress_ratios": ratios if ratios is not None else REAL_RATIOS,
        "deepseek4.attention.compress_rope_freq_base": 160000.0,
        "deepseek4.rope.freq_base": 10000.0,
        "deepseek4.rope.scaling.type": "yarn",
        "deepseek4.rope.scaling.factor": 16.0,
        "deepseek4.rope.scaling.original_context_length": 65536,
        "deepseek4.expert_count": 256,
        "deepseek4.expert_used_count": 6,
        "deepseek4.expert_feed_forward_length": 2048,
        "deepseek4.expert_shared_count": 1,
        "tokenizer.ggml.tokens": [f"tok{i}" for i in range(129280)],
    }


TYPES_IQ2_Q8 = frozenset({"IQ2_XS", "Q8_0", "F32", "BF16", "I32"})


def test_producer_layer_composition() -> None:
    spec = parse_dsv4_gguf_architecture(synthetic_kv(), tensor_type_names=TYPES_IQ2_Q8)
    assert spec.architecture == "DeepseekV4ForCausalLM"
    assert spec.model_type == "deepseek4"
    assert spec.num_hidden_layers == 43
    assert spec.hidden_size == 4096
    assert spec.vocab_size == 129280
    assert spec.max_position_embeddings == 1048576
    assert spec.num_attention_heads == 64
    assert spec.num_key_value_heads == 1
    assert spec.head_dim == 512
    assert spec.sliding_window == 128
    # measured layout: layers 0,1 pure window; even 2..42 ratio-4 (CSA);
    # odd 3..41 ratio-128 (HCA)
    assert spec.count_attention("sliding_attention") == 2
    assert spec.count_attention("csa_attention") == 21
    assert spec.count_attention("hca_attention") == 20
    assert spec.layers[0].attention == "sliding_attention"
    assert spec.layers[2].attention == "csa_attention"
    assert spec.layers[3].attention == "hca_attention"
    assert spec.layers[40].attention == "csa_attention"
    assert spec.layers[41].attention == "hca_attention"
    assert spec.layers[42].attention == "csa_attention"
    assert spec.paged_kv_layers == tuple(range(43))
    assert spec.recurrent_layers == ()
    assert not spec.needs_two_cache_families
    assert spec.is_moe and spec.moe is not None
    assert spec.moe.num_experts == 256 and spec.moe.top_k == 6
    assert spec.moe.intermediate_size == 2048
    assert not spec.has_mtp
    assert not spec.has_vision_tower


def test_producer_rope_and_quant() -> None:
    spec = parse_dsv4_gguf_architecture(synthetic_kv(), tensor_type_names=TYPES_IQ2_Q8)
    assert set(spec.rope) == {"default", "compressed"}
    assert spec.rope["default"].theta == 10000.0
    assert spec.rope["default"].rope_type == "yarn"
    assert spec.rope["default"].factor == 16.0
    assert spec.rope["default"].original_max_position_embeddings == 65536
    assert spec.rope["compressed"].theta == 160000.0
    assert spec.quant.method == "gguf"
    assert spec.quant.format == "iq2_xs+q8_0"
    assert spec.quant.format in SUPPORTED_QUANT_FORMATS["gguf"]


def test_producer_rejects_unknown_ratio() -> None:
    ratios = REAL_RATIOS.copy()
    ratios[5] = 7
    with pytest.raises(UnsupportedArchitectureError, match="compress_ratio 7"):
        parse_dsv4_gguf_architecture(synthetic_kv(ratios=ratios), tensor_type_names=TYPES_IQ2_Q8)


def test_producer_rejects_short_ratios() -> None:
    with pytest.raises(UnsupportedArchitectureError, match="shorter"):
        parse_dsv4_gguf_architecture(synthetic_kv(ratios=[0, 4]), tensor_type_names=TYPES_IQ2_Q8)


def test_loader_gate_rejects_other_gguf_quant_mixes() -> None:
    # IQ3_XXS is a real sibling quant of the same model; a different point on
    # the size/quality curve is not silently interchangeable.
    spec = parse_dsv4_gguf_architecture(
        synthetic_kv(), tensor_type_names=frozenset({"IQ2_XS", "IQ3_XXS", "Q8_0", "F32"})
    )
    assert spec.quant.format == "iq2_xs+q8_0+iq3_xxs"
    with pytest.raises(UnsupportedArchitectureError, match="no loader adapter"):
        _loader_for(spec)


# --- GGUF file-level resolution -------------------------------------------


def _kv_bytes(key: str) -> bytes:
    return struct.pack("<Q", len(key)) + key.encode()


def write_synthetic_deepseek4_gguf(path: Path, quant_types: tuple[int, ...]) -> None:
    """Header-only GGUF: deepseek4 KV + one tensor entry per given ggml type."""
    out = bytearray(b"GGUF" + struct.pack("<I", 3))
    out += struct.pack("<QQ", len(quant_types), 4)  # tensors, kv_count
    out += _kv_bytes("general.architecture") + struct.pack("<I", 8) + _kv_bytes("deepseek4")
    out += _kv_bytes("deepseek4.block_count") + struct.pack("<I", 4) + struct.pack("<I", 3)
    ratios = [0, 4, 128]
    out += _kv_bytes("deepseek4.attention.compress_ratios") + struct.pack("<I", 9)
    out += struct.pack("<I", 5) + struct.pack("<Q", len(ratios))
    out += struct.pack(f"<{len(ratios)}i", *ratios)
    out += _kv_bytes("tokenizer.ggml.tokens") + struct.pack("<I", 9)
    out += struct.pack("<I", 8) + struct.pack("<Q", 2)
    out += _kv_bytes("a") + _kv_bytes("b")
    for index, type_id in enumerate(quant_types):
        name = f"t{index}"
        out += struct.pack("<Q", len(name)) + name.encode()
        out += struct.pack("<I", 1) + struct.pack("<Q", 256)
        out += struct.pack("<IQ", type_id, index * 4096)
    path.write_bytes(bytes(out))


def test_resolve_synthetic_gguf_resolves_to_deepseek_v4(tmp_path: Path) -> None:
    target = tmp_path / "mini.gguf"
    # 17 = IQ2_XS, 8 = Q8_0, 0 = F32
    write_synthetic_deepseek4_gguf(target, quant_types=(17, 8, 0))
    assert "deepseek_v4" in IMPLEMENTED_BACKENDS
    resolution = resolve_checkpoint(target)
    assert resolution.backend == "deepseek_v4"
    assert resolution.spec.architecture == "DeepseekV4ForCausalLM"


def test_resolve_rejects_non_deepseek4_gguf(tmp_path: Path) -> None:
    target = tmp_path / "other.gguf"
    out = bytearray(b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 0, 1))
    out += _kv_bytes("general.architecture") + struct.pack("<I", 8) + _kv_bytes("llama")
    target.write_bytes(bytes(out))
    with pytest.raises(UnsupportedArchitectureError, match="only serves 'deepseek4'"):
        resolve_checkpoint(target)


def test_resolve_rejects_plain_weight_files(tmp_path: Path) -> None:
    target = tmp_path / "weights.safetensors"
    target.write_bytes(b"{}")
    with pytest.raises(UnsupportedArchitectureError, match="only .gguf files"):
        resolve_checkpoint(target)


@pytest.mark.skipif(not REAL_GGUF.exists(), reason="GGUF download not present")
def test_resolve_real_gguf_header() -> None:
    # Phase 4: deepseek_v4 is implemented; resolution must succeed with the
    # right identity and no speculative strategy (the main GGUF carries no
    # dspark/mtp tensors).
    resolution = resolve_checkpoint(REAL_GGUF)
    assert resolution.backend == "deepseek_v4"
    assert resolution.spec.architecture == "DeepseekV4ForCausalLM"
    assert resolution.speculative is None
