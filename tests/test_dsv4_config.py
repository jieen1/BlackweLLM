"""Dsv4Config: GGUF-KV-driven runner config for the DSV4 model graph."""

from __future__ import annotations

from pathlib import Path

import pytest

from loader.gguf_header import read_gguf_header
from runtime.architecture import UnsupportedArchitectureError
from runtime.model.dsv4_config import Dsv4Config, config_from_gguf_kv

REAL_GGUF = Path(
    "/home/bot/models/DeepSeek-V4-Flash-0731-GGUF/DeepSeek-V4-Flash-0731-IQ2_XS-Experts-Q8_0.gguf"
)

REAL_RATIOS = [0, 0] + [4, 128] * 20 + [4] + [0, 0, 0]


def kv_with(ratios=None, **overrides) -> dict:
    kv = {
        "deepseek4.block_count": 43,
        "deepseek4.context_length": 1048576,
        "deepseek4.embedding_length": 4096,
        "deepseek4.attention.layer_norm_rms_epsilon": 1e-6,
        "deepseek4.attention.head_count": 64,
        "deepseek4.attention.head_count_kv": 1,
        "deepseek4.attention.key_length": 512,
        "deepseek4.rope.dimension_count": 64,
        "deepseek4.attention.q_lora_rank": 1024,
        "deepseek4.attention.output_group_count": 8,
        "deepseek4.attention.output_lora_rank": 1024,
        "deepseek4.attention.sliding_window": 128,
        "deepseek4.attention.compress_ratios": ratios if ratios is not None else REAL_RATIOS,
        "deepseek4.attention.compress_rope_freq_base": 160000.0,
        "deepseek4.rope.freq_base": 10000.0,
        "deepseek4.rope.scaling.factor": 16.0,
        "deepseek4.rope.scaling.original_context_length": 65536,
        "deepseek4.rope.scaling.yarn_beta_fast": 32,
        "deepseek4.rope.scaling.yarn_beta_slow": 1,
        "deepseek4.attention.indexer.head_count": 64,
        "deepseek4.attention.indexer.key_length": 128,
        "deepseek4.attention.indexer.top_k": 512,
        "deepseek4.hyper_connection.count": 4,
        "deepseek4.hyper_connection.sinkhorn_iterations": 20,
        "deepseek4.hyper_connection.epsilon": 1e-6,
        "deepseek4.expert_count": 256,
        "deepseek4.expert_shared_count": 1,
        "deepseek4.expert_used_count": 6,
        "deepseek4.expert_feed_forward_length": 2048,
        "deepseek4.expert_weights_scale": 1.5,
        "deepseek4.hash_layer_count": 3,
        "tokenizer.ggml.tokens": ["x"] * 129280,
    }
    kv.update(overrides)
    return kv


def test_config_reads_the_verified_facts() -> None:
    config = config_from_gguf_kv(kv_with())
    assert config.vocab_size == 129280
    assert config.hidden_size == 4096
    assert config.num_layers == 43
    assert config.num_heads == 64 and config.head_dim == 512
    assert config.nope_dim == 448
    assert config.rope_head_dim == 64
    assert config.q_lora_rank == 1024
    assert config.o_groups == 8 and config.o_lora_rank == 1024
    assert config.window_size == 128
    assert config.rope_theta == 10000.0 and config.rope_factor == 16.0
    assert config.compress_rope_theta == 160000.0
    assert config.index_n_heads == 64 and config.index_head_dim == 128
    assert config.index_topk == 512
    assert config.hc_mult == 4 and config.hc_sinkhorn_iters == 20
    assert config.hc_dim == 16384 and config.hc_mix_dim == 24
    assert config.n_routed_experts == 256 and config.n_activated_experts == 6
    assert config.moe_intermediate_size == 2048
    assert config.route_scale == 1.5 and config.swiglu_limit == 10.0
    assert config.n_hash_layers == 3 and config.hash_layer_ids == (0, 1, 2)


def test_layer_geometry_matches_the_verified_layout() -> None:
    config = config_from_gguf_kv(kv_with())
    assert not config.has_compressor(0) and not config.has_compressor(1)
    assert config.layer_ratio(0) == 0 and config.layer_ratio(1) == 0
    for layer_id in range(2, 43):
        expected = 4 if layer_id % 2 == 0 else 128
        assert config.layer_ratio(layer_id) == expected
        assert config.has_compressor(layer_id)
        assert config.has_indexer(layer_id) == (expected == 4)
        assert config.compressor_coeff(layer_id) == (2 if expected == 4 else 1)
    # the three trailing zeros belong to MTP stages and must be cut off
    assert len(config.compress_ratios) == 43


def test_config_rejects_bad_ratios() -> None:
    ratios = REAL_RATIOS.copy()
    ratios[10] = 8
    with pytest.raises(UnsupportedArchitectureError, match="compress_ratio 8"):
        config_from_gguf_kv(kv_with(ratios=ratios))
    with pytest.raises(UnsupportedArchitectureError, match="shorter"):
        config_from_gguf_kv(kv_with(ratios=[0, 4]))


@pytest.mark.skipif(not REAL_GGUF.exists(), reason="GGUF download not present")
def test_config_from_real_gguf_header() -> None:
    header = read_gguf_header(REAL_GGUF)
    config = config_from_gguf_kv(header.kv)
    assert config.num_layers == 43
    assert config.vocab_size == 129280
    assert config.compress_ratios == tuple([0, 0] + [4, 128] * 20 + [4])
    assert config.norm_eps == pytest.approx(1e-6)
    # sanity: defaults survive where the file carries no surprises
    assert Dsv4Config(compress_ratios=config.compress_ratios).hc_mult == 4
