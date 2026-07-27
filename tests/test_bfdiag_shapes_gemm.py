"""Tests for bfdiag.shapes.gemm: dense GEMM M/N/K shapes, cross-checked
against the real safetensors weight shapes on this machine (read via
safetensors' header metadata, not materialized) -- see
notes/2026-07-27-bfdiag-shape-derivation.md for how those numbers were
obtained.
"""

from __future__ import annotations

from bfdiag.shapes.gemm import (
    draft_attention_gemms,
    draft_dense_gemms,
    draft_fc_gemm,
    draft_qkv_proj_gemm,
    target_dense_gemms,
)
from bfdiag.shapes.model import load_draft_config, load_laguna_config


def test_target_attention_proj_gemms_match_real_weight_shapes():
    """Real weights on this machine:
    layer 0 (full):    q_proj=[6144,3072] k/v_proj=[1024,3072] o_proj=[3072,6144] g_proj=[48,3072]
    layer 1 (sliding):  q_proj=[9216,3072] k/v_proj=[1024,3072] o_proj=[3072,9216] g_proj=[72,3072]
    """
    config = load_laguna_config()
    gemms = {g.name: g for g in target_dense_gemms(config, num_tokens=7)}

    assert gemms["full.q_proj"].weight_shape == (6144, 3072)
    assert gemms["full.k_proj"].weight_shape == (1024, 3072)
    assert gemms["full.v_proj"].weight_shape == (1024, 3072)
    assert gemms["full.o_proj"].weight_shape == (3072, 6144)
    assert gemms["full.g_proj"].weight_shape == (48, 3072)

    assert gemms["sliding.q_proj"].weight_shape == (9216, 3072)
    assert gemms["sliding.k_proj"].weight_shape == (1024, 3072)
    assert gemms["sliding.v_proj"].weight_shape == (1024, 3072)
    assert gemms["sliding.o_proj"].weight_shape == (3072, 9216)
    assert gemms["sliding.g_proj"].weight_shape == (72, 3072)

    for g in gemms.values():
        assert g.m == 7


def test_target_dense_mlp_and_moe_gemms_match_real_weight_shapes():
    """layer 0 dense mlp: gate/up=[12288,3072] down=[3072,12288]
    router gate: [256,3072]; shared_expert: gate/up=[1024,3072] down=[3072,1024]
    lm_head: [100352,3072]"""
    config = load_laguna_config()
    gemms = {g.name: g for g in target_dense_gemms(config, num_tokens=1)}

    assert gemms["layer0.dense_mlp.gate_proj"].weight_shape == (12288, 3072)
    assert gemms["layer0.dense_mlp.up_proj"].weight_shape == (12288, 3072)
    assert gemms["layer0.dense_mlp.down_proj"].weight_shape == (3072, 12288)

    assert gemms["moe.router_gate"].weight_shape == (256, 3072)

    assert gemms["moe_layer.shared_expert.gate_proj"].weight_shape == (1024, 3072)
    assert gemms["moe_layer.shared_expert.up_proj"].weight_shape == (1024, 3072)
    assert gemms["moe_layer.shared_expert.down_proj"].weight_shape == (3072, 1024)

    assert gemms["lm_head"].weight_shape == (100352, 3072)


def test_draft_qkv_proj_is_fused_and_matches_real_weight_shape():
    """Real draft checkpoint: self_attn.qkv_proj.weight = [11264, 3072]
    (72*128 q + 8*128 k + 8*128 v = 9216 + 1024 + 1024), a real architecture
    difference from the target model's separate q/k/v_proj."""
    draft = load_draft_config()
    gemm = draft_qkv_proj_gemm(draft, num_tokens=3)
    assert gemm.weight_shape == (11264, 3072)
    assert gemm.n == 9216 + 1024 + 1024
    assert gemm.m == 3


def test_draft_attention_gemms_match_real_weight_shapes():
    draft = load_draft_config()
    gemms = {g.name: g for g in draft_attention_gemms(draft, num_tokens=1)}
    assert gemms["draft.qkv_proj"].weight_shape == (11264, 3072)
    assert gemms["draft.o_proj"].weight_shape == (3072, 9216)
    assert gemms["draft.g_proj"].weight_shape == (72, 3072)


def test_draft_fc_gemm_matches_real_weight_shape():
    """Real draft checkpoint: fc.weight = [3072, 18432] = [hidden, 6*hidden]
    (EAGLE-style fusion of the 6 aux hidden states)."""
    draft = load_draft_config()
    gemm = draft_fc_gemm(draft, num_tokens=1)
    assert gemm.weight_shape == (3072, 18432)
    assert gemm.k == 6 * 3072


def test_draft_dense_gemms_has_no_lm_head():
    """The draft checkpoint has no lm_head/embed_tokens tensor -- it reuses
    the target model's tied lm_head. draft_dense_gemms must not invent one."""
    draft = load_draft_config()
    gemms = {g.name for g in draft_dense_gemms(draft, num_tokens=1)}
    assert not any("lm_head" in name for name in gemms)
    assert "draft.mlp.gate_proj" in gemms
    assert "draft.mlp.down_proj" in gemms


def test_gemm_shape_x_weight_y_shapes():
    from bfdiag.shapes.gemm import GemmShape

    g = GemmShape("test", m=4, n=8, k=16)
    shapes = g.shapes()
    assert shapes["x"] == (4, 16)
    assert shapes["weight"] == (8, 16)
    assert shapes["y"] == (4, 8)
