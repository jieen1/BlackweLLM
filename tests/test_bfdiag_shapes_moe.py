"""Tests for bfdiag.shapes.moe: NVFP4-packed expert weight shapes, cross-
checked against the real safetensors header on this machine (shape/dtype
metadata only, no tensor data materialized) -- see
runtime/backends/laguna_sparkinfer_moe.py for the loading code this mirrors.
"""

from __future__ import annotations

import pytest

from bfdiag.shapes.model import load_laguna_config
from bfdiag.shapes.moe import (
    expert_projection_shapes,
    router_shapes,
    sparkinfer_w13_shapes,
    stacked_expert_shapes,
)


def test_expert_projection_shapes_match_real_checkpoint():
    """Real checkpoint (read directly, safetensors header only):
    gate_proj/up_proj: weight_packed=[1024,1536] weight_scale=[1024,192]
    down_proj:         weight_packed=[3072, 512] weight_scale=[3072, 64]
    """
    config = load_laguna_config()
    projs = expert_projection_shapes(config)

    assert projs["gate_proj"].weight_packed_shape == (1024, 1536)
    assert projs["gate_proj"].weight_scale_shape == (1024, 192)
    assert projs["up_proj"].weight_packed_shape == (1024, 1536)
    assert projs["up_proj"].weight_scale_shape == (1024, 192)
    assert projs["down_proj"].weight_packed_shape == (3072, 512)
    assert projs["down_proj"].weight_scale_shape == (3072, 64)


def test_packing_halves_in_features_and_scale_divides_by_group_size():
    config = load_laguna_config()
    projs = expert_projection_shapes(config)
    for proj in projs.values():
        assert proj.weight_packed_shape[1] == proj.in_features // 2
        assert proj.weight_scale_shape[1] == proj.in_features // proj.group_size
        assert proj.weight_packed_shape[0] == proj.out_features


def test_stacked_expert_shapes_prepend_num_experts():
    config = load_laguna_config()
    stacked = stacked_expert_shapes(config)
    assert stacked["gate_proj.weight_packed"] == (256, 1024, 1536)
    assert stacked["gate_proj.weight_scale"] == (256, 1024, 192)
    assert stacked["down_proj.weight_packed"] == (256, 3072, 512)
    assert stacked["down_proj.weight_scale"] == (256, 3072, 64)


def test_sparkinfer_w13_fuses_gate_and_up():
    """prepare_sparkinfer_layer concatenates [up_w, gate_w] along dim=1 --
    the fused w13 out-dim doubles moe_intermediate_size."""
    config = load_laguna_config()
    w = sparkinfer_w13_shapes(config)
    assert w["w13_fp4"] == (256, 2 * 1024, 1536)
    assert w["w2_fp4"] == (256, 3072, 512)


def test_router_shapes():
    config = load_laguna_config()
    r = router_shapes(config, num_tokens=5)
    assert r["router_logits"] == (5, 256)
    assert r["topk_ids"] == (5, 10)
    assert r["topk_weights"] == (5, 10)


def test_expert_projection_shapes_requires_group_size(monkeypatch):
    config = load_laguna_config()
    broken = config.__class__(
        **{**config.__dict__, "nvfp4_group_size": None},
    )
    with pytest.raises(ValueError, match="group_size"):
        expert_projection_shapes(broken)


def test_expert_projection_shapes_requires_experts():
    """The DFlash draft model has num_experts=0 -- moe.py should refuse to
    derive expert shapes for it rather than silently returning something."""
    config = load_laguna_config()
    dense_only = config.__class__(**{**config.__dict__, "num_experts": 0})
    with pytest.raises(ValueError, match="num_experts=0"):
        expert_projection_shapes(dense_only)
