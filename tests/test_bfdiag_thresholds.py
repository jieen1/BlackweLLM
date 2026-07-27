"""Unit tests for bfdiag.divergence.thresholds's composite/depth-relaxed bars."""

from __future__ import annotations

import pytest

from bfdiag.divergence.thresholds import (
    ATTN_OUT,
    ENV_THRESHOLD_OVERRIDE,
    HIDDEN_STATE,
    INPUT_LAYERNORM,
    MOE_OUT,
    POST_ATTENTION_LAYERNORM,
    ROUTER_LOGITS,
    kind_for_submodule,
    threshold_for,
)


@pytest.mark.parametrize(
    ("name", "expected_kind"),
    [
        ("self_attn", ATTN_OUT),
        ("model.layers.5.self_attn", ATTN_OUT),
        ("mlp", MOE_OUT),
        ("model.layers.12.mlp", MOE_OUT),
        ("mlp.experts", MOE_OUT),
        ("mlp.shared_expert", MOE_OUT),
        ("mlp.gate", ROUTER_LOGITS),
        ("model.layers.3.mlp.gate", ROUTER_LOGITS),
        ("input_layernorm", INPUT_LAYERNORM),
        ("model.layers.0.input_layernorm", INPUT_LAYERNORM),
        ("post_attention_layernorm", POST_ATTENTION_LAYERNORM),
        ("hidden_state", HIDDEN_STATE),
    ],
)
def test_kind_for_submodule_matches_real_module_names(name: str, expected_kind: str) -> None:
    assert kind_for_submodule(name) == expected_kind


def test_kind_for_submodule_falls_back_to_name_for_unknown_suffix() -> None:
    assert kind_for_submodule("some_custom_probe") == "some_custom_probe"


def test_threshold_relaxes_monotonically_with_depth() -> None:
    shallow = threshold_for(ATTN_OUT, 0)
    mid = threshold_for(ATTN_OUT, 10)
    deep = threshold_for(ATTN_OUT, 47)
    assert shallow.min_cosine >= mid.min_cosine >= deep.min_cosine
    assert shallow.max_rel_abs_error <= mid.max_rel_abs_error <= deep.max_rel_abs_error
    assert shallow.min_top1_agreement >= mid.min_top1_agreement >= deep.min_top1_agreement


def test_layer_zero_matches_the_documented_base_floor() -> None:
    # At layer 0, growth == 1.0, so the composite threshold should equal the
    # kind's own layer-0 floor exactly (see the _BASE_THRESHOLDS table).
    threshold = threshold_for(INPUT_LAYERNORM, 0)
    assert threshold.min_cosine == pytest.approx(0.999999)
    assert threshold.min_top1_agreement == pytest.approx(1.0)


def test_relaxation_growth_is_capped_at_deep_layers() -> None:
    # Growth saturates well before layer ~200; thresholds should stop
    # loosening once the cap is hit (no runaway relaxation hiding real bugs).
    far = threshold_for(MOE_OUT, 200)
    farther = threshold_for(MOE_OUT, 5000)
    assert far == farther


def test_moe_out_never_relaxes_tighter_than_attn_out() -> None:
    # MoE has a real, evidence-backed looser baseline (NVFP4 quantization
    # noise) than attention; the depth model must preserve that ordering at
    # every depth, not just at layer 0.
    for layer_idx in (0, 1, 17, 47):
        assert threshold_for(MOE_OUT, layer_idx).min_cosine <= threshold_for(
            ATTN_OUT, layer_idx
        ).min_cosine


def test_env_override_forces_min_cosine_uniformly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_THRESHOLD_OVERRIDE, "0.42")
    for kind in (ATTN_OUT, MOE_OUT, ROUTER_LOGITS, INPUT_LAYERNORM):
        for layer_idx in (0, 17, 47):
            assert threshold_for(kind, layer_idx).min_cosine == pytest.approx(0.42)


def test_no_env_override_by_default() -> None:
    assert threshold_for(ATTN_OUT, 0).min_cosine == pytest.approx(0.9999)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
