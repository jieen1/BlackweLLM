"""Tests for bfdiag.shapes.model: config parsing, layer grouping, hard errors
on missing/incomplete config.json.

Two kinds of fixtures are used on purpose:
- the REAL config.json on this machine (~/.cache/huggingface/hub/...) for the
  acceptance-criteria assertions the task calls out explicitly (12 full + 36
  sliding layers, full positions 0,4,...,44) -- these need to be true against
  the actual checkpoint, not a synthetic stand-in.
- a small synthetic config (written to tmp_path) for structural edge cases
  (uniform head-count fallback, missing fields, missing file) that shouldn't
  depend on exactly what's in the real checkpoint.
"""

from __future__ import annotations

import json

import pytest

from bfdiag.shapes.model import (
    DEFAULT_DRAFT_MODEL_ID,
    DEFAULT_MODEL_ID,
    LagunaConfigError,
    load_draft_config,
    load_laguna_config,
)

# ---------------------------------------------------------------------------
# Real checkpoint (must be present on this machine per the task brief)
# ---------------------------------------------------------------------------


@pytest.mark.requires_hf_snapshot(DEFAULT_MODEL_ID)
def test_real_config_layer_grouping():
    """Acceptance criterion #2: 12 full_attention + 36 sliding_attention,
    full layer positions exactly 0,4,8,...,44."""
    config = load_laguna_config()
    assert len(config.full_layer_indices) == 12
    assert len(config.sliding_layer_indices) == 36
    assert config.full_layer_indices == tuple(range(0, 48, 4))
    assert len(config.full_layer_indices) + len(config.sliding_layer_indices) == 48


@pytest.mark.requires_hf_snapshot(DEFAULT_MODEL_ID)
def test_real_config_per_layer_heads():
    """Real per-layer head counts (safetensors-verified, see
    notes/2026-07-27-laguna-real-shapes-correction-and-page-size-migration-plan.md):
    full_attention=48 Q heads, sliding_attention=72 Q heads, both 8 KV heads."""
    config = load_laguna_config()
    assert config.heads_per_layer_source == "config_per_layer"
    full = config.groups["full"]
    sliding = config.groups["sliding"]
    assert full.num_qo_heads == 48
    assert sliding.num_qo_heads == 72
    assert full.num_kv_heads == sliding.num_kv_heads == 8
    assert full.head_dim == sliding.head_dim == 128
    assert sliding.window == 512
    assert full.window is None
    for i in config.full_layer_indices:
        assert config.heads_per_layer[i] == 48
    for i in config.sliding_layer_indices:
        assert config.heads_per_layer[i] == 72


@pytest.mark.requires_hf_snapshot(DEFAULT_MODEL_ID)
def test_real_config_moe_and_dense_layers():
    config = load_laguna_config()
    assert config.dense_mlp_layer_indices == (0,)
    assert 0 not in config.moe_layer_indices
    assert len(config.moe_layer_indices) == 47
    assert config.num_experts == 256
    assert config.num_experts_per_tok == 10
    assert config.moe_intermediate_size == 1024
    assert config.shared_expert_intermediate_size == 1024
    assert config.nvfp4_group_size == 16


@pytest.mark.requires_hf_snapshot(DEFAULT_MODEL_ID)
def test_real_config_kv_cache_dtype_is_fp8():
    config = load_laguna_config()
    assert config.kv_cache_dtype == "fp8_e4m3"


@pytest.mark.requires_hf_snapshot(DEFAULT_DRAFT_MODEL_ID)
def test_real_draft_config():
    draft = load_draft_config()
    assert draft.num_hidden_layers == 6
    assert draft.num_attention_heads == 72
    assert draft.num_key_value_heads == 8
    assert draft.head_dim == 128
    assert draft.sliding_window == 512
    assert len(draft.eagle_aux_hidden_state_layer_ids) == 6


@pytest.mark.requires_hf_snapshot(DEFAULT_DRAFT_MODEL_ID)
def test_dflash_constants_agree_with_draft_config():
    """Cross-check the runtime's hardcoded dflash_constants.py against the
    real draft config.json -- exactly the kind of drift this package exists
    to catch."""
    from runtime.backends.dflash_constants import (
        DRAFT_HEAD_DIM,
        DRAFT_NUM_KV_HEADS,
        DRAFT_NUM_LAYERS,
        DRAFT_NUM_QO_HEADS,
        DRAFT_WINDOW,
    )

    draft = load_draft_config()
    assert DRAFT_NUM_LAYERS == draft.num_hidden_layers
    assert DRAFT_WINDOW == draft.sliding_window
    assert DRAFT_NUM_QO_HEADS == draft.num_attention_heads
    assert DRAFT_NUM_KV_HEADS == draft.num_key_value_heads
    assert DRAFT_HEAD_DIM == draft.head_dim


# ---------------------------------------------------------------------------
# Synthetic fixtures: hard-error and fallback paths
# ---------------------------------------------------------------------------


def _minimal_config(**overrides) -> dict:
    base = {
        "num_hidden_layers": 4,
        "layer_types": [
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
        ],
        "sliding_window": 8,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "hidden_size": 32,
        "intermediate_size": 64,
        "vocab_size": 100,
        "num_attention_heads": 4,
        "mlp_only_layers": [0],
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 16,
        "shared_expert_intermediate_size": 16,
    }
    base.update(overrides)
    return base


def test_config_missing_directory_raises(tmp_path):
    """Acceptance criterion #4: missing config.json raises, does not fall
    back to a hardcoded default."""
    with pytest.raises(LagunaConfigError, match="no config.json"):
        load_laguna_config("fake/model", path_override=tmp_path / "nonexistent")


def test_config_missing_required_field_raises(tmp_path):
    cfg = _minimal_config()
    del cfg["hidden_size"]
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(LagunaConfigError, match="hidden_size"):
        load_laguna_config("fake/model", path_override=tmp_path)


def test_config_malformed_json_raises(tmp_path):
    (tmp_path / "config.json").write_text("{not valid json")
    with pytest.raises(LagunaConfigError, match="JSON"):
        load_laguna_config("fake/model", path_override=tmp_path)


def test_config_layer_types_length_mismatch_raises(tmp_path):
    cfg = _minimal_config(layer_types=["full_attention", "sliding_attention"])  # len=2 != 4
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(LagunaConfigError, match="layer_types"):
        load_laguna_config("fake/model", path_override=tmp_path)


def test_config_uniform_heads_fallback(tmp_path):
    """When num_attention_heads_per_layer is absent (a non-Laguna-shaped
    config), fall back to the uniform num_attention_heads field -- this is
    still config-derived, not a hardcoded number, and is flagged via
    heads_per_layer_source."""
    cfg = _minimal_config()
    assert "num_attention_heads_per_layer" not in cfg
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    config = load_laguna_config("fake/model", path_override=tmp_path)
    assert config.heads_per_layer_source == "config_uniform"
    assert config.groups["full"].num_qo_heads == 4
    assert config.groups["sliding"].num_qo_heads == 4


def test_config_per_layer_heads_used_when_present(tmp_path):
    cfg = _minimal_config(num_attention_heads_per_layer=[4, 8, 8, 8])
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    config = load_laguna_config("fake/model", path_override=tmp_path)
    assert config.heads_per_layer_source == "config_per_layer"
    assert config.groups["full"].num_qo_heads == 4
    assert config.groups["sliding"].num_qo_heads == 8


def test_config_inconsistent_heads_within_pattern_raises(tmp_path):
    """If sliding_attention layers don't all share one head count, the
    module refuses to guess which one is "the" group head count."""
    cfg = _minimal_config(num_attention_heads_per_layer=[4, 8, 9, 8])
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(LagunaConfigError, match="inconsistent head counts"):
        load_laguna_config("fake/model", path_override=tmp_path)


def test_config_path_override_accepts_config_json_file_directly(tmp_path):
    cfg = _minimal_config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg))
    config = load_laguna_config("fake/model", path_override=config_path)
    assert config.hidden_size == 32
