"""Torch-free contracts for the compressed-tensors loader adapter (Track A
step 6). Ported to their new home when ``remap_kv_scale_name``/
``IGNORE_WEIGHT_SUFFIXES`` moved out of ``runtime/model/laguna_model.py`` /
``runtime/model/_weight_loading.py`` -- this is the first direct unit-test
coverage either has had; previously they were only exercised indirectly
through ``LagunaModelSelfBuilt.load_weights`` on a real checkpoint.
"""

from __future__ import annotations

import pytest

from runtime.loading.compressed_tensors import (
    IGNORE_WEIGHT_SUFFIXES,
    QUANT_ALGO_MP_FP8_CHANNEL,
    QUANT_ALGO_MP_NVFP4,
    MixedPrecisionQuantMap,
    mixed_precision_quant_map,
    remap_kv_scale_name,
)


class TestRemapKvScaleName:
    def test_names_already_in_params_dict_pass_through_unchanged(self):
        # The real fast path: most checkpoint keys already match a model
        # parameter name exactly (e.g. plain Linear weights) and never need
        # remapping at all.
        params_dict = {"model.layers.0.self_attn.qkv_proj.weight": object()}
        assert (
            remap_kv_scale_name("model.layers.0.self_attn.qkv_proj.weight", params_dict)
            == "model.layers.0.self_attn.qkv_proj.weight"
        )

    def test_k_scale_is_remapped_into_the_attn_submodule(self):
        # The one real pattern this runtime's checkpoints produce: a
        # checkpoint key ending in .k_scale/.v_scale directly under
        # self_attn, remapped to match SelfBuiltAttentionPlaceholder's own
        # self.attn submodule nesting.
        params_dict = {"model.layers.0.self_attn.attn.k_scale": object()}
        assert (
            remap_kv_scale_name("model.layers.0.self_attn.k_scale", params_dict)
            == "model.layers.0.self_attn.attn.k_scale"
        )

    def test_v_scale_is_remapped_the_same_way(self):
        params_dict = {"model.layers.3.self_attn.attn.v_scale": object()}
        assert (
            remap_kv_scale_name("model.layers.3.self_attn.v_scale", params_dict)
            == "model.layers.3.self_attn.attn.v_scale"
        )

    def test_remap_target_missing_from_params_dict_returns_none(self):
        # E.g. the DFlash draft model's checkpoint, which never creates
        # k_scale/v_scale Parameters at all -- the remapped name can never
        # exist, and the caller (LagunaModelSelfBuilt.load_weights) treats
        # None as "skip this checkpoint tensor", not an error.
        params_dict: dict = {}
        assert remap_kv_scale_name("model.layers.0.self_attn.k_scale", params_dict) is None

    def test_unrelated_suffix_passes_through_for_the_caller_to_reject_or_accept(self):
        # Anything not ending in .k_scale/.v_scale and not already an exact
        # params_dict match is returned as-is -- this function only knows
        # about the one remap pattern, the caller decides what to do with
        # a name that still doesn't match anything.
        params_dict: dict = {}
        assert remap_kv_scale_name("model.layers.0.mlp.gate_proj.weight", params_dict) == (
            "model.layers.0.mlp.gate_proj.weight"
        )

    def test_exact_match_is_checked_before_the_scale_suffix_pattern(self):
        # If a checkpoint key ending in .k_scale already matches a param
        # name directly (not the realistic case today, but the function's
        # own documented order), the exact match wins and no remap happens.
        params_dict = {"model.layers.0.self_attn.k_scale": object()}
        assert (
            remap_kv_scale_name("model.layers.0.self_attn.k_scale", params_dict)
            == "model.layers.0.self_attn.k_scale"
        )


class TestIgnoreWeightSuffixes:
    def test_contains_the_generic_bias_suffixes(self):
        assert ".bias" in IGNORE_WEIGHT_SUFFIXES
        assert "_bias" in IGNORE_WEIGHT_SUFFIXES

    def test_contains_compressed_tensors_own_scale_suffixes(self):
        # weight_scale/input_scale are compressed-tensors' own naming
        # convention (docs/architecture.md §3.2-D) -- this is the part of
        # the tuple that is genuinely format-specific, unlike the bias
        # suffixes above.
        for suffix in (".weight_scale", "_weight_scale", ".input_scale", "_input_scale"):
            assert suffix in IGNORE_WEIGHT_SUFFIXES

    def test_contains_the_kv_scale_suffixes_remap_kv_scale_name_also_knows_about(self):
        # Not a coincidence: a name ending in .k_scale/.v_scale that never
        # matches params_dict (e.g. wrong layer index, or the draft model's
        # checkpoint) still needs to be recognized as an intentional skip
        # rather than a missing-tensor bug -- both this tuple and
        # remap_kv_scale_name agree on the same four suffixes.
        for suffix in (".k_scale", "_k_scale", ".v_scale", "_v_scale"):
            assert suffix in IGNORE_WEIGHT_SUFFIXES


# ---------------------------------------------------------------------------
# "mixed-precision" sub-format classification (unsloth's Qwen3.6-27B-NVFP4).
# Pure string/regex logic -- no torch needed here, matching this module's
# own "runtime/loading/compressed_tensors.py stays torch-free" design (see
# that module's docstring). Fixture shapes below are the real checkpoint's
# ``config_groups``/``ignore`` entries, verified 2026-08-02 against its
# actual config.json, not invented.
# ---------------------------------------------------------------------------

_UNSLOTH_LIKE_QUANT_CONFIG = {
    "quant_method": "compressed-tensors",
    "format": "mixed-precision",
    "ignore": [
        "model.language_model.layers.0.linear_attn.in_proj_a",
        "model.language_model.layers.0.linear_attn.in_proj_b",
        "model.language_model.layers.0.linear_attn.norm",
        r"re:^mtp.*",
    ],
    "config_groups": {
        "group_0": {
            "format": "float-quantized",
            "targets": [
                r"re:.*self_attn\.(q|k|v|o)_proj$",
                r"re:.*linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$",
                r"re:.*lm_head",
                r"re:.*layers\.(56|57|58|59|60|61|62|63)\.mlp\.(gate|up|down)_proj$",
            ],
        },
        "group_1": {
            "format": "nvfp4-pack-quantized",
            "targets": [r"re:.*mlp\.(gate|up|down)_proj$"],
        },
    },
}


class TestMixedPrecisionQuantMap:
    def test_fp8_group_classifies_self_attn_and_lm_head(self):
        qmap = MixedPrecisionQuantMap(_UNSLOTH_LIKE_QUANT_CONFIG)
        assert (
            qmap.get("model.language_model.layers.3.self_attn.q_proj")
            == QUANT_ALGO_MP_FP8_CHANNEL
        )
        assert qmap.get("lm_head") == QUANT_ALGO_MP_FP8_CHANNEL

    def test_fp8_group_classifies_linear_attn_projections(self):
        qmap = MixedPrecisionQuantMap(_UNSLOTH_LIKE_QUANT_CONFIG)
        for suffix in ("in_proj_qkv", "in_proj_z", "out_proj"):
            assert (
                qmap.get(f"model.language_model.layers.1.linear_attn.{suffix}")
                == QUANT_ALGO_MP_FP8_CHANNEL
            )

    def test_nvfp4_group_classifies_early_layer_mlp(self):
        qmap = MixedPrecisionQuantMap(_UNSLOTH_LIKE_QUANT_CONFIG)
        for proj in ("gate_proj", "up_proj", "down_proj"):
            assert (
                qmap.get(f"model.language_model.layers.0.mlp.{proj}") == QUANT_ALGO_MP_NVFP4
            )

    def test_fp8_wins_the_layer_56_63_mlp_overlap(self):
        # The one real overlap: group_1's blanket mlp regex also matches
        # layers 56-63, which group_0 explicitly carves out. Verified
        # against the real checkpoint's safetensors headers (module
        # docstring) that FP8 is what actually got baked into those
        # tensors -- this test pins that measured precedence, not an
        # assumption about config_groups dict order.
        qmap = MixedPrecisionQuantMap(_UNSLOTH_LIKE_QUANT_CONFIG)
        for layer in (56, 63):
            assert (
                qmap.get(f"model.language_model.layers.{layer}.mlp.gate_proj")
                == QUANT_ALGO_MP_FP8_CHANNEL
            )

    def test_ignore_list_wins_over_a_matching_target(self):
        qmap = MixedPrecisionQuantMap(_UNSLOTH_LIKE_QUANT_CONFIG)
        assert qmap.get("model.language_model.layers.0.linear_attn.in_proj_a") is None
        assert qmap.get("model.language_model.layers.0.linear_attn.in_proj_b") is None

    def test_ignore_regex_entry_matches(self):
        qmap = MixedPrecisionQuantMap(_UNSLOTH_LIKE_QUANT_CONFIG)
        assert qmap.get("mtp.fc") is None
        assert qmap.get("mtp.layers.0.self_attn.q_proj") is None

    def test_module_matching_no_target_is_unquantized(self):
        qmap = MixedPrecisionQuantMap(_UNSLOTH_LIKE_QUANT_CONFIG)
        assert qmap.get("model.language_model.embed_tokens") is None
        assert qmap.get("model.language_model.layers.0.input_layernorm") is None

    def test_get_default_is_returned_not_none_literal(self):
        qmap = MixedPrecisionQuantMap(_UNSLOTH_LIKE_QUANT_CONFIG)
        sentinel = object()
        assert qmap.get("model.language_model.embed_tokens", sentinel) is sentinel

    def test_unknown_group_format_raises(self):
        config = {
            "config_groups": {"group_0": {"format": "some-future-format", "targets": ["x"]}}
        }
        with pytest.raises(ValueError, match="some-future-format"):
            MixedPrecisionQuantMap(config)

    def test_missing_ignore_and_config_groups_keys_default_empty(self):
        # A minimal config_groups-only dict (no "ignore" key at all) must
        # not raise -- every real key here is optional per compressed-
        # tensors' own schema.
        qmap = MixedPrecisionQuantMap({"config_groups": {}})
        assert qmap.get("anything") is None


class TestMixedPrecisionQuantMapFn:
    def test_empty_when_no_quantization_config(self):
        assert mixed_precision_quant_map({}) == {}
        assert mixed_precision_quant_map({"quantization_config": None}) == {}

    def test_empty_when_format_is_not_mixed_precision(self):
        # modelopt checkpoints (or a hypothetical single-format
        # compressed-tensors checkpoint) must not be misrouted here.
        config = {"quantization_config": {"quant_method": "modelopt"}}
        assert mixed_precision_quant_map(config) == {}

    def test_builds_a_working_classifier_for_a_real_shaped_config(self):
        config = {"quantization_config": _UNSLOTH_LIKE_QUANT_CONFIG}
        qmap = mixed_precision_quant_map(config)
        assert isinstance(qmap, MixedPrecisionQuantMap)
        assert qmap.get("lm_head") == QUANT_ALGO_MP_FP8_CHANNEL
