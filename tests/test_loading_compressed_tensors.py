"""Torch-free contracts for the compressed-tensors loader adapter (Track A
step 6). Ported to their new home when ``remap_kv_scale_name``/
``IGNORE_WEIGHT_SUFFIXES`` moved out of ``runtime/model/laguna_model.py`` /
``runtime/model/_weight_loading.py`` -- this is the first direct unit-test
coverage either has had; previously they were only exercised indirectly
through ``LagunaModelSelfBuilt.load_weights`` on a real checkpoint.
"""

from __future__ import annotations

from runtime.loading.compressed_tensors import IGNORE_WEIGHT_SUFFIXES, remap_kv_scale_name


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
