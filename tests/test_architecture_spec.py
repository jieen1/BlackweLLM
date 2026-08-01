"""Step 3 shadow tests: parsing must reproduce what the runtime hardcodes.

Shadow mode makes exactly one claim -- that reading ``config.json`` yields the
values already baked into the source. Nothing drives off the parser yet, so a
failure here means the parser is wrong, never that the runtime is.

Two layers, for two different reasons:

* Inline-dict tests pin the parser's contract and run everywhere, including
  CI, which has no checkpoints.
* Checkpoint tests are the ones that carry evidentiary weight, because a
  hand-written fixture can only confirm what its author already believed.
  They skip when the checkpoint is absent rather than being deleted, since
  the machine that has the weights is where the claim can be checked.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from runtime.architecture import (
    CACHE_PAGED_KV,
    CACHE_RECURRENT,
    UnsupportedArchitectureError,
    parse_architecture,
    validate_text_only,
)

HUB = Path.home() / ".cache" / "huggingface" / "hub"

LAGUNA = "models--poolside--Laguna-S-2.1-NVFP4"
QWEN_OFFICIAL = "models--nvidia--Qwen3.6-27B-NVFP4"
QWEN_TEXT_MTP = "models--sakamakismile--Qwen3.6-27B-Text-NVFP4-MTP"
QWEN_UNSLOTH = "models--unsloth--Qwen3.6-27B-NVFP4"
QWEN_THINKINGCAP = "models--morosystems--ThinkingCap-Qwen3.6-27B-NVFP4"


def load_config(repo: str) -> dict:
    """Return a local checkpoint's config.json, or skip if it is not here."""
    matches = sorted((HUB / repo).glob("snapshots/*/config.json"))
    if not matches:
        pytest.skip(f"{repo} not present in the local HF cache")
    return json.loads(matches[0].read_text())


def minimal_config(**overrides) -> dict:
    config = {
        "architectures": ["TestForCausalLM"],
        "model_type": "test",
        "num_hidden_layers": 2,
        "layer_types": ["full_attention", "sliding_attention"],
        "hidden_size": 8,
        "vocab_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
    }
    config.update(overrides)
    return config


class TestParserContract:
    def test_layer_types_drive_the_cache_requirement(self):
        spec = parse_architecture(
            minimal_config(
                num_hidden_layers=3,
                layer_types=["full_attention", "linear_attention", "sliding_attention"],
            )
        )
        assert [layer.cache for layer in spec.layers] == [
            CACHE_PAGED_KV,
            CACHE_RECURRENT,
            CACHE_PAGED_KV,
        ]
        assert spec.paged_kv_layers == (0, 2)
        assert spec.recurrent_layers == (1,)
        assert spec.needs_two_cache_families is True

    def test_attention_only_model_needs_one_cache_family(self):
        spec = parse_architecture(minimal_config())
        assert spec.recurrent_layers == ()
        assert spec.needs_two_cache_families is False

    def test_missing_layer_types_is_rejected_not_guessed(self):
        config = minimal_config()
        del config["layer_types"]
        with pytest.raises(UnsupportedArchitectureError, match="layer_types"):
            parse_architecture(config)

    def test_layer_count_disagreement_is_rejected(self):
        with pytest.raises(UnsupportedArchitectureError, match="contradicts itself"):
            parse_architecture(minimal_config(num_hidden_layers=5))

    def test_unknown_attention_kind_names_itself_in_the_error(self):
        with pytest.raises(UnsupportedArchitectureError, match="quantum_attention"):
            parse_architecture(
                minimal_config(num_hidden_layers=1, layer_types=["quantum_attention"])
            )

    def test_nested_text_config_is_read(self):
        # Reading Qwen3.6's fields at the top level yields nothing, silently.
        nested = {
            "architectures": ["NestedForCausalLM"],
            "model_type": "nested",
            "text_config": minimal_config(num_hidden_layers=1, layer_types=["full_attention"]),
        }
        spec = parse_architecture(nested)
        assert spec.num_hidden_layers == 1
        assert spec.hidden_size == 8

    def test_missing_num_hidden_layers_says_where_it_looked(self):
        nested = {
            "architectures": ["NestedForCausalLM"],
            "model_type": "nested",
            "text_config": {"hidden_size": 8},
        }
        with pytest.raises(UnsupportedArchitectureError, match="text_config"):
            parse_architecture(nested)


class TestVisionTowerPolicy:
    """RK8, at config level rather than after a surprising tensor name."""

    def test_declared_text_only_is_accepted(self):
        spec = parse_architecture(minimal_config(language_model_only=True))
        validate_text_only(spec)

    def test_vision_config_is_rejected_with_an_actionable_message(self):
        spec = parse_architecture(minimal_config(vision_config={"depth": 1}))
        with pytest.raises(UnsupportedArchitectureError) as excinfo:
            validate_text_only(spec)
        message = str(excinfo.value)
        assert "vision_config" in message
        assert "language_model_only" in message

    def test_an_explicit_text_only_claim_outranks_a_leftover_vision_config(self):
        spec = parse_architecture(
            minimal_config(vision_config={"depth": 1}, language_model_only=True)
        )
        validate_text_only(spec)


class TestLagunaShadowAgreement:
    """The step-3 claim: parsing reproduces today's hardcoded values."""

    @staticmethod
    @pytest.fixture(scope="class")
    def spec():
        return parse_architecture(load_config(LAGUNA))

    def test_router_constants_are_derivable_from_the_checkpoint(self, spec):
        # runtime/laguna_router.py hardcodes these at module scope (S6). If the
        # checkpoint disagreed with them, generalizing the router for a second
        # MoE shape would start from a false premise.
        from runtime.laguna_router import EXPERTS, TOP_K

        assert spec.moe is not None
        assert spec.moe.num_experts == EXPERTS
        assert spec.moe.top_k == TOP_K

    def test_laguna_has_no_recurrent_layers(self, spec):
        # laguna.py constructs its ModelSpec with gdn_layer_names=[]. That is
        # an assertion about the checkpoint, and this is it holding.
        assert spec.recurrent_layers == ()
        assert spec.needs_two_cache_families is False

    def test_attention_split_matches_the_documented_shape(self, spec):
        assert spec.num_hidden_layers == 48
        assert len(spec.layers) == 48
        assert (
            spec.count_attention("full_attention") + spec.count_attention("sliding_attention") == 48
        )
        assert spec.sliding_window == 512

    def test_quantization_is_compressed_tensors_with_fp8_kv(self, spec):
        assert spec.quant.method == "compressed-tensors"
        assert spec.quant.kv_num_bits == 8
        assert spec.quant.kv_type == "float"

    def test_rope_is_per_layer_type(self, spec):
        # Laguna uses yarn for full attention and plain RoPE for the sliding
        # layers -- a single global RoPE setting cannot describe it.
        assert set(spec.rope) == {"full_attention", "sliding_attention"}
        assert spec.rope["full_attention"].rope_type == "yarn"
        assert spec.rope["full_attention"].partial_rotary_factor == 0.5
        assert spec.rope["sliding_attention"].rope_type == "default"

    def test_laguna_is_accepted_as_text_only(self, spec):
        validate_text_only(spec)


class TestAgainstRealCheckpoints:
    """Facts observed across four local Qwen3.6 checkpoints on 2026-08-01."""

    def test_qwen36_is_hybrid_and_needs_both_cache_families(self):
        spec = parse_architecture(load_config(QWEN_OFFICIAL))
        assert spec.num_hidden_layers == 64
        assert len(spec.recurrent_layers) == 48
        assert len(spec.paged_kv_layers) == 16
        assert spec.needs_two_cache_families is True
        assert spec.has_mtp

    @pytest.mark.parametrize("repo", [QWEN_OFFICIAL, QWEN_TEXT_MTP, QWEN_UNSLOTH, QWEN_THINKINGCAP])
    def test_architecture_name_cannot_distinguish_these_checkpoints(self, repo):
        # All four say Qwen3_5ForConditionalGeneration / qwen3_5, yet they
        # differ in vision tower and in quantization method. A registry keyed
        # on the architecture string alone would treat them as interchangeable.
        spec = parse_architecture(load_config(repo))
        assert spec.architecture == "Qwen3_5ForConditionalGeneration"
        assert spec.model_type == "qwen3_5"

    def test_quantization_method_varies_between_same_architecture_checkpoints(self):
        # Three of the four are modelopt; unsloth's is compressed-tensors. The
        # loader adapter must therefore be chosen per checkpoint, not per
        # architecture -- see architecture.md §3.2-D.
        official = parse_architecture(load_config(QWEN_OFFICIAL))
        unsloth = parse_architecture(load_config(QWEN_UNSLOTH))
        assert official.quant.method == "modelopt"
        assert unsloth.quant.method == "compressed-tensors"

    def test_only_the_text_build_survives_the_vision_check(self):
        validate_text_only(parse_architecture(load_config(QWEN_TEXT_MTP)))
        for repo in (QWEN_OFFICIAL, QWEN_UNSLOTH, QWEN_THINKINGCAP):
            with pytest.raises(UnsupportedArchitectureError):
                validate_text_only(parse_architecture(load_config(repo)))
