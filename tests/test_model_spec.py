"""E1 Phase 1: tests for runtime/model_spec.py — ModelSpec frozen dataclass."""

import pytest

from runtime.model_spec import ModelSpec


class TestModelSpecConstruction:
    def test_basic_construction(self):
        spec = ModelSpec(
            model_id="test/model",
            architecture="TestArch",
            attn_layer_names=("layer.0", "layer.3"),
            gdn_layer_names=("layer.1", "layer.2"),
        )
        assert spec.model_id == "test/model"
        assert spec.architecture == "TestArch"
        assert spec.num_attn_layers == 2
        assert spec.num_gdn_layers == 2
        assert spec.num_layers == 4

    def test_from_runner_init(self):
        spec = ModelSpec.from_runner_init(
            model_id="unsloth/Qwen3.6-27B-NVFP4",
            architecture="Qwen3_5ForConditionalGeneration",
            attn_layer_names=["l.0", "l.3", "l.6"],
            gdn_layer_names=["l.1", "l.2", "l.4", "l.5"],
            kv_dtype="fp8_e4m3",
            block_size=64,
        )
        assert spec.num_attn_layers == 3
        assert spec.num_gdn_layers == 4
        assert spec.num_layers == 7
        assert spec.kv_dtype == "fp8_e4m3"
        assert spec.block_size == 64

    def test_frozen_immutability(self):
        spec = ModelSpec(
            model_id="test",
            architecture="test",
            attn_layer_names=("a",),
            gdn_layer_names=("g",),
        )
        with pytest.raises(AttributeError):
            spec.model_id = "changed"

    def test_defaults(self):
        spec = ModelSpec(
            model_id="test",
            architecture="test",
            attn_layer_names=(),
            gdn_layer_names=(),
        )
        assert spec.mtp_model_id is None
        assert spec.num_speculative_tokens == 0
        assert spec.kv_dtype == "fp8_e4m3"
        assert spec.block_size == 64
        assert spec.has_mtp is False


class TestModelSpecMTP:
    def test_has_mtp_true(self):
        spec = ModelSpec(
            model_id="test",
            architecture="test",
            attn_layer_names=("a",),
            gdn_layer_names=("g",),
            mtp_model_id="mtp_draft",
            mtp_attn_layer_names=("mtp.0",),
            num_speculative_tokens=3,
        )
        assert spec.has_mtp is True
        assert spec.verify_qo_len == 4  # K+1

    def test_has_mtp_false_without_tokens(self):
        spec = ModelSpec(
            model_id="test",
            architecture="test",
            attn_layer_names=("a",),
            gdn_layer_names=("g",),
            mtp_model_id="mtp_draft",
            num_speculative_tokens=0,
        )
        assert spec.has_mtp is False

    def test_verify_qo_len(self):
        spec = ModelSpec(
            model_id="test",
            architecture="test",
            attn_layer_names=("a",),
            gdn_layer_names=("g",),
            num_speculative_tokens=5,
        )
        assert spec.verify_qo_len == 6


class TestModelSpecLayerNames:
    def test_tuples_not_lists(self):
        """Layer names should be stored as tuples (immutable)."""
        spec = ModelSpec.from_runner_init(
            model_id="test",
            architecture="test",
            attn_layer_names=["a", "b"],
            gdn_layer_names=["g"],
        )
        assert isinstance(spec.attn_layer_names, tuple)
        assert isinstance(spec.gdn_layer_names, tuple)

    def test_empty_layers(self):
        spec = ModelSpec(
            model_id="test",
            architecture="test",
            attn_layer_names=(),
            gdn_layer_names=(),
        )
        assert spec.num_layers == 0
