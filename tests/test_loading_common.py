"""Format-agnostic loader building blocks (Track A step 6). Needs torch
(``assert_all_params_loaded``/``apply_kv_cache_scale_post_load`` inspect
real ``torch.nn.Module`` parameter trees; ``iterate_safetensors_checkpoint``
reads real safetensors files) -- skipped under the CPU-only CI job via
``pytest.importorskip``, same pattern as ``tests/test_laguna_config.py``.

Most of what's below is new direct coverage: before this split,
``assert_all_params_loaded``/``apply_kv_cache_scale_post_load``/
``iterate_safetensors_checkpoint`` (formerly private, single-underscore
names in ``runtime/model_loading.py``) had no unit tests at all -- only
indirect exercise through a real Laguna checkpoint load.
"""

# ruff: noqa: E402

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import save_file  # noqa: E402

from runtime.loading.common import (
    apply_kv_cache_scale_post_load,
    assert_all_params_loaded,
    default_torch_dtype,
    iterate_safetensors_checkpoint,
)
from runtime.model.plain_attention import SelfBuiltAttentionPlaceholder


class TestDefaultTorchDtype:
    def test_sets_and_restores_default_dtype(self):
        original = torch.get_default_dtype()
        with default_torch_dtype(torch.float16):
            assert torch.get_default_dtype() == torch.float16
        assert torch.get_default_dtype() == original

    def test_restores_even_if_the_body_raises(self):
        original = torch.get_default_dtype()
        with pytest.raises(ValueError):
            with default_torch_dtype(torch.bfloat16):
                assert torch.get_default_dtype() == torch.bfloat16
                raise ValueError("boom")
        assert torch.get_default_dtype() == original


class TestAssertAllParamsLoaded:
    @staticmethod
    def _model() -> torch.nn.Module:
        model = torch.nn.Module()
        model.a = torch.nn.Parameter(torch.zeros(1))
        model.b = torch.nn.Parameter(torch.zeros(1))
        return model

    def test_passes_when_every_parameter_was_loaded(self):
        model = self._model()
        assert_all_params_loaded(model, {"a", "b"}, context="test")

    def test_raises_naming_the_context_and_missing_parameter(self):
        model = self._model()
        with pytest.raises(RuntimeError) as excinfo:
            assert_all_params_loaded(model, {"a"}, context="load_laguna_model")
        message = str(excinfo.value)
        assert "load_laguna_model" in message
        assert "'b'" in message

    def test_expected_unloaded_carves_out_legitimately_unloaded_params(self):
        # The DFlash draft model's real case: embed_tokens/lm_head are
        # shared-by-reference from the target model, never loaded from the
        # draft checkpoint at all.
        model = self._model()
        assert_all_params_loaded(
            model,
            {"a"},
            context="load_laguna_dflash_draft_model",
            expected_unloaded=frozenset({"b"}),
        )

    def test_expected_unloaded_does_not_hide_a_genuinely_different_missing_param(self):
        model = self._model()
        model.c = torch.nn.Parameter(torch.zeros(1))
        with pytest.raises(RuntimeError, match="'c'"):
            assert_all_params_loaded(
                model, {"a"}, context="test", expected_unloaded=frozenset({"b"})
            )


class TestApplyKvCacheScalePostLoad:
    @staticmethod
    def _attention_layer(*, has_kv_scale: bool) -> SelfBuiltAttentionPlaceholder:
        quant_config = SimpleNamespace(kv_cache_scheme={"num_bits": 8} if has_kv_scale else None)
        return SelfBuiltAttentionPlaceholder(
            num_heads=2,
            head_size=4,
            scale=1.0,
            num_kv_heads=1,
            cache_config=SimpleNamespace(),
            quant_config=quant_config,
        )

    def test_copies_the_loaded_k_and_v_scale_into_the_runtime_buffers(self):
        layer = self._attention_layer(has_kv_scale=True)
        # Simulate load_weights having copied real checkpoint values into
        # the loadable k_scale/v_scale Parameters.
        with torch.no_grad():
            layer.k_scale.copy_(torch.tensor([2.5]))
            layer.v_scale.copy_(torch.tensor([3.5]))

        model = torch.nn.Module()
        model.attn = layer
        apply_kv_cache_scale_post_load(model)

        assert torch.equal(layer._k_scale, torch.tensor([2.5]))
        assert torch.equal(layer._v_scale, torch.tensor([3.5]))

    def test_leaves_the_default_scale_alone_when_the_checkpoint_has_none(self):
        # The DFlash draft model's real case: no quantization_config, no
        # k_scale/v_scale Parameters ever created, so there is nothing to
        # copy and the hardcoded 1.0 default must survive untouched.
        layer = self._attention_layer(has_kv_scale=False)
        model = torch.nn.Module()
        model.attn = layer
        apply_kv_cache_scale_post_load(model)

        assert torch.equal(layer._k_scale, torch.tensor([1.0]))
        assert torch.equal(layer._v_scale, torch.tensor([1.0]))
        assert not hasattr(layer, "k_scale")


class TestIterateSafetensorsCheckpoint:
    def test_single_unsharded_file_with_no_index(self, tmp_path):
        # The DFlash draft model's real layout: one model.safetensors, no
        # model.safetensors.index.json at all.
        save_file(
            {"weight_a": torch.arange(4, dtype=torch.float32)},
            str(tmp_path / "model.safetensors"),
        )
        result = dict(iterate_safetensors_checkpoint(str(tmp_path)))
        assert set(result) == {"weight_a"}
        assert torch.equal(result["weight_a"], torch.arange(4, dtype=torch.float32))

    def test_sharded_checkpoint_with_an_index_reads_every_shard(self, tmp_path):
        # The main model's real layout: multiple shard files plus
        # model.safetensors.index.json mapping tensor name -> shard file.
        save_file(
            {"model.layers.0.weight": torch.ones(2)},
            str(tmp_path / "model-00001-of-00002.safetensors"),
        )
        save_file(
            {"model.layers.1.weight": torch.zeros(2)},
            str(tmp_path / "model-00002-of-00002.safetensors"),
        )
        index = {
            "weight_map": {
                "model.layers.0.weight": "model-00001-of-00002.safetensors",
                "model.layers.1.weight": "model-00002-of-00002.safetensors",
            }
        }
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

        result = dict(iterate_safetensors_checkpoint(str(tmp_path)))
        assert set(result) == {"model.layers.0.weight", "model.layers.1.weight"}
        assert torch.equal(result["model.layers.0.weight"], torch.ones(2))
        assert torch.equal(result["model.layers.1.weight"], torch.zeros(2))
