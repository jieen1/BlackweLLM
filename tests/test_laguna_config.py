"""CPU-only contracts for the owned Laguna configuration surface."""

# ruff: noqa: E402

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from runtime.laguna_config import (
    LagunaRuntimeConfig,
    SelfBuiltCacheConfig,
    SelfBuiltCompilationConfig,
    SelfBuiltDeviceConfig,
    SelfBuiltKernelConfig,
    SelfBuiltLoadConfig,
    SelfBuiltModelConfig,
    SelfBuiltParallelConfig,
    SelfBuiltQuantConfig,
    build_laguna_config,
    build_laguna_dflash_config,
    load_laguna_draft_hf_config,
)


def _runtime_config() -> LagunaRuntimeConfig:
    model_config = SelfBuiltModelConfig(
        hf_config=SimpleNamespace(hidden_size=3072, vocab_size=151936, num_hidden_layers=48),
        dtype=torch.bfloat16,
        model="target-model",
        max_model_len=65536,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        quantization="nvfp4",
        tokenizer="target-tokenizer",
        tokenizer_mode="slow",
        seed=17,
    )
    return LagunaRuntimeConfig(
        model_config=model_config,
        cache_config=SelfBuiltCacheConfig(),
        quant_config=SelfBuiltQuantConfig(),
        parallel_config=SelfBuiltParallelConfig(),
        compilation_config=SelfBuiltCompilationConfig(),
        load_config=SelfBuiltLoadConfig(),
        device_config=SelfBuiltDeviceConfig(),
        kernel_config=SelfBuiltKernelConfig(),
    )


def test_dflash_config_is_owned_copy_with_required_draft_fields() -> None:
    runtime_config = _runtime_config()
    draft_hf_config = SimpleNamespace(draft_vocab_size=151936)

    draft_config = build_laguna_dflash_config(
        runtime_config,
        model="draft-model",
        hf_config=draft_hf_config,
        num_speculative_tokens=15,
        max_model_len=656,
    )

    assert draft_config is not runtime_config
    assert draft_config.model_config is runtime_config.model_config
    assert draft_config.cache_config is runtime_config.cache_config
    assert draft_config.speculative_config.method == "dflash"
    assert draft_config.speculative_config.num_speculative_tokens == 15
    assert draft_config.speculative_config.target_parallel_config is runtime_config.parallel_config

    draft_model_config = draft_config.speculative_config.draft_model_config
    assert draft_model_config.model == "draft-model"
    assert draft_model_config.hf_config is draft_hf_config
    assert draft_model_config.dtype is torch.bfloat16
    assert draft_model_config.tokenizer == "target-tokenizer"
    assert draft_model_config.tokenizer_mode == "slow"
    assert draft_model_config.trust_remote_code is True
    assert draft_model_config.seed == 17
    assert draft_model_config.max_model_len == 656
    assert draft_model_config.spec_target_max_model_len == 65536
    assert draft_model_config.enforce_eager is True


def test_draft_config_preserves_draft_rope_and_adds_legacy_defaults(monkeypatch) -> None:
    from runtime import laguna_config

    class DraftConfig:
        rope_parameters = {"sliding_attention": {"rope_theta": 10_000.0}}

    captured: dict[str, object] = {}

    def load(model: str, *, trust_remote_code: bool) -> DraftConfig:
        captured.update(model=model, trust_remote_code=trust_remote_code)
        return DraftConfig()

    monkeypatch.setattr(laguna_config.AutoConfig, "from_pretrained", load)

    draft_config = load_laguna_draft_hf_config("draft-model")

    assert captured == {"model": "draft-model", "trust_remote_code": True}
    assert draft_config.rope_parameters["sliding_attention"]["rope_theta"] == 10_000.0
    assert draft_config.qkv_bias is False
    assert draft_config.decoder_sparse_step == 1
    assert draft_config.mlp_only_layers == [0]
    assert draft_config.norm_topk_prob is True
    assert draft_config.num_attention_heads_per_layer is None
    assert draft_config.partial_rotary_factor == 1.0
    assert draft_config.swa_attention_sink_enabled is False
    assert draft_config.swa_rope_parameters is None


def test_build_config_resolves_cached_repo_id_to_local_snapshot(monkeypatch, tmp_path) -> None:
    from runtime import laguna_config

    (tmp_path / "config.json").write_text('{"quantization_config": {"quant_method": "nvfp4"}}')
    loaded: dict[str, object] = {}

    monkeypatch.setattr(laguna_config, "snapshot_download", lambda **_: str(tmp_path))

    def load(model, **_):
        loaded["model"] = model
        return SimpleNamespace()

    monkeypatch.setattr(laguna_config.AutoConfig, "from_pretrained", load)

    config = build_laguna_config("poolside/Laguna-S-2.1-NVFP4", max_model_len=64)

    assert loaded["model"] == str(tmp_path)
    assert config.model_config.model == str(tmp_path)
