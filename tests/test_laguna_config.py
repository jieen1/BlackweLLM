from __future__ import annotations

import json

import pytest

from runtime.laguna_config import (
    LagunaConfig,
    LagunaConfigError,
    SelfBuiltModelConfig,
    build_laguna_config,
    build_selfbuilt_model_config,
    build_selfbuilt_speculative_config,
    replace,
    resolve_laguna_checkpoint_dir,
)


def _laguna_config(**overrides) -> dict[str, object]:
    base: dict[str, object] = {
        "architectures": ["LagunaForCausalLM"],
        "model_type": "laguna",
        "hidden_size": 3072,
        "intermediate_size": 12288,
        "num_hidden_layers": 4,
        "num_attention_heads": 48,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "max_position_embeddings": 8192,
        "layer_types": [
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
        ],
        "sliding_window": 512,
        "num_attention_heads_per_layer": [48, 72, 72, 72],
        "num_experts_per_tok": 10,
        "norm_topk_prob": True,
        "moe_router_logit_softcapping": 0.0,
        "moe_apply_router_weight_on_input": False,
        "rope_parameters": {
            "full_attention": {"rope_type": "yarn"},
            "sliding_attention": {"rope_type": "default"},
        },
        "quantization_config": {
            "kv_cache_scheme": {"num_bits": 8},
            "config_groups": {"group_0": {"weights": {"group_size": 16}}},
        },
    }
    base.update(overrides)
    return base


def _write_config(dir_path, **overrides):
    dir_path.mkdir(parents=True, exist_ok=True)
    config_path = dir_path / "config.json"
    config_path.write_text(json.dumps(_laguna_config(**overrides)), encoding="utf-8")
    return config_path


def test_build_laguna_config_from_local_directory(tmp_path) -> None:
    _write_config(tmp_path)

    config = build_laguna_config(tmp_path)

    assert isinstance(config, LagunaConfig)
    assert config.architectures == ["LagunaForCausalLM"]
    assert config.layer_types[1] == "sliding_attention"
    assert config.num_attention_heads_per_layer == [48, 72, 72, 72]
    assert config.rope_parameters["full_attention"]["rope_type"] == "yarn"
    assert config.quantization_config["kv_cache_scheme"]["num_bits"] == 8


def test_build_laguna_config_accepts_config_json_file(tmp_path) -> None:
    config_path = _write_config(tmp_path)

    config = build_laguna_config(config_path)

    assert config.hidden_size == 3072
    assert config.sliding_window == 512


def test_build_laguna_config_resolves_hf_cache_root(tmp_path, monkeypatch) -> None:
    snapshot_dir = (
        tmp_path
        / "hf-cache"
        / "models--fake-org--fake-laguna"
        / "snapshots"
        / "abc123"
    )
    _write_config(snapshot_dir, hidden_size=2048)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf-cache"))

    config = build_laguna_config("fake-org/fake-laguna")

    assert config.hidden_size == 2048


def test_resolve_laguna_checkpoint_prefers_refs_main(tmp_path, monkeypatch) -> None:
    model_dir = tmp_path / "hub" / "models--fake-org--fake-laguna"
    _write_config(model_dir / "snapshots" / "older", hidden_size=1111)
    _write_config(model_dir / "snapshots" / "newer", hidden_size=2222)
    refs_dir = model_dir / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    (refs_dir / "main").write_text("older\n", encoding="utf-8")
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "hub"))

    resolved = resolve_laguna_checkpoint_dir("fake-org/fake-laguna")
    config = build_laguna_config("fake-org/fake-laguna")

    assert resolved.name == "older"
    assert config.hidden_size == 1111


def test_build_laguna_config_rejects_missing_config(tmp_path) -> None:
    with pytest.raises(LagunaConfigError, match="does not exist"):
        build_laguna_config(tmp_path / "missing")


def test_build_laguna_config_rejects_bad_layer_topology(tmp_path) -> None:
    _write_config(tmp_path, layer_types=["full_attention"])

    with pytest.raises(LagunaConfigError, match="layer_types length"):
        build_laguna_config(tmp_path)


def test_build_selfbuilt_model_config_loads_hf_shape(tmp_path) -> None:
    _write_config(tmp_path)

    model_config = build_selfbuilt_model_config(
        model="fake-org/fake-laguna",
        path_override=tmp_path,
        seed=7,
        dtype="float16",
    )

    assert isinstance(model_config, SelfBuiltModelConfig)
    assert model_config.model == "fake-org/fake-laguna"
    assert model_config.tokenizer == "fake-org/fake-laguna"
    assert model_config.dtype == "float16"
    assert model_config.seed == 7
    assert model_config.max_model_len == 8192
    assert model_config.get_hidden_size() == 3072


def test_build_selfbuilt_speculative_config_and_replace(tmp_path) -> None:
    _write_config(tmp_path)
    target_model_config = build_selfbuilt_model_config(
        model="fake-org/fake-laguna",
        path_override=tmp_path,
        tokenizer_mode="slow",
        trust_remote_code=False,
        max_model_len=4096,
    )
    draft_model_config = build_selfbuilt_model_config(
        model="fake-org/fake-draft",
        hf_config=target_model_config.hf_config,
        runner="draft",
        tokenizer=target_model_config.tokenizer,
        tokenizer_mode=target_model_config.tokenizer_mode,
        trust_remote_code=target_model_config.trust_remote_code,
        dtype=target_model_config.dtype,
        seed=target_model_config.seed,
        max_model_len=640,
        spec_target_max_model_len=target_model_config.max_model_len,
        enforce_eager=True,
    )

    spec_config = build_selfbuilt_speculative_config(
        model="fake-org/fake-draft",
        method="dflash",
        num_speculative_tokens=16,
        target_model_config=target_model_config,
        target_parallel_config={"tp": 1},
        draft_model_config=draft_model_config,
    )
    updated = replace(spec_config, num_speculative_tokens=32)

    assert spec_config.draft_model_config is draft_model_config
    assert draft_model_config.runner == "draft"
    assert draft_model_config.spec_target_max_model_len == 4096
    assert updated.num_speculative_tokens == 32
    assert spec_config.num_speculative_tokens == 16
