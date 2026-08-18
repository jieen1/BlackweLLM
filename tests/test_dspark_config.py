"""DSpark config evidence and target/draft compatibility gates.

These tests are deliberately torch-free.  The first failure point for a
separate draft checkpoint should be a precise config error, before a 2.7 GB
draft file is streamed into GPU memory or a CUDA graph is captured.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from runtime.dspark_config import (
    DSparkConfigError,
    DSparkDraftConfig,
    load_and_validate_dspark_pair,
    validate_dspark_target,
)

HUB = Path.home() / ".cache" / "huggingface" / "hub"
QWEN38_TARGET = HUB / "models--unsloth--Qwen3.8-27B-NVFP4"


def _official_qwen38_dspark_config() -> dict:
    """The exact shape contract read from RadixArk's official draft config."""

    return {
        "architectures": ["DSparkDraftModel"],
        "block_size": 7,
        "confidence_head_with_markov": True,
        "dflash_config": {
            "attention_mode": "gqa",
            "confidence_head_alpha": 1.0,
            "confidence_head_with_markov": True,
            "enable_confidence_head": True,
            "markov_head_type": "vanilla",
            "markov_rank": 256,
            "mask_token_id": 248077,
            "projector_type": "dspark",
            "target_layer_ids": [4, 16, 28, 40, 52],
        },
        "dtype": "bfloat16",
        "enable_confidence_head": True,
        "hidden_size": 5120,
        "intermediate_size": 10240,
        "layer_types": ["full_attention"] * 5,
        "max_position_embeddings": 262144,
        "max_window_layers": 5,
        "model_type": "qwen3",
        "num_attention_heads": 40,
        "num_hidden_layers": 5,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "markov_head_type": "vanilla",
        "markov_rank": 256,
        "rms_norm_eps": 1e-6,
        "vocab_size": 248320,
        "rope_parameters": {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 32,
            "original_max_position_embeddings": 8192,
            "rope_theta": 10000000,
            "rope_type": "yarn",
        },
    }


def test_official_dspark_shape_is_parsed_exactly() -> None:
    draft = DSparkDraftConfig.from_dict(_official_qwen38_dspark_config())

    assert draft.block_size == 7
    assert draft.target_layer_ids == (4, 16, 28, 40, 52)
    assert draft.num_hidden_layers == len(draft.target_layer_ids) == 5
    assert draft.markov_rank == 256
    assert draft.mask_token_id == 248077
    assert draft.confidence_head is True


def test_non_dspark_architecture_is_rejected() -> None:
    config = _official_qwen38_dspark_config()
    config["architectures"] = ["Qwen3ForCausalLM"]

    with pytest.raises(DSparkConfigError, match="DSparkDraftModel"):
        DSparkDraftConfig.from_dict(config)


def test_target_pair_rejects_mismatched_vocab() -> None:
    target = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {
            "hidden_size": 5120,
            "vocab_size": 123,
            "num_hidden_layers": 64,
        },
    }
    draft = DSparkDraftConfig.from_dict(_official_qwen38_dspark_config())

    with pytest.raises(DSparkConfigError, match="vocab_size"):
        validate_dspark_target(target, draft)


def test_real_qwen38_target_has_native_mtp_but_no_dspark_fields() -> None:
    config_paths = sorted(QWEN38_TARGET.glob("snapshots/*/config.json"))
    if not config_paths:
        pytest.skip("the local Qwen3.8 target checkpoint is not cached")
    config = json.loads(config_paths[0].read_text())
    text_config = config["text_config"]
    assert text_config["mtp_num_hidden_layers"] == 1
    assert "dspark_block_size" not in text_config
    assert "dspark_target_layer_ids" not in text_config


def test_real_qwen38_target_accepts_the_official_draft_shape(tmp_path: Path) -> None:
    config_paths = sorted(QWEN38_TARGET.glob("snapshots/*/config.json"))
    if not config_paths:
        pytest.skip("the local Qwen3.8 target checkpoint is not cached")
    target_path = config_paths[0]
    draft_path = tmp_path / "config.json"
    draft_path.write_text(json.dumps(_official_qwen38_dspark_config()))
    draft = load_and_validate_dspark_pair(target_path, draft_path)
    assert draft.target_layer_ids == (4, 16, 28, 40, 52)
