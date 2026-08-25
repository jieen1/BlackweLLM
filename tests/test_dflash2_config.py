from pathlib import Path

import pytest

from runtime.dflash2_config import (
    DFlash2ConfigError,
    DFlash2DraftConfig,
    load_dflash2_config,
    validate_dflash2_target,
)


def _config(**dflash_overrides):
    dflash = {
        "block_size": 8,
        "conv_group_size": 4,
        "conv_kernel_size": 2,
        "mask_token_id": 31,
        "selector_rank": 4,
        "selector_top_k": 3,
        "target_layer_ids": [1, 2],
    }
    dflash.update(dflash_overrides)
    return {
        "architectures": ["DFlash2DraftModel"],
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_target_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "vocab_size": 32,
        "layer_types": ["sliding_attention", "sliding_attention"],
        "sliding_window": 16,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 10000, "rope_type": "default"},
        "dflash_config": dflash,
    }


def test_dflash2_config_parses_and_validates_target():
    draft = DFlash2DraftConfig.from_dict(_config())
    validate_dflash2_target(
        {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "weight_format": "gguf",
            "hidden_size": 16,
            "vocab_size": 32,
            "num_hidden_layers": 4,
        },
        draft,
    )
    assert draft.block_size == 8
    assert draft.target_layer_ids == (1, 2)
    assert draft.input_embedding_scale == 1.0


def test_dflash2_accepts_compressed_tensors_qwen_target():
    draft = DFlash2DraftConfig.from_dict(_config())
    validate_dflash2_target(
        {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "quantization_config": {"quant_method": "compressed-tensors"},
            "hidden_size": 16,
            "vocab_size": 32,
            "num_hidden_layers": 4,
        },
        draft,
    )


def test_dflash2_accepts_flattened_nvfp4_text_config():
    draft = DFlash2DraftConfig.from_dict(_config())
    validate_dflash2_target(
        {
            "model_type": "qwen3_5_text",
            "quantization_config": {"quant_method": "compressed-tensors"},
            "hidden_size": 16,
            "vocab_size": 32,
            "num_hidden_layers": 4,
        },
        draft,
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"selector_top_k": 33},
        {"conv_group_size": 3},
        {"mask_token_id": 32},
        {"target_layer_ids": [1]},
        {"input_embedding_scale": 0.0},
    ],
)
def test_dflash2_config_rejects_invalid_contract(overrides):
    with pytest.raises(DFlash2ConfigError):
        DFlash2DraftConfig.from_dict(_config(**overrides))


def test_real_dflash2_config_if_downloaded():
    config_path = Path("/home/bot/models/Qwen3.8-27B-DFlash2/config.json")
    if not config_path.is_file():
        pytest.skip("local DFlash2 checkpoint is not present")
    config = load_dflash2_config(config_path)
    assert config.hidden_size == 5120
    assert config.block_size == 8
    assert config.target_layer_ids == (5, 19, 33, 47, 61)


def test_dflash2_loader_rejects_checkpoint_file_path(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"not a JSON document")

    with pytest.raises(DFlash2ConfigError, match="model directory or its config.json"):
        load_dflash2_config(checkpoint)
