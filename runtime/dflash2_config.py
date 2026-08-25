"""Torch-free configuration contract for Qwen3.8 DFlash2 drafts.

DFlash2 is a separate five-layer block-diffusion drafter.  Its checkpoint is
not a second set of target-model weights: it shares the target embedding and
LM head and consumes hidden-state taps from the target trunk.  Keeping this
parser independent of torch lets callers reject an incompatible pair before
allocating either model on the GPU.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DFlash2ConfigError(ValueError):
    """Raised when a DFlash2 draft or target/draft pair is incompatible."""


@dataclass(frozen=True)
class DFlash2DraftConfig:
    """Fields consumed by the local DFlash2 draft implementation."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_target_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    block_size: int
    conv_kernel_size: int
    conv_group_size: int
    selector_rank: int
    selector_top_k: int
    mask_token_id: int
    target_layer_ids: tuple[int, ...]
    sliding_window: int
    rms_norm_eps: float
    rope_parameters: Mapping[str, Any]
    output_multiplier: float
    final_logit_softcapping: float | None
    input_embedding_scale: float

    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> DFlash2DraftConfig:
        """Parse the official ``DFlash2DraftModel`` JSON contract."""

        architectures = config.get("architectures")
        if architectures != ["DFlash2DraftModel"]:
            raise DFlash2ConfigError(
                f"DFlash2 draft architectures must be ['DFlash2DraftModel']; got {architectures!r}"
            )
        dflash = config.get("dflash_config")
        if not isinstance(dflash, Mapping):
            raise DFlash2ConfigError("DFlash2 draft requires an object-valued dflash_config")

        def required_int(name: str, source: Mapping[str, Any] = config) -> int:
            value = source.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise DFlash2ConfigError(f"DFlash2 field {name!r} must be a positive integer")
            return value

        target_ids_raw = dflash.get("target_layer_ids")
        if not isinstance(target_ids_raw, list) or not target_ids_raw:
            raise DFlash2ConfigError(
                "DFlash2 dflash_config.target_layer_ids must be a non-empty list"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in target_ids_raw
        ):
            raise DFlash2ConfigError("DFlash2 target_layer_ids must contain non-negative integers")
        target_layer_ids = tuple(target_ids_raw)

        mask_token_id = dflash.get("mask_token_id")
        if (
            isinstance(mask_token_id, bool)
            or not isinstance(mask_token_id, int)
            or mask_token_id < 0
        ):
            raise DFlash2ConfigError(
                "DFlash2 dflash_config.mask_token_id must be a non-negative integer"
            )

        layer_types = config.get("layer_types")
        if not isinstance(layer_types, list):
            raise DFlash2ConfigError("DFlash2 layer_types must be a list")
        num_hidden_layers = required_int("num_hidden_layers")
        if layer_types != ["sliding_attention"] * num_hidden_layers:
            raise DFlash2ConfigError(
                "the local DFlash2 path requires every draft layer to be sliding_attention"
            )

        rope_parameters = config.get("rope_parameters", {})
        if not isinstance(rope_parameters, Mapping):
            raise DFlash2ConfigError("DFlash2 rope_parameters must be an object")

        softcap_raw = dflash.get("final_logit_softcapping", config.get("final_logit_softcapping"))
        softcap = None if softcap_raw is None else float(softcap_raw)
        if softcap is not None and softcap <= 0:
            raise DFlash2ConfigError("DFlash2 final_logit_softcapping must be positive when set")
        embedding_scale_raw = dflash.get(
            "input_embedding_scale", config.get("input_embedding_scale", 1.0)
        )
        try:
            input_embedding_scale = float(embedding_scale_raw)
        except (TypeError, ValueError) as exc:
            raise DFlash2ConfigError(
                "DFlash2 input_embedding_scale must be a finite positive number"
            ) from exc
        if not math.isfinite(input_embedding_scale) or input_embedding_scale <= 0:
            raise DFlash2ConfigError(
                "DFlash2 input_embedding_scale must be a finite positive number"
            )

        result = cls(
            hidden_size=required_int("hidden_size"),
            intermediate_size=required_int("intermediate_size"),
            num_hidden_layers=num_hidden_layers,
            num_target_layers=required_int("num_target_layers"),
            num_attention_heads=required_int("num_attention_heads"),
            num_key_value_heads=required_int("num_key_value_heads"),
            head_dim=required_int("head_dim"),
            vocab_size=required_int("vocab_size"),
            block_size=required_int("block_size", dflash),
            conv_kernel_size=required_int("conv_kernel_size", dflash),
            conv_group_size=required_int("conv_group_size", dflash),
            selector_rank=required_int("selector_rank", dflash),
            selector_top_k=required_int("selector_top_k", dflash),
            mask_token_id=mask_token_id,
            target_layer_ids=target_layer_ids,
            sliding_window=required_int("sliding_window"),
            rms_norm_eps=float(config.get("rms_norm_eps", 1e-6)),
            rope_parameters=dict(rope_parameters),
            output_multiplier=float(dflash.get("output_multiplier", 1.0)),
            final_logit_softcapping=softcap,
            input_embedding_scale=input_embedding_scale,
        )
        result._validate_internal()
        return result

    def _validate_internal(self) -> None:
        if self.num_hidden_layers != len(self.target_layer_ids):
            raise DFlash2ConfigError(
                "DFlash2 draft layer count must equal target hidden-state tap count: "
                f"layers={self.num_hidden_layers}, taps={len(self.target_layer_ids)}"
            )
        if self.num_target_layers <= max(self.target_layer_ids):
            raise DFlash2ConfigError(
                "DFlash2 target hidden-state tap exceeds num_target_layers: "
                f"taps={self.target_layer_ids}, target_layers={self.num_target_layers}"
            )
        if self.hidden_size % self.num_attention_heads:
            raise DFlash2ConfigError("DFlash2 hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise DFlash2ConfigError(
                "DFlash2 num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.hidden_size % self.conv_group_size:
            raise DFlash2ConfigError("DFlash2 hidden_size must be divisible by conv_group_size")
        if self.conv_kernel_size < 1:
            raise DFlash2ConfigError("DFlash2 conv_kernel_size must be positive")
        if self.selector_top_k > self.vocab_size:
            raise DFlash2ConfigError("DFlash2 selector_top_k cannot exceed vocab_size")
        if self.mask_token_id >= self.vocab_size:
            raise DFlash2ConfigError(
                f"DFlash2 mask_token_id={self.mask_token_id} is outside "
                f"vocab_size={self.vocab_size}"
            )
        if self.output_multiplier <= 0:
            raise DFlash2ConfigError("DFlash2 output_multiplier must be positive")
        if not math.isfinite(self.output_multiplier):
            raise DFlash2ConfigError("DFlash2 output_multiplier must be finite")


def load_dflash2_config(path: str | Path) -> DFlash2DraftConfig:
    """Read and validate ``config.json`` from a local DFlash2 directory."""

    config_path = Path(path)
    if config_path.is_dir():
        config_path /= "config.json"
    if not config_path.is_file():
        raise DFlash2ConfigError(f"DFlash2 draft config not found: {config_path}")
    if config_path.name != "config.json":
        raise DFlash2ConfigError(
            f"DFlash2 draft path must be the model directory or its config.json; got {config_path}"
        )
    try:
        raw = json.loads(config_path.read_text())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DFlash2ConfigError(f"invalid DFlash2 draft JSON: {config_path}") from exc
    if not isinstance(raw, Mapping):
        raise DFlash2ConfigError("DFlash2 draft config.json must contain an object")
    return DFlash2DraftConfig.from_dict(raw)


def _target_text_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    text_config = config.get("text_config")
    return text_config if isinstance(text_config, Mapping) else config


def validate_dflash2_target(target_config: Mapping[str, Any], draft: DFlash2DraftConfig) -> None:
    """Validate the shared hidden/vocabulary/tap contract before loading.

    DFlash2 shares the Qwen3.8 target graph, not a particular weight
    encoding.  The first integration only admitted the GGUF target because
    that was the first caller; rejecting the compressed-tensors NVFP4 target
    here made it impossible to measure DFlash2's own cost independently of
    Q6.  Keep the architecture and shape checks strict, while accepting the
    two quantized target formats that this runtime actually loads.
    """

    target = _target_text_config(target_config)
    architectures = target_config.get("architectures")
    model_type = target_config.get("model_type") or target.get("model_type")
    if architectures is not None and architectures != ["Qwen3_5ForConditionalGeneration"]:
        raise DFlash2ConfigError(
            f"DFlash2 target must be Qwen3_5ForConditionalGeneration; got {architectures!r}"
        )
    if architectures is None and model_type not in {"qwen3_5", "qwen3_5_text"}:
        raise DFlash2ConfigError(
            "DFlash2 target must be a Qwen3.5 target config; got "
            f"model_type={model_type!r}, architectures={architectures!r}"
        )
    quantization_config = target_config.get("quantization_config")
    quant_method = (
        quantization_config.get("quant_method")
        if isinstance(quantization_config, Mapping)
        else None
    )
    weight_format = target.get("weight_format") or target_config.get("weight_format")
    if weight_format != "gguf" and quant_method not in {"compressed-tensors", "modelopt"}:
        raise DFlash2ConfigError(
            "DFlash2 target must use a supported quantized format: GGUF, "
            f"compressed-tensors, or modelopt; got weight_format={weight_format!r}, "
            f"quant_method={quant_method!r}"
        )
    for name in ("hidden_size", "vocab_size", "num_hidden_layers"):
        value = target.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise DFlash2ConfigError(f"target config field {name!r} is missing or invalid")
    if draft.hidden_size != target["hidden_size"]:
        raise DFlash2ConfigError(
            f"DFlash2 hidden_size={draft.hidden_size} does not match target "
            f"hidden_size={target['hidden_size']}"
        )
    if draft.vocab_size != target["vocab_size"]:
        raise DFlash2ConfigError(
            f"DFlash2 vocab_size={draft.vocab_size} does not match target "
            f"vocab_size={target['vocab_size']}"
        )
    if draft.num_target_layers != target["num_hidden_layers"]:
        raise DFlash2ConfigError(
            f"DFlash2 target_layers={draft.num_target_layers} does not match target "
            f"layers={target['num_hidden_layers']}"
        )


def load_and_validate_dflash2_pair(
    target_path: str | Path, draft_path: str | Path
) -> DFlash2DraftConfig:
    """Load both local configs and return the validated draft contract."""

    target_config_path = Path(target_path)
    if target_config_path.is_dir():
        target_config_path /= "config.json"
    try:
        target_raw = json.loads(target_config_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise DFlash2ConfigError(f"invalid DFlash2 target config: {target_config_path}") from exc
    if not isinstance(target_raw, Mapping):
        raise DFlash2ConfigError("DFlash2 target config.json must contain an object")
    draft = load_dflash2_config(draft_path)
    validate_dflash2_target(target_raw, draft)
    return draft
