"""Configuration contract for external Qwen DSpark draft checkpoints.

DSpark is not an attribute of the Qwen3.8 target checkpoint.  The target
checkpoint carries MTP tensors, while DSpark uses a separate dense draft
checkpoint whose ``config.json`` contains a small DFlash backbone, a Markov
head, and the target-layer tap list.  Keeping this parser independent of
torch makes the distinction testable before loading either set of weights.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DSparkConfigError(ValueError):
    """Raised when a DSpark draft or target/draft pair is incompatible."""


@dataclass(frozen=True)
class DSparkDraftConfig:
    """The fields the local DSpark runtime must consume from a draft config."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    block_size: int
    mask_token_id: int
    target_layer_ids: tuple[int, ...]
    markov_rank: int
    markov_head_type: str
    attention_mode: str
    projector_type: str
    confidence_head: bool
    confidence_head_with_markov: bool
    rms_norm_eps: float
    rope_parameters: Mapping[str, Any]

    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> DSparkDraftConfig:
        """Parse and validate one official-style ``DSparkDraftModel`` config."""

        architectures = config.get("architectures")
        if architectures != ["DSparkDraftModel"]:
            raise DSparkConfigError(
                f"DSpark draft architectures must be ['DSparkDraftModel']; got {architectures!r}"
            )
        dflash = config.get("dflash_config")
        if not isinstance(dflash, Mapping):
            raise DSparkConfigError("DSpark draft config requires an object-valued dflash_config")

        def required_int(name: str, source: Mapping[str, Any] = config) -> int:
            value = source.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise DSparkConfigError(f"DSpark draft field {name!r} must be a positive integer")
            return value

        target_layer_ids_raw = dflash.get("target_layer_ids")
        if not isinstance(target_layer_ids_raw, list) or not target_layer_ids_raw:
            raise DSparkConfigError(
                "DSpark draft dflash_config.target_layer_ids must be a non-empty list"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in target_layer_ids_raw
        ):
            raise DSparkConfigError("DSpark target_layer_ids must contain non-negative integers")
        target_layer_ids = tuple(target_layer_ids_raw)

        block_size = required_int("block_size")
        mask_token_id = required_int("mask_token_id", dflash)
        markov_rank = required_int("markov_rank")
        markov_head_type = config.get("markov_head_type")
        if not isinstance(markov_head_type, str) or not markov_head_type:
            raise DSparkConfigError("DSpark draft markov_head_type must be a non-empty string")
        attention_mode = dflash.get("attention_mode")
        projector_type = dflash.get("projector_type")
        if not isinstance(attention_mode, str) or not attention_mode:
            raise DSparkConfigError("DSpark draft dflash_config.attention_mode is required")
        if not isinstance(projector_type, str) or not projector_type:
            raise DSparkConfigError("DSpark draft dflash_config.projector_type is required")

        confidence_head = bool(
            config.get("enable_confidence_head", False)
            or dflash.get("enable_confidence_head", False)
            or config.get("confidence_head_with_markov", False)
            or dflash.get("confidence_head_with_markov", False)
        )
        confidence_head_with_markov = bool(
            config.get(
                "confidence_head_with_markov",
                dflash.get("confidence_head_with_markov", markov_rank > 0),
            )
        )
        rope_parameters = config.get("rope_parameters", {})
        if not isinstance(rope_parameters, Mapping):
            raise DSparkConfigError("DSpark draft rope_parameters must be an object")

        return cls(
            hidden_size=required_int("hidden_size"),
            intermediate_size=required_int("intermediate_size"),
            num_hidden_layers=required_int("num_hidden_layers"),
            num_attention_heads=required_int("num_attention_heads"),
            num_key_value_heads=required_int("num_key_value_heads"),
            head_dim=required_int("head_dim"),
            vocab_size=required_int("vocab_size"),
            block_size=block_size,
            mask_token_id=mask_token_id,
            target_layer_ids=target_layer_ids,
            markov_rank=markov_rank,
            markov_head_type=markov_head_type,
            attention_mode=attention_mode,
            projector_type=projector_type,
            confidence_head=confidence_head,
            confidence_head_with_markov=confidence_head_with_markov,
            rms_norm_eps=float(config.get("rms_norm_eps", 1e-6)),
            rope_parameters=dict(rope_parameters),
        )


def load_dspark_draft_config(path: str | Path) -> DSparkDraftConfig:
    """Read ``config.json`` from a local DSpark draft directory."""

    config_path = Path(path)
    if config_path.is_dir():
        config_path /= "config.json"
    if not config_path.is_file():
        raise DSparkConfigError(f"DSpark draft config not found: {config_path}")
    try:
        raw = json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        raise DSparkConfigError(f"invalid DSpark draft JSON: {config_path}") from exc
    if not isinstance(raw, Mapping):
        raise DSparkConfigError("DSpark draft config.json must contain an object")
    return DSparkDraftConfig.from_dict(raw)


def target_text_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the language-model section used by hybrid Qwen checkpoints."""

    text_config = config.get("text_config")
    return text_config if isinstance(text_config, Mapping) else config


def validate_dspark_target(target_config: Mapping[str, Any], draft: DSparkDraftConfig) -> None:
    """Validate the target/draft shape contract before any GPU allocation.

    DSpark shares the target embedding and LM head and reads hidden states
    after the configured target layers.  Attention head geometry may differ
    between target and draft; it is intentionally not compared here.
    """

    target = target_text_config(target_config)
    target_architectures = target_config.get("architectures")
    if target_architectures != ["Qwen3_5ForConditionalGeneration"]:
        raise DSparkConfigError(
            "Qwen3.8 DSpark target must be a Qwen3_5ForConditionalGeneration checkpoint; "
            f"got {target_architectures!r}"
        )
    for name in ("hidden_size", "vocab_size", "num_hidden_layers"):
        value = target.get(name)
        if not isinstance(value, int) or value <= 0:
            raise DSparkConfigError(f"target config field {name!r} is missing or invalid")
    if draft.hidden_size != target["hidden_size"]:
        raise DSparkConfigError(
            f"DSpark hidden_size={draft.hidden_size} does not match target "
            f"hidden_size={target['hidden_size']}"
        )
    if draft.vocab_size != target["vocab_size"]:
        raise DSparkConfigError(
            f"DSpark vocab_size={draft.vocab_size} does not match target "
            f"vocab_size={target['vocab_size']}"
        )
    if draft.mask_token_id >= draft.vocab_size:
        raise DSparkConfigError(
            f"DSpark mask_token_id={draft.mask_token_id} is outside vocab_size={draft.vocab_size}"
        )
    target_layers = target["num_hidden_layers"]
    if draft.num_hidden_layers != len(draft.target_layer_ids):
        raise DSparkConfigError(
            "DSpark draft layer count must equal the number of target hidden-state taps: "
            f"layers={draft.num_hidden_layers}, taps={len(draft.target_layer_ids)}"
        )
    if any(layer_id > target_layers for layer_id in draft.target_layer_ids):
        raise DSparkConfigError(
            f"DSpark target hidden-state taps {draft.target_layer_ids} exceed "
            f"target layer count {target_layers}"
        )
    if draft.markov_head_type != "vanilla":
        raise DSparkConfigError(
            "the local DSpark implementation currently supports only vanilla Markov heads; "
            f"got {draft.markov_head_type!r}"
        )
    if draft.projector_type != "dspark":
        raise DSparkConfigError(
            "the local DSpark implementation requires dflash_config.projector_type='dspark'; "
            f"got {draft.projector_type!r}"
        )


def load_and_validate_dspark_pair(
    target_path: str | Path, draft_path: str | Path
) -> DSparkDraftConfig:
    """Load both configs and return the validated draft contract."""

    target_config_path = Path(target_path)
    if target_config_path.is_dir():
        target_config_path /= "config.json"
    try:
        target_raw = json.loads(target_config_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise DSparkConfigError(f"invalid Qwen DSpark target config: {target_config_path}") from exc
    if not isinstance(target_raw, Mapping):
        raise DSparkConfigError("Qwen DSpark target config.json must contain an object")
    draft = load_dspark_draft_config(draft_path)
    validate_dspark_target(target_raw, draft)
    return draft
