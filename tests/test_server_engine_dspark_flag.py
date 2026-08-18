"""Construction-time contract for the external Qwen3.8 DSpark path."""

from __future__ import annotations

import pytest

from server.engine import ServerEngine


def test_dspark_rejects_non_qwen_backend_before_tokenizer_or_gpu() -> None:
    with pytest.raises(ValueError, match="enable_dspark requires backend='qwen36'"):
        ServerEngine(
            backend="laguna",
            capacity=1,
            num_slots=1,
            enable_cudagraph=False,
            enable_dspark=True,
        )


def test_dspark_is_mutually_exclusive_with_mtp() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        ServerEngine(
            backend="qwen36",
            capacity=1,
            num_slots=1,
            enable_cudagraph=False,
            enable_dspark=True,
            enable_mtp=True,
        )


def test_dspark_allows_dynamic_target_kv_arena() -> None:
    engine = ServerEngine(
        backend="qwen36",
        capacity=1,
        num_slots=1,
        enable_cudagraph=False,
        enable_dspark=True,
        enable_prefix_cache=True,
        qwen_kv_mode="strict",
    )
    assert engine.qwen_kv_mode == "strict"
    assert engine.enable_prefix_cache is True


def test_dspark_requires_positive_speculative_width() -> None:
    with pytest.raises(ValueError, match="dspark_num_speculative_tokens"):
        ServerEngine(
            backend="qwen36",
            capacity=1,
            num_slots=1,
            enable_cudagraph=False,
            enable_dspark=True,
            dspark_num_speculative_tokens=0,
        )
