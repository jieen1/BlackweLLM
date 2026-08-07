"""Prove the GGUF-embedded tokenizer equals the official tokenizer.json.

Requires the real artifacts (GGUF download + notes/dsv4flash-ref/tokenizer.json);
skips wherever they are absent so the CI jobs stay hermetic. Synthetic unit
coverage for the comparison logic lives in test_gguf_tokenizer_unit below the
skip-guarded real-artifact test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from loader.gguf_header import read_gguf_header
from loader.gguf_tokenizer import (
    compare_gguf_and_hf_tokenizer,
    gguf_tokenizer_summary,
)

GGUF_PATH = Path(
    "/home/bot/models/DeepSeek-V4-Flash-0731-GGUF/DeepSeek-V4-Flash-0731-IQ2_XS-Experts-Q8_0.gguf"
)
HF_TOKENIZER_PATH = (
    Path(__file__).resolve().parents[1] / "notes" / "dsv4flash-ref" / "tokenizer.json"
)


@pytest.mark.skipif(not GGUF_PATH.exists(), reason="GGUF download not present")
@pytest.mark.skipif(not HF_TOKENIZER_PATH.exists(), reason="official tokenizer.json not present")
def test_gguf_tokenizer_matches_official() -> None:
    header = read_gguf_header(GGUF_PATH)
    hf_tokenizer = json.loads(HF_TOKENIZER_PATH.read_text())
    mismatches = compare_gguf_and_hf_tokenizer(header.kv, hf_tokenizer)
    assert mismatches == [], f"tokenizer disagreements: {mismatches[:10]}"


@pytest.mark.skipif(not GGUF_PATH.exists(), reason="GGUF download not present")
def test_gguf_tokenizer_summary_sane() -> None:
    header = read_gguf_header(GGUF_PATH)
    summary = gguf_tokenizer_summary(header.kv)
    assert summary["vocab_size"] == 129280
    assert summary["eos"] == 1
    assert summary["bos"] == 0
    assert summary["add_bos"] is False


def test_comparison_logic_synthetic() -> None:
    """The comparator itself, on hand-built inputs (runs everywhere)."""
    gguf_kv = {
        "tokenizer.ggml.tokens": ["<s>", "</s>", "a", "b", "ab"],
        "tokenizer.ggml.token_type": [3, 3, 1, 1, 1],
        "tokenizer.ggml.merges": ["a b"],
        "tokenizer.ggml.bos_token_id": 0,
        "tokenizer.ggml.eos_token_id": 1,
    }
    hf_ok = {
        "model": {
            "type": "BPE",
            "vocab": {"a": 2, "b": 3, "ab": 4},
            "merges": ["a b"],
        },
        "added_tokens": [
            {"id": 0, "content": "<s>"},
            {"id": 1, "content": "</s>"},
        ],
    }
    assert compare_gguf_and_hf_tokenizer(gguf_kv, hf_ok) == []

    hf_bad = json.loads(json.dumps(hf_ok))
    hf_bad["model"]["vocab"]["b"] = 3
    hf_bad["added_tokens"][1]["content"] = "<eos-different>"
    hf_bad["model"]["merges"] = ["b a"]
    mismatches = compare_gguf_and_hf_tokenizer(gguf_kv, hf_bad)
    kinds = {m.kind for m in mismatches}
    assert "token_text" in kinds
    assert "merge" in kinds
