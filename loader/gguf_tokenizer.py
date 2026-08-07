"""Cross-check a GGUF-embedded tokenizer against an HF tokenizer.json.

The serving path will use the official tokenizer.json (downloaded into
notes/dsv4flash-ref/), while llama.cpp — our end-to-end oracle — uses the
tokenizer embedded in the GGUF. If those two disagree, greedy token-stream
comparisons become meaningless, so this module exists to prove they are the
same tokenizer. Pure stdlib: both inputs are JSON/KV structures already
parsed by loader.gguf_header / the json module.

GGUF tokenizer KV keys (gpt2-style BPE):
  tokenizer.ggml.model        = "gpt2"
  tokenizer.ggml.tokens       = str[vocab_size]  (id -> token text)
  tokenizer.ggml.scores       = f32[vocab_size]  (optional)
  tokenizer.ggml.token_type   = i32[vocab_size]  (1=normal, 2=unused/byte-ish, 3=user-defined)
  tokenizer.ggml.merges       = str[num_merges]  ("a b" pairs)
  tokenizer.ggml.{bos,eos,padding}_token_id, add_bos_token, add_eos_token
"""

from __future__ import annotations

import collections
from typing import Any

# token_type values used by llama.cpp's GGUF writer
TOKEN_TYPE_NORMAL = 1
TOKEN_TYPE_USER_DEFINED = 3


class TokenizerMismatch:
    """One concrete disagreement between the GGUF and HF tokenizers."""

    def __init__(self, kind: str, detail: str) -> None:
        self.kind = kind
        self.detail = detail

    def __repr__(self) -> str:
        return f"TokenizerMismatch({self.kind}: {self.detail})"


def gguf_tokenizer_arrays(kv: dict[str, Any]) -> dict[str, Any]:
    """Pull the tokenizer arrays out of parsed GGUF metadata."""
    required = ("tokenizer.ggml.tokens", "tokenizer.ggml.merges")
    missing = [key for key in required if key not in kv]
    if missing:
        raise ValueError(f"GGUF metadata lacks tokenizer keys: {missing}")
    return {
        "tokens": list(kv["tokenizer.ggml.tokens"]),
        "token_type": list(kv.get("tokenizer.ggml.token_type", [])),
        "merges": list(kv["tokenizer.ggml.merges"]),
        "bos": kv.get("tokenizer.ggml.bos_token_id"),
        "eos": kv.get("tokenizer.ggml.eos_token_id"),
        "pad": kv.get("tokenizer.ggml.padding_token_id"),
        "add_bos": kv.get("tokenizer.ggml.add_bos_token"),
        "add_eos": kv.get("tokenizer.ggml.add_eos_token"),
    }


def compare_gguf_and_hf_tokenizer(
    gguf_kv: dict[str, Any],
    hf_tokenizer: dict[str, Any],
    *,
    max_reported: int = 20,
) -> list[TokenizerMismatch]:
    """Return every disagreement; empty list means the tokenizers match."""
    mismatches: list[TokenizerMismatch] = []

    def report(kind: str, detail: str) -> None:
        if len(mismatches) < max_reported:
            mismatches.append(TokenizerMismatch(kind, detail))

    gguf = gguf_tokenizer_arrays(gguf_kv)
    gguf_tokens: list[str] = gguf["tokens"]
    gguf_merges: list[str] = gguf["merges"]

    model = hf_tokenizer.get("model", {})
    if model.get("type") != "BPE":
        report("model_type", f"HF model type is {model.get('type')!r}, expected BPE")
    hf_vocab: dict[str, int] = dict(model.get("vocab", {}))
    hf_added = {int(t["id"]): t["content"] for t in hf_tokenizer.get("added_tokens", [])}
    # added_tokens may re-declare base-vocab ids (HF convention for bos/eos/pad),
    # so the id space is the union, not the sum.
    hf_total = len(set(hf_vocab.values()) | set(hf_added))

    if len(gguf_tokens) != hf_total:
        report(
            "vocab_size",
            f"GGUF has {len(gguf_tokens)} tokens, HF has {hf_total} "
            f"({len(hf_vocab)} vocab + {len(hf_added)} added)",
        )

    # ids covered by the regular HF vocab
    id_to_hf_token = {idx: token for token, idx in hf_vocab.items()}
    compared = min(len(gguf_tokens), hf_total)
    for token_id in range(compared):
        gguf_token = gguf_tokens[token_id]
        if token_id in hf_added:
            hf_token = hf_added[token_id]
        else:
            hf_token = id_to_hf_token.get(token_id)
        if hf_token is None:
            report("missing_id", f"id {token_id} absent from HF tokenizer")
            continue
        if gguf_token != hf_token:
            report(
                "token_text",
                f"id {token_id}: GGUF {gguf_token!r} != HF {hf_token!r}",
            )

    hf_merges = [
        pair if isinstance(pair, str) else " ".join(pair) for pair in model.get("merges", [])
    ]
    if len(gguf_merges) != len(hf_merges):
        report(
            "merge_count",
            f"GGUF has {len(gguf_merges)} merges, HF has {len(hf_merges)}",
        )
    for index in range(min(len(gguf_merges), len(hf_merges))):
        if gguf_merges[index] != hf_merges[index]:
            report(
                "merge",
                f"merge {index}: GGUF {gguf_merges[index]!r} != HF {hf_merges[index]!r}",
            )
        if len(mismatches) >= max_reported:
            break

    return mismatches


def gguf_tokenizer_summary(gguf_kv: dict[str, Any]) -> dict[str, Any]:
    """Compact description useful for logs and fact-baseline notes."""
    gguf = gguf_tokenizer_arrays(gguf_kv)
    type_counts = collections.Counter(gguf["token_type"]) if gguf["token_type"] else {}
    return {
        "vocab_size": len(gguf["tokens"]),
        "merges": len(gguf["merges"]),
        "bos": gguf["bos"],
        "eos": gguf["eos"],
        "pad": gguf["pad"],
        "add_bos": gguf["add_bos"],
        "add_eos": gguf["add_eos"],
        "token_type_counts": dict(type_counts),
    }
