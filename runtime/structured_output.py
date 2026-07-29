"""C3: Structured output (JSON mode / json_schema) via xgrammar logits masking.

Provides grammar-constrained decoding for the BlackweLLM runtime:
- ``json_object`` mode: output is guaranteed valid JSON
- ``json_schema`` mode: output conforms to a user-provided JSON Schema

Integration points:
- Engine creates a GrammarState per request with response_format
- Each decode round: fill bitmask → apply to logits → sample → accept token
- MTP verify: draft tokens are checked against the grammar; rejected drafts
  that violate the grammar are treated as mismatches (existing accept/reject
  logic handles this naturally since verify logits are also masked)

Design constraints:
- xgrammar bitmask is CPU-side (int32 packed); apply via logits mask on GPU
- Grammar state is per-slot, reset on slot release
- Greedy path: mask logits before argmax (bit-identical when grammar allows argmax)
- Overhead: ~0.1ms per token for bitmask fill + GPU mask application
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# Lazy-loaded xgrammar singleton (avoid import cost when not used)
_xgr = None
_compiler = None
_tokenizer_info = None


def _ensure_xgrammar(tokenizer) -> None:
    """Initialize xgrammar compiler with the model tokenizer (once)."""
    global _xgr, _compiler, _tokenizer_info
    if _compiler is not None:
        return
    import xgrammar as xgr

    _xgr = xgr
    _tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=tokenizer.vocab_size)
    _compiler = xgr.GrammarCompiler(_tokenizer_info, max_threads=4, cache_enabled=True)
    logger.info("xgrammar compiler initialized (vocab_size=%d)", tokenizer.vocab_size)


@dataclass
class ResponseFormat:
    """Parsed response_format from the API request."""

    type: str = "text"  # "text" | "json_object" | "json_schema"
    json_schema: dict[str, Any] | None = None

    @property
    def is_constrained(self) -> bool:
        return self.type in ("json_object", "json_schema")

    @classmethod
    def from_api(cls, response_format: dict | None) -> ResponseFormat:
        if response_format is None:
            return cls(type="text")
        fmt_type = response_format.get("type", "text")
        if fmt_type == "json_object":
            return cls(type="json_object")
        elif fmt_type == "json_schema":
            schema_def = response_format.get("json_schema", {})
            schema = schema_def.get("schema", {})
            return cls(type="json_schema", json_schema=schema)
        return cls(type="text")


def _unpack_bitmask_to_mask(bitmask_row: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Unpack packed int32 bitmask to a bool mask of shape [vocab_size].

    Vectorized: uses bitwise ops on the full int32 tensor, no Python loops.
    """
    import torch

    # bitmask_row: [ceil(vocab/32)] int32
    # Expand each int32 into 32 bits using bitwise AND with powers of 2
    # Create bit position tensor [32]
    bit_positions = torch.arange(32, dtype=torch.int32)
    # Expand bitmask to [num_words, 32] via right-shift and AND
    expanded = (bitmask_row.unsqueeze(1).to(torch.int32) >> bit_positions.unsqueeze(0)) & 1
    # Flatten to [num_words * 32] and truncate to vocab_size
    flat = expanded.reshape(-1)[:vocab_size]
    return flat.bool()


class GrammarState:
    """Per-request grammar state for constrained decoding.

    Lifecycle:
      1. Created at admission with the response_format
      2. Each decode step: apply_mask(logits) → sample → accept(token_id)
      3. Destroyed when request finishes
    """

    def __init__(self, response_format: ResponseFormat, tokenizer) -> None:
        _ensure_xgrammar(tokenizer)
        self._response_format = response_format
        self._vocab_size = tokenizer.vocab_size
        self._matcher = None
        self._bitmask = None
        self._finished = False

        if response_format.type == "json_object":
            compiled = _compiler.compile_builtin_json_grammar()
            self._matcher = _xgr.GrammarMatcher(compiled)
        elif response_format.type == "json_schema":
            schema_str = json.dumps(response_format.json_schema)
            compiled = _compiler.compile_json_schema(schema_str)
            self._matcher = _xgr.GrammarMatcher(compiled)

        if self._matcher is not None:
            self._bitmask = _xgr.allocate_token_bitmask(1, self._vocab_size)

    @property
    def is_active(self) -> bool:
        return self._matcher is not None and not self._finished

    def apply_mask(self, logits: torch.Tensor) -> None:
        """Apply grammar bitmask to logits in-place (single request, shape [vocab])."""
        if not self.is_active:
            return
        self._matcher.fill_next_token_bitmask(self._bitmask)
        mask_bool = _unpack_bitmask_to_mask(self._bitmask[0], self._vocab_size)
        logits[~mask_bool.to(logits.device)] = float("-inf")

    def apply_mask_batch(self, logits: torch.Tensor, batch_idx: int) -> None:
        """Apply grammar bitmask to one row of a batch logits tensor [batch, vocab]."""
        if not self.is_active:
            return
        self._matcher.fill_next_token_bitmask(self._bitmask)
        mask_bool = _unpack_bitmask_to_mask(self._bitmask[0], self._vocab_size)
        logits[batch_idx][~mask_bool.to(logits.device)] = float("-inf")

    def accept(self, token_id: int) -> None:
        """Accept a committed token into the grammar state."""
        if not self.is_active:
            return
        self._matcher.accept_token(token_id)
        if self._matcher.is_terminated():
            self._finished = True

    def reset(self) -> None:
        """Reset grammar state (for slot reuse)."""
        if self._matcher is not None:
            self._matcher.reset()
            self._finished = False

    def rollback(self, num_tokens: int) -> None:
        """Rollback grammar state by num_tokens (for MTP reject)."""
        if not self.is_active:
            return
        self._matcher.rollback(num_tokens)
