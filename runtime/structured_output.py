"""C3: Structured output (JSON mode / json_schema) via xgrammar logits masking.

**Status (N1, docs/roadmap.md Track E -- resolved 2026-08-01): NOT WIRED IN,
BY DESIGN, NOT A BUG.** ``server/app.py`` rejects any request whose
``response_format`` is ``json_object``/``json_schema`` with a 400 before it
ever reaches the engine (``_reject_unsupported_response_format``) --
``server/engine.py`` no longer references this module at all. This is the
"fail loudly" branch, not the "wire it in" branch, and the reasoning is
recorded here so nobody re-derives it (or worse, re-wires the dead path
below without re-checking these constraints still hold):

- ``apply_mask``/``apply_mask_batch`` (below) are logically sound and were
  never the blocker -- the blocker is that this decode loop has no
  reachable *hook* to call them from for the paths that matter:
  - The prefill anchor token (the FIRST token of every request, whether or
    not it asked for structured output) is a raw, unconstrained
    ``argmax`` deep inside ``runtime/backends/laguna.py``'s
    ``prefill_chunked_begin``/``_forward`` call chain -- no
    ``SamplingParams``/grammar object even reaches that code.
  - CUDA-Graph decode replay (``LagunaCudaGraphDecode``) bakes greedy
    argmax directly into the captured graph; there is no per-token logits
    tensor to mask against without re-architecting the graph itself.
  - Eager decode's own greedy shortcut
    (``if params.is_greedy: argmax(...)`` in
    ``decode_batch_sampled``) bypasses ``sample_from_logits`` --  the one
    seam this module COULD hook into -- entirely.
  - This runtime's default temperature is 0.0 (greedy) when unset, so
    "the paths that matter" above are exactly the paths a typical
    "give me guaranteed JSON" request (no explicit temperature) takes,
    for every token including the first.
  - All three of the above live in ``runtime/backends/laguna.py`` /
    ``runtime/backends/laguna_cuda_graph.py``, out of scope for the pass
    that made this determination (file-ownership boundary with parallel
    work on those files) -- but the constraint is architectural, not just
    a scope artifact: see docs/api-layer-design.md §5.1 for the full
    writeup, including why wiring only the reachable slice
    (temperature > 0, decode tokens 2+) would silently leave the default
    case unconstrained while looking wired-in -- worse than today's
    "obviously not connected" state, not better.

**What would need to change to revive this**: a masking hook inside
``runtime/backends/laguna.py`` reachable from (a) the prefill anchor
computation and (b) either disabling CUDA Graph replay for
grammar-constrained slots or baking a per-token bitmask into the graph
itself. Until then, this module is kept as a correct, tested foundation
(``ResponseFormat``, the bitmask unpacking, the matcher wrapper) for
whoever picks that up -- not speculative deletion, but also not claiming
to work today.

Provides grammar-constrained decoding for the BlackweLLM runtime:
- ``json_object`` mode: output is guaranteed valid JSON
- ``json_schema`` mode: output conforms to a user-provided JSON Schema

Integration points (aspirational -- see status above):
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
