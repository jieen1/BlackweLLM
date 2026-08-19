"""Reasoning (``<think>``) span detection and metadata-artifact stripping.

Historical note: this module's docstring used to claim "the Qwen3.6 chat
template injects a ``<think>`` tag"; that model was removed from this repo
(see docs/roadmap.md R7) and the current production model, Laguna, does NOT
inject ``<think>`` via its chat template (empirically confirmed: real GPU
output for Laguna carries no ``<think>``/``</think>`` at all unless the
model chooses to emit one itself -- see notes/2026-07-27-p1-http-e2e-and-
thinking-strip-bug.md). Laguna has been observed to voluntarily emit a
``<think>...</think>`` block as the first thing it generates, exactly like a
model that opens its own reasoning block.

``find_reasoning_span`` is the single source of truth for "does this text
carry a reasoning span, and where does it end" -- ``server/formats/
stream.py``'s ``StreamProcessor`` calls it for both the streaming and the
non-streaming code path (there is exactly one implementation, not two).

THE RULE THIS MODULE ENFORCES (this is the fix for docs/roadmap.md R4):
a ``<think>``/``</think>`` pair is only ever treated as a reasoning span
when ``<think>`` is the very first thing in the text. A ``<think>`` or
``</think>`` appearing anywhere else is ordinary visible content (e.g. the
model explaining how the tag works, which is a high-frequency request for a
runtime that mostly serves code/agent workloads) and must be left
byte-for-byte untouched. The previous implementation used two unanchored
regexes -- ``_ORPHAN_CLOSE_RE`` (``r"\\A.*?</think>"``, deleted everything up
to the FIRST ``</think>`` anywhere in the response) and ``_UNCLOSED_THINK_RE``
(``r"<think>.*\\Z"``, deleted everything after any ``<think>`` anywhere) --
which is exactly what let a request like "explain how the <think> tag is
used" get silently truncated.

``<usage>...</usage>`` blocks are a DIFFERENT category of problem: a rare
model artifact (leaked Claude sub-agent output format from training data),
not a legitimate structured-content convention users ask the model to
discuss. Removing every occurrence, wherever it appears, is the deliberate
tested behavior (unlike ``<think>``, we have no evidence of models being
asked to discuss literal ``<usage>`` tags) -- kept via ``strip_usage_
artifacts`` unchanged from its pre-existing behavior.
"""

from __future__ import annotations

import os
import re

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def apply_qwen_default_reasoning_effort(tokenizer, default: str | None = None) -> str | None:
    """Set Qwen's template default without rewriting individual requests.

    The official Qwen3.8 template uses ``reasoning_effort|default('xhigh')``.
    Replacing only that Jinja default keeps the request contract simple:
    requests that omit effort use the service default, while an explicit
    ``reasoning_effort`` still wins through the template variable.  Non-Qwen
    or already-customized templates are left unchanged.
    """
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or "reasoning_effort" not in template:
        return None

    configured = default
    if configured is None:
        configured = os.environ.get("QSR_DEFAULT_REASONING_EFFORT", "medium")
    configured = configured.strip().lower()
    if configured == "high":
        configured = "xhigh"
    if configured not in {"low", "medium", "xhigh"}:
        raise ValueError(
            "QSR_DEFAULT_REASONING_EFFORT must be one of low, medium, xhigh"
        )

    for marker in ("reasoning_effort|default('xhigh')", 'reasoning_effort|default("xhigh")'):
        if marker in template:
            tokenizer.chat_template = template.replace(
                marker,
                f"reasoning_effort|default('{configured}')",
                1,
            )
            return configured
    return None

# <usage>...</usage>: paired and unclosed (hit max_tokens mid-block) forms.
_USAGE_BLOCK_RE = re.compile(r"<usage>.*?</usage>\s*", re.DOTALL)
_UNCLOSED_USAGE_RE = re.compile(r"<usage>.*\Z", re.DOTALL)


def find_reasoning_span(text: str) -> tuple[int, int, bool] | None:
    """Locate a reasoning span at the START of ``text``.

    Returns ``None`` if ``text`` does not literally start with ``<think>``
    (there is no reasoning span at all -- the whole text is content, even
    if a ``<think>``/``</think>`` appears later).

    Otherwise returns ``(start, end, closed)``: ``text[start:end]`` is the
    reasoning body (leading newline right after the open tag skipped, to
    match how models format the block). ``closed`` is ``False`` when no
    matching ``</think>`` was found -- e.g. the request hit ``max_tokens``
    mid-thought -- in which case ``end == len(text)``: the entire remainder
    is reasoning and there is no visible content at all.
    """
    if not text.startswith(THINK_OPEN):
        return None
    body = text[len(THINK_OPEN) :]
    offset = len(THINK_OPEN)
    if body.startswith("\n"):
        body = body[1:]
        offset += 1
    close_idx = body.find(THINK_CLOSE)
    if close_idx < 0:
        return offset, len(text), False
    return offset, offset + close_idx, True


def strip_usage_artifacts(text: str) -> str:
    """Remove ``<usage>...</usage>`` metadata-artifact blocks (see module
    docstring for why this is handled separately from ``<think>``)."""
    text = _USAGE_BLOCK_RE.sub("", text)
    text = _UNCLOSED_USAGE_RE.sub("", text)
    return text


def strip_thinking(text: str) -> str:
    """Text-only removal of a leading reasoning span plus ``<usage>``
    artifacts, and U+FFFD cleanup.

    This is the ``QSR_REASONING_MODE=strip`` degenerate case, and a
    convenience for callers that only have the final text (no token
    stream). Prefer ``server.formats.stream.StreamProcessor`` when the
    token stream is available: it shares this exact span-finding logic
    (via ``find_reasoning_span``) but also supports a chat template that
    injects ``<think>`` into the PROMPT (so it never appears literally in
    the generated text), which this function has no way to infer from
    text alone.
    """
    # Stray byte-level BPE tokens decode to incomplete UTF-8 / replacement
    # characters (e.g. token ids 246873/246883/247033/247081 in Qwen-family
    # vocabs); unrelated to thinking/usage, just decode-level noise cleanup.
    text = text.replace("\ufffd", "")
    span = find_reasoning_span(text)
    if span is not None:
        start, end, closed = span
        text = "" if not closed else text[end + len(THINK_CLOSE) :]
    text = strip_usage_artifacts(text)
    return text.strip()
