"""Stateful stream processor for model output.

Handles the two-phase nature of Qwen3.6 streaming output:
1. Thinking phase: content between <think> and </think> -- streamed as thinking
2. Content phase: visible text after </think>, possibly followed by tool-call XML

The Qwen3.6 chat template ALWAYS injects <think> at the END of the prompt
(add_generation_prompt=True). Therefore the GENERATED tokens start directly
with thinking content (no <think> prefix in generated text). The model
eventually produces </think> followed by the actual answer.

We prepend <think> to the decoded generated text so that the thinking
detection logic works correctly.

For API compatibility:
- Anthropic: thinking is streamed as "thinking" content blocks
- OpenAI: thinking is streamed as "reasoning_content" in delta
"""

from __future__ import annotations

import json

from server.formats.thinking import strip_thinking
from server.formats.tool_parsers import ToolCallParser, get_active_parser
from server.formats.tools import find_tool_call_start, parse_tool_calls

_THINK_OPEN = chr(60) + "think" + chr(62)
_THINK_CLOSE = chr(60) + "/think" + chr(62)
_USAGE_OPEN = chr(60) + "usage" + chr(62)


class StreamProcessor:
    """Accumulates token IDs and produces safe content deltas.

    Usage::

        proc = StreamProcessor(tokenizer)
        for token_batch in stream:
            proc.add_tokens(token_batch)
            for delta in proc.drain_thinking():
                yield delta  # thinking text
            for delta in proc.drain_content():
                yield delta  # safe visible text, no thinking, no tool XML
        # After stream ends:
        visible_text, tool_calls = proc.finalize()
    """

    def __init__(
        self,
        tokenizer,
        thinking_capable: bool = True,
        tool_parser: ToolCallParser | None = None,
    ):
        self._tok = tokenizer
        self._all_ids: list[int] = []
        self._thinking_capable = thinking_capable
        # Defaults to the process's active parser (set at server startup
        # from the loaded model's config) -- pass one explicitly to pin a
        # specific shape regardless of what's active (e.g. in tests).
        self._tool_parser = tool_parser or get_active_parser()
        # Non-thinking backends (e.g. Laguna) never emit a <think> block, so
        # there is no thinking phase to wait through -- start "done" so
        # drain_content() treats every token as immediately visible content
        # instead of stalling forever waiting for a </think> that never comes.
        self._thinking_done = not thinking_capable
        self._tool_call_started = False
        self._emitted_len = 0
        self._thinking_emitted_len = 0
        self._last_decode_len = 0
        self._cached_raw = ""
        self._tool_names_emitted: set[int] = set()
        self._tool_args_emitted: set[int] = set()

    def add_tokens(self, token_ids: list[int]) -> None:
        self._all_ids.extend(token_ids)

    @property
    def all_ids(self) -> list[int]:
        return self._all_ids

    def _get_raw(self) -> str:
        """Decode all accumulated tokens with <think> prepended.

        The chat template injects <think> at the end of the prompt,
        so generated tokens start with thinking content directly.
        We prepend <think> to make the thinking detection logic work.
        """
        n = len(self._all_ids)
        if n == self._last_decode_len:
            return self._cached_raw
        decoded = self._tok.decode(self._all_ids, skip_special_tokens=True)
        # Strip U+FFFD from stray byte-level BPE tokens (Qwen3.6 vocab has
        # ~14 tokens that decode to incomplete UTF-8 / replacement chars).
        decoded = decoded.replace("\ufffd", "")
        # Prepend <think> since the chat template already injected it in the
        # prompt -- only for thinking-capable backends. For a backend that
        # never emits <think> at all, synthesizing one here would make every
        # downstream "still thinking" check see it as permanently open.
        if self._thinking_capable and not decoded.startswith(_THINK_OPEN):
            self._cached_raw = _THINK_OPEN + "\n" + decoded
        else:
            self._cached_raw = decoded
        self._last_decode_len = n
        return self._cached_raw

    @property
    def thinking_done(self) -> bool:
        return self._thinking_done

    def _visible_text(self, raw: str) -> str:
        """Apply thinking filtering only for a thinking-capable model."""
        if self._thinking_capable:
            return strip_thinking(raw)
        return raw

    def drain_thinking(self) -> list[str]:
        """Return thinking text deltas since last call.

        Returns the raw text inside <think> tags as it accumulates.
        Returns empty list once thinking phase is complete or if
        no thinking block was detected.
        """
        if self._thinking_done:
            return []
        raw = self._get_raw()
        if _THINK_OPEN not in raw:
            return []
        start = raw.index(_THINK_OPEN) + len(_THINK_OPEN)
        # Skip leading newline after <think>
        if start < len(raw) and raw[start] == "\n":
            start += 1
        if _THINK_CLOSE in raw:
            end = raw.index(_THINK_CLOSE)
            thinking = raw[start:end]
        else:
            thinking = raw[start:]
        if len(thinking) > self._thinking_emitted_len:
            delta = thinking[self._thinking_emitted_len :]
            self._thinking_emitted_len = len(thinking)
            return [delta]
        return []

    def drain_content(self) -> list[str]:
        """Return list of safe content deltas since last call.

        Returns empty list if still in thinking phase or if tool call
        XML has started (content is frozen at that point).
        """
        if self._tool_call_started:
            return []

        raw = self._get_raw()

        # Phase 1: detect thinking completion
        if not self._thinking_done:
            if _THINK_CLOSE in raw:
                # Normal case: think block closed
                self._thinking_done = True
            elif _THINK_OPEN in raw:
                # Think block opened but not yet closed -- still thinking
                return []
            else:
                # No think tags at all -- should not happen with Qwen3.6
                # but handle gracefully
                self._thinking_done = True

        visible = self._visible_text(raw)

        # Check for tool call XML start
        tc_start = find_tool_call_start(visible, open_tag=self._tool_parser.open_tag)
        if tc_start >= 0:
            self._tool_call_started = True
            safe = visible[:tc_start]
            if len(safe) > self._emitted_len:
                delta = safe[self._emitted_len :]
                self._emitted_len = len(safe)
                return [delta]
            return []

        # Hold back <usage> metadata blocks (model artifact)
        usage_idx = visible.find(_USAGE_OPEN)
        if usage_idx >= 0:
            safe = visible[:usage_idx]
            if len(safe) > self._emitted_len:
                delta = safe[self._emitted_len :]
                self._emitted_len = len(safe)
                return [delta]
            return []
        # Partial <usage> prefix at end of buffer (streaming edge)
        for plen in range(len(_USAGE_OPEN) - 1, 0, -1):
            if visible.endswith(_USAGE_OPEN[:plen]):
                safe = visible[: len(visible) - plen]
                if len(safe) > self._emitted_len:
                    delta = safe[self._emitted_len :]
                    self._emitted_len = len(safe)
                    return [delta]
                return []

        # Normal content delta
        if len(visible) > self._emitted_len:
            delta = visible[self._emitted_len :]
            self._emitted_len = len(visible)
            return [delta]
        return []

    def drain_tool_deltas(self) -> list[dict]:
        """Return incremental tool call deltas since last call.

        Returns a list of delta events:
          - {"type": "name", "index": i, "name": "func_name", "id": "call_xxx"}
          - {"type": "arguments_delta", "index": i, "delta": "...json..."}

        The ``name`` event streams as soon as the function name is known.
        ``arguments_delta`` is emitted exactly once per tool call, once its
        block fully closes (``</function>`` / ``</tool_call>``) -- NOT
        incrementally character-by-character. The model's on-the-wire shape
        (Qwen's ``<parameter=K>V</parameter>``, Poolside's
        ``<arg_key>K</arg_key><arg_value>V</arg_value>``) is XML-ish, not
        JSON; streaming raw slices of it as "arguments_delta" (the previous
        implementation) handed clients a concatenated string like
        ``<arg_key>path</arg_key><arg_value>.</arg_value>`` where the
        OpenAI/Anthropic wire formats require a JSON object string --
        every OpenAI-compatible client fails to json-decode that. Emitting
        the fully-parsed, ``json.dumps``-encoded arguments as a single delta
        once the block is known-complete is what OpenAI's own "arguments
        arrive as one or more chunks that concatenate to valid JSON"
        contract requires; one whole-JSON chunk trivially satisfies it (and
        remains valid even for a client that parses eagerly on every chunk,
        unlike a genuinely-partial JSON prefix would).

        Delegates all shape knowledge to the active ``ToolCallParser``
        (``server/formats/tool_parsers/``) -- this loop only knows about the
        shared ``open_tag``/``close_tag`` wrapper and calls into the parser
        for the name boundary and the final block parse, so it works
        unchanged for any registered parser.
        """
        if not self._tool_call_started:
            return []

        parser = self._tool_parser
        raw = self._get_raw()
        visible = self._visible_text(raw)
        deltas = []

        search_start = 0
        tc_idx = 0
        while True:
            tc_pos = visible.find(parser.open_tag, search_start)
            if tc_pos < 0:
                break
            after_open = tc_pos + len(parser.open_tag)
            close_pos = visible.find(parser.close_tag, after_open)
            block_closed = close_pos >= 0
            interior_so_far = (
                visible[after_open:] if not block_closed else visible[after_open:close_pos]
            )

            call_name = parser.find_name_boundary(interior_so_far, block_closed)
            if call_name is None:
                break  # not enough arrived yet to even know the name

            if tc_idx not in self._tool_names_emitted:
                self._tool_names_emitted.add(tc_idx)
                deltas.append(
                    {
                        "type": "name",
                        "index": tc_idx,
                        "name": call_name,
                        "id": f"call_{call_name}_{tc_idx}",
                    }
                )

            if block_closed and tc_idx not in self._tool_args_emitted:
                self._tool_args_emitted.add(tc_idx)
                parsed = parser.parse_block(interior_so_far)
                arguments = parsed["arguments"] if parsed is not None else {}
                deltas.append(
                    {
                        "type": "arguments_delta",
                        "index": tc_idx,
                        "delta": json.dumps(arguments, ensure_ascii=False),
                    }
                )

            if not block_closed:
                break
            search_start = close_pos + len(parser.close_tag)
            tc_idx += 1

        return deltas

    def finalize(self) -> tuple[str, list[dict]]:
        """Called after stream ends. Returns (visible_text, tool_calls)."""
        raw = self._tok.decode(self._all_ids, skip_special_tokens=True)
        # Prepend <think> for consistent processing -- thinking-capable
        # backends only (see the matching guard in _get_raw()).
        if self._thinking_capable and not raw.startswith(_THINK_OPEN):
            raw = _THINK_OPEN + "\n" + raw
        visible = self._visible_text(raw)
        visible_text, tool_calls = parse_tool_calls(visible, parser=self._tool_parser)
        return visible_text, tool_calls
