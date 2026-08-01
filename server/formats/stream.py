"""Stateful stream processor for model output.

Splits a generation into (optional) reasoning, visible content, and tool
calls, incrementally as tokens arrive. This is the ONE state machine for
that job -- both the streaming (SSE) and non-streaming response paths in
``server/app.py`` build on it (the non-streaming path just calls
``add_tokens`` once with everything, then ``content_text()`` /
``reasoning_content()`` / ``finalize()``), instead of each maintaining its
own separate parsing logic.

Two ways a generation can carry a reasoning phase:

1. ``thinking_capable=True`` (opt-in, no backend uses this today): the
   chat template injects ``<think>`` into the PROMPT
   (``add_generation_prompt=True``), so the model's own generated tokens
   start directly with the reasoning body -- no literal ``<think>`` in the
   output. We synthesize the tag ourselves (``_get_raw()``) so the rest of
   the pipeline has one uniform representation to look at.
2. ``thinking_capable=False`` (Laguna's mode, the default in production):
   no template injection. A reasoning phase exists ONLY if the model's own
   generated text literally starts with ``<think>`` (Laguna has been
   observed to do this voluntarily) -- see ``server/formats/thinking.py``
   for why this is anchored to position 0 rather than a blind scan: a
   ``<think>``/``</think>`` appearing anywhere else is ordinary visible
   content (e.g. the model explaining how the tag is used) and must not be
   touched.

For API compatibility, reasoning is exposed alongside (not mixed into)
visible content: Anthropic gets a custom ``reasoning_content_delta`` SSE
event (see server/app.py for why NOT the spec ``thinking`` content block --
directive in commit f13fd4a), OpenAI gets ``reasoning_content`` in
``delta``/``message``.
"""

from __future__ import annotations

from server.formats.thinking import THINK_CLOSE as _THINK_CLOSE
from server.formats.thinking import THINK_OPEN as _THINK_OPEN
from server.formats.thinking import find_reasoning_span, strip_usage_artifacts
from server.formats.tools import find_tool_call_start, parse_tool_calls

_USAGE_OPEN = chr(60) + "usage" + chr(62)


def _trim_ambiguous_tail(text: str, marker: str) -> str:
    """Drop a trailing strict prefix of ``marker`` from ``text``, if any.

    Used mid-stream to avoid emitting bytes that might still turn into
    ``marker`` once more tokens arrive (e.g. ``text`` ending in ``"</thin"``
    -- four more characters and it's ``</think>``, at which point those
    bytes must NOT have already been sent as reasoning/content text).
    """
    for plen in range(min(len(marker) - 1, len(text)), 0, -1):
        if text.endswith(marker[:plen]):
            return text[: len(text) - plen]
    return text


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

    def __init__(self, tokenizer, thinking_capable: bool = True):
        self._tok = tokenizer
        self._all_ids: list[int] = []
        self._thinking_capable = thinking_capable
        # Whether a reasoning phase is present is resolved lazily (see
        # _reasoning_span): for thinking_capable=False we don't know for
        # sure until either <think> literally appears at position 0, or
        # enough tokens have decoded to rule that out.
        self._thinking_done = False
        self._tool_call_started = False
        self._emitted_len = 0
        self._thinking_emitted_len = 0
        self._last_decode_len = 0
        self._cached_raw = ""

    def add_tokens(self, token_ids: list[int]) -> None:
        self._all_ids.extend(token_ids)

    @property
    def all_ids(self) -> list[int]:
        return self._all_ids

    def _get_raw(self) -> str:
        """Decode all accumulated tokens, with <think> synthesized for a
        template-injecting backend (``thinking_capable=True``) if it isn't
        already there.

        For Laguna (``thinking_capable=False``) the decoded text is
        returned as-is: a <think> synthesized here would make every
        downstream "still thinking" check see it as permanently open, even
        though Laguna's template never injects one -- see the class
        docstring.
        """
        n = len(self._all_ids)
        if n == self._last_decode_len:
            return self._cached_raw
        decoded = self._tok.decode(self._all_ids, skip_special_tokens=True)
        # Stray byte-level BPE tokens decode to incomplete UTF-8 /
        # replacement chars; unrelated to thinking, just decode-level noise.
        decoded = decoded.replace("\ufffd", "")
        if self._thinking_capable and not decoded.startswith(_THINK_OPEN):
            self._cached_raw = _THINK_OPEN + "\n" + decoded
        else:
            self._cached_raw = decoded
        self._last_decode_len = n
        return self._cached_raw

    @property
    def thinking_done(self) -> bool:
        return self._thinking_done

    def _reasoning_span(self, raw: str, *, final: bool):
        """Locate the reasoning span in ``raw``, or report ambiguity.

        Returns ``find_reasoning_span(raw)``'s result (``None`` or
        ``(start, end, closed)``), except mid-stream (``final=False``) for
        a ``thinking_capable=False`` processor where ``raw`` is still
        shorter than ``<think>`` itself AND is a strict prefix of it (e.g.
        raw == "<thi") -- in that narrow window we genuinely cannot tell
        yet whether more tokens will complete the tag, so we return the
        sentinel ``"pending"`` and the caller must wait. At ``final=True``
        (end of generation, e.g. non-streaming or a max_tokens cutoff) this
        can never happen: whatever is there is final.
        """
        if (
            not self._thinking_capable
            and not final
            and len(raw) < len(_THINK_OPEN)
            and _THINK_OPEN.startswith(raw)
        ):
            return "pending"
        return find_reasoning_span(raw)

    def _visible_text(self, raw: str) -> str:
        """``raw`` with any reasoning span removed.

        <usage> artifacts and tool-call XML are left INTACT here --
        drain_content() needs to see them (to decide whether to freeze
        streaming output at the first occurrence); finalize()/content_text()
        strip <usage> as a separate, explicit step.
        """
        span = self._reasoning_span(raw, final=True)
        if span in (None, "pending"):
            return raw
        start, end, closed = span
        if not closed:
            return ""  # unclosed at generation end: it was ALL reasoning
        tail = raw[end + len(_THINK_CLOSE) :]
        return tail[1:] if tail.startswith("\n") else tail

    def drain_thinking(self) -> list[str]:
        """Return thinking text deltas since last call.

        Returns the raw text inside <think> tags as it accumulates.
        Returns empty list once thinking phase is complete or if
        no thinking block was detected.
        """
        if self._thinking_done:
            return []
        raw = self._get_raw()
        span = self._reasoning_span(raw, final=False)
        if span in (None, "pending"):
            return []
        start, end, closed = span
        thinking = raw[start:end]
        if not closed:
            # `end` may currently include a trailing partial "</think>"
            # that hasn't fully arrived -- withhold those bytes until we
            # know whether they complete the close tag (they must not be
            # sent as reasoning text and then "un-sent" once they do).
            thinking = _trim_ambiguous_tail(thinking, _THINK_CLOSE)
        if len(thinking) > self._thinking_emitted_len:
            delta = thinking[self._thinking_emitted_len :]
            self._thinking_emitted_len = len(thinking)
            return [delta]
        return []

    def reasoning_content(self) -> str | None:
        """Full reasoning text seen so far, or ``None`` if this generation
        carries no reasoning span. Safe to call any time (including after
        ``finalize()``); does not mutate drain state. The non-streaming
        counterpart to incrementally draining via ``drain_thinking()``."""
        raw = self._get_raw()
        span = self._reasoning_span(raw, final=True)
        if span in (None, "pending"):
            return None
        start, end, _closed = span
        text = raw[start:end].strip()
        return text or None

    def content_text(self) -> str:
        """Full visible content: reasoning removed, <usage> artifacts
        removed, tool-call XML left INTACT (callers do their own tool-call
        parsing, e.g. ``server/formats/openai.py``'s ``build_response``) --
        the non-streaming counterpart to incrementally draining via
        ``drain_content()``."""
        raw = self._get_raw()
        return strip_usage_artifacts(self._visible_text(raw))

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
            span = self._reasoning_span(raw, final=False)
            if span == "pending":
                return []  # not enough decoded text to tell yet
            if span is None:
                self._thinking_done = True  # no reasoning phase at all
            else:
                _start, _end, closed = span
                if closed:
                    self._thinking_done = True
                else:
                    return []  # still thinking, unclosed so far

        visible = self._visible_text(raw)

        # Check for tool call XML start
        tc_start = find_tool_call_start(visible)
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
        safe = _trim_ambiguous_tail(visible, _USAGE_OPEN)
        if len(safe) < len(visible):
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
          - {"type": "arguments_delta", "index": i, "delta": "...partial json..."}

        Enables streaming tool_call arguments as they are generated,
        rather than freezing until the complete tool call is parsed.

        Recognizes both interior shapes (see server/formats/tools.py):
        Qwen's ``<function=NAME>...</function>`` and Laguna's poolside_v1
        ``NAME<arg_key>...</arg_key><arg_value>...</arg_value>`` (bare name,
        no wrapper tag) -- detected per block the same way the final
        ``parse_tool_calls`` does, so a block isn't misread as one shape
        while genuinely being the other.
        """
        if not self._tool_call_started:
            return []

        raw = self._get_raw()
        visible = self._visible_text(raw)
        deltas = []

        tc_open = "<tool_call>"
        tc_close = "</tool_call>"
        func_open = "<function="
        func_close = "</function>"
        arg_key_open = "<arg_key>"

        if not hasattr(self, "_tool_names_emitted"):
            self._tool_names_emitted = set()
        if not hasattr(self, "_tool_args_emitted_len"):
            self._tool_args_emitted_len = {}

        search_start = 0
        tc_idx = 0
        while True:
            tc_pos = visible.find(tc_open, search_start)
            if tc_pos < 0:
                break
            after_open = tc_pos + len(tc_open)

            func_pos = visible.find(func_open, after_open)
            arg_key_pos = visible.find(arg_key_open, after_open)
            tc_close_pos = visible.find(tc_close, after_open)
            # Qwen shape: <function=NAME> present, and not preceded by a
            # poolside delimiter -- otherwise this is a poolside block.
            is_qwen = (
                func_pos >= 0
                and (arg_key_pos < 0 or func_pos < arg_key_pos)
                and (tc_close_pos < 0 or func_pos < tc_close_pos)
            )

            if is_qwen:
                name_end = visible.find(">", func_pos + len(func_open))
                if name_end < 0:
                    break
                func_name = visible[func_pos + len(func_open) : name_end].strip()
                args_start = name_end + 1
                args_end = visible.find(func_close, args_start)
                args_so_far = visible[args_start:] if args_end < 0 else visible[args_start:args_end]
                block_end = args_end + len(func_close) if args_end >= 0 else -1
            else:
                # Poolside shape: bare NAME up to the first <arg_key> or the
                # closing </tool_call> (zero-argument call). Neither has
                # arrived yet -- wait for more tokens before guessing.
                name_end = arg_key_pos if arg_key_pos >= 0 else tc_close_pos
                if name_end < 0:
                    break
                func_name = visible[after_open:name_end].strip()
                args_so_far = (
                    visible[name_end:] if tc_close_pos < 0 else visible[name_end:tc_close_pos]
                )
                block_end = tc_close_pos + len(tc_close) if tc_close_pos >= 0 else -1

            if tc_idx not in self._tool_names_emitted:
                self._tool_names_emitted.add(tc_idx)
                deltas.append(
                    {
                        "type": "name",
                        "index": tc_idx,
                        "name": func_name,
                        "id": f"call_{func_name}_{tc_idx}",
                    }
                )

            prev_len = self._tool_args_emitted_len.get(tc_idx, 0)
            if len(args_so_far) > prev_len:
                delta_text = args_so_far[prev_len:]
                self._tool_args_emitted_len[tc_idx] = len(args_so_far)
                deltas.append(
                    {
                        "type": "arguments_delta",
                        "index": tc_idx,
                        "delta": delta_text,
                    }
                )

            if block_end < 0:
                break
            search_start = block_end
            tc_idx += 1

        return deltas

    def finalize(self) -> tuple[str, list[dict]]:
        """Called after stream ends. Returns (visible_text, tool_calls).

        Reuses ``content_text()`` (same state machine as the streaming
        path -- see module docstring) then parses tool calls out of it, so
        a caller that wants tool calls does not need to re-derive the
        reasoning-stripped text itself.
        """
        return parse_tool_calls(self.content_text())
