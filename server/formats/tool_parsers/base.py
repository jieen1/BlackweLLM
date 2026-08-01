"""Tool-call parser interface.

Each model family that has been fine-tuned to emit tool calls has its own
on-the-wire text shape for them -- there is no config shipped with a model
that declares this shape (checked: none of the locally-cached HF repos
carry one); the shape only exists as literal text inside the model's
``chat_template.jinja``. So a new model means writing a new ``ToolCallParser``
subclass by reading that one template file, not discovering it at runtime.

Modeled on vLLM's own solution to the same problem (a registry of
hand-written parsers selected by ``--tool-call-parser NAME``, e.g. ``hermes``,
``qwen3_coder``, ``mistral`` -- vLLM does not auto-detect the shape either):
each parser lives in its own module and is only responsible for its own
model's shape. Adding model C means adding ``tool_parsers/model_c.py`` and
one line in ``registry.py`` -- it never touches ``poolside_v1.py`` or
``qwen3_coder.py``.
"""

from __future__ import annotations

import abc


class ToolCallParser(abc.ABC):
    """One model family's tool-call wire format.

    ``open_tag``/``close_tag`` delimit a tool call block in the model's
    generated text. Every model observed so far uses the same
    ``<tool_call>...</tool_call>`` outer wrapper -- these are class
    attributes (not hardcoded into the scanning loop) so a future model
    using a different wrapper only needs to override them in its own
    subclass, not touch shared code.
    """

    name: str
    open_tag: str = "<tool_call>"
    close_tag: str = "</tool_call>"

    @abc.abstractmethod
    def parse_block(self, interior: str) -> dict | None:
        """Parse one CLOSED tool-call block's interior (the text between
        ``open_tag`` and ``close_tag``, exclusive of both) into
        ``{"name": str, "arguments": dict}``.

        Returns None if ``interior`` doesn't match this parser's shape at
        all (e.g. garbled model output) -- the caller leaves the block as
        untouched visible text, same as a non-match.
        """

    @abc.abstractmethod
    def find_name_boundary(self, interior_so_far: str, block_closed: bool) -> str | None:
        """Best-effort function-name extraction from a block that may
        still be streaming in.

        ``interior_so_far`` is everything decoded so far between this
        block's ``open_tag`` and (once arrived) its ``close_tag``.
        ``block_closed`` is True once the caller has located this block's
        own ``close_tag`` (so ``interior_so_far`` is then this block's
        complete interior). Returns None if the name isn't determinable
        yet from what has arrived so far -- the caller waits for more
        tokens rather than guessing.
        """
