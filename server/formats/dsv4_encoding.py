"""DSV4 chat encoding adapter (serving contract, plan D9 / §7.2).

DeepSeek-V4-Flash does not carry a Jinja chat template; the official
message encoder (``encoding_dsv4.py``, vendored verbatim from
``notes/dsv4flash-ref/encoding/``) defines the prompt format.  This
adapter is the server's only import point for it, so the vendored
module can stay untouched.

Serving contract (plan §7.2): EOS=1, no BOS added.  The official
encoder defaults to ``add_default_bos_token=True``; the runtime
contract drops BOS, so the adapter passes ``add_default_bos_token=False``
(the tokenizer itself also defaults ``add_bos_token=False``).
"""

from __future__ import annotations

from typing import Any

from server.formats.encoding_dsv4 import encode_messages


def encode_messages_dsv4(
    messages: list[dict[str, Any]],
    tools: Any = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> str:
    """Encode OpenAI-style messages into the DSV4 prompt string.

    The official encoder expects request-level tools on the first system or
    developer message.  Copy the request before injecting them so FastAPI's
    parsed body is never mutated.  ``chat_template_kwargs`` maps the common
    API's ``enable_thinking`` and DSV4's ``reasoning_effort`` onto the
    official encoder rather than silently ignoring them.
    """
    encoded_messages = [dict(message) for message in messages]
    if tools:
        target = next(
            (
                message
                for message in encoded_messages
                if message.get("role") in {"system", "developer"}
            ),
            None,
        )
        if target is None:
            target = {"role": "system", "content": ""}
            encoded_messages.insert(0, target)
        target["tools"] = tools

    template_kwargs = dict(chat_template_kwargs or {})
    thinking_mode = template_kwargs.get("thinking_mode")
    if thinking_mode is None:
        thinking_mode = "thinking" if template_kwargs.get("enable_thinking", False) else "chat"
    return encode_messages(
        encoded_messages,
        thinking_mode=thinking_mode,
        add_default_bos_token=False,
        reasoning_effort=template_kwargs.get("reasoning_effort"),
    )
