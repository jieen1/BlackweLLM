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
    messages: list[dict[str, Any]], tools: Any = None
) -> str:
    """Encode OpenAI-style messages into the DSV4 prompt string.

    ``tools`` is accepted for signature compatibility with the server's
    chat-tokenization call sites; tool calling for DSV4 is follow-up work
    (the official encoder supports ``tools=`` inside message dicts once the
    request layer maps them in).
    """
    return encode_messages(
        messages,
        thinking_mode="chat",
        add_default_bos_token=False,
    )
