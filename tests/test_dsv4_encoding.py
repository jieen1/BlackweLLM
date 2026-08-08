"""DSV4 serving-format tests: message encoding + tokenizer contract.

The deepseek_v4 backend does not carry a Jinja chat template; the official
encoding_dsv4.py message encoder defines the prompt format, and the serving
contract is EOS=1, no BOS added (plan §7.2 / D9).  These tests pin the
adapter (server/formats/dsv4_encoding.py) and the ServerEngine tokenizer
branch without any model weights.
"""

from __future__ import annotations

import pytest

from server.formats.dsv4_encoding import encode_messages_dsv4

DSV4_TOKENIZER_DIR = (
    "/home/bot/project/qwen-sm120-runtime/notes/dsv4flash-ref"
)


def test_encodes_simple_chat() -> None:
    prompt = encode_messages_dsv4(
        [{"role": "user", "content": "Hello"}]
    )
    assert "<｜User｜>" in prompt
    assert "Hello" in prompt
    assert "<｜Assistant｜>" in prompt


def test_encodes_multi_turn() -> None:
    prompt = encode_messages_dsv4(
        [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "Thanks"},
        ]
    )
    assert prompt.count("<｜User｜>") == 2
    assert prompt.count("<｜Assistant｜>") == 2  # both turns, per the official encoder
    assert "What is 2+2?" in prompt and "Thanks" in prompt
    assert prompt.count("<｜end▁of▁sentence｜>") == 1  # prior assistant turn is closed


def test_no_bos_prefix() -> None:
    # Serving contract: no BOS token is added by the encoder.
    prompt = encode_messages_dsv4([{"role": "user", "content": "Hi"}])
    assert not prompt.startswith("<｜begin▁of▁sentence｜>")


@pytest.mark.skipif(
    not __import__("pathlib").Path(DSV4_TOKENIZER_DIR).is_dir(),
    reason="DSV4 tokenizer dir not present",
)
def test_tokenizer_contract_eos_one_no_bos() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(DSV4_TOKENIZER_DIR)
    assert tok.eos_token_id == 1
    assert tok.bos_token_id is None or tok.bos_token_id != 1


@pytest.mark.skipif(
    not __import__("pathlib").Path(DSV4_TOKENIZER_DIR).is_dir(),
    reason="DSV4 tokenizer dir not present",
)
def test_encode_then_tokenize_roundtrip() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(DSV4_TOKENIZER_DIR)
    prompt = encode_messages_dsv4([{"role": "user", "content": "Hello"}])
    ids = tok.encode(prompt, add_special_tokens=False)
    assert ids  # non-empty
    assert 1 not in ids  # no EOS injected mid-prompt


@pytest.mark.skipif(
    not __import__("pathlib").Path(DSV4_TOKENIZER_DIR).is_dir(),
    reason="DSV4 tokenizer dir not present",
)
def test_server_engine_tokenizer_branch() -> None:
    """ServerEngine(backend='deepseek_v4') must load the official tokenizer
    and pin the serving contract: EOS=1, no BOS added."""
    import os

    os.environ["QSR_DSV4_TOKENIZER_DIR"] = DSV4_TOKENIZER_DIR
    try:
        from server.engine import ServerEngine

        engine = ServerEngine(
            backend="deepseek_v4",
            model="/nonexistent/DeepSeek-V4-Flash-0731.gguf",
            capacity=1,
            num_slots=2,
            enable_cudagraph=False,
            production=True,
        )
    finally:
        os.environ.pop("QSR_DSV4_TOKENIZER_DIR", None)
    assert engine.eos_token_id == 1
    assert engine.eos_token_ids == frozenset({1})
    # tokenizer loaded from the official dir (not the GGUF path).
    assert engine.tok is not None
    # encoding a chat prompt adds no BOS/EOS.
    prompt = encode_messages_dsv4([{"role": "user", "content": "Hi"}])
    ids = engine.tok.encode(prompt, add_special_tokens=False)
    assert 1 not in ids
