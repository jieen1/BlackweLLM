"""B3/serving: ``ServerEngine``'s ``enable_mtp`` construction-time contract.

Torch-free by construction: every assertion here fires before
``ServerEngine.__init__`` ever touches ``AutoTokenizer.from_pretrained`` or
any GPU state (the checks all sit ahead of that call, same as
``enable_session_affinity``'s own N8 guard -- see
``tests/test_engine_session_affinity.py::TestSessionAffinityRejectedAtStartup``
for the established pattern this file follows). Landing MTP without a
backend guard would mean ``ServerEngine(backend="laguna", enable_mtp=True)``
either silently no-ops (an operator thinks MTP is on; it never runs) or
crashes deep inside ``_load_laguna_model`` with a confusing
``AttributeError`` the first time a request is served -- this is the same
"fail loud at construction, before any GPU work" discipline N8 established
for the identical ``enable_session_affinity``/``warm_continue`` mismatch.
"""

from __future__ import annotations

import pytest

from server.engine import ServerEngine


class TestMtpRejectedForWrongBackend:
    def test_rejects_mtp_for_laguna_backend(self) -> None:
        # Raises before ServerEngine.__init__ reaches AutoTokenizer.from_
        # pretrained (this module's own docstring) -- genuinely torch- and
        # transformers-free, unlike the two tests below.
        with pytest.raises(ValueError, match="enable_mtp requires backend='qwen36'"):
            ServerEngine(
                backend="laguna",
                capacity=1,
                num_slots=1,
                enable_cudagraph=False,
                enable_mtp=True,
            )

    def test_default_backend_with_mtp_off_is_unaffected(self) -> None:
        # The default (and only shipped-by-default) configuration must still
        # construct without needing any qwen36-specific state at all. Needs
        # a real tokenizer load (Laguna's), unlike the rejection test above.
        pytest.importorskip("transformers")
        engine = ServerEngine(backend="laguna", capacity=1, num_slots=1, enable_cudagraph=False)
        assert engine.enable_mtp is False
        assert engine.K == 0

    def test_mtp_k_is_recorded_even_when_disabled(self) -> None:
        # mtp_num_speculative_tokens is stored regardless of enable_mtp, but
        # self.K (the capacity headroom the admission path reserves) must
        # stay 0 unless MTP is actually on -- a non-MTP request must not pay
        # capacity_ok()'s headroom for a feature it never uses.
        pytest.importorskip("transformers")
        engine = ServerEngine(
            backend="laguna",
            capacity=1,
            num_slots=1,
            enable_cudagraph=False,
            mtp_num_speculative_tokens=8,
        )
        assert engine.mtp_num_speculative_tokens == 8
        assert engine.K == 0
