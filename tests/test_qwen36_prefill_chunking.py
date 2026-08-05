"""A long prompt must not reach the model as one oversized forward.

`Qwen36Backend.prefill_chunked_begin` documented itself as "one-shot" and
accepted the caller's `chunk_size` only to satisfy the protocol signature --
`server/engine.py:607` even carried the comment "unused: Qwen36Backend prefill
is one-shot". So `_prefill_forward` ran the whole prompt suffix through a
single `self.model(...)` call regardless of length.

Above ~61,681 tokens that call cannot succeed. sparkinfer's w4a16 fused MoE
builds a cutlass DSL memref whose element count is `m * fc1_cols`, and with
this model's `fc1_cols = 2 * 17408 = 34816` that crosses int32 at m = 61,682.
The kernel raises OverflowError from inside the launch.

The failure was worse than a crash. `ServerEngine` swallowed the exception and
the client received HTTP 200, `finish=stop`, and **zero tokens** -- no error
field, no log line at the API layer, nothing to distinguish it from a model
that simply chose to say nothing. The model advertises `max_context=131072`
per slot, so every prompt between roughly 61.7k and 131k returned silence.

It was found by accident: a long-context throughput measurement reported
"no tokens streamed", and only the server-side log carried the OverflowError.

Two properties make this worth a permanent gate rather than a one-line fix:

- **The bound is invisible from this repo.** It comes from a descriptor width
  inside a dependency's kernel launch. Nothing here would flag a regression to
  one-shot prefill, and the symptom is silence rather than a stack trace.
- **The chunk size is derived, not pinned.** `_prefill_chunk_tokens` computes
  the hard cap from `intermediate_size`, so a model with a wider MLP tightens
  it automatically. A test that hardcodes 8192 would pass while the derivation
  rotted; these assert the relationship instead.

CPU-only: a fake model records the shapes it is called with. No GPU, no
checkpoint, no real weights.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch-free CI job")

from runtime.backends.qwen36 import (  # noqa: E402
    _PREFERRED_PREFILL_CHUNK_TOKENS,
    _W4A16_MEMREF_ELEMENT_LIMIT,
    Qwen36Backend,
)


class _FakeModel:
    """Records every forward's token count; returns correctly-shaped hidden.

    ``config`` is a plain dict because that is what
    ``Qwen36ForCausalLMSelfBuilt.config`` actually is -- the raw checkpoint
    config, not an attribute object. Modelling it as an object here is what
    let the first version of this change pass its own tests while failing
    against the real model.
    """

    def __init__(self, intermediate_size: int = 17408, hidden_size: int = 8):
        self.config = {"intermediate_size": intermediate_size}
        self.hidden_size = hidden_size
        self.calls: list[int] = []

    def __call__(self, input_ids, state):  # noqa: D102
        n = int(input_ids.shape[1])
        self.calls.append(n)
        return torch.zeros(1, n, self.hidden_size)


def _chunk_tokens_for(intermediate_size: int) -> int:
    backend = Qwen36Backend.__new__(Qwen36Backend)
    backend.model = _FakeModel(intermediate_size=intermediate_size)
    return Qwen36Backend._prefill_chunk_tokens(backend)


class TestTheDerivedCap:
    def test_this_model_uses_the_preferred_size(self):
        """At 17408 the hard cap (61,680) is far above the preferred 8192."""
        assert _chunk_tokens_for(17408) == _PREFERRED_PREFILL_CHUNK_TOKENS

    def test_a_wide_mlp_tightens_the_cap_automatically(self):
        """The whole point of deriving it: a wider MLP must shrink the chunk.

        At intermediate_size = 2,000,000 the cap falls below the preferred
        size, so the derivation -- not the constant -- has to win.
        """
        wide = 2_000_000
        got = _chunk_tokens_for(wide)
        assert got < _PREFERRED_PREFILL_CHUNK_TOKENS
        assert got == _W4A16_MEMREF_ELEMENT_LIMIT // (2 * wide)

    @pytest.mark.parametrize("intermediate_size", [1024, 17408, 65536, 2_000_000])
    def test_the_cap_never_permits_an_overflowing_forward(self, intermediate_size):
        """The invariant the bound exists for, checked directly."""
        chunk = _chunk_tokens_for(intermediate_size)
        assert chunk * (2 * intermediate_size) <= _W4A16_MEMREF_ELEMENT_LIMIT

    def test_it_never_returns_zero(self):
        """A pathological config must still make progress, not divide to 0."""
        assert _chunk_tokens_for(10**12) >= 1

    def test_a_model_without_config_falls_back_not_crashes(self):
        """Test doubles standing in for the model have no `config`.

        The cap bounds a REAL long prefill; a stub that never prefills 61k
        tokens cannot trip it. Falling back keeps unrelated suites from having
        to carry a field they have no other use for -- which is exactly what
        the first version of this change forced, breaking 18 tests across
        three files that had nothing to do with prefill.
        """

        class _NoConfig:
            pass

        backend = Qwen36Backend.__new__(Qwen36Backend)
        backend.model = _NoConfig()
        assert Qwen36Backend._prefill_chunk_tokens(backend) == _PREFERRED_PREFILL_CHUNK_TOKENS


class TestPrefillActuallyChunks:
    def _run(self, n_tokens: int, intermediate_size: int = 17408) -> list[int]:
        backend = Qwen36Backend.__new__(Qwen36Backend)
        model = _FakeModel(intermediate_size=intermediate_size)
        backend.model = model
        model.compute_logits = lambda h: torch.zeros(h.shape[0], 4)

        class _Pool:
            slot_kv_len = {0: 0}

            def slot_state(self, slot):
                return object()

        backend.pool = _Pool()
        backend.device = torch.device("cpu")
        Qwen36Backend._prefill_forward(backend, 0, list(range(n_tokens)), prefix_hit=0)
        return model.calls

    def test_a_long_prompt_is_split(self):
        calls = self._run(100_000)
        assert len(calls) > 1, "the whole prompt still went through in one forward"
        assert max(calls) <= _PREFERRED_PREFILL_CHUNK_TOKENS
        assert sum(calls) == 100_000, "chunking must not drop or duplicate tokens"

    def test_a_prompt_past_the_old_silent_limit(self):
        """61,682 tokens is the first length the single-forward path could not do."""
        calls = self._run(61_682)
        assert max(calls) * 34816 <= _W4A16_MEMREF_ELEMENT_LIMIT
        assert sum(calls) == 61_682

    def test_a_short_prompt_is_still_one_forward(self):
        """Chunking must not add round-trips to the common case."""
        assert self._run(512) == [512]

    def test_chunks_are_contiguous_and_ordered(self):
        """Order matters -- each forward continues the previous one's state."""
        calls = self._run(20_000)
        assert calls[:-1] == [_PREFERRED_PREFILL_CHUNK_TOKENS] * (len(calls) - 1)
        assert 0 < calls[-1] <= _PREFERRED_PREFILL_CHUNK_TOKENS
