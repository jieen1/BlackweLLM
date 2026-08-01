"""Real (non-monkeypatched) coverage for
``bfdiag.daemon.session.reset_laguna_engine``.

``session.py``'s own module docstring notes it is "real, GPU-touching
code... never executed" against the actual production engine (hard no-GPU
constraint for that task). Every existing caller-side test
(``tests/test_bfdiag_workloads.py``, ``tests/test_bfdiag_daemon.py``)
monkeypatches ``reset_laguna_engine`` itself away rather than calling it, so
until now nothing exercised its actual body at all.

``bfdiag.checkpoint.testing``'s ``FakeBackend``/``FakeDFlashEngine`` are
pure-CPU, duck-typed stand-ins built for exactly this kind of thing (no GPU,
no ``runtime.*`` import) -- this module reuses them to call the REAL
function for real, rather than adding another synthetic pattern demo.

Background for why this matters (see
notes/2026-08-01-bfdiag-assertion-audit.md): ``reset_laguna_engine`` used to
be able to rely on ``backend.reset_slot(slot)`` to zero a slot's
full-attention/SWA-ring KV. The real ``reset_slot`` was rewritten to
preserve KV across resets for Laguna's own persistent prefix cache, so
``reset_laguna_engine`` now does that zeroing itself -- these tests are the
regression gate for that fix, on the actual function, not a copy of its
logic.
"""

from __future__ import annotations

# Optional torch is intentionally a collection-time skip in CPU-only CI.
# ruff: noqa: E402, I001

import pytest

torch = pytest.importorskip("torch")

from bfdiag.checkpoint.testing import FakeBackend, FakeDFlashEngine
from bfdiag.daemon.session import reset_laguna_engine


def _fresh_engine(**kwargs) -> FakeDFlashEngine:
    defaults = dict(num_slots=2, block_size=16, blocks_per_slot=8, swa_window=40)
    defaults.update(kwargs)
    backend = FakeBackend(**defaults)
    return FakeDFlashEngine(backend, draft_window=40, num_draft_layers=2)


def _prime_slot(engine: FakeDFlashEngine, slot: int, prompt_len: int = 90, rounds: int = 8) -> None:
    prompt = list(range(1, prompt_len + 1))
    boot = engine.dflash_prefill_bootstrap(slot, prompt)
    anchor, draft_tokens = boot["anchor"], boot["draft_tokens"]
    for _ in range(rounds):
        dec = engine.dflash_round(slot, anchor, draft_tokens)
        anchor, draft_tokens = dec["next_anchor"], dec["next_draft_tokens"]


def test_reset_laguna_engine_zeros_full_attention_kv_for_every_slot() -> None:
    engine = _fresh_engine(num_slots=2)
    _prime_slot(engine, slot=0)
    _prime_slot(engine, slot=1, prompt_len=60, rounds=5)
    backend = engine.backend
    assert all(bool(t.any()) for t in backend.kv_caches.values()), "test setup: expected dirty KV"

    reset_laguna_engine(engine)

    for tensor in backend.kv_caches.values():
        assert not bool(tensor.any()), "reset_laguna_engine left non-zero full-attention/SWA KV"
    for tensor in engine._draft_kv_caches.values():
        assert not bool(tensor.any())
    assert backend.slot_kv_len == [0, 0]
    assert backend.slot_committed_tokens == [[], []]


def test_reset_laguna_engine_clears_persistent_prefix_cache_for_every_slot() -> None:
    """reset_slot() alone would have just POPULATED
    _prefix_cache_tokens/_prefix_cache_kv_len from each slot's live state
    (its own contract) -- reset_laguna_engine must clear them back out so a
    canary run right after cannot spuriously prefix-hit leftover state from
    whatever ran before it."""
    engine = _fresh_engine(num_slots=2)
    _prime_slot(engine, slot=0)
    _prime_slot(engine, slot=1, prompt_len=60, rounds=5)
    backend = engine.backend

    performed = reset_laguna_engine(engine)

    assert backend._prefix_cache_tokens == [None, None]
    assert backend._prefix_cache_kv_len == [0, 0]
    assert any("prefix cache" in step for step in performed)


def test_reset_laguna_engine_returns_the_steps_it_actually_performed() -> None:
    engine = _fresh_engine(num_slots=1)
    performed = reset_laguna_engine(engine)
    assert any("KV cache blocks" in step for step in performed)
    assert any("draft KV cache" in step for step in performed)
