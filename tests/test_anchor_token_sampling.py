"""E-N1-b0 (docs/e2e-and-quality-plan.md §2.3): the anchor/first token of
every prefill must honor ``SamplingParams`` instead of always being an
unconstrained ``argmax``.

CPU-only: no GPU or model weights required. Both production prefill entry
points -- ``LagunaBackend.prefill_with_aux`` (DFlash off) and
``DFlashEngine.dflash_prefill_bootstrap`` (DFlash on, the path production
actually takes) -- are exercised with a hand-built ``Backend``/``logits``
stub standing in for the real forward pass, following the same pattern
``tests/test_dflash_engine.py::TestPrefixChunkSnapshot`` and
``tests/test_bfdiag_ring.py`` already use for CPU-only LagunaBackend/
DFlashEngine method tests.

Two claims must both hold:
  1. ``temperature == 0`` (greedy) is byte-for-byte unchanged: the anchor is
     always exactly ``int(logits[-1].argmax(dim=-1).item())``, regardless of
     whether ``params`` is ``None``, omitted, or an explicit greedy
     ``SamplingParams``.
  2. ``temperature > 0`` (non-greedy) actually varies the anchor across
     requests for the SAME prompt -- proven with varying seeds (deterministic
     and reproducible, not a flaky "hope the RNG diverges" assertion).
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest


@pytest.fixture(autouse=True)
def _force_cpu_generator(monkeypatch):
    """Never touch CUDA in this file: the claim under test is
    SamplingParams wiring, not device placement, and this sandbox's real
    GPU may be busy with another agent's workload (B1/7-g). Forcing
    make_generator's auto-detected device to "cpu" also makes these tests
    hermetic across GPU-having and GPU-less machines alike."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _tiered_logits(vocab: int = 8):
    """Row where argmax is unique (index 2) but index 5 has real mass under
    temperature>0 -- greedy always picks 2; sampling sometimes picks 5."""
    torch = pytest.importorskip("torch")

    row = torch.full((vocab,), -50.0)
    row[2] = 3.5  # unique greedy argmax
    row[5] = 3.0  # close second: real sampling mass, not a rounding artifact
    return row.unsqueeze(0)  # shape [1, vocab], mirrors a 1-token prefill


class TestSelectAnchorToken:
    """Direct tests of the extracted helper both prefill paths route
    through -- runtime/backends/laguna.py's ``_select_anchor_token``."""

    def test_none_params_matches_argmax_bit_exact(self):
        pytest.importorskip("torch")
        from runtime.backends.laguna import _select_anchor_token

        logits = _tiered_logits()
        expected = int(logits[-1].argmax(dim=-1).item())
        assert _select_anchor_token(logits, None) == expected == 2

    def test_explicit_greedy_params_matches_argmax_bit_exact(self):
        pytest.importorskip("torch")
        from runtime.backends.laguna import _select_anchor_token
        from runtime.sampling import SamplingParams

        logits = _tiered_logits()
        expected = int(logits[-1].argmax(dim=-1).item())
        assert _select_anchor_token(logits, SamplingParams(temperature=0.0)) == expected == 2

    def test_greedy_is_stable_across_repeated_calls_and_seeds(self):
        pytest.importorskip("torch")
        from runtime.backends.laguna import _select_anchor_token
        from runtime.sampling import SamplingParams

        logits = _tiered_logits()
        results = {
            _select_anchor_token(logits, SamplingParams(temperature=0.0, seed=seed))
            for seed in range(20)
        }
        assert results == {2}

    def test_non_greedy_diversifies_across_seeds(self):
        pytest.importorskip("torch")
        from runtime.backends.laguna import _select_anchor_token
        from runtime.sampling import SamplingParams

        logits = _tiered_logits()
        tokens = [
            _select_anchor_token(logits, SamplingParams(temperature=1.0, seed=seed))
            for seed in range(40)
        ]
        assert len(set(tokens)) > 1, f"expected diversity, got {tokens}"
        # And the only real contenders given the logits are the two
        # deliberately-close candidates.
        assert set(tokens) <= {2, 5}


class _StubBackend:
    """Stands in for LagunaBackend in prefill_with_aux/_prefill_with_prefix_hit
    tests: only the attributes those two methods actually touch on the
    no-SWA, single-chunk path."""

    _swa_scratch = None
    _prefill_chunk_tokens = 10_000
    _swa_window = 0

    def __init__(self, logits):
        self.slot_kv_len = [0]
        self.slot_committed_tokens = [[]]
        self._prefix_chunk_snapshots = [None]
        self._logits = logits

    def _forward_with_aux(self, *args, **kwargs):
        return self._logits, None

    @contextmanager
    def _swa_scratch_context(self, **kwargs):
        # Real LagunaBackend._swa_scratch_context is a no-op when
        # self._swa_scratch is None (exactly this stub's case) -- this
        # mirrors that, not a shortcut that changes behavior.
        yield


class TestPrefillWithAuxAnchorSampling:
    """runtime/backends/laguna.py: LagunaBackend.prefill_with_aux, the
    DFlash-off production prefill path."""

    def test_no_prefix_hit_greedy_matches_argmax_bit_exact(self):
        pytest.importorskip("torch")
        from runtime.backends.laguna import LagunaBackend

        logits = _tiered_logits()
        expected = int(logits[-1].argmax(dim=-1).item())
        backend = _StubBackend(logits)

        first_token, _aux = LagunaBackend.prefill_with_aux(backend, 0, [1, 2, 3])

        assert first_token == expected == 2
        assert backend.slot_kv_len[0] == 3
        assert backend.slot_committed_tokens[0] == [1, 2, 3, 2]

    def test_no_prefix_hit_greedy_params_still_matches_argmax(self):
        pytest.importorskip("torch")
        from runtime.backends.laguna import LagunaBackend
        from runtime.sampling import SamplingParams

        logits = _tiered_logits()
        backend = _StubBackend(logits)

        first_token, _aux = LagunaBackend.prefill_with_aux(
            backend, 0, [1, 2, 3], params=SamplingParams(temperature=0.0)
        )
        assert first_token == 2

    def test_no_prefix_hit_non_greedy_diversifies_across_requests(self):
        pytest.importorskip("torch")
        from runtime.backends.laguna import LagunaBackend
        from runtime.sampling import SamplingParams

        logits = _tiered_logits()
        tokens = []
        for seed in range(40):
            backend = _StubBackend(logits)  # fresh "request": fresh slot state
            first_token, _aux = LagunaBackend.prefill_with_aux(
                backend, 0, [1, 2, 3], params=SamplingParams(temperature=1.0, seed=seed)
            )
            tokens.append(first_token)

        assert len(set(tokens)) > 1, f"expected diversity across requests, got {tokens}"

    def test_prefix_hit_greedy_matches_argmax_bit_exact(self):
        """Exercises _prefill_with_prefix_hit directly (the prefix_hit>0
        branch prefill_with_aux delegates to) -- same helper stub, called
        the same way prefill_with_aux's `if prefix_hit > 0:` line does."""
        pytest.importorskip("torch")
        from runtime.backends.laguna import LagunaBackend

        logits = _tiered_logits()
        expected = int(logits[-1].argmax(dim=-1).item())
        backend = _StubBackend(logits)
        backend.slot_kv_len = [4]  # 4 tokens already cached

        first_token, _aux = LagunaBackend._prefill_with_prefix_hit(
            backend, 0, [9, 9, 9, 9, 1, 2], 4
        )

        assert first_token == expected == 2

    def test_prefix_hit_non_greedy_diversifies_across_requests(self):
        pytest.importorskip("torch")
        from runtime.backends.laguna import LagunaBackend
        from runtime.sampling import SamplingParams

        logits = _tiered_logits()
        tokens = []
        for seed in range(40):
            backend = _StubBackend(logits)
            backend.slot_kv_len = [4]
            first_token, _aux = LagunaBackend._prefill_with_prefix_hit(
                backend,
                0,
                [9, 9, 9, 9, 1, 2],
                4,
                params=SamplingParams(temperature=1.0, seed=seed),
            )
            tokens.append(first_token)

        assert len(set(tokens)) > 1, f"expected diversity across requests, got {tokens}"


class _FakeDflashBackend:
    """Mirrors LagunaBackend.prefill_with_aux's anchor-selection contract
    exactly (same _select_anchor_token call) without any of the real
    forward-pass machinery -- proves dflash_prefill_bootstrap's wiring, not
    prefill_with_aux's internals (already covered above)."""

    def __init__(self, logits):
        self.slot_kv_len = [0]
        self.slot_committed_tokens = [[]]
        self._logits = logits

    def prefill_with_aux(self, slot, prompt_ids, *, prefix_hit=0, params=None):
        from runtime.backends.laguna import _select_anchor_token

        first_token = _select_anchor_token(self._logits, params)
        self.slot_kv_len[slot] = len(prompt_ids)
        self.slot_committed_tokens[slot] = list(prompt_ids) + [first_token]
        return first_token, None


def _bare_dflash_engine(backend, *, draft_logits_vocab=8):
    torch = pytest.importorskip("torch")
    from runtime.backends.laguna_dflash import DFlashEngine

    engine = object.__new__(DFlashEngine)
    engine.backend = backend
    engine._pending_draft_probs = {}
    engine._draft_cg = None
    engine._verify_cg = None
    engine._use_cuda_graph = False
    engine._cg_captured = True
    engine._draft_forward = lambda slot, anchor, kv_len: [10] * 15
    engine._draft_forward_logits = lambda slot, anchor, kv_len: torch.zeros(
        15, draft_logits_vocab
    )
    return engine


class TestDflashPrefillBootstrapAnchorSampling:
    """runtime/backends/laguna_dflash.py: DFlashEngine.dflash_prefill_bootstrap,
    the path production actually takes today (DFlash on)."""

    def test_greedy_anchor_matches_argmax_bit_exact(self):
        pytest.importorskip("torch")

        logits = _tiered_logits()
        expected = int(logits[-1].argmax(dim=-1).item())
        backend = _FakeDflashBackend(logits)
        engine = _bare_dflash_engine(backend)

        result = engine.dflash_prefill_bootstrap(0, [1, 2, 3])

        assert result["anchor"] == expected == 2
        assert result["draft_tokens"] == [10] * 15  # greedy draft path unchanged

    def test_none_params_matches_argmax_bit_exact(self):
        """Explicitly passing params=None (prefill_chunked_begin's default
        for a slot with no entry in params_per_slot) must be indistinguishable
        from omitting it."""
        pytest.importorskip("torch")

        logits = _tiered_logits()
        expected = int(logits[-1].argmax(dim=-1).item())
        backend = _FakeDflashBackend(logits)
        engine = _bare_dflash_engine(backend)

        result = engine.dflash_prefill_bootstrap(0, [1, 2, 3], params=None)
        assert result["anchor"] == expected == 2

    def test_non_greedy_anchor_diversifies_across_requests(self):
        """The regression this whole change fixes: before E-N1-b0,
        dflash_prefill_bootstrap never forwarded params to
        backend.prefill_with_aux, so this test would have failed (anchor
        pinned to argmax=2 for every seed) on the pre-fix code."""
        pytest.importorskip("torch")
        from runtime.sampling import SamplingParams

        logits = _tiered_logits()
        anchors = []
        for seed in range(40):
            backend = _FakeDflashBackend(logits)
            engine = _bare_dflash_engine(backend)
            result = engine.dflash_prefill_bootstrap(
                0, [1, 2, 3], params=SamplingParams(temperature=1.0, seed=seed)
            )
            anchors.append(result["anchor"])

        assert len(set(anchors)) > 1, f"expected diversity across requests, got {anchors}"

    def test_non_greedy_draft_tokens_still_sampled_and_cached(self):
        """Confirms the anchor fix is purely additive: E2-b's draft-token
        sampling (bonus_token now possibly-sampled instead of always-argmax)
        still runs and still populates _pending_draft_probs."""
        pytest.importorskip("torch")
        from runtime.sampling import SamplingParams

        logits = _tiered_logits()
        backend = _FakeDflashBackend(logits)
        engine = _bare_dflash_engine(backend)

        result = engine.dflash_prefill_bootstrap(
            0, [1, 2, 3], params=SamplingParams(temperature=1.0, seed=7)
        )

        assert 0 in engine._pending_draft_probs
        assert len(result["draft_tokens"]) == 15
