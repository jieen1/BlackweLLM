"""B3/serving: :class:`runtime.backends.qwen36_mtp.Qwen36MTPEngine`.

The failure mode this file exists to catch: a (token, hidden) pairing bug
that produces the RIGHT SHAPE and the RIGHT TYPE at every step -- never a
crash, never a NaN, never a vocab-range violation -- and is invisible to
any test that only checks "did a round run and return well-formed output".
Every prior standalone B3 script (``scripts/b3_mtp_e2e_acceptance_
throughput.py``, ``scripts/b3b_*.py``) had exactly this bug for weeks and
none of them caught it, because none of them asserted WHICH hidden tensor
fed the draft head's first step -- only that A hidden tensor of the right
shape did. See ``runtime/backends/qwen36_mtp.py``'s module docstring for
the full derivation (cross-checked against vLLM's own native Qwen3.6 MTP
kernel).

These tests make that content assertion directly: a stub model whose
hidden state VALUE always equals the token id that produced it (rather
than modeling anything numerically real) turns "which hidden was used" into
a plain equality check, discriminating the fix from the bug it replaces --
the OLD (buggy) code path is reconstructable at each assertion site and
verified to disagree with what these tests require.

CPU-only throughout (``device="cpu"``, stub model) -- no GPU, no real
checkpoint. What is deliberately NOT covered here: whether the REAL
``Qwen36MTPHead`` (real weights) achieves any particular acceptance rate --
that is a GPU claim (``scripts/b3_mtp_e2e_acceptance_throughput.py``, run
through the server), not something a stub can measure.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fla")
pytest.importorskip("sparkinfer")

from runtime.backends.qwen36 import Qwen36Backend  # noqa: E402
from runtime.backends.qwen36_mtp import Qwen36MTPEngine, Qwen36MTPGDNRows  # noqa: E402
from runtime.sampling import SamplingParams  # noqa: E402

_VOCAB = 1000
_CONV_DIM = 8
_CONV_K = 4
_V_HEADS = 2
_HEAD_DIM = 4


class _MTPCache:
    """Stand-in for ``Qwen36PagedAttentionCache`` -- only ``seq_len`` is
    read/written by ``Qwen36MTPEngine``."""

    def __init__(self) -> None:
        self.seq_len = 0


class _StubMTPModel:
    """A model-shaped object whose hidden states are traceable by
    construction: ``hidden(token) == token``'s own value, always. This
    turns "which hidden fed this call" into a plain float comparison,
    which is exactly the property the pairing-fix tests below need and
    which no realistic model could offer directly.

    ``compute_logits`` picks ``(hidden_value + 1) % vocab`` as the argmax
    token -- same convention ``tests/test_qwen36_backend.py``'s own
    ``_StubModel`` uses, generalized to work on any leading-dim shape
    (unlike that one, which is hardcoded 2-D) since
    ``Qwen36MTPEngine.round`` calls it on both ``[1, 1, H]`` (anchor) and
    ``[1, K, H]`` (verify) tensors.
    """

    def __init__(self) -> None:
        layers = [
            SimpleNamespace(
                layer_idx=0,
                layer_type="full_attention",
                linear_attn=None,
                self_attn=SimpleNamespace(num_kv_heads=2, head_dim=_HEAD_DIM, num_heads=4),
            ),
            SimpleNamespace(
                layer_idx=1,
                layer_type="linear_attention",
                linear_attn=SimpleNamespace(
                    conv_dim=_CONV_DIM,
                    conv_kernel_size=_CONV_K,
                    num_v_heads=_V_HEADS,
                    head_k_dim=_HEAD_DIM,
                    head_v_dim=_HEAD_DIM,
                ),
                self_attn=None,
            ),
        ]
        self.model = SimpleNamespace(layers=layers)
        self.mtp = object()  # any non-None sentinel; Qwen36MTPEngine only checks "is None"
        # `intermediate_size` is not MTP's business, but this fake stands in
        # for the whole model and `Qwen36Backend._prefill_chunk_tokens` reads
        # it to derive the prefill chunk cap (the w4a16 int32 memref bound --
        # see tests/test_qwen36_prefill_chunking.py). A fake that omits a key
        # the real config always carries makes these tests pass against a
        # model shape that does not exist.
        self.config = {"vocab_size": _VOCAB, "intermediate_size": 17408}

        self.mtp_step_calls: list[dict] = []
        self.mtp_resync_calls: list[dict] = []
        self.verify_forward_calls: list[dict] = []

    # -- backbone (Qwen36Backend's own calling convention) ------------------

    def __call__(self, input_ids: torch.Tensor, state) -> torch.Tensor:
        seq_len = int(input_ids.shape[1])
        state.num_tokens_seen += seq_len
        for cache in state.attn_caches:
            if cache is not None:
                cache.seq_len += seq_len
        return input_ids.to(torch.float32).unsqueeze(-1)  # [1, seq, 1]

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(*hidden.shape[:-1], _VOCAB)
        flat_hidden = hidden.reshape(-1)
        flat_out = out.reshape(-1, _VOCAB)
        for i in range(flat_hidden.shape[0]):
            flat_out[i, (int(flat_hidden[i].item()) + 1) % _VOCAB] = 1.0
        return out

    # -- MTP (B3) -------------------------------------------------------

    def mtp_new_cache(self, *, device, dtype) -> _MTPCache:
        return _MTPCache()

    def verify_forward(self, draft_token_ids: torch.Tensor, state):
        self.verify_forward_calls.append(
            {"drafts": draft_token_ids[0].tolist(), "past_len": state.num_tokens_seen}
        )
        hidden = draft_token_ids.to(torch.float32).unsqueeze(-1)  # [1, K, 1]
        return hidden, {}

    def commit_verify(self, state, gdn_snapshots, *, past_len: int, accepted_count: int) -> None:
        del gdn_snapshots
        state.num_tokens_seen = past_len + accepted_count

    def mtp_step(self, next_token_ids, prev_hidden, position: int, cache: _MTPCache):
        self.mtp_step_calls.append(
            {
                "token": int(next_token_ids.item()),
                "prev_hidden": float(prev_hidden.reshape(-1)[0].item()),
                "position": position,
            }
        )
        cache.seq_len += 1
        draft_token = torch.tensor([(int(next_token_ids.item()) + 1) % _VOCAB])
        hidden = next_token_ids.to(torch.float32).view(1, 1, 1)
        return draft_token, hidden

    def mtp_resync_step(self, next_token_ids, prev_hidden, start_pos: int, cache: _MTPCache):
        tokens = next_token_ids[0].tolist()
        hiddens = prev_hidden.reshape(-1).tolist()
        self.mtp_resync_calls.append({"tokens": tokens, "hiddens": hiddens, "start_pos": start_pos})
        cache.seq_len = start_pos + len(tokens)
        hidden = next_token_ids.to(torch.float32).unsqueeze(-1)
        logits = self.compute_logits(hidden)
        return logits, hidden


def _backend(num_slots: int = 3) -> tuple[Qwen36Backend, _StubMTPModel]:
    model = _StubMTPModel()
    backend = Qwen36Backend(
        model,
        num_slots=num_slots,
        max_seq_len=512,
        block_size=64,
        device="cpu",
        dtype=torch.float32,
        enable_prefix_cache=False,
    )
    return backend, model


class TestBackendWiring:
    """enable_mtp/capabilities/has_speculative_decode/reset_slot -- the
    protocol-level surface ``server/engine.py`` reaches through."""

    def test_capabilities_can_do_mtp_but_it_is_off_until_enabled(self) -> None:
        backend, _ = _backend()
        assert backend.capabilities.speculative_decode is True
        assert backend.has_speculative_decode is False

    def test_enable_mtp_flips_has_speculative_decode(self) -> None:
        backend, _ = _backend()
        backend.enable_mtp(num_speculative_tokens=4, enable_resync=False)
        assert backend.has_speculative_decode is True

    def test_enable_mtp_is_idempotent(self) -> None:
        backend, _ = _backend()
        backend.enable_mtp(num_speculative_tokens=4)
        first = backend._mtp
        backend.enable_mtp(num_speculative_tokens=8)  # second call is a no-op
        assert backend._mtp is first
        assert backend._mtp.k == 4

    def test_enable_mtp_requires_a_model_loaded_with_enable_mtp(self) -> None:
        model = _StubMTPModel()
        model.mtp = None  # simulates load_qwen36_model(..., enable_mtp=False)
        backend = Qwen36Backend(
            model,
            num_slots=2,
            max_seq_len=512,
            block_size=64,
            device="cpu",
            dtype=torch.float32,
            enable_prefix_cache=False,
        )
        with pytest.raises(ValueError, match="enable_mtp=True"):
            backend.enable_mtp(num_speculative_tokens=4)

    def test_mtp_verify_and_commit_batch_requires_enable_mtp_first(self) -> None:
        backend, _ = _backend()
        with pytest.raises(RuntimeError, match="enable_mtp"):
            backend.mtp_verify_and_commit_batch([0], {0: 1}, {0: [2, 3]})

    def test_reset_slot_clears_the_mtp_cache_too(self) -> None:
        backend, _ = _backend()
        backend.enable_mtp(num_speculative_tokens=2, enable_resync=False)
        cache = backend._mtp._caches[0]
        cache.seq_len = 7
        backend.reset_slot(0)
        assert cache.seq_len == 0

    def test_cpu_device_never_attempts_cuda_graph_capture(self) -> None:
        """2026-08-03 CUDA-Graph follow-up
        (runtime.backends.qwen36_mtp_cudagraph): capture is only ever
        attempted on a real CUDA device -- this stub model's ``model.mtp``
        is a bare ``object()`` sentinel with no ``.layers[0].self_attn``
        geometry to pool from, so an unconditional capture attempt would
        crash every test in this file, not just this one. Pinning this
        stays green even if a future edit accidentally drops the device
        guard.
        """
        backend, _ = _backend()
        backend.enable_mtp(num_speculative_tokens=2, enable_resync=False)
        engine: Qwen36MTPEngine = backend._mtp
        assert engine._use_cuda_graph is False
        assert engine.cg_status == {}
        assert engine._anchor_cg is None
        assert engine._draft_cg is None
        assert engine.cuda_graphs_healthy() is True  # vacuous: nothing attempted

    def test_spec_rows_are_fixed_and_disjoint_from_the_live_pool(self) -> None:
        """MTP state must have a stable K+2 address for every slot/column.

        A shape-only test would miss the dangerous regression: reusing the
        live row for a candidate makes rejection overwrite the state that the
        next round must continue from, while copying snapshots back hides the
        mistake behind large ``aten::copy_`` costs. This checks the actual
        pool views and the column-zero bootstrap copy without a CUDA kernel.

        K+2 columns, not K+1, since the round folded the anchor into the
        verify forward: that forward runs k+1 positions and ``spec_forward``
        requires ``seq_len + 1`` rows. Column 0 is the incoming state;
        columns 1..k+1 hold the state after each position, and accepting
        ``m`` drafts commits column ``m + 1``. See ``Qwen36MTPGDNRows``.
        """
        backend, _ = _backend()
        live_before = backend.pool.slot_state(0).gdn_states[1].conv_state
        rows = Qwen36MTPGDNRows(backend, num_speculative_tokens=3)
        rows.sync_from_live(0)
        columns = rows.rows_for_slot(0)[1]
        assert int(columns[0].conv_state.data_ptr()) != int(live_before.data_ptr())
        assert backend.pool.slot_state(0).gdn_states[1] is columns[0]
        assert len(columns) == 5  # K+2 for K=3
        assert len({int(state.conv_state.data_ptr()) for state in columns}) == 5
        assert len({int(state.recurrent_state.data_ptr()) for state in columns}) == 5
        assert rows.row_for_slot(0, 0) == 0
        assert rows.row_for_slot(0, 1) > backend.pool.num_slots


class TestPairingFix:
    """The core claim: the NEXT round's draft loop is seeded with the
    hidden state that PREDICTED the new anchor, not the hidden state
    produced by processing it. See module docstring."""

    def test_full_accept_seeds_from_verify_hidden_at_m_minus_1(self) -> None:
        """K=3, every draft accepted (m=k=3): the historically-correct
        seed is ``verify_hidden[m-1] == verify_hidden[2]`` (the hidden
        that predicted the bonus token), never ``anchor_hidden`` of the
        NEW anchor itself (which is what every prior B3 script used --
        see module docstring) and never any other row.
        """
        backend, model = _backend()
        backend.enable_mtp(num_speculative_tokens=3, enable_resync=False)
        engine: Qwen36MTPEngine = backend._mtp

        anchor_token = 10
        drafts = [11, 12, 13]  # each one more than the last: designed to
        # all be accepted under the stub's (value+1)%vocab argmax rule.
        engine._caches[0].seq_len = 3  # simulate the prior draft_after_prefill/round call

        model.mtp_step_calls.clear()
        result = engine.round(0, anchor_token, drafts)

        assert result["num_accepted"] == 3
        assert result["committed"] == [11, 12, 13, 14]
        assert result["next_anchor"] == 14

        # The re-draft step's FIRST mtp_step call is what matters here.
        first_call = model.mtp_step_calls[0]
        assert first_call["token"] == 14  # new_anchor
        assert first_call["prev_hidden"] == 13.0  # verify_hidden[m-1] = verify_hidden[2]
        # The bug this fix replaces would have used anchor_hidden of the
        # NEW anchor (14.0, from re-forwarding new_anchor through the
        # target) -- explicitly NOT what was used.
        assert first_call["prev_hidden"] != 14.0

    def test_immediate_reject_seeds_from_this_rounds_own_anchor_hidden(self) -> None:
        """m=0 (first draft already wrong): the correct seed is THIS
        round's own anchor_hidden (the hidden that predicted new_anchor,
        since new_anchor == the target's own recovery prediction from
        that exact hidden) -- row 0 of ``all_hiddens``, never
        ``verify_hidden`` at all (there is no accepted draft to source a
        verify row from).
        """
        backend, model = _backend()
        backend.enable_mtp(num_speculative_tokens=3, enable_resync=False)
        engine: Qwen36MTPEngine = backend._mtp

        anchor_token = 20
        drafts = [99, 98, 97]  # deliberately NOT anchor_argmax (21) at position 0
        engine._caches[0].seq_len = 3

        model.mtp_step_calls.clear()
        result = engine.round(0, anchor_token, drafts)

        assert result["num_accepted"] == 0
        assert result["committed"] == [21]  # the target's own recovery prediction
        assert result["next_anchor"] == 21

        first_call = model.mtp_step_calls[0]
        assert first_call["token"] == 21
        assert first_call["prev_hidden"] == 20.0  # THIS round's own anchor_hidden
        # The bug this replaces would have re-forwarded new_anchor (21)
        # through the target and used ITS hidden (21.0) instead.
        assert first_call["prev_hidden"] != 21.0

    def test_partial_accept_seeds_from_verify_hidden_at_m_minus_1(self) -> None:
        """m=1 (one accepted, then a mismatch): exercises the general
        ``verify_hidden[m-1]`` branch with ``m`` strictly between 0 and k.
        """
        backend, model = _backend()
        backend.enable_mtp(num_speculative_tokens=3, enable_resync=False)
        engine: Qwen36MTPEngine = backend._mtp

        anchor_token = 30
        # anchor_argmax = 31 -> matches drafts[0]=31 (accepted).
        # verify_hidden[0] (from drafts[0]=31) argmax = 32, but drafts[1]=50
        # does not match -> reject at p=1, m=1.
        drafts = [31, 50, 51]
        engine._caches[0].seq_len = 3

        model.mtp_step_calls.clear()
        result = engine.round(0, anchor_token, drafts)

        assert result["num_accepted"] == 1
        assert result["committed"] == [31, 32]  # accepted draft + recovery token
        assert result["next_anchor"] == 32

        first_call = model.mtp_step_calls[0]
        assert first_call["token"] == 32
        assert first_call["prev_hidden"] == 31.0  # verify_hidden[m-1] = verify_hidden[0]
        assert first_call["prev_hidden"] != 32.0

    def test_kv_bookkeeping_matches_committed_ahead_of_kv_by_one(self) -> None:
        """Same invariant DFlash's own round already keeps for Laguna
        (``runtime/backends/laguna_dflash.py``): ``slot_committed_tokens``
        includes the just-decided recovery/bonus token immediately;
        ``slot_kv_len`` only advances by what ``commit_verify`` actually
        wrote (the accepted prefix), one token behind."""
        backend, _ = _backend()
        backend.enable_mtp(num_speculative_tokens=3, enable_resync=False)
        engine: Qwen36MTPEngine = backend._mtp
        backend.pool.slot_kv_len[0] = 7
        backend.pool.slot_committed_tokens[0] = [1, 2, 3, 4, 5, 6, 10]
        backend.pool.slot_state(0).num_tokens_seen = 7
        engine._caches[0].seq_len = 3

        result = engine.round(0, 10, [11, 12, 13])

        assert backend.pool.slot_kv_len[0] == 7 + 1 + result["num_accepted"]
        assert (
            backend.pool.slot_committed_tokens[0][-len(result["committed"]) :]
            == result["committed"]
        )


class TestPrefillSeed:
    def test_draft_after_prefill_seeds_from_last_prompt_position_hidden(self) -> None:
        """``Qwen36Backend.prefill_chunked_begin`` -> ``draft_after_prefill``:
        the seed must be the hidden at the LAST prompt position (the one
        whose argmax IS ``first_token``), not a hidden produced by
        forwarding ``first_token`` itself (which prefill never even
        computes -- there is no such value to mistakenly reuse here,
        unlike the round-to-round case ``TestPairingFix`` covers)."""
        backend, model = _backend()
        backend.enable_mtp(num_speculative_tokens=2, enable_resync=False)
        model.mtp_step_calls.clear()

        prompt = [5, 6, 7]  # last position's hidden value == 7 (stub convention)
        state = backend.prefill_chunked_begin([0], [prompt], params_per_slot={})
        anchor = state.result[0]["anchor"]
        assert anchor == 8  # (7 + 1) % vocab, the stub's own argmax rule

        first_call = model.mtp_step_calls[0]
        assert first_call["token"] == anchor
        assert first_call["prev_hidden"] == 7.0  # hidden at the LAST prompt position

    def test_draft_after_prefill_requires_a_freshly_reset_slot(self) -> None:
        backend, _ = _backend()
        backend.enable_mtp(num_speculative_tokens=2, enable_resync=False)
        engine: Qwen36MTPEngine = backend._mtp
        engine._caches[0].seq_len = 5  # not fresh
        with pytest.raises(RuntimeError, match="must reset_slot first"):
            engine.draft_after_prefill(0, first_token=1, pred_hidden=torch.zeros(1, 1, 1))


class TestResync:
    """QSR_SERVER_MTP_RESYNC (default off): re-grounds the m-1 interior
    accepted positions with real verify_hidden. Indexing-only test (the
    stub cannot say anything about whether this improves acceptance on
    real weights -- that is a GPU/A-B question the coordinator schedules
    separately)."""

    def test_resync_off_by_default_does_not_call_mtp_resync_step(self) -> None:
        backend, model = _backend()
        backend.enable_mtp(num_speculative_tokens=3, enable_resync=False)
        engine: Qwen36MTPEngine = backend._mtp
        engine._caches[0].seq_len = 3
        engine.round(0, 10, [11, 12, 13])
        assert model.mtp_resync_calls == []

    def test_resync_on_rewrites_interior_positions_with_shifted_pairing(self) -> None:
        backend, model = _backend()
        backend.enable_mtp(num_speculative_tokens=3, enable_resync=True)
        engine: Qwen36MTPEngine = backend._mtp
        round_mtp_start = 5
        engine._caches[0].seq_len = round_mtp_start + 3  # as if 3 draft steps already ran

        result = engine.round(0, 10, [11, 12, 13])
        assert result["num_accepted"] == 3  # m=3 >= 2, resync should have fired

        assert len(model.mtp_resync_calls) == 1
        call = model.mtp_resync_calls[0]
        # drafts[1:m] = drafts[1:3] = [12, 13]; verify_hidden[0:m-1] =
        # verify_hidden[0:2], whose values equal drafts[0:2] = [11, 12]
        # under this stub's convention (hidden(token) == token).
        assert call["tokens"] == [12, 13]
        assert call["hiddens"] == [11.0, 12.0]
        assert call["start_pos"] == round_mtp_start + 1

    def test_resync_does_not_fire_for_m_below_2(self) -> None:
        backend, model = _backend()
        backend.enable_mtp(num_speculative_tokens=3, enable_resync=True)
        engine: Qwen36MTPEngine = backend._mtp
        engine._caches[0].seq_len = 3
        # m=1 case from TestPairingFix.
        engine.round(0, 30, [31, 50, 51])
        assert model.mtp_resync_calls == []


class TestSampledRoundStaysCorrect:
    """E2-b's non-greedy composition (params.temperature>0): reuses
    ``runtime.mtp_accept.sample_accept_reject`` with a degenerate
    (one-hot) draft distribution. That function's own docstring proves
    the output marginal always equals the target distribution regardless
    of what the draft distribution is -- correctness does not depend on
    the draft head sampling faithfully, only on target_probs being right.
    This test checks the round does not crash and returns a well-formed
    result under sampling; the acceptance-rate consequence of a
    degenerate q is a real, expected, and out-of-scope-here performance
    property, not a correctness one.
    """

    def test_sampled_round_returns_a_well_formed_result(self) -> None:
        backend, _ = _backend()
        backend.enable_mtp(num_speculative_tokens=3, enable_resync=False)
        engine: Qwen36MTPEngine = backend._mtp
        engine._caches[0].seq_len = 3
        params = SamplingParams(temperature=1.0, seed=0)

        result = engine.round(0, 10, [11, 12, 13], params=params)

        assert 0 <= result["num_accepted"] <= 3
        assert len(result["committed"]) == result["num_accepted"] + 1
        assert len(result["next_draft_tokens"]) == 3
        assert backend._mtp.stats["sampled_rounds"] == 1


class TestResyncFlagRefusesWithoutAnImplementation:
    """`QSR_SERVER_MTP_RESYNC=1` used to be a flag that could only crash.

    `Qwen36MTPEngine._resync` calls `self.model.mtp_resync_step(...)`. On the
    real model that method does not exist -- it was written only on the
    unmerged branch `work/mtp-resync-20260802` (@ aed0e2d). So setting the flag
    brought the server up normally, answered requests, and then raised
    AttributeError partway through the first round that accepted more than one
    draft token: as far from the cause as a failure can get, on a path a
    startup config flag had opted into.

    Note the stub in this file DOES define `mtp_resync_step`, which is exactly
    why nothing here caught it -- every other test in this file constructs the
    engine against a model more capable than the real one. These tests use a
    model without the method, i.e. the shape production actually has.

    The engine now refuses at construction. That is the right trade for a flag
    with no implementation behind it: porting it means ~166 lines that also
    reimplement `mtp_step` in terms of themselves, on the MTP hot path, for an
    optimization with no A/B measurement behind it.
    """

    def _engine(self, *, enable_resync: bool) -> Qwen36MTPEngine:
        backend, _ = _backend()
        backend.model.mtp = object()
        return Qwen36MTPEngine(backend, num_speculative_tokens=4, enable_resync=enable_resync)

    def test_it_refuses_when_the_model_cannot_resync(self):
        """The real model has no `mtp_resync_step`; the stub in this file does,
        which is precisely why nothing here caught the flag being unusable."""
        saved = _StubMTPModel.mtp_resync_step
        try:
            del _StubMTPModel.mtp_resync_step  # type: ignore[attr-defined]
            with pytest.raises(RuntimeError, match="mtp_resync_step"):
                self._engine(enable_resync=True)
        finally:
            _StubMTPModel.mtp_resync_step = saved  # type: ignore[attr-defined]

    def test_the_default_path_is_unaffected(self):
        """Resync off must construct exactly as before -- the guard is scoped."""
        self._engine(enable_resync=False)

    def test_a_model_that_can_resync_is_accepted(self):
        """The guard checks capability, not a version string, so a real port of
        `mtp_resync_step` turns the flag on without touching this code."""
        self._engine(enable_resync=True)
