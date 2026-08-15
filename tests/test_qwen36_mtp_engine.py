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
pytest.importorskip("b12x")

from runtime.backends.qwen36 import Qwen36Backend  # noqa: E402
from runtime.backends.qwen36_mtp import Qwen36MTPEngine, Qwen36MTPGDNRows  # noqa: E402
from runtime.backends.qwen36_mtp_cudagraph import Qwen36MTPBatchedSync  # noqa: E402
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
        self.mtp_forward_calls: list[dict] = []
        self.mtp_forward_last_logits_only_calls: list[bool] = []
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

    def mtp_forward(
        self,
        next_token_ids,
        prev_hidden,
        start_position: int,
        cache: _MTPCache,
        *,
        logits_last_position_only: bool = False,
    ):
        tokens = next_token_ids[0].tolist()
        self.mtp_forward_calls.append(
            {
                "tokens": tokens,
                "hiddens": prev_hidden.reshape(-1).tolist(),
                "start_position": start_position,
            }
        )
        assert cache.seq_len == start_position
        cache.seq_len += len(tokens)
        hidden = next_token_ids.to(torch.float32).unsqueeze(-1)
        self.mtp_forward_last_logits_only_calls.append(logits_last_position_only)
        logits_hidden = hidden[:, -1:] if logits_last_position_only else hidden
        return self.compute_logits(logits_hidden), hidden

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

    def test_unused_anchor_does_not_hide_a_healthy_verify_graph_pair(self) -> None:
        """The anchor is folded into verify, so ``unused`` is not failure."""
        backend, _ = _backend()
        backend.enable_mtp(num_speculative_tokens=2, enable_resync=False)
        engine: Qwen36MTPEngine = backend._mtp
        engine.cg_status = {"anchor": "unused", "draft": "captured", "verify": "captured"}
        assert engine.cuda_graphs_healthy() is True
        engine.cg_status["verify"] = "failed"
        assert engine.cuda_graphs_healthy() is False

    def test_snapshot_exposes_mtp_capture_outcome_to_metrics(self) -> None:
        """A successful capture is not useful if production cannot observe it."""
        backend, _ = _backend()
        backend.cg_status["decode"] = "captured"
        backend.enable_mtp(num_speculative_tokens=2, enable_resync=False)
        backend._mtp.cg_status = {  # noqa: SLF001 - asserts public snapshot result
            "anchor": "unused",
            "draft": "captured",
            "verify": "failed",
        }
        assert backend.snapshot().dflash_cg_status == (
            ("decode", "captured"),
            ("mtp_draft", "captured"),
            ("mtp_verify", "failed"),
        )

    def test_spec_rows_are_fixed_and_disjoint_from_the_live_pool(self) -> None:
        """MTP state must have a stable K+1 address for every slot/column.

        A shape-only test would miss the dangerous regression: reusing the
        live row for a candidate makes rejection overwrite the state that the
        next round must continue from, while copying snapshots back hides the
        mistake behind large ``aten::copy_`` costs. This checks the actual
        pool views and the column-zero alias without a CUDA kernel.

        The anchor-plus-K verify owns K+1 output rows.  Before the first
        verify, column 0 IS the ordinary prefill state; the forward gathers
        it then overwrites column 0 with the anchor result. Thereafter,
        accepting ``m`` drafts selects column ``m``. See ``Qwen36MTPGDNRows``.
        """
        backend, _ = _backend()
        live_before = backend.pool.slot_state(0).gdn_states[1].conv_state
        live_before.fill_(7)
        rows = Qwen36MTPGDNRows(backend, num_speculative_tokens=3)
        rows.sync_from_live(0)
        columns = rows.rows_for_slot(0)[1]
        live_after = backend.pool.slot_state(0).gdn_states[1]
        assert live_after is columns[0]
        assert int(live_after.conv_state.data_ptr()) == int(columns[0].conv_state.data_ptr())
        assert torch.equal(columns[0].conv_state, live_before)
        assert len(columns) == 4  # K+1 for K=3
        assert len({int(state.conv_state.data_ptr()) for state in columns}) == 4
        assert len({int(state.recurrent_state.data_ptr()) for state in columns}) == 4
        assert rows.row_for_slot(0, 0) == 0
        assert rows.row_for_slot(0, 1) > backend.pool.num_slots

    def test_enable_mtp_after_decode_graph_is_refused(self) -> None:
        """A graph cannot survive MTP's aliased GDN-pool extension."""
        backend, _ = _backend()
        backend._decode_graphs[1] = object()  # noqa: SLF001 - lifecycle gate
        with pytest.raises(RuntimeError, match="before capture_decode_cuda_graph"):
            backend.enable_mtp(num_speculative_tokens=3, enable_resync=False)


class TestBatchedSyncDispatch:
    def test_ragged_sync_prefers_the_one_pass_verify_body(self) -> None:
        helper = object.__new__(Qwen36MTPBatchedSync)
        helper._verify_supported = True
        helper._verify_graphs = {}
        helper._graphs = {}
        helper._fill_verify_ragged = (
            lambda slots, tokens, hidden_rows, starts: 4  # noqa: ARG005
        )
        helper._fill_ragged = lambda *args, **kwargs: (_ for _ in ()).throw(  # noqa: ARG005
            AssertionError("ragged verify path must not fall back to decode-step fill")
        )
        helper._forward_all_steps = lambda *args, **kwargs: (_ for _ in ()).throw(  # noqa: ARG005
            AssertionError("ragged verify path must not use the q-step decode loop")
        )
        calls: list[tuple[int, int]] = []

        def _forward_verify_body(batch: int, query_len: int) -> None:
            calls.append((batch, query_len))

        helper._forward_verify_body = _forward_verify_body
        helper._gather_last = lambda batch: (  # noqa: ARG005
            torch.tensor([15, 22], dtype=torch.long),
            torch.tensor([[[115.0]], [[221.0]]], dtype=torch.float32),
        )

        first_drafts, first_hidden = helper.replay_ragged(
            [0, 1],
            [[11, 12, 13, 14], [21]],
            [torch.tensor([[[10.0], [11.0], [12.0], [13.0]]]), torch.tensor([[[20.0]]])],
            [0, 0],
        )

        assert calls == [(2, 4)]
        assert first_drafts.tolist() == [15, 22]
        assert first_hidden.reshape(-1).tolist() == [115.0, 221.0]

    def test_ragged_sync_falls_back_to_q_step_loop_when_verify_support_is_unavailable(
        self,
    ) -> None:
        helper = object.__new__(Qwen36MTPBatchedSync)
        helper._verify_supported = False
        helper._verify_graphs = {}
        helper._graphs = {}
        helper._fill_verify_ragged = lambda *args, **kwargs: (_ for _ in ()).throw(  # noqa: ARG005
            AssertionError("legacy fallback must not touch verify-fill state")
        )
        helper._fill_ragged = (
            lambda slots, tokens, hidden_rows, starts: 4  # noqa: ARG005
        )
        calls: list[tuple[int, int]] = []

        def _forward_all_steps(batch: int, query_len: int) -> None:
            calls.append((batch, query_len))

        helper._forward_all_steps = _forward_all_steps
        helper._forward_verify_body = lambda *args, **kwargs: (_ for _ in ()).throw(  # noqa: ARG005
            AssertionError("legacy fallback must not use verify body")
        )
        helper._gather_last = lambda batch: (  # noqa: ARG005
            torch.tensor([15, 22], dtype=torch.long),
            torch.tensor([[[115.0]], [[221.0]]], dtype=torch.float32),
        )

        first_drafts, first_hidden = helper.replay_ragged(
            [0, 1],
            [[11, 12, 13, 14], [21]],
            [torch.tensor([[[10.0], [11.0], [12.0], [13.0]]]), torch.tensor([[[20.0]]])],
            [0, 0],
        )

        assert calls == [(2, 4)]
        assert first_drafts.tolist() == [15, 22]
        assert first_hidden.reshape(-1).tolist() == [115.0, 221.0]

    def test_ragged_sync_uses_decode_body_for_q1_even_when_verify_is_available(
        self,
    ) -> None:
        """q=1 shares decode's attention workspace and must not use verify."""
        helper = object.__new__(Qwen36MTPBatchedSync)
        helper._verify_supported = True
        helper._verify_graphs = {}
        helper._graphs = {}
        helper._fill_verify_ragged = lambda *args, **kwargs: (_ for _ in ()).throw(  # noqa: ARG005
            AssertionError("q=1 must not allocate or dispatch verify attention")
        )
        helper._forward_verify_body = lambda *args, **kwargs: (_ for _ in ()).throw(  # noqa: ARG005
            AssertionError("q=1 must not dispatch the verify body")
        )
        helper._fill_ragged = (
            lambda slots, tokens, hidden_rows, starts: 1  # noqa: ARG005
        )
        calls: list[tuple[int, int]] = []

        def _forward_all_steps(batch: int, query_len: int) -> None:
            calls.append((batch, query_len))

        helper._forward_all_steps = _forward_all_steps
        helper._gather_last = lambda batch: (  # noqa: ARG005
            torch.tensor([15, 22], dtype=torch.long),
            torch.tensor([[[115.0]], [[221.0]]], dtype=torch.float32),
        )

        first_drafts, first_hidden = helper.replay_ragged(
            [0, 1],
            [[11], [21]],
            [torch.tensor([[[10.0]]]), torch.tensor([[[20.0]]])],
            [0, 0],
        )

        assert calls == [(2, 1)]
        assert first_drafts.tolist() == [15, 22]
        assert first_hidden.reshape(-1).tolist() == [115.0, 221.0]


class TestTeacherForcedSync:
    """The MTP cache mirrors real target context before each draft tail."""

    def test_full_accept_teacher_forces_every_real_verify_position(self) -> None:
        backend, model = _backend()
        backend.enable_mtp(num_speculative_tokens=3, enable_resync=False)
        engine: Qwen36MTPEngine = backend._mtp

        anchor_token = 10
        drafts = [11, 12, 13]  # each one more than the last: designed to
        # all be accepted under the stub's (value+1)%vocab argmax rule.
        engine._caches[0].seq_len = 2  # K=3 means two physical continuation rows

        model.mtp_step_calls.clear()
        result = engine.round(0, anchor_token, drafts)

        assert result["num_accepted"] == 3
        assert result["committed"] == [11, 12, 13, 14]
        assert result["next_anchor"] == 14

        assert model.mtp_forward_calls == [
            {"tokens": [11, 12, 13, 14], "hiddens": [10.0, 11.0, 12.0, 13.0], "start_position": 0}
        ]
        assert model.mtp_forward_last_logits_only_calls == [True]
        # The continuation starts *after* the MTP step-0 output.  The old
        # tail-only implementation instead began with new_anchor=14 and
        # never wrote the real [anchor, accepted, bonus] suffix.
        first_call = model.mtp_step_calls[0]
        assert first_call["token"] == 15
        assert first_call["prev_hidden"] == 14.0

    def test_multi_slot_round_uses_one_batched_verify_replay(self) -> None:
        """M-2 must batch the target verify without cross-slot state reuse."""
        backend, model = _backend(num_slots=2)
        backend.enable_mtp(num_speculative_tokens=3, enable_resync=False)
        engine: Qwen36MTPEngine = backend._mtp

        class _BatchedVerify:
            def __init__(self) -> None:
                self.calls: list[tuple[list[int], list[list[int]], list[int]]] = []

            def replay(self, slots, tokens, past_lens):
                self.calls.append((slots, tokens, past_lens))
                hidden = torch.tensor(tokens, dtype=torch.float32).unsqueeze(-1)
                # The real graph now returns (hidden, graph-owned logits)
                # with the lm_head captured inside the graph; the stub
                # reproduces the same pair via its own compute_logits.
                return hidden, model.compute_logits(hidden)

        verify = _BatchedVerify()
        engine._verify_cg = verify
        for slot in (0, 1):
            engine._caches[slot].seq_len = 2
            backend.pool.slot_kv_len[slot] = 7
            backend.pool.slot_state(slot).num_tokens_seen = 7

        model.mtp_step_calls.clear()
        result = engine.round_batch(
            [0, 1],
            {0: 10, 1: 20},
            {0: [11, 12, 13], 1: [21, 22, 23]},
        )

        assert verify.calls == [([0, 1], [[10, 11, 12, 13], [20, 21, 22, 23]], [7, 7])]
        assert result[0]["committed"] == [11, 12, 13, 14]
        assert result[1]["committed"] == [21, 22, 23, 24]
        assert backend.pool.slot_kv_len[:2] == [11, 11]
        assert [call["tokens"] for call in model.mtp_forward_calls] == [
            [11, 12, 13, 14],
            [21, 22, 23, 24],
        ]
        assert [model.mtp_step_calls[index]["token"] for index in (0, 2)] == [15, 25]
        assert engine.stats["batched_verify_replays"] == 1
        assert backend.stats["mtp_verify_graph_slots"] == 2

    def test_multi_slot_round_with_device_drafts_fills_preallocated_verify_tokens(
        self,
    ) -> None:
        """The device-draft hot path fills the preallocated [B, K+1] buffer.

        The batched production path receives per-slot device ``[K]`` draft
        rows and must stage them into ``_verify_tokens_buf`` without any
        per-round allocation.  Regression: the pinned anchor staging was
        shaped ``[B, 1]`` and the ``[B, K+1]`` column copy used to fail with
        a broadcast error on the real server (2026-08-06), which the
        host-draft tests could not catch.
        """
        backend, model = _backend(num_slots=2)
        backend.enable_mtp(num_speculative_tokens=3, enable_resync=False)
        engine: Qwen36MTPEngine = backend._mtp

        class _BatchedVerify:
            def __init__(self) -> None:
                self.calls: list[tuple[list[int], torch.Tensor, list[int]]] = []

            def replay(self, slots, tokens, past_lens):
                self.calls.append((list(slots), tokens.clone(), list(past_lens)))
                hidden = tokens.to(torch.float32).unsqueeze(-1)
                return hidden, model.compute_logits(hidden)

        verify = _BatchedVerify()
        engine._verify_cg = verify
        for slot in (0, 1):
            engine._caches[slot].seq_len = 2
            backend.pool.slot_kv_len[slot] = 7
            backend.pool.slot_state(slot).num_tokens_seen = 7

        result = engine.round_batch(
            [0, 1],
            {0: 10, 1: 20},
            {
                0: torch.tensor([11, 12, 13]),
                1: torch.tensor([21, 22, 23]),
            },
        )

        (slots, tokens, past_lens) = verify.calls[0]
        assert slots == [0, 1]
        assert past_lens == [7, 7]
        assert tokens.tolist() == [[10, 11, 12, 13], [20, 21, 22, 23]]
        assert result[0]["committed"] == [11, 12, 13, 14]
        assert result[1]["committed"] == [21, 22, 23, 24]

    def test_multi_slot_round_batches_the_chained_draft_replay_too(self) -> None:
        """M-4 follows the historical draft funnel: B-wide at each K step.

        A correct target verify alone still leaves a serial B=1 graph replay
        per slot.  This test pins the handoff after accept/reject: one
        batched draft replay sees both new anchors and exactly the hidden row
        that predicted each one.
        """
        backend, model = _backend(num_slots=2)
        backend.enable_mtp(num_speculative_tokens=3, enable_resync=False)
        engine: Qwen36MTPEngine = backend._mtp

        class _BatchedVerify:
            def replay(self, slots, tokens, past_lens):
                del slots, past_lens
                hidden = torch.tensor(tokens, dtype=torch.float32).unsqueeze(-1)
                return hidden, model.compute_logits(hidden)

        class _BatchedDraft:
            def __init__(self) -> None:
                self.calls: list[tuple[list[int], list[int], torch.Tensor, list[int]]] = []

            def replay_batch(self, slots, seed_tokens, seed_hiddens, start_positions):
                self.calls.append(
                    (list(slots), list(seed_tokens), seed_hiddens.clone(), list(start_positions))
                )
                return {slot: [token + 1, token + 2] for slot, token in zip(slots, seed_tokens)}

        engine._verify_cg = _BatchedVerify()
        draft = _BatchedDraft()
        engine._draft_cg = draft
        for slot in (0, 1):
            engine._caches[slot].seq_len = 2
            backend.pool.slot_kv_len[slot] = 7
            backend.pool.slot_state(slot).num_tokens_seen = 7

        result = engine.round_batch(
            [0, 1],
            {0: 10, 1: 20},
            {0: [11, 12, 13], 1: [21, 22, 23]},
        )

        assert len(draft.calls) == 1
        slots, tokens, hiddens, starts = draft.calls[0]
        assert slots == [0, 1]
        assert tokens == [15, 25]
        assert hiddens.reshape(-1).tolist() == [14.0, 24.0]
        assert starts == [4, 4]
        assert result[0]["next_draft_tokens"] == [15, 16, 17]
        assert result[1]["next_draft_tokens"] == [25, 26, 27]
        assert len(result[0]["next_draft_tokens"]) == engine.k
        assert len(result[1]["next_draft_tokens"]) == engine.k
        assert engine.stats["batched_draft_replays"] == 1
        assert backend.stats["mtp_draft_graph_slots"] == 2

    def test_ragged_sync_hands_device_seeds_to_the_draft_graph_without_tolist(self) -> None:
        """The ragged-sync graph path must not round-trip first-draft seeds
        through the host before the draft graph is enqueued.

        Measured 2026-08-05 at 128K: ``first_drafts.tolist()`` between the
        sync replay and the draft replay blocked the host on a second D2H
        synchronisation point every round (5-15 ms/round).  The batched
        ragged path now returns the device ``_step_tokens`` row and
        ``_continue_draft_batch`` stages it D2D; the int conversion happens
        only after the draft replay has already synchronized the stream.
        """
        backend, _model = _backend(num_slots=2)
        backend.enable_mtp(num_speculative_tokens=3, enable_resync=False)
        engine: Qwen36MTPEngine = backend._mtp

        class _RaggedSync:
            def __init__(self) -> None:
                self.calls: list[list[int]] = []

            def replay_ragged(self, slots, shifted_token_ids, target_hidden_rows, starts):
                self.calls.append(list(slots))
                first_drafts = torch.tensor(
                    [(tokens[-1] + 1) % _VOCAB for tokens in shifted_token_ids],
                    dtype=torch.long,
                )
                mtp_hidden = torch.stack(
                    [row[:, -1:] for row in target_hidden_rows], dim=0
                )  # [B, 1, H]
                return first_drafts, mtp_hidden

        class _DeviceSeedDraft:
            def __init__(self) -> None:
                self.seed_kind: list[str] = []
                self.seed_values: list[int] = []

            def replay_batch(self, slots, seed_tokens, seed_hiddens, start_positions):
                del seed_hiddens, start_positions
                self.seed_kind = [
                    "tensor" if isinstance(seed, torch.Tensor) else "int" for seed in seed_tokens
                ]
                self.seed_values = [int(seed.item()) for seed in seed_tokens]
                return {
                    slot: [(value + 1) % _VOCAB, (value + 2) % _VOCAB]
                    for slot, value in zip(slots, self.seed_values)
                }

        engine._batched_sync = _RaggedSync()
        engine._draft_cg = _DeviceSeedDraft()
        for slot in (0, 1):
            engine._caches[slot].seq_len = 2
            engine._sync_len[slot] = 2

        h0 = torch.tensor([[[13.0], [14.0]]], dtype=torch.float32)
        h1 = torch.tensor([[[23.0], [24.0]]], dtype=torch.float32)
        first_by_slot = engine._sync_real_suffix_batch_ragged(
            [0, 1],
            [[11, 12], [21, 22]],
            [h0, h1],
        )
        # Device seeds, not host ints, leave the sync phase.
        assert isinstance(first_by_slot[0][0], torch.Tensor)
        assert first_by_slot[0][0].item() == 13
        assert isinstance(first_by_slot[1][0], torch.Tensor)
        assert first_by_slot[1][0].item() == 23

        first_drafts = [first_by_slot[slot][0] for slot in (0, 1)]
        first_hiddens = torch.cat([first_by_slot[slot][1] for slot in (0, 1)], dim=0)
        next_drafts = engine._continue_draft_batch([0, 1], first_drafts, first_hiddens)
        assert next_drafts == {0: [13, 14, 15], 1: [23, 24, 25]}
        assert engine._draft_cg.seed_kind == ["tensor", "tensor"]
        assert engine._draft_cg.seed_values == [13, 23]

    def test_equal_acceptance_lengths_fall_back_to_one_grouped_batched_sync(self) -> None:
        """The pre-ragged grouped fast path remains a valid fallback."""
        backend, model = _backend(num_slots=2)
        backend.enable_mtp(num_speculative_tokens=3, enable_resync=False)
        engine: Qwen36MTPEngine = backend._mtp

        class _BatchedVerify:
            def replay(self, slots, tokens, past_lens):
                del slots, past_lens
                hidden = torch.tensor(tokens, dtype=torch.float32).unsqueeze(-1)
                return hidden, model.compute_logits(hidden)

        class _BatchedSync:
            def __init__(self) -> None:
                self.calls: list[tuple[list[int], list[list[int]], torch.Tensor, list[int]]] = []

            def replay(self, slots, tokens, hiddens, starts):
                self.calls.append((list(slots), list(tokens), hiddens.clone(), list(starts)))
                first_drafts = torch.tensor(
                    [(token_ids[-1] + 1) % _VOCAB for token_ids in tokens], dtype=torch.long
                )
                return first_drafts, hiddens[:, -1:]

        engine._verify_cg = _BatchedVerify()
        sync = _BatchedSync()
        engine._batched_sync = sync
        for slot in (0, 1):
            engine._caches[slot].seq_len = 2

        result = engine.round_batch(
            [0, 1],
            {0: 10, 1: 20},
            {0: [11, 12, 13], 1: [21, 22, 23]},
        )

        assert len(sync.calls) == 1
        slots, tokens, hiddens, starts = sync.calls[0]
        assert slots == [0, 1]
        assert tokens == [[11, 12, 13, 14], [21, 22, 23, 24]]
        assert hiddens.reshape(-1).tolist() == [10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0]
        assert starts == [0, 0]
        assert model.mtp_forward_calls == []
        assert result[0]["next_draft_tokens"] == [15, 16, 17]
        assert result[1]["next_draft_tokens"] == [25, 26, 27]
        assert engine.stats["batched_sync_replays"] == 1
        assert backend.stats["mtp_batched_sync_slots"] == 2

    def test_ragged_acceptance_lengths_use_one_all_slot_batched_sync_and_rewind_real_boundaries(
        self,
    ) -> None:
        """The ragged fast path pads once, then resumes each slot at its real end."""
        backend, model = _backend(num_slots=2)
        backend.enable_mtp(num_speculative_tokens=3, enable_resync=False)
        engine: Qwen36MTPEngine = backend._mtp

        class _BatchedVerify:
            def replay(self, slots, tokens, past_lens):
                del slots, past_lens
                hidden = torch.tensor(tokens, dtype=torch.float32).unsqueeze(-1)
                return hidden, model.compute_logits(hidden)

        class _RaggedSync:
            def __init__(self) -> None:
                self.calls: list[
                    tuple[list[int], list[list[int]], list[torch.Tensor], list[int]]
                ] = []

            def replay_ragged(self, slots, tokens, hidden_rows, starts):
                self.calls.append(
                    (list(slots), list(tokens), [row.clone() for row in hidden_rows], list(starts))
                )
                first_drafts = torch.tensor([15, 22], dtype=torch.long)
                first_hidden = torch.tensor([[[115.0]], [[221.0]]], dtype=torch.float32)
                return first_drafts, first_hidden

        class _BatchedDraft:
            def __init__(self) -> None:
                self.calls: list[tuple[list[int], list[int], torch.Tensor, list[int]]] = []

            def replay_batch(self, slots, seed_tokens, seed_hiddens, start_positions):
                self.calls.append(
                    (list(slots), list(seed_tokens), seed_hiddens.clone(), list(start_positions))
                )
                return {
                    slot: [token + 1, token + 2]
                    for slot, token in zip(slots, seed_tokens, strict=True)
                }

        engine._verify_cg = _BatchedVerify()
        sync = _RaggedSync()
        draft = _BatchedDraft()
        engine._batched_sync = sync
        engine._draft_cg = draft
        for slot in (0, 1):
            engine._caches[slot].seq_len = 2

        result = engine.round_batch(
            [0, 1],
            {0: 10, 1: 20},
            {0: [11, 12, 13], 1: [99, 98, 97]},
        )

        assert result[0]["num_accepted"] == 3
        assert result[1]["num_accepted"] == 0
        assert len(sync.calls) == 1
        slots, tokens, hidden_rows, starts = sync.calls[0]
        assert slots == [0, 1]
        assert tokens == [[11, 12, 13, 14], [21]]
        assert [row.reshape(-1).tolist() for row in hidden_rows] == [
            [10.0, 11.0, 12.0, 13.0],
            [20.0],
        ]
        assert starts == [0, 0]
        assert model.mtp_forward_calls == []
        assert len(draft.calls) == 1
        draft_slots, seed_tokens, seed_hiddens, draft_starts = draft.calls[0]
        assert draft_slots == [0, 1]
        assert seed_tokens == [15, 22]
        assert seed_hiddens.reshape(-1).tolist() == [115.0, 221.0]
        assert draft_starts == [4, 1]
        assert result[0]["next_draft_tokens"] == [15, 16, 17]
        assert result[1]["next_draft_tokens"] == [22, 23, 24]
        assert engine.stats["batched_sync_replays"] == 1

    def test_ragged_lengths_fall_back_to_per_slot_sync_when_all_slot_helper_is_missing(
        self,
    ) -> None:
        """Missing ragged support must preserve the old per-slot-correct path."""
        backend, model = _backend(num_slots=2)
        backend.enable_mtp(num_speculative_tokens=3, enable_resync=False)
        engine: Qwen36MTPEngine = backend._mtp

        class _BatchedVerify:
            def replay(self, slots, tokens, past_lens):
                del slots, past_lens
                hidden = torch.tensor(tokens, dtype=torch.float32).unsqueeze(-1)
                return hidden, model.compute_logits(hidden)

        class _LegacyOnly:
            def replay(self, *args, **kwargs):
                raise AssertionError("singleton fallback must not try the grouped helper")

        engine._verify_cg = _BatchedVerify()
        engine._batched_sync = _LegacyOnly()
        for slot in (0, 1):
            engine._caches[slot].seq_len = 2

        result = engine.round_batch(
            [0, 1],
            {0: 10, 1: 20},
            {0: [11, 12, 13], 1: [99, 98, 97]},
        )

        assert result[0]["num_accepted"] == 3
        assert result[1]["num_accepted"] == 0
        assert [call["tokens"] for call in model.mtp_forward_calls] == [
            [11, 12, 13, 14],
            [21],
        ]
        assert engine.stats["batched_sync_replays"] == 0

    def test_multi_slot_round_rejects_a_graph_that_reemits_step0(self) -> None:
        """Teacher-forced sync owns step 0, so graph replay must return only K-1 tail tokens."""
        backend, model = _backend(num_slots=2)
        backend.enable_mtp(num_speculative_tokens=3, enable_resync=False)
        engine: Qwen36MTPEngine = backend._mtp

        class _BatchedVerify:
            def replay(self, slots, tokens, past_lens):
                del slots, past_lens
                hidden = torch.tensor(tokens, dtype=torch.float32).unsqueeze(-1)
                return hidden, model.compute_logits(hidden)

        class _TooLongDraft:
            def __init__(self) -> None:
                self.calls: list[tuple[list[int], list[int], torch.Tensor, list[int]]] = []

            def replay_batch(self, slots, seed_tokens, seed_hiddens, start_positions):
                self.calls.append(
                    (list(slots), list(seed_tokens), seed_hiddens.clone(), list(start_positions))
                )
                return {
                    slot: [token, token + 1, token + 2]
                    for slot, token in zip(slots, seed_tokens, strict=True)
                }

        engine._verify_cg = _BatchedVerify()
        draft = _TooLongDraft()
        engine._draft_cg = draft
        for slot in (0, 1):
            engine._caches[slot].seq_len = 2
            backend.pool.slot_kv_len[slot] = 7
            backend.pool.slot_state(slot).num_tokens_seen = 7

        with pytest.raises(RuntimeError, match="expected K-1=2"):
            engine.round_batch(
                [0, 1],
                {0: 10, 1: 20},
                {0: [11, 12, 13], 1: [21, 22, 23]},
            )

        assert len(draft.calls) == 1
        slots, tokens, hiddens, starts = draft.calls[0]
        assert slots == [0, 1]
        assert tokens == [15, 25]
        assert hiddens.reshape(-1).tolist() == [14.0, 24.0]
        assert starts == [4, 4]

    def test_single_slot_round_uses_the_same_batched_verify_entrypoint(self) -> None:
        """Historical verify and suffix-sync batches also own the B=1 call."""
        backend, model = _backend(num_slots=1)
        backend.enable_mtp(num_speculative_tokens=3, enable_resync=False)
        engine: Qwen36MTPEngine = backend._mtp

        class _BatchedVerify:
            def __init__(self) -> None:
                self.calls: list[tuple[list[int], list[list[int]], list[int]]] = []

            def replay(self, slots, tokens, past_lens):
                self.calls.append((slots, tokens, past_lens))
                hidden = torch.tensor(tokens, dtype=torch.float32).unsqueeze(-1)
                return hidden, model.compute_logits(hidden)

        verify = _BatchedVerify()

        class _BatchedSync:
            def __init__(self) -> None:
                self.calls: list[tuple[list[int], list[list[int]], torch.Tensor, list[int]]] = []

            def replay_ragged(self, slots, tokens, hidden_rows, starts):
                hiddens = torch.cat(hidden_rows, dim=0)
                self.calls.append((list(slots), list(tokens), hiddens.clone(), list(starts)))
                return torch.tensor([tokens[0][-1] + 1]), hiddens[:, -1:]

        sync = _BatchedSync()
        engine._verify_cg = verify
        engine._batched_sync = sync
        engine._caches[0].seq_len = 2
        backend.pool.slot_kv_len[0] = 7
        backend.pool.slot_state(0).num_tokens_seen = 7

        result = backend.mtp_verify_and_commit_batch([0], {0: 10}, {0: [11, 12, 13]})

        assert verify.calls == [([0], [[10, 11, 12, 13]], [7])]
        assert sync.calls[0][0] == [0]
        assert sync.calls[0][1] == [[11, 12, 13, 14]]
        assert model.mtp_forward_calls == []
        assert result[0]["committed"] == [11, 12, 13, 14]
        assert engine.stats["verify_graph_replays"] == 1
        assert engine.stats["batched_verify_replays"] == 0
        assert engine.stats["batched_sync_replays"] == 1
        assert engine.stats["batched_sync_slots"] == 1

    def test_immediate_reject_syncs_anchor_and_recovery_before_redraft(self) -> None:
        backend, model = _backend()
        backend.enable_mtp(num_speculative_tokens=3, enable_resync=False)
        engine: Qwen36MTPEngine = backend._mtp

        anchor_token = 20
        drafts = [99, 98, 97]  # deliberately NOT anchor_argmax (21) at position 0
        engine._caches[0].seq_len = 2

        model.mtp_step_calls.clear()
        result = engine.round(0, anchor_token, drafts)

        assert result["num_accepted"] == 0
        assert result["committed"] == [21]  # the target's own recovery prediction
        assert result["next_anchor"] == 21

        first_call = model.mtp_step_calls[0]
        assert model.mtp_forward_calls[-1] == {
            "tokens": [21],
            "hiddens": [20.0],
            "start_position": 0,
        }
        assert first_call["token"] == 22
        assert first_call["prev_hidden"] == 21.0

    def test_partial_accept_syncs_accepted_prefix_and_recovery(self) -> None:
        backend, model = _backend()
        backend.enable_mtp(num_speculative_tokens=3, enable_resync=False)
        engine: Qwen36MTPEngine = backend._mtp

        anchor_token = 30
        # anchor_argmax = 31 -> matches drafts[0]=31 (accepted).
        # verify_hidden[0] (from drafts[0]=31) argmax = 32, but drafts[1]=50
        # does not match -> reject at p=1, m=1.
        drafts = [31, 50, 51]
        engine._caches[0].seq_len = 2

        model.mtp_step_calls.clear()
        result = engine.round(0, anchor_token, drafts)

        assert result["num_accepted"] == 1
        assert result["committed"] == [31, 32]  # accepted draft + recovery token
        assert result["next_anchor"] == 32

        first_call = model.mtp_step_calls[0]
        assert model.mtp_forward_calls[-1] == {
            "tokens": [31, 32],
            "hiddens": [30.0, 31.0],
            "start_position": 0,
        }
        assert first_call["token"] == 33
        assert first_call["prev_hidden"] == 32.0

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
        engine._caches[0].seq_len = 2

        result = engine.round(0, 10, [11, 12, 13])

        assert backend.pool.slot_kv_len[0] == 7 + 1 + result["num_accepted"]
        assert (
            backend.pool.slot_committed_tokens[0][-len(result["committed"]) :]
            == result["committed"]
        )


class TestPrefillSync:
    def test_prefill_teacher_forces_the_whole_prompt_plus_anchor(self) -> None:
        backend, model = _backend()
        backend.enable_mtp(num_speculative_tokens=2, enable_resync=False)
        model.mtp_step_calls.clear()

        prompt = [5, 6, 7]  # last position's hidden value == 7 (stub convention)
        state = backend.prefill_chunked_begin([0], [prompt], params_per_slot={})
        anchor = state.result[0]["anchor"]
        assert anchor == 8  # (7 + 1) % vocab, the stub's own argmax rule

        assert model.mtp_forward_calls == [
            {"tokens": [6, 7, 8], "hiddens": [5.0, 6.0, 7.0], "start_position": 0}
        ]
        engine: Qwen36MTPEngine = backend._mtp
        assert engine._sync_len[0] == len(prompt)
        # The final sync row conditions on the anchor and emits draft 0;
        # only the remaining K-1 drafts have physical cache rows.
        assert engine._caches[0].seq_len == len(prompt) + engine.k - 1
        first_call = model.mtp_step_calls[0]
        assert first_call["token"] == 9
        assert first_call["prev_hidden"] == 8.0

    def test_sync_rewinds_an_old_speculative_tail_to_the_real_boundary(self) -> None:
        backend, model = _backend()
        backend.enable_mtp(num_speculative_tokens=2, enable_resync=False)
        engine: Qwen36MTPEngine = backend._mtp
        engine._caches[0].seq_len = 5
        engine.sync_prefill_chunk(
            0,
            shifted_token_ids=[2, 3],
            target_hidden=torch.tensor([[[1.0], [2.0]]]),
            final=False,
        )
        assert model.mtp_forward_calls[-1]["start_position"] == 0
        assert engine._sync_len[0] == 2
        assert engine._caches[0].seq_len == 2

    def test_same_slot_prefix_restore_reinstates_the_mtp_real_boundary(self) -> None:
        backend, _ = _backend()
        backend.enable_mtp(num_speculative_tokens=2, enable_resync=False)
        engine: Qwen36MTPEngine = backend._mtp
        engine._sync_len[0] = 64
        engine._caches[0].seq_len = 65  # one K-1 continuation row

        engine.preserve_prefix(0, 64)
        engine.reset_slot(0)
        assert engine.can_restore_prefix(0, 64)

        engine.restore_prefix(0, 64)
        assert engine._sync_len[0] == 64
        assert engine._caches[0].seq_len == 64

    def test_cross_slot_prefix_copy_moves_mtp_kv_without_aliasing_source(self) -> None:
        # Keep this at the MTP-engine boundary rather than relying only on
        # Qwen36Backend's facade: a future cache-layout change can otherwise
        # leave target/GDN copies correct while the MTP causal context stays
        # stale.  The non-contiguous tables mirror pooled CUDA-graph storage.
        engine = Qwen36MTPEngine.__new__(Qwen36MTPEngine)
        source = SimpleNamespace(
            page_size=128,
            page_table=torch.tensor([[1, 0]], dtype=torch.int32),
            k_cache=torch.zeros(4, 128, 1, 1),
            v_cache=torch.zeros(4, 128, 1, 1),
            seq_len=0,
        )
        target = SimpleNamespace(
            page_size=128,
            page_table=torch.tensor([[3, 2]], dtype=torch.int32),
            k_cache=torch.zeros(4, 128, 1, 1),
            v_cache=torch.zeros(4, 128, 1, 1),
            seq_len=0,
        )
        source.k_cache[1].fill_(5.0)
        source.k_cache[0].fill_(6.0)
        source.v_cache[1].fill_(7.0)
        source.v_cache[0].fill_(8.0)
        engine._caches = [source, target]
        engine._cached_prefix_sync_len = [129, 0]
        engine._sync_len = [0, 0]
        engine._spec_rows = None
        engine._spec_state_col = [0, 0]

        engine.copy_prefix(0, 1, 129)

        assert torch.all(target.k_cache[3] == 5.0)
        assert torch.all(target.k_cache[2] == 6.0)
        assert torch.all(target.v_cache[3] == 7.0)
        assert torch.all(target.v_cache[2] == 8.0)
        target.k_cache[3].fill_(9.0)
        assert torch.all(source.k_cache[1] == 5.0)
        assert engine._sync_len[1] == 129
        assert target.seq_len == 129

    def test_scratch_prefix_snapshot_restores_mtp_context_after_source_reuse(self) -> None:
        engine = Qwen36MTPEngine.__new__(Qwen36MTPEngine)
        caches = [
            SimpleNamespace(
                page_size=128,
                page_table=torch.tensor([[0, 1]], dtype=torch.int32),
                k_cache=torch.zeros(2, 128, 1, 1),
                v_cache=torch.zeros(2, 128, 1, 1),
                seq_len=0,
            )
            for _ in range(3)
        ]
        caches[0].k_cache[0].fill_(5.0)
        caches[0].k_cache[1].fill_(6.0)
        caches[0].v_cache[0].fill_(7.0)
        caches[0].v_cache[1].fill_(8.0)
        engine._caches = caches
        engine.device = torch.device("cpu")
        engine.scratch_row = 2
        engine.mtp_page_size = 128
        engine._sync_len = [129, 0]
        engine._cached_prefix_sync_len = [0, 0]
        engine._spec_rows = None
        engine._spec_state_col = [0, 0]

        assert engine.snapshot_prefix_to_scratch(0, 129)
        caches[0].k_cache[0].zero_()
        assert engine.restore_prefix_from_scratch(1, 129)
        assert torch.all(caches[1].k_cache[0] == 5.0)
        assert torch.all(caches[1].k_cache[1] == 6.0)
        assert torch.all(caches[1].v_cache[0] == 7.0)
        assert torch.all(caches[1].v_cache[1] == 8.0)
        assert engine._sync_len[1] == 129

    def test_restore_from_scratch_never_clears_the_live_gdn_column(self) -> None:
        """A persistent restore must preserve the target's live GDN state.

        The backend restores the recurrent checkpoint into column zero
        BEFORE this call, so resetting the spec rows here -- column zero
        included -- would start the first verify from an empty GDN
        recurrence and emit wrong logits that nothing downstream can detect
        (the full-hit corruption seen on 2026-08-05).  Candidate columns are
        always overwritten by the next verify, so only the source-column
        pointer needs pinning.
        """
        engine = Qwen36MTPEngine.__new__(Qwen36MTPEngine)
        caches = [
            SimpleNamespace(
                page_size=128,
                page_table=torch.tensor([[0, 1]], dtype=torch.int32),
                k_cache=torch.zeros(2, 128, 1, 1),
                v_cache=torch.zeros(2, 128, 1, 1),
                seq_len=0,
            )
            for _ in range(3)
        ]
        caches[0].k_cache[0].fill_(5.0)
        caches[0].k_cache[1].fill_(6.0)
        caches[0].v_cache[0].fill_(7.0)
        caches[0].v_cache[1].fill_(8.0)
        engine._caches = caches
        engine.device = torch.device("cpu")
        engine.scratch_row = 2
        engine.mtp_page_size = 128
        engine._sync_len = [129, 0]
        engine._cached_prefix_sync_len = [0, 0]
        engine._spec_state_col = [0, 0]

        class _SpecRows:
            def __init__(self) -> None:
                self.resets = 0
                self.activations: list[tuple[int, int]] = []

            def reset_slot(self, slot: int) -> None:
                self.resets += 1

            def activate(self, slot: int, col: int) -> None:
                self.activations.append((slot, col))

        spec = _SpecRows()
        engine._spec_rows = spec

        assert engine.snapshot_prefix_to_scratch(0, 129)
        assert engine.restore_prefix_from_scratch(1, 129)
        assert spec.resets == 0
        assert spec.activations == [(1, 0)]
        assert engine._spec_state_col == [0, 0]

    def test_scratch_watermark_survives_a_later_shorter_store(self) -> None:
        """A shorter later snapshot must not lower the scratch watermark.

        ``scratch.seq_len`` guards the whole persistent arena, but its
        entries live at disjoint page offsets.  Measured 2026-08-06 in the
        real 256K grid: after 64K/128K entries were stored, the 4K/32K
        stores dropped the watermark below the long entries and every
        64K/128K restore failed with "persistent MTP prefix disappeared".
        The long entry's bytes are still in the arena, so the watermark
        must only ever rise.
        """
        engine = Qwen36MTPEngine.__new__(Qwen36MTPEngine)
        caches = [
            SimpleNamespace(
                page_size=128,
                page_table=torch.tensor([[0, 1]], dtype=torch.int32),
                k_cache=torch.zeros(4, 128, 1, 1),
                v_cache=torch.zeros(4, 128, 1, 1),
                seq_len=0,
            )
            for _ in range(3)
        ]
        caches[0].k_cache[0].fill_(5.0)
        caches[0].k_cache[1].fill_(6.0)
        caches[0].k_cache[2].fill_(9.0)
        caches[0].v_cache[0].fill_(7.0)
        caches[0].v_cache[1].fill_(8.0)
        caches[0].v_cache[2].fill_(10.0)
        engine._caches = caches
        engine.device = torch.device("cpu")
        engine.scratch_row = 2
        engine.mtp_page_size = 128
        engine._sync_len = [129, 0]
        engine._cached_prefix_sync_len = [0, 0]
        engine._spec_rows = None
        engine._spec_state_col = [0, 0]

        # Long entry first: pages 0-1 of the scratch arena.
        assert engine.snapshot_prefix_to_scratch(0, 129, scratch_pages=(0, 1))
        # Shorter entry afterwards: disjoint page 2.  The pre-fix code set
        # scratch.seq_len = 64 here and permanently hid the 129 entry.
        assert engine.snapshot_prefix_to_scratch(0, 64, scratch_pages=(2,))
        caches[0].k_cache[0].zero_()
        caches[0].k_cache[1].zero_()
        assert engine.restore_prefix_from_scratch(1, 129)
        assert torch.all(caches[1].k_cache[0] == 5.0)
        assert torch.all(caches[1].k_cache[1] == 6.0)
        assert torch.all(caches[1].v_cache[0] == 7.0)
        assert torch.all(caches[1].v_cache[1] == 8.0)
        assert engine._sync_len[1] == 129


class TestHistoricalSync:
    """Every round synchronises the full accepted suffix; no optional
    interior-only repair is needed for the real-prefix state contract."""

    def test_resync_off_by_default_does_not_call_mtp_resync_step(self) -> None:
        backend, model = _backend()
        backend.enable_mtp(num_speculative_tokens=3, enable_resync=False)
        engine: Qwen36MTPEngine = backend._mtp
        engine._caches[0].seq_len = 2
        engine.round(0, 10, [11, 12, 13])
        assert model.mtp_resync_calls == []
        assert model.mtp_forward_calls[-1]["tokens"] == [11, 12, 13, 14]

    def test_sync_preserves_a_nonzero_real_prefix_boundary(self) -> None:
        backend, model = _backend()
        backend.enable_mtp(num_speculative_tokens=3, enable_resync=False)
        engine: Qwen36MTPEngine = backend._mtp
        engine._sync_len[0] = 5
        engine._caches[0].seq_len = 7

        result = engine.round(0, 10, [11, 12, 13])
        assert result["num_accepted"] == 3
        assert model.mtp_forward_calls[-1]["start_position"] == 5
        assert engine._sync_len[0] == 9

    def test_partial_accept_still_uses_full_sync_without_legacy_resync(self) -> None:
        backend, model = _backend()
        backend.enable_mtp(num_speculative_tokens=3, enable_resync=False)
        engine: Qwen36MTPEngine = backend._mtp
        engine._caches[0].seq_len = 2
        engine.round(0, 30, [31, 50, 51])
        assert model.mtp_resync_calls == []
        assert model.mtp_forward_calls[-1]["tokens"] == [31, 32]


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
        engine._caches[0].seq_len = 2
        params = SamplingParams(temperature=1.0, seed=0)

        result = engine.round(0, 10, [11, 12, 13], params=params)

        assert 0 <= result["num_accepted"] <= 3
        assert len(result["committed"]) == result["num_accepted"] + 1
        assert len(result["next_draft_tokens"]) == 3
        assert backend._mtp.stats["sampled_rounds"] == 1


class TestResyncFlagRefusesWithoutAnImplementation:
    """`QSR_SERVER_MTP_RESYNC` is retired now that every committed suffix is teacher-forced."""

    def _engine(self, *, enable_resync: bool) -> Qwen36MTPEngine:
        backend, _ = _backend()
        backend.model.mtp = object()
        return Qwen36MTPEngine(backend, num_speculative_tokens=4, enable_resync=enable_resync)

    def test_it_refuses_when_resync_is_requested(self) -> None:
        with pytest.raises(ValueError, match="retired"):
            self._engine(enable_resync=True)

    def test_the_default_path_is_unaffected(self) -> None:
        engine = self._engine(enable_resync=False)
        assert engine.enable_resync is False

    def test_retirement_is_independent_of_model_capability(self) -> None:
        with pytest.raises(ValueError, match="teacher-forces every newly committed target suffix"):
            self._engine(enable_resync=True)
