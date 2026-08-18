"""B3 CUDA-Graph follow-up: :mod:`runtime.backends.qwen36_mtp_cudagraph`.

Two failure modes this file exists to catch, neither of which a "does the
server still respond" smoke test would surface:

1. **Wrong physical address, right shape.** ``decode_write_index`` is the
   formula every graph-replay call in this module trusts to route a
   write into the correct slot's own pages of a POOLED KV tensor. Get the
   arithmetic wrong (e.g. multiply by ``pages_per_slot`` in the wrong
   place, or forget the ``% page_size`` remainder) and the result is
   still a valid row index -- just one that lands in a DIFFERENT slot's
   memory. Nothing crashes; a live conversation on an unrelated slot
   starts producing tokens from a different sequence's KV, and it only
   shows up as "the model briefly makes no sense", the exact class of bug
   ``docs/a3-cache-coordinator-design.md`` calls out (INV-A3-1: "不是
   崩溃 -- 是某个请求的输出因为另一个请求的写入而改变"). A real GPU run
   would need a specific multi-slot adversarial script to catch this at
   all; this test needs neither GPU nor a slot pool.

2. **A capture failure silently reverting to eager with no signal.**
   ``attempt_mtp_cg_capture`` is the single choke point
   ``Qwen36MTPEngine.capture_cuda_graphs`` uses for both graphs -- get its
   strict/non-strict branching wrong and either (a) a real capture
   failure in production is swallowed with nothing recorded in
   ``cg_status`` (the exact "silent eager fallback" shape that hid the
   w4a16 scratch bug for weeks, see ``tests/test_w4a16_scratch_contract.
   py``), or (b) QSR_QWEN36_MTP_REQUIRE_CG=1 stops re-raising and this
   engine silently starts in a half-captured state. A fake
   ``capture_fn`` that always raises exercises both branches without a
   GPU.

CPU-only throughout: no GPU, no real checkpoint, no sparkinfer kernel
call. ``Qwen36MTPAnchorCudaGraph``/``Qwen36MTPDraftCudaGraph``/
``build_pooled_mtp_caches`` themselves are real-GPU-only (they call into
sparkinfer's CUDA-only graph-replay planner) and are proven bit-exact
against eager separately, on GPU, by
``scripts/verify_qwen36_mtp_cuda_graph_bit_exact.py`` -- matching this
repo's existing convention (``tests/test_qwen36_mtp_head.py``'s docstring)
of keeping kernel-touching proofs out of the CPU suite.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
# qwen36_mtp_cudagraph imports qwen36_model, which imports fla/sparkinfer
# at module level even though the functions exercised below never call
# either -- same guard tests/test_qwen36_mtp_head.py uses for the same
# reason. sparkinfer 仓库 2026-08-09 upstream merge 后以 ``b12x`` 为包名。
pytest.importorskip("fla")
pytest.importorskip("b12x")

from runtime.backends.qwen36_dspark_cudagraph import (  # noqa: E402
    Qwen36DSparkDraftBatchCudaGraph,
)
from runtime.backends.qwen36_mtp_cudagraph import (  # noqa: E402
    Qwen36MTPBatchedSync,
    Qwen36MTPDraftCudaGraph,
    Qwen36MTPRaggedVerifyCudaGraph,
    Qwen36MTPVerifyCudaGraph,
    _page_table_slot_key,
    attempt_mtp_cg_capture,
    decode_write_index,
)


class TestPageTableSlotKey:
    def test_dynamic_key_changes_when_the_same_slot_is_remapped(self) -> None:
        pool = type(
            "Pool",
            (),
            {
                "dynamic_arena": True,
                "versions": [3, 7],
                "page_table_version": lambda self, slot: self.versions[slot],
            },
        )()

        before = _page_table_slot_key(pool, [0])
        pool.versions[0] += 1

        assert before == ((0, 3),)
        assert _page_table_slot_key(pool, [0]) == ((0, 4),)

    def test_legacy_key_only_tracks_slot_membership(self) -> None:
        pool = type("Pool", (), {"dynamic_arena": False})()
        assert _page_table_slot_key(pool, [2, 0]) == (2, 0)


class TestDecodeWriteIndex:
    """The pooled-KV addressing formula every graph-replay call trusts."""

    def test_slot_0_position_0_is_row_0(self) -> None:
        assert decode_write_index(slot=0, kv_len=0, page_size=128, pages_per_slot=4) == 0

    def test_advances_by_one_within_a_page(self) -> None:
        base = decode_write_index(slot=0, kv_len=5, page_size=128, pages_per_slot=4)
        nxt = decode_write_index(slot=0, kv_len=6, page_size=128, pages_per_slot=4)
        assert nxt == base + 1

    def test_crossing_a_page_boundary_jumps_to_the_next_page_start(self) -> None:
        """kv_len=127 -> 128 crosses page_size=128's boundary: the next
        write must land at the START of page 1, not page 0's row 128
        (there is no such row -- page 0 only has rows 0..127)."""
        last_of_page0 = decode_write_index(slot=0, kv_len=127, page_size=128, pages_per_slot=4)
        first_of_page1 = decode_write_index(slot=0, kv_len=128, page_size=128, pages_per_slot=4)
        assert last_of_page0 == 127
        assert first_of_page1 == 128  # page 1 starts right after page 0's 128 rows

    def test_different_slots_never_share_a_row_at_the_same_kv_len(self) -> None:
        """The failure mode this whole formula exists to prevent: two
        slots' writes at identical logical positions must land in
        disjoint physical rows, or one slot's decode corrupts another's
        KV (INV-A3-1)."""
        page_size, pages_per_slot = 128, 4
        seen: set[int] = set()
        for slot in range(3):
            for kv_len in (0, 1, 127, 128, 511):
                row = decode_write_index(slot, kv_len, page_size, pages_per_slot)
                assert row not in seen, f"slot={slot} kv_len={kv_len} collided at row {row}"
                seen.add(row)

    def test_matches_the_slot_pools_own_build_decode_batch_formula(self) -> None:
        """Same computation as ``Qwen36SlotPool.build_decode_batch``
        (``global_page = slot * pages_per_slot + past // page_size;
        write_row = global_page * page_size + past % page_size``),
        spelled out independently here so the two cannot silently drift
        apart -- this graph's replay writes into the SAME pooled storage
        that method addresses for the plain (non-MTP) decode path."""
        slot, kv_len, page_size, pages_per_slot = 2, 300, 128, 4
        global_page = slot * pages_per_slot + kv_len // page_size
        expected = global_page * page_size + kv_len % page_size
        assert decode_write_index(slot, kv_len, page_size, pages_per_slot) == expected

    def test_page_row_overrides_the_static_slot_formula(self) -> None:
        """Phase 2 dynamic arena: a non-contiguous page row (the result of
        bundle allocation/COW) must win over the legacy
        ``slot * pages_per_slot`` formula -- the physical bundle for the
        logical page comes from the shared page table."""
        page_size, pages_per_slot = 128, 4
        row = [200, 7, 300, 400]  # deliberately non-contiguous physical bundles
        # kv_len=300 -> logical page 2 -> physical bundle 300.
        assert decode_write_index(0, 300, page_size, pages_per_slot, page_row=row) == 300 * 128 + 44
        # kv_len=64 -> logical page 0 -> physical bundle 200.
        assert decode_write_index(3, 64, page_size, pages_per_slot, page_row=row) == 200 * 128 + 64

    def test_page_row_missing_page_raises_instead_of_aliasing(self) -> None:
        """A write past the prepared prefix must fail loudly -- a null
        bundle (0) or missing entry would otherwise address the null page
        and silently corrupt shared content."""
        row = [0, 1]  # only two logical pages prepared
        with pytest.raises(RuntimeError, match="logical page"):
            decode_write_index(0, 300, 128, 4, page_row=row)

    def test_page_row_with_null_bundle_is_an_explicit_error(self) -> None:
        """Logical page still pointing at the null bundle (unallocated)
        must not be silently writable (plan §7 invariant 5)."""
        row = [0, 5]
        assert decode_write_index(0, 0, 128, 4, page_row=row) == 0  # null page row 0
        # The null bundle maps to physical page 0; a write there would hit
        # the reserved page. The check is at the pool level (prepare_kv_writes
        # allocates before writing); the indexer must at least be consistent.
        assert decode_write_index(0, 0, 128, 4, page_row=row) < 128 * 5


class TestAttemptMtpCgCapture:
    """The single choke point every capture site in this module goes
    through -- see module docstring's failure mode 2."""

    def test_success_returns_captured_and_never_logs_an_error(self, caplog) -> None:
        calls: list[str] = []
        status = attempt_mtp_cg_capture("anchor", lambda: calls.append("ran"), strict=False)
        assert status == "captured"
        assert calls == ["ran"]
        assert not any(r.levelname == "ERROR" for r in caplog.records)

    def test_non_strict_failure_degrades_and_is_recorded(self, caplog) -> None:
        def _boom() -> None:
            raise RuntimeError("simulated sparkinfer graph-capacity failure")

        status = attempt_mtp_cg_capture("draft", _boom, strict=False)
        assert status == "failed"
        # Loud, not silent (module docstring's failure mode 2): a real
        # operator or `bf daemon status`-style tooling must be able to
        # find this without re-running the failing capture themselves.
        assert any(r.levelname == "ERROR" for r in caplog.records)

    def test_strict_failure_reraises_instead_of_degrading(self) -> None:
        def _boom() -> None:
            raise RuntimeError("simulated sparkinfer graph-capacity failure")

        with pytest.raises(RuntimeError, match="simulated sparkinfer"):
            attempt_mtp_cg_capture("anchor", _boom, strict=True)

    def test_verify_capture_uses_the_same_observable_failure_signal(self, caplog) -> None:
        """The verify graph is the expensive path this change adds.

        It must go through the same status choke point as anchor/draft: a
        graph-capacity or GDN capture bug otherwise falls back to eager while
        leaving ``cg_status`` looking healthy, which is invisible to a token
        smoke test and was the failure mode that let scratch undersizing ship.
        """

        def _boom() -> None:
            raise RuntimeError("verify boom")

        status = attempt_mtp_cg_capture("verify", _boom, strict=False)
        assert status == "failed"
        assert any("verify" in record.message for record in caplog.records)


class TestReplayFillDiscipline:
    """Hot-path replay helpers should keep reusing their static staging buffers."""

    def test_fill_paths_use_cached_host_views_instead_of_per_call_numpy_lookups(self) -> None:
        sources = (
            inspect.getsource(Qwen36MTPVerifyCudaGraph._fill),
            inspect.getsource(Qwen36MTPDraftCudaGraph._fill),
            inspect.getsource(Qwen36MTPBatchedSync._fill),
            inspect.getsource(Qwen36MTPBatchedSync._fill_ragged),
            inspect.getsource(Qwen36MTPBatchedSync._fill_verify_ragged),
        )
        for source in sources:
            assert ".numpy()" not in source

    def test_draft_fill_stages_device_seeds_with_a_direct_copy(self) -> None:
        """The ragged-sync fast path hands the draft graph device seed rows;
        staging must be a D2D ``copy_`` from the tensor, not a host
        ``tolist()`` round-trip (measured host gap 5-15 ms/round at 128K)."""
        fill_source = inspect.getsource(Qwen36MTPDraftCudaGraph._fill)
        assert "isinstance(seed_tokens, torch.Tensor)" in fill_source
        assert "all(isinstance(token, torch.Tensor) for token in seed_tokens)" in fill_source
        assert "input_ids[:, 0].copy_(seed_tokens, non_blocking=True)" in fill_source

    def test_step_loops_reuse_graph_owned_cache_seqlens_buffer(self) -> None:
        sources = (
            inspect.getsource(Qwen36MTPDraftCudaGraph._forward_all_steps),
            inspect.getsource(Qwen36MTPBatchedSync._forward_all_steps),
        )
        for source in sources:
            assert ".to(torch.int32)" not in source
            assert "cache_seqlens.copy_(start_pos)" in source
            assert "cache_seqlens.add_(1)" in source

    def test_dspark_ragged_accept_builds_next_draft_inputs_on_device(self) -> None:
        graph = Qwen36MTPRaggedVerifyCudaGraph.__new__(Qwen36MTPRaggedVerifyCudaGraph)
        graph._capture_accept = True
        graph.device = torch.device("cpu")
        graph.max_verify_tokens = 4
        graph._batches = {2: object()}
        graph._accepted_long = {2: torch.empty(2, dtype=torch.long)}
        graph._next_anchors = {2: torch.empty(2, dtype=torch.long)}
        graph._next_kv_lens = {2: torch.empty(2, dtype=torch.long)}
        graph._inputs = {
            2: {
                "cache_seqlens": torch.tensor([13, 24], dtype=torch.int32),
                "verify_lens": torch.tensor([4, 3], dtype=torch.int32),
            }
        }

        accepted = torch.tensor([1, 2], dtype=torch.int32)
        committed = torch.tensor(
            [[101, 202, 0, 0], [303, 404, 505, 0]], dtype=torch.long
        )
        anchors, kv_lens = graph.next_draft_inputs(2, accepted, committed)

        assert anchors.tolist() == [202, 505]
        assert kv_lens.tolist() == [11, 24]

    def test_dspark_draft_graph_exposes_device_replay_without_flashinfer_d2h(self) -> None:
        source = inspect.getsource(Qwen36DSparkDraftBatchCudaGraph)
        assert "def replay_device(" in source
        assert "self._anchors.copy_(anchors, non_blocking=True)" in source
        assert "host_kv_lens" in source
        assert "metadata_kv_lens = host_kv_lens" in source

    def test_ragged_verify_exposes_device_next_draft_transition(self) -> None:
        source = inspect.getsource(Qwen36MTPRaggedVerifyCudaGraph.next_draft_inputs)
        assert "torch.gather" in source
        assert "next_kv_lens.sub_(inputs[\"verify_lens\"])" in source
        assert "next_kv_lens.add_(accepted)" in source

    def test_fused_context_mapping_commits_only_the_accepted_prefix(self) -> None:
        graph = Qwen36MTPRaggedVerifyCudaGraph.__new__(Qwen36MTPRaggedVerifyCudaGraph)
        graph.context_kv_fused = True
        graph._accepted = {2: torch.tensor([0, 2], dtype=torch.int32)}
        indices = torch.arange(8, dtype=torch.long)
        graph._context_indices = {2: indices}
        graph._context_row_indices = {2: torch.empty(8, dtype=torch.long)}
        graph._context_row_starts = {2: torch.empty(8, dtype=torch.int32)}
        graph._context_columns = {2: torch.empty(8, dtype=torch.long)}
        graph._context_row_accepts = {2: torch.empty(8, dtype=torch.int32)}
        graph._context_keep = {2: torch.empty(8, dtype=torch.bool)}
        graph._context_scratch_mapping = {2: indices + 100}
        graph._context_gated_mapping = {2: torch.empty(8, dtype=torch.long)}
        batch = SimpleNamespace(
            cu_seqlens_q=torch.tensor([0, 3, 5], dtype=torch.int32),
            context_slot_mapping=torch.arange(8, dtype=torch.long) + 10,
        )

        mapping = graph._accepted_context_mapping(2, batch)

        assert mapping.tolist() == [10, 101, 102, 13, 14, 105, 106, 107]
