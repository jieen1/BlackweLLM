"""B3 follow-up (2026-08-03): CUDA-Graph capture for MTP's verify/draft path.

**The regression this exists to fix** (``notes/2026-08-03-*``, commit
"MTP is wired in but must not ship: it makes decode 3.6x slower"):
``Qwen36MTPEngine.round`` is entirely eager and never reaches
``Qwen36Backend.decode_batch_sampled``, the only path
``Qwen36Backend.capture_decode_cuda_graph`` ever replays. Turning MTP on
does not merely fail to use the decode graph -- it makes the decode graph
*unreachable*, forfeiting the whole CUDA-Graph speedup
(``notes/2026-08-03-cudagraph-vs-eager-decode-throughput.md``: 4.71x) to
gain 1.54-accepted-of-4. This module gives MTP its own graphs, the same
way ``runtime/backends/laguna_dflash.py``'s ``DFlashEngine`` gives DFlash
its own (``_capture_verify_cg``/``_capture_draft_cg``/``_ensure_decode_cg``)
instead of trying to reuse Laguna's plain-decode graph.

**What DFlash's shape does NOT transfer, and why this module looks
different**: DFlash's draft model predicts all 15 tokens in ONE parallel
(mask-token) forward, so ``DFlashDraftCudaGraph`` captures a single
``verify``-mode (M=16) sparkinfer call. Qwen3.6's MTP head
(``Qwen36MTPHead``/``mtp_step``) is genuinely autoregressive -- each draft
step conditions on the PREVIOUS step's own sampled token and hidden state
(see ``runtime.backends.qwen36_mtp``'s module docstring) -- so there is no
parallel-mask equivalent here. :class:`Qwen36MTPDraftCudaGraph` instead
captures ``K`` chained ``decode``-mode (M=1) steps as ONE graph, unrolled
at capture time; that shape already exists in this codebase
(``Qwen36DecodeGraphAttention``, built for
``Qwen36Backend.capture_decode_cuda_graph``'s M=1 batched decode), so this
module reuses it rather than inventing a new sparkinfer graph-replay mode.

**What is captured, and what is not (say so explicitly, per the B3
follow-up brief)**:

* :class:`Qwen36MTPAnchorCudaGraph` -- the anchor-advance step
  (``Qwen36MTPEngine.round``'s ``self.model(anchor_input, state)``): one
  M=1 forward through the full 64-layer backbone. This is EXACTLY the
  shape ``Qwen36Backend``'s own decode graph already solves, so this class
  reuses the backend's pooled KV/GDN storage
  (``Qwen36SlotPool.k_pools``/``v_pools``/``conv_pools``/``recurrent_pools``
  -- the same physical bytes ``Qwen36GenerationState.attn_caches``/
  ``gdn_states`` read via a per-slot VIEW, see that class's module
  docstring's "one allocation, two addressings") through a SEPARATE,
  dedicated M=1 graph rather than the backend's own shared
  ``_decode_graphs[1]`` -- reusing that graph directly is not safe: it is
  captured to also stamp ``slot_committed_tokens``/``slot_kv_len``
  (``Qwen36SlotPool.build_decode_batch``), and the anchor token this graph
  processes was ALREADY committed by the previous round (see
  ``runtime.backends.qwen36_mtp.Qwen36MTPEngine.round``'s "committed ahead
  of kv by one" comment) -- reusing the shared graph's bookkeeping would
  double-count it. This mirrors DFlash's OWN precedent exactly:
  ``DFlashEngine._ensure_decode_cg``/``LagunaCudaGraphDecode`` is ALSO a
  separate M=1 decode graph from ``LagunaBackend``'s own, because DFlash
  needs ``aux_hidden_states`` the shared graph does not expose -- here MTP
  needs the post-norm hidden state (pre-``lm_head``) the shared graph
  does not expose either.

* :class:`Qwen36MTPDraftCudaGraph` -- the ``K`` chained
  ``Qwen36MTPEngine._draft_loop`` steps: one self-attention layer, M=1
  each, chained. Its own dedicated pooled KV cache
  (:func:`build_pooled_mtp_caches`) is required, not merely convenient:
  each real slot's ``mtp_step`` calls must reach a SINGLE captured graph
  (there are no spare graphs per slot -- capturing ``num_slots`` separate
  draft graphs would multiply capture time and memory for zero benefit),
  and a CUDA Graph bakes in the exact tensor ADDRESSES its kernels read --
  a graph captured against one slot's OWN standalone
  ``Qwen36PagedAttentionCache`` (the pre-existing per-slot allocation)
  could only ever be replayed correctly for that one slot. Pooling with a
  global page table (mirroring ``Qwen36SlotPool``'s own "one allocation,
  two addressings" design, scaled down to MTP's tiny 1-layer head) is what
  lets ONE graph serve every slot, exactly like the backbone's own
  ``Qwen36DecodeGraphAttention``.

* :class:`Qwen36MTPVerifyCudaGraph` -- the full K-token target verify body.
  It uses a graph-mode ``verify`` workspace with sparkinfer's
  ``prepare_prefill_graph_replay_state``/
  ``update_prefill_graph_replay_metadata`` pair, so the worst-case plan is
  captured once while page ids, cache length, query positions, and K/V write
  rows stay replay inputs. GDN's ``spec_forward`` writes fixed ``spec_row``
  destinations instead of allocating snapshots. One graph is captured for
  each physical slot; a device source-row index selects the accepted column
  before replay. This is the captured equivalent of ``verify_forward`` -- the
  old eager method remains the fallback and the bit-exact oracle.

**Discipline**: every capture site goes through
:func:`attempt_mtp_cg_capture`, the same "record status, log loud, degrade
to eager rather than crash (unless a strict env var says otherwise)"
shape ``runtime.backends.laguna_dflash._attempt_cg_capture`` established
for DFlash (C7-2) -- reimplemented here, not imported, because that
function's own docstring and defaults are DFlash-specific (a documented,
still-open correctness gap in DFlash's eager fallback that has no MTP
analogue: MTP's eager round() is the SAME code
``scripts/b3_mtp_e2e_acceptance_throughput.py`` already token-matches
against non-speculative decode). ``QSR_QWEN36_MTP_REQUIRE_CG`` therefore
defaults to ``"0"`` (degrade, don't refuse to start) -- the opposite of
DFlash's default -- because degrading here falls back to an
already-proven-correct path, not a suspect one.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from runtime.model.qwen36_model import (
    _PAGED_ATTENTION_PAGE_SIZE,
    Qwen36DecodeBatch,
    Qwen36DecodeGraphAttention,
    Qwen36PagedAttentionCache,
    Qwen36VerifyBatch,
    Qwen36VerifyGraphAttention,
)

if TYPE_CHECKING:
    from runtime.backends.qwen36 import Qwen36Backend
    from runtime.backends.qwen36_mtp import Qwen36MTPEngine

logger = logging.getLogger("qwen_sm120_runtime.qwen36_mtp_cudagraph")


def attempt_mtp_cg_capture(name: str, capture_fn: Callable[[], None], *, strict: bool) -> str:
    """Run one MTP CUDA Graph capture attempt; return ``"captured"`` or ``"failed"``.

    See module docstring's "Discipline" section for why this mirrors
    ``laguna_dflash._attempt_cg_capture`` in shape but not in code or
    default.
    """
    try:
        capture_fn()
        return "captured"
    except Exception:
        action = (
            "refusing to finish enable_mtp() (QSR_QWEN36_MTP_REQUIRE_CG=1)"
            if strict
            else "degrading to eager (QSR_QWEN36_MTP_REQUIRE_CG=0, the default) -- "
            "the same round()/_draft_loop path scripts/b3_mtp_e2e_acceptance_"
            "throughput.py already token-matches against non-speculative decode"
        )
        logger.error(
            "Qwen3.6 MTP: %s CUDA Graph capture failed -- %s.", name, action, exc_info=True
        )
        if strict:
            raise
        return "failed"


def decode_write_index(slot: int, kv_len: int, page_size: int, pages_per_slot: int) -> int:
    """Row into a pooled ``[total_pages, page_size, ...]`` KV tensor's
    flattened ``view(-1, kv_heads, head_dim)`` for the token about to be
    written at ``(slot, kv_len)``.

    The exact formula ``Qwen36SlotPool.build_decode_batch`` uses (module
    docstring's "one allocation, two addressings"), extracted here as a
    pure function so it is unit-testable without a GPU, a real pool, or
    even torch: a wrong constant here addresses the wrong physical bytes,
    which no shape assertion catches and which corrupts a DIFFERENT
    slot's KV silently -- exactly the failure class this repo's own
    INV-A3-1 warns about, and precisely the kind of bug that would go
    unnoticed until a long-running conversation on an adjacent slot
    started producing wrong tokens.
    """
    global_page = slot * pages_per_slot + kv_len // page_size
    return global_page * page_size + kv_len % page_size


def build_pooled_mtp_caches(
    model, *, num_slots: int, device: torch.device, dtype: torch.dtype
) -> tuple[list[Qwen36PagedAttentionCache], torch.Tensor, torch.Tensor, int, int]:
    """Build ``num_slots + 1`` (the ``+1`` scratch row, matching
    ``Qwen36SlotPool``'s own capture-safety reasoning) per-slot MTP
    self-attention caches as VIEWS over one pooled ``[k|v]`` allocation,
    instead of ``num_slots`` independent
    ``Qwen36PagedAttentionCache.__init__`` allocations.

    Required for :class:`Qwen36MTPDraftCudaGraph` to exist at all (see
    module docstring): a CUDA Graph bakes in the exact address its
    kernels read, so a graph captured against one independently-allocated
    cache's own storage could only ever replay correctly against that one
    cache. Returns ``(caches, k_pool, v_pool, page_size, pages_per_slot)``.
    Caller (:class:`runtime.backends.qwen36_mtp.Qwen36MTPEngine`) keeps
    ``caches[0:num_slots]`` as the real per-slot caches (indexed exactly
    as the old per-slot allocation was) and ``caches[num_slots]`` as the
    scratch row used only for capture-time warmup.
    """
    mtp_attn = model.mtp.layers[0].self_attn
    page_size = _PAGED_ATTENTION_PAGE_SIZE
    pages_per_slot = (mtp_attn.max_seq_len + page_size - 1) // page_size
    num_rows = num_slots + 1
    total_pages = num_rows * pages_per_slot
    kv_shape = (total_pages, page_size, mtp_attn.num_kv_heads, mtp_attn.head_dim)
    k_pool = torch.zeros(kv_shape, dtype=mtp_attn.kv_cache_dtype, device=device)
    v_pool = torch.zeros(kv_shape, dtype=mtp_attn.kv_cache_dtype, device=device)
    marker = getattr(torch._dynamo, "mark_static_address", None)
    if marker is not None:  # pragma: no branch - present in every supported torch
        marker(k_pool)
        marker(v_pool)
    caches = [
        Qwen36PagedAttentionCache.wrap(
            k_cache=k_pool[row * pages_per_slot : (row + 1) * pages_per_slot],
            v_cache=v_pool[row * pages_per_slot : (row + 1) * pages_per_slot],
            page_size=page_size,
        )
        for row in range(num_rows)
    ]
    return caches, k_pool, v_pool, page_size, pages_per_slot


class Qwen36MTPAnchorCudaGraph:
    """CUDA Graph for ``Qwen36MTPEngine.round``'s anchor-advance step: one
    M=1 forward through the full backbone, returning the post-norm hidden
    state (``[1, 1, H]``). See module docstring for why this is a
    dedicated graph rather than a reuse of ``Qwen36Backend``'s own shared
    decode graph.
    """

    def __init__(
        self,
        backend: Qwen36Backend,
        *,
        conv_pools: list[torch.Tensor | None] | None = None,
        recurrent_pools: list[torch.Tensor | None] | None = None,
        state_row: Callable[[int, int], int] | None = None,
        reset_spec_state: Callable[[int], None] | None = None,
    ) -> None:
        self.backend = backend
        self.pool = backend.pool
        self.device = backend.device
        self.dtype = backend.dtype
        self._conv_pools = self.pool.conv_pools if conv_pools is None else conv_pools
        self._recurrent_pools = (
            self.pool.recurrent_pools if recurrent_pools is None else recurrent_pools
        )
        self._state_row = state_row
        self._reset_spec_state = reset_spec_state

        self._input_ids = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        self._positions = torch.zeros(1, dtype=torch.long, device=self.device)
        self._write_index = torch.zeros(1, dtype=torch.long, device=self.device)
        self._slot_index = torch.zeros(1, dtype=torch.long, device=self.device)
        self._attn: Qwen36DecodeGraphAttention = self.pool.build_graph_attention_driver(1)
        # Dedicated output scratch (module docstring: never shared with
        # Qwen36Backend._decode_graphs' own attn_outputs, to keep this
        # graph's captured addresses independent of the shared one's).
        self._attn_outputs: list[torch.Tensor | None] = [
            None
            if self.pool.attn_outputs[i] is None
            else torch.zeros(
                1,
                *self.pool.attn_outputs[i].shape[1:],
                dtype=self.pool.dtype,
                device=self.pool.device,
            )
            for i in range(self.pool.num_layers)
        ]

        self._graph: torch.cuda.CUDAGraph | None = None
        self._hidden: torch.Tensor | None = None
        self._captured = False

    def _batch(self) -> Qwen36DecodeBatch:
        return Qwen36DecodeBatch(
            input_ids=self._input_ids,
            positions=self._positions,
            write_index=self._write_index,
            slot_index=self._slot_index,
            attn=self._attn,
            k_pools=self.pool.k_pools,
            v_pools=self.pool.v_pools,
            conv_pools=self._conv_pools,
            recurrent_pools=self._recurrent_pools,
            attn_outputs=self._attn_outputs,
        )

    def _fill(self, slot: int, token: int, kv_len: int, *, state_col: int = 0) -> None:
        self._input_ids[0, 0] = token
        self._positions[0] = kv_len
        self._write_index[0] = decode_write_index(
            slot, kv_len, self.pool.page_size, self.pool.pages_per_slot
        )
        self._slot_index[0] = slot if self._state_row is None else self._state_row(slot, state_col)
        self._attn.page_table[0].copy_(self.pool._global_page_table[slot])  # noqa: SLF001
        self._attn.cache_seqlens[0] = kv_len + 1

    def capture(self) -> None:
        if self._captured:
            return
        scratch = self.pool.scratch_row
        self.pool.reset_slot(scratch)
        if self._reset_spec_state is not None:
            self._reset_spec_state(scratch)

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._fill(scratch, 0, 0, state_col=0)
                self.pool.model.model.decode_batch(self._batch())
        torch.cuda.current_stream().wait_stream(side)

        self._fill(scratch, 0, 0, state_col=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._hidden = self.pool.model.model.decode_batch(self._batch())
        self._graph = graph
        self._captured = True
        self.pool.reset_slot(scratch)
        if self._reset_spec_state is not None:
            self._reset_spec_state(scratch)

    def replay(self, slot: int, token: int, kv_len: int, *, state_col: int = 0) -> torch.Tensor:
        """Advance ``slot``'s backbone state by exactly one token
        (``token``, written at position ``kv_len``) and return its
        post-norm hidden state, ``[1, 1, H]``.

        Performs ONLY the forward computation + the KV/GDN pool write
        ``Qwen36SlotPool.build_decode_batch`` would also do -- NOT that
        method's OWN ``slot_kv_len``/``slot_committed_tokens`` advance
        (which would double-count the anchor token; see module docstring)
        or ``state.num_tokens_seen``/``cache.seq_len`` (the caller,
        ``Qwen36MTPEngine.round``, owns those exactly as it already does
        for the eager path -- this call must be followed by the same
        ``+= 1`` advance the eager ``self.model(anchor_input, state)``
        call performs internally).
        """
        self._fill(slot, token, kv_len, state_col=state_col)
        self._graph.replay()
        return self._hidden


class Qwen36MTPVerifyCudaGraph:
    """CUDA Graphs for the K-token target verify body.

    One graph is captured for each physical slot. The recurrent source row is
    selected through a device index that is updated before replay, so the
    source address set remains fixed while the accepted column changes. This
    removes the old K+1 graph variants and makes the graph agree with the
    fixed pooled-row mechanism used by the eager path.
    """

    def __init__(self, engine: Qwen36MTPEngine) -> None:
        if engine._spec_rows is None:  # pragma: no cover - guarded by caller
            raise RuntimeError("verify graph requires spec-row GDN state")
        self.engine = engine
        self.backend = engine.backend
        self.pool = engine.backend.pool
        self.model = engine.model
        self.device = engine.device
        self.dtype = engine.dtype
        self.k = engine.k
        self.scratch = self.pool.scratch_row
        self._draft_ids = torch.zeros(1, self.k, dtype=torch.long, device=self.device)
        self._positions = torch.zeros(self.k, dtype=torch.long, device=self.device)
        self._write_index = torch.zeros(self.k, dtype=torch.long, device=self.device)
        self._page_table_input = torch.zeros(
            1, self.pool.pages_per_slot, dtype=torch.int32, device=self.device
        )
        self._cache_seqlens_input = torch.zeros(1, dtype=torch.int32, device=self.device)
        self._cu_seqlens_q = torch.tensor([0, self.k], dtype=torch.int32, device=self.device)

        attn_drivers: list[Qwen36VerifyGraphAttention | None] = [None] * self.pool.num_layers
        attn_outputs: list[torch.Tensor | None] = [None] * self.pool.num_layers
        for layer in self.pool.model.model.layers:
            if layer.layer_type != "full_attention":
                continue
            k_pool = self.pool.k_pools[layer.layer_idx]
            assert k_pool is not None
            attn = layer.self_attn
            attn_drivers[layer.layer_idx] = Qwen36VerifyGraphAttention(
                num_q_heads=attn.num_heads,
                num_kv_heads=attn.num_kv_heads,
                head_dim=attn.head_dim,
                page_size=self.pool.page_size,
                pages_per_slot=self.pool.pages_per_slot,
                num_cache_pages=k_pool.shape[0],
                max_seq_len=self.pool.max_seq_len,
                verify_tokens=self.k,
                dtype=self.dtype,
                kv_dtype=k_pool.dtype,
                device=self.device,
            )
            attn_outputs[layer.layer_idx] = torch.zeros(
                self.k,
                attn.num_heads,
                attn.head_dim,
                dtype=self.dtype,
                device=self.device,
            )
        self._attn_drivers = attn_drivers
        self._attn_outputs = attn_outputs
        self._gdn_source_index = torch.zeros(1, dtype=torch.long, device=self.device)
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._hidden: dict[int, torch.Tensor] = {}
        self._batches: dict[int, Qwen36VerifyBatch] = {}
        # The graph captures this index tensor's address, not a particular
        # source row. Replay changes its value to select the accepted row from
        # the fixed GDN pools, just as the anchor graph does.
        for slot in range(self.pool.num_slots + 1):
            rows = engine._spec_rows.rows_for_slot(slot)
            destinations = {
                layer_idx: [states[0], *states[1:]] for layer_idx, states in rows.items()
            }
            self._batches[slot] = Qwen36VerifyBatch(
                input_ids=self._draft_ids,
                positions=self._positions,
                write_index=self._write_index,
                k_pools=self.pool.k_pools,
                v_pools=self.pool.v_pools,
                attn_drivers=self._attn_drivers,
                attn_outputs=self._attn_outputs,
                gdn_source_index=self._gdn_source_index,
                gdn_conv_pools=engine._spec_rows.conv_pools,
                gdn_recurrent_pools=engine._spec_rows.recurrent_pools,
                gdn_state_rows=destinations,
            )
        self._captured = False

    def _fill(
        self, slot: int, drafts: list[int], past_len: int, *, state_col: int | None = None
    ) -> None:
        if state_col is None:
            state_col = self.engine._spec_state_col[slot]  # noqa: SLF001
        self._gdn_source_index[0] = self.engine._spec_rows.row_for_slot(  # noqa: SLF001
            slot, state_col
        )
        self._draft_ids[0].copy_(torch.tensor(drafts, dtype=torch.long, device=self.device))
        self._positions.copy_(
            torch.arange(past_len, past_len + self.k, dtype=torch.long, device=self.device)
        )
        self._page_table_input[0].copy_(self.pool._global_page_table[slot])  # noqa: SLF001
        self._cache_seqlens_input[0] = past_len + self.k
        rows = [
            decode_write_index(slot, past_len + i, self.pool.page_size, self.pool.pages_per_slot)
            for i in range(self.k)
        ]
        self._write_index.copy_(torch.tensor(rows, dtype=torch.long, device=self.device))
        for driver in self._attn_drivers:
            if driver is not None:
                driver.update_metadata(
                    self._page_table_input,
                    self._cache_seqlens_input,
                    self._cu_seqlens_q,
                )

    def capture(self) -> None:
        if self._captured:
            return
        try:
            for slot, batch in self._batches.items():
                self.engine._spec_rows.reset_slot(slot)  # noqa: SLF001
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(3):
                        self._fill(slot, [0] * self.k, 0, state_col=0)
                        self.model.verify_batch(batch)
                torch.cuda.current_stream().wait_stream(side)

                self._fill(slot, [0] * self.k, 0, state_col=0)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    hidden = self.model.verify_batch(batch)
                self._graphs[slot] = graph
                self._hidden[slot] = hidden
        finally:
            for slot in range(self.pool.num_slots + 1):
                self.engine._spec_rows.reset_slot(slot)  # noqa: SLF001
        self._captured = True

    def replay(self, slot: int, drafts: list[int], past_len: int) -> torch.Tensor:
        if not self._captured:
            raise RuntimeError("verify CUDA Graph replay requested before capture")
        if len(drafts) != self.k:
            raise ValueError(f"verify graph expects K={self.k}, got {len(drafts)} drafts")
        self._fill(slot, drafts, past_len)
        self._graphs[slot].replay()
        return self._hidden[slot]


class Qwen36MTPDraftCudaGraph:
    """CUDA Graph for ``Qwen36MTPEngine._draft_loop``: ``K`` chained M=1
    forwards through the (1-layer) MTP head, unrolled at capture time.

    Requires the pooled MTP cache (:func:`build_pooled_mtp_caches`), not
    the per-slot standalone allocation -- see module docstring.
    """

    def __init__(self, engine: Qwen36MTPEngine) -> None:
        self.engine = engine
        self.device = engine.device
        self.dtype = engine.dtype
        self.k = engine.k
        model = engine.model
        self.mtp_head = model.mtp
        self.mtp_layer = model.mtp.layers[0]
        attn = self.mtp_layer.self_attn
        hidden_size = model.model.hidden_size

        self._seed_token = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        self._seed_hidden = torch.zeros(1, 1, hidden_size, dtype=self.dtype, device=self.device)
        self._slot_buf = torch.zeros(1, dtype=torch.long, device=self.device)
        self._start_pos = torch.zeros(1, dtype=torch.long, device=self.device)

        pages_per_slot = engine.mtp_pages_per_slot
        page_size = engine.mtp_page_size
        self._driver = Qwen36DecodeGraphAttention(
            batch=1,
            num_q_heads=attn.num_heads,
            num_kv_heads=attn.num_kv_heads,
            head_dim=attn.head_dim,
            page_size=page_size,
            pages_per_slot=pages_per_slot,
            num_cache_pages=engine.mtp_k_pool.shape[0],
            max_seq_len=pages_per_slot * page_size,
            dtype=self.dtype,
            kv_dtype=attn.kv_cache_dtype,
            device=self.device,
        )
        self._attn_output = torch.zeros(
            1, attn.num_heads, attn.head_dim, dtype=self.dtype, device=self.device
        )

        self._graph: torch.cuda.CUDAGraph | None = None
        self._draft_tokens: torch.Tensor | None = None  # [k], captured output
        self._captured = False

    def _rebind_before_replay(self, slot: int, start_pos: int) -> None:
        """Rebind the buffers that stay CONSTANT across all ``K`` steps of
        one replay (slot identity, round start position). The per-step
        position/write_index/cache_seqlens values are computed INSIDE the
        captured graph itself from ``self._slot_buf``/``self._start_pos``
        (see :meth:`_forward_all_steps`) -- only these two need a
        host->device write before each replay.
        """
        self._slot_buf.fill_(slot)
        self._start_pos.fill_(start_pos)
        pages_per_slot = self.engine.mtp_pages_per_slot
        base = slot * pages_per_slot
        self._driver.page_table[0, :pages_per_slot].copy_(
            torch.arange(base, base + pages_per_slot, dtype=torch.int32, device=self.device)
        )

    def _forward_all_steps(self) -> torch.Tensor:
        cos_sin_cache = self.engine.model.model.cos_sin_cache
        page_size = self.engine.mtp_page_size
        pages_per_slot = self.engine.mtp_pages_per_slot
        next_input = self._seed_token
        prev_hidden = self._seed_hidden
        draft_tokens: list[torch.Tensor] = []
        for step in range(self.k):
            pos = self._start_pos + step  # [1] int64, LOCAL position (module docstring)
            embeds = self.engine.model.model.embed_tokens(next_input)  # [1, 1, H]
            fused = torch.cat(
                [
                    self.mtp_head.pre_fc_norm_embedding(embeds),
                    self.mtp_head.pre_fc_norm_hidden(prev_hidden),
                ],
                dim=-1,
            )
            hidden = self.mtp_head.fc(fused)

            residual = hidden
            hidden = self.mtp_layer.input_layernorm(hidden)
            write_index = (self._slot_buf * pages_per_slot + pos // page_size) * page_size + (
                pos % page_size
            )
            self._driver.cache_seqlens.copy_((pos + 1).to(torch.int32))
            hidden = self.mtp_layer.self_attn.decode_batch(
                hidden,
                pos,
                cos_sin_cache,
                k_pool=self.engine.mtp_k_pool,
                v_pool=self.engine.mtp_v_pool,
                write_index=write_index,
                attn=self._driver,
                output=self._attn_output,
            )
            hidden = residual + hidden

            residual = hidden
            hidden = self.mtp_layer.post_attention_layernorm(hidden)
            hidden = self.mtp_layer.mlp(hidden)
            hidden = residual + hidden

            post_norm = self.mtp_head.norm(hidden)
            logits = self.engine.model.lm_head(post_norm)
            draft_token = logits[:, -1, :].argmax(dim=-1)  # [1]
            draft_tokens.append(draft_token)

            next_input = draft_token.view(1, 1)
            prev_hidden = post_norm

        return torch.cat(draft_tokens, dim=0)  # [k]

    def capture(self) -> None:
        if self._captured:
            return
        scratch = self.engine.scratch_row
        self.engine._caches[scratch].seq_len = 0  # noqa: SLF001 -- same object, own class

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._seed_token.zero_()
                self._seed_hidden.zero_()
                self._rebind_before_replay(scratch, 0)
                self._forward_all_steps()
        torch.cuda.current_stream().wait_stream(side)

        self._seed_token.zero_()
        self._seed_hidden.zero_()
        self._rebind_before_replay(scratch, 0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._draft_tokens = self._forward_all_steps()
        self._graph = graph
        self._captured = True
        self.engine._caches[scratch].seq_len = 0  # noqa: SLF001

    def replay(
        self, slot: int, seed_token: int, seed_hidden: torch.Tensor, start_pos: int
    ) -> list[int]:
        """Draft ``self.k`` tokens for ``slot``, chained, starting from
        ``seed_token``/``seed_hidden`` at cache position ``start_pos``
        (``Qwen36MTPEngine._draft_loop``'s own contract). Advances
        ``slot``'s MTP cache's ``seq_len`` by ``self.k`` -- the same side
        effect the eager ``mtp_step`` loop produces one step at a time via
        ``Qwen36PagedAttentionCache.append``, replicated here in one shot
        since this graph bypasses ``.append`` entirely (writes happen via
        the pooled ``write_index`` mechanism instead).
        """
        self._seed_token[0, 0] = seed_token
        self._seed_hidden.copy_(seed_hidden)
        self._rebind_before_replay(slot, start_pos)
        self._graph.replay()
        self.engine._caches[slot].seq_len = start_pos + self.k
        return self._draft_tokens.tolist()
