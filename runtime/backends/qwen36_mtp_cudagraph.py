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
captures the remaining ``K-1`` chained ``decode``-mode (M=1) continuation
steps as ONE graph, unrolled at capture time; step 0 is emitted by the
teacher-forced sync that writes the real target suffix. That shape already
exists in this codebase
(``Qwen36DecodeGraphAttention``, built for
``Qwen36Backend.capture_decode_cuda_graph``'s M=1 batched decode), so this
module reuses it rather than inventing a new sparkinfer graph-replay mode.

**What is captured, and what is not (say so explicitly, per the B3
follow-up brief)**:

* :class:`Qwen36MTPAnchorCudaGraph` is kept only as the reference
  implementation for a single-token MTP backbone step over the pooled
  KV/GDN storage. It is NOT captured by production anymore: M-3 folded the
  anchor into the ``K+1`` verify body, so replaying a dedicated anchor graph
  would only duplicate capture cost and reintroduce bookkeeping hazards
  around an already-committed token. ``Qwen36MTPEngine.capture_cuda_graphs``
  therefore marks ``anchor=unused`` instead of attempting this graph.

* :class:`Qwen36MTPDraftCudaGraph` -- the ``K-1`` chained continuation
  steps after teacher-forced step 0: one self-attention layer, M=1 each,
  chained. Its own dedicated pooled KV cache
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
  each request-count bucket; device source and destination row-id buffers
  select the live slots and accepted columns before replay. This is the
  captured equivalent of ``verify_forward`` -- the old eager method remains
  the fallback and the bit-exact oracle.

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
from collections.abc import Callable, Sequence
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
from runtime.round_profile import round_profile

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
            "the same round()/teacher-forced-sync path scripts/b3_mtp_e2e_acceptance_"
            "throughput.py already token-matches against non-speculative decode"
        )
        logger.error(
            "Qwen3.6 MTP: %s CUDA Graph capture failed -- %s.", name, action, exc_info=True
        )
        if strict:
            raise
        return "failed"


def decode_write_index(
    slot: int,
    kv_len: int,
    page_size: int,
    pages_per_slot: int,
    page_row: Sequence[int] | None = None,
) -> int:
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

    Phase 2 dynamic arena: ``page_row`` (the slot's logical-to-physical
    bundle row) overrides the static ``slot * pages_per_slot`` formula --
    the physical bundle for the logical page is looked up in the shared
    page table, exactly as ``Qwen36SlotPool.write_index`` does. ``None``
    keeps the legacy contiguous-row behavior.
    """
    if page_row is None:
        global_page = slot * pages_per_slot + kv_len // page_size
    else:
        logical_page = kv_len // page_size
        if logical_page >= len(page_row):
            raise RuntimeError(
                f"MTP write at {kv_len} needs logical page {logical_page} "
                f"but the page row has {len(page_row)} entries"
            )
        global_page = page_row[logical_page]
    return global_page * page_size + kv_len % page_size


def _page_table_slot_key(pool: object, slots: list[int]) -> tuple[object, ...]:
    """Key a copied graph page table by slot and dynamic mapping epoch."""
    if getattr(pool, "dynamic_arena", False):
        return tuple((slot, pool.page_table_version(slot)) for slot in slots)
    return tuple(slots)


def build_pooled_mtp_caches(
    model,
    *,
    num_slots: int,
    device: torch.device,
    dtype: torch.dtype,
    pool_bundles: int | None = None,
    page_table: torch.Tensor | None = None,
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

    Phase 2 dynamic arena (``pool_bundles`` given): the MTP KV pool spans
    the same GLOBAL physical bundle count as the backbone, and every row's
    page table aliases the backbone's ``page_table`` rows -- the bundle
    mapping (allocate/COW/evict) is shared, so MTP KV and backbone KV are
    lock-step by construction (plan §6.1). Legacy (``pool_bundles=None``):
    the historical ``(num_slots + 1) * pages_per_slot`` fixed rows with
    identity mapping.
    """
    mtp_attn = model.mtp.layers[0].self_attn
    page_size = _PAGED_ATTENTION_PAGE_SIZE
    pages_per_slot = (mtp_attn.max_seq_len + page_size - 1) // page_size
    num_rows = num_slots + 1
    if pool_bundles is None:
        total_pages = num_rows * pages_per_slot
    else:
        total_pages = pool_bundles
    kv_shape = (total_pages, page_size, mtp_attn.num_kv_heads, mtp_attn.head_dim)
    k_pool = torch.zeros(kv_shape, dtype=mtp_attn.kv_cache_dtype, device=device)
    v_pool = torch.zeros(kv_shape, dtype=mtp_attn.kv_cache_dtype, device=device)
    marker = getattr(torch._dynamo, "mark_static_address", None)
    if marker is not None:  # pragma: no branch - present in every supported torch
        marker(k_pool)
        marker(v_pool)
    if pool_bundles is not None:
        if page_table is None or page_table.shape[0] < num_rows:
            raise ValueError(
                f"dynamic MTP caches need a backbone page table with >= {num_rows} rows"
            )
        caches = [
            Qwen36PagedAttentionCache.wrap(
                k_cache=k_pool,
                v_cache=v_pool,
                page_size=page_size,
                page_table=page_table[row : row + 1],
            )
            for row in range(num_rows)
        ]
    else:
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
        self.pool.prepare_kv_writes(slot, kv_len, 1)
        self._input_ids[0, 0] = token
        self._positions[0] = kv_len
        self._write_index[0] = self.pool.write_index(slot, kv_len)
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
    """CUDA Graphs for the (anchor + K drafts) target verify body.

    The query is ``k+1`` tokens, not ``k``: position 0 is the already-committed
    anchor, positions 1..k are the drafts. Folding the anchor in here is what
    removes the separate anchor forward the round used to run first. Decode on
    this model is bandwidth-bound, so a ``k+1``-token forward costs about what
    a 1-token one does, which is why the historical implementation did exactly
    this (``oracle/.../direct_model_runner.py::verify_batch_spec``, qo_len=k+1;
    see notes/2026-08-03-mtp-verify-mode.md).

    The ``k+1`` GDN candidate rows fit this shape exactly rather than by luck:
    the kernel ``_ssm_spec_row`` is modelled on writes a state for every
    candidate position ``t`` in the batch (here ``0..k``) and reads its
    incoming state from column ``num_accepted - 1``. With the anchor always
    accepted, ``num_accepted = m + 1``, so the read column is ``m`` -- which
    is why ``Qwen36MTPGDNRows.activate(slot, m)`` is unchanged by this.

    One graph is captured for every request-count bucket. Replays update
    device source and destination row ids, so slot identity and the accepted
    column remain data rather than graph-captured addresses. This is what lets
    a batch-2 graph safely serve any two live slots, not only the pair used
    during warmup.
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
        # qo_len, not k: this graph verifies the anchor token FOLLOWED by
        # the K drafts in one pass (the historical qo_len=k+1 shape,
        # oracle direct_model_runner.py::verify_batch_spec). Every
        # query-length quantity below is k+1, not k.
        self.qo_len = engine.k + 1
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._hidden: dict[int, torch.Tensor] = {}
        #: One shared graph-private memory pool for every batch bucket in
        #: this family (plan §4.6 P0-M3): the B=1..num_slots verify graphs
        #: are mutually exclusive (exactly one replays per round, and its
        #: pool-owned hidden/logits are consumed by acceptance before the
        #: next replay), so they can share one pool exactly the way
        #: ``Qwen36Backend._capture_decode_graphs`` shares ``_graph_pool``
        #: across the main decode buckets. NOT shared with the draft/sync
        #: families: this family's hidden/logits are still referenced by
        #: the acceptance path when other families replay.
        self._graph_pool: object | None = None
        #: Graph-owned ``[B, qo_len, vocab]`` lm_head output captured
        #: alongside the verify body.  ``compute_logits`` used to run
        #: outside the graph and allocate an ~8 MiB logits tensor every
        #: round; capturing it makes the buffer graph-owned so the hot
        #: path performs zero logits allocations (measured host-side
        #: replay fill ~10 ms/round under 94/96 GiB resident memory).
        self._logits: dict[int, torch.Tensor] = {}
        self._batches: dict[int, Qwen36VerifyBatch] = {}
        self._inputs: dict[
            int,
            tuple[
                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
            ],
        ] = {}
        self._host_inputs: dict[
            int,
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
        ] = {}
        self._host_input_views: dict[
            int, tuple[object, object, object, object, object, object]
        ] = {}
        for batch in range(1, self.pool.num_slots + 1):
            self._batches[batch] = self._new_batch(batch)
        # The target pool's page table is a pure function of a physical slot
        # id.  It never grows or rebinds after construction, unlike a
        # prefix-cache block table.  Avoid repeating the same D2D copies for
        # the common stable active-slot set while retaining arbitrary order
        # on the first replay after membership changes.
        self._page_table_slots: dict[int, tuple[object, ...] | None] = {
            batch: None for batch in self._batches
        }
        self._captured = False

    def _new_batch(self, batch: int) -> Qwen36VerifyBatch:
        input_ids = torch.zeros(batch, self.qo_len, dtype=torch.long, device=self.device)
        positions = torch.zeros(batch * self.qo_len, dtype=torch.long, device=self.device)
        write_index = torch.zeros(batch * self.qo_len, dtype=torch.long, device=self.device)
        page_table = torch.zeros(
            batch, self.pool.pages_per_slot, dtype=torch.int32, device=self.device
        )
        cache_seqlens = torch.zeros(batch, dtype=torch.int32, device=self.device)
        source_index = torch.zeros(batch, dtype=torch.long, device=self.device)
        destination_index = torch.zeros(batch, self.qo_len, dtype=torch.long, device=self.device)
        attn_drivers: list[Qwen36VerifyGraphAttention | None] = [None] * self.pool.num_layers
        attn_outputs: list[torch.Tensor | None] = [None] * self.pool.num_layers
        for layer in self.pool.model.model.layers:
            if layer.layer_type != "full_attention":
                continue
            k_pool = self.pool.k_pools[layer.layer_idx]
            assert k_pool is not None
            attn = layer.self_attn
            attn_drivers[layer.layer_idx] = Qwen36VerifyGraphAttention(
                batch=batch,
                num_q_heads=attn.num_heads,
                num_kv_heads=attn.num_kv_heads,
                head_dim=attn.head_dim,
                page_size=self.pool.page_size,
                pages_per_slot=self.pool.pages_per_slot,
                num_cache_pages=k_pool.shape[0],
                max_seq_len=self.pool.max_seq_len,
                verify_tokens=self.qo_len,
                dtype=self.dtype,
                kv_dtype=k_pool.dtype,
                device=self.device,
            )
            attn_outputs[layer.layer_idx] = torch.zeros(
                batch * self.qo_len,
                attn.num_heads,
                attn.head_dim,
                dtype=self.dtype,
                device=self.device,
            )
        self._inputs[batch] = (
            input_ids,
            positions,
            write_index,
            page_table,
            cache_seqlens,
            destination_index,
        )
        host_inputs = (
            torch.zeros(batch, dtype=torch.long, device="cpu", pin_memory=True),
            torch.zeros(batch, self.qo_len, dtype=torch.long, device="cpu", pin_memory=True),
            torch.zeros(batch * self.qo_len, dtype=torch.long, device="cpu", pin_memory=True),
            torch.zeros(batch * self.qo_len, dtype=torch.long, device="cpu", pin_memory=True),
            torch.zeros(batch, dtype=torch.int32, device="cpu", pin_memory=True),
            torch.zeros(batch, self.qo_len, dtype=torch.long, device="cpu", pin_memory=True),
        )
        self._host_inputs[batch] = host_inputs
        self._host_input_views[batch] = tuple(tensor.numpy() for tensor in host_inputs)
        return Qwen36VerifyBatch(
            input_ids=input_ids,
            positions=positions,
            write_index=write_index,
            k_pools=self.pool.k_pools,
            v_pools=self.pool.v_pools,
            attn_drivers=attn_drivers,
            attn_outputs=attn_outputs,
            gdn_source_index=source_index,
            gdn_destination_index=destination_index,
            gdn_conv_pools=self.engine._spec_rows.conv_pools,  # noqa: SLF001
            gdn_recurrent_pools=self.engine._spec_rows.recurrent_pools,  # noqa: SLF001
            gdn_state_rows=None,
        )

    def _fill(
        self,
        slots: list[int],
        tokens: list[list[int]] | torch.Tensor,
        past_lens: list[int],
    ) -> None:
        batch = len(slots)
        descriptor = self._batches[batch]
        input_ids, positions, write_index, page_table, cache_seqlens, destination_index = (
            self._inputs[batch]
        )
        (
            source_host,
            tokens_host,
            positions_host,
            write_index_host,
            cache_seqlens_host,
            destination_host,
        ) = self._host_inputs[batch]
        (
            source_view,
            tokens_view,
            positions_view,
            write_index_view,
            cache_seqlens_view,
            destination_view,
        ) = self._host_input_views[batch]
        if isinstance(tokens, torch.Tensor):
            if tuple(tokens.shape) != (batch, self.qo_len):
                raise ValueError(
                    f"verify graph expects [{batch}, anchor+K={self.qo_len}] device tokens"
                )
        elif any(len(row) != self.qo_len for row in tokens):
            raise ValueError(f"verify graph expects anchor+K={self.qo_len} tokens per slot")
        for slot, past_len in zip(slots, past_lens, strict=True):
            self.pool.prepare_kv_writes(slot, past_len, self.qo_len)
        source_cols = [self.engine._spec_state_col[slot] for slot in slots]  # noqa: SLF001
        source_view[:] = [
            self.engine._spec_rows.row_for_slot(slot, col)  # noqa: SLF001
            for slot, col in zip(slots, source_cols, strict=True)
        ]
        destination_view[:] = [
            [
                self.engine._spec_rows.row_for_slot(slot, col)  # noqa: SLF001
                for col in range(self.qo_len)
            ]
            for slot in slots
        ]
        if isinstance(tokens, torch.Tensor):
            input_ids.copy_(tokens, non_blocking=True)
        else:
            tokens_view[:] = tokens
        positions_view[:] = [
            position
            for past_len in past_lens
            for position in range(past_len, past_len + self.qo_len)
        ]
        cache_seqlens_view[:] = [past_len + self.qo_len for past_len in past_lens]
        write_index_view[:] = [
            self.pool.write_index(slot, past_len + offset)
            for slot, past_len in zip(slots, past_lens, strict=True)
            for offset in range(self.qo_len)
        ]
        descriptor.gdn_source_index.copy_(source_host, non_blocking=True)
        destination_index.copy_(destination_host, non_blocking=True)
        if not isinstance(tokens, torch.Tensor):
            input_ids.copy_(tokens_host, non_blocking=True)
        positions.copy_(positions_host, non_blocking=True)
        cache_seqlens.copy_(cache_seqlens_host, non_blocking=True)
        write_index.copy_(write_index_host, non_blocking=True)
        slot_key = _page_table_slot_key(self.pool, slots)
        if self._page_table_slots[batch] != slot_key:
            for row, slot in enumerate(slots):
                page_table[row].copy_(self.pool._global_page_table[slot])  # noqa: SLF001
            self._page_table_slots[batch] = slot_key
        for driver in descriptor.attn_drivers:
            if driver is not None:
                driver.update_metadata(
                    page_table,
                    cache_seqlens,
                    driver._cu_seqlens_q,  # noqa: SLF001 - graph-owned metadata buffer
                    host_cache_seqlens=cache_seqlens_view,
                )

    def capture(self) -> None:
        if self._captured:
            return
        try:
            for batch_size, batch in self._batches.items():
                slots = list(range(batch_size))
                for slot in slots:
                    self.engine._spec_rows.reset_slot(slot)  # noqa: SLF001
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(3):
                        self._fill(slots, [[0] * self.qo_len for _ in slots], [0] * batch_size)
                        self.model.verify_batch(batch)
                torch.cuda.current_stream().wait_stream(side)

                self._fill(slots, [[0] * self.qo_len for _ in slots], [0] * batch_size)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=self._graph_pool):
                    hidden = self.model.verify_batch(batch)
                    logits = self.model.compute_logits(hidden)
                if self._graph_pool is None:
                    self._graph_pool = graph.pool()
                self._graphs[batch_size] = graph
                self._hidden[batch_size] = hidden
                self._logits[batch_size] = logits
        finally:
            for slot in range(self.pool.num_slots + 1):
                self.engine._spec_rows.reset_slot(slot)  # noqa: SLF001
        self._captured = True

    def replay(
        self, slots: list[int], tokens: list[list[int]], past_lens: list[int]
    ) -> torch.Tensor:
        if not self._captured:
            raise RuntimeError("verify CUDA Graph replay requested before capture")
        batch = len(slots)
        self._fill(slots, tokens, past_lens)
        self._graphs[batch].replay()
        return self._hidden[batch], self._logits[batch]


class Qwen36MTPDraftCudaGraph:
    """CUDA Graphs for ``K-1`` speculative MTP continuation steps.

    A draft is autoregressive *within* a slot, but step ``j`` is independent
    across slots once step ``j - 1`` has completed.  Historical Qwen3.6
    therefore ran one B-wide MTP-head forward per chained step, rather than
    replaying B separate B=1 graphs.  Capture one unrolled graph for each
    B=1..num_slots and retain pooled KV addressing so any live slot subset
    can use the matching graph.
    """

    def __init__(self, engine: Qwen36MTPEngine) -> None:
        self.engine = engine
        self.device = engine.device
        self.dtype = engine.dtype
        # The teacher-forced MTP sync emits draft step 0 while writing the
        # real prefix.  Graph only the remaining speculative continuation;
        # otherwise replay would append an extra token and reintroduce the
        # tail-only state model this class is meant to accelerate.
        self.steps = max(engine.k - 1, 0)
        model = engine.model
        self.mtp_head = model.mtp
        self.mtp_layer = model.mtp.layers[0]
        attn = self.mtp_layer.self_attn
        hidden_size = model.model.hidden_size

        pages_per_slot = engine.mtp_pages_per_slot
        if getattr(engine.backend.pool, "dynamic_arena", False):
            # Phase 2 dynamic arena: MTP shares the backbone's bundle
            # mapping. Each draft graph's page-table row source aliases the
            # backbone pool's device page table (a stable, graph-compatible
            # tensor; only its contents change at replay).
            self._page_table_by_slot = engine.backend.pool._global_page_table  # noqa: SLF001
        else:
            page_ids = torch.arange(
                (engine.backend.num_slots + 1) * pages_per_slot,
                dtype=torch.int32,
                device=self.device,
            )
            self._page_table_by_slot = page_ids.reshape(
                engine.backend.num_slots + 1, pages_per_slot
            )
        self._inputs: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._host_inputs: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._host_input_views: dict[int, tuple[object, object, object]] = {}
        self._drivers: dict[int, Qwen36DecodeGraphAttention] = {}
        self._attn_outputs: dict[int, torch.Tensor] = {}
        self._page_table_slots: dict[int, tuple[object, ...] | None] = {}
        for batch in range(1, engine.backend.num_slots + 1):
            self._inputs[batch] = (
                torch.zeros(batch, 1, dtype=torch.long, device=self.device),
                torch.zeros(batch, 1, hidden_size, dtype=self.dtype, device=self.device),
                torch.zeros(batch, dtype=torch.long, device=self.device),
                torch.zeros(batch, dtype=torch.long, device=self.device),
            )
            host_inputs = (
                torch.zeros(batch, dtype=torch.long, device="cpu", pin_memory=True),
                torch.zeros(batch, dtype=torch.long, device="cpu", pin_memory=True),
                torch.zeros(batch, dtype=torch.long, device="cpu", pin_memory=True),
            )
            self._host_inputs[batch] = host_inputs
            self._host_input_views[batch] = tuple(tensor.numpy() for tensor in host_inputs)
            self._drivers[batch] = Qwen36DecodeGraphAttention(
                batch=batch,
                num_q_heads=attn.num_heads,
                num_kv_heads=attn.num_kv_heads,
                head_dim=attn.head_dim,
                page_size=engine.mtp_page_size,
                pages_per_slot=pages_per_slot,
                num_cache_pages=engine.mtp_k_pool.shape[0],
                max_seq_len=pages_per_slot * engine.mtp_page_size,
                dtype=self.dtype,
                kv_dtype=attn.kv_cache_dtype,
                device=self.device,
            )
            self._attn_outputs[batch] = torch.zeros(
                batch, attn.num_heads, attn.head_dim, dtype=self.dtype, device=self.device
            )
            self._page_table_slots[batch] = None
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._draft_tokens: dict[int, torch.Tensor] = {}  # B -> [B, K-1]
        #: One shared graph-private memory pool for every batch bucket in
        #: this family (plan §4.6 P0-M3): the B=1..num_slots draft graphs
        #: are mutually exclusive (exactly one replays per round, and its
        #: pool-owned draft-token rows are consumed by the same round's
        #: verify fill), same sharing discipline as the main decode graphs.
        self._graph_pool: object | None = None
        self._captured = False

    def _fill(
        self,
        slots: list[int],
        seed_tokens: list[int] | torch.Tensor,
        seed_hiddens: torch.Tensor,
        start_positions: list[int],
    ) -> None:
        """Stage one B-wide draft round's inputs.

        ``seed_tokens`` may be a host ``list[int]`` (historical path) or a
        device tensor ``[B]`` (the ragged-sync fast path: the sync graph's
        ``_step_tokens`` already lives on device, so copying it D2D into the
        graph-owned buffer avoids the mid-round D2H ``.tolist()`` + H2D
        re-upload that made the host block before the draft graph could be
        enqueued -- measured 2026-08-05 as a 5-15 ms/round host gap at 128K).
        """
        batch = len(slots)
        input_ids, prev_hidden, slot_buf, start_pos = self._inputs[batch]
        if not (len(seed_tokens) == len(start_positions) == batch):
            raise ValueError("MTP draft graph slots/tokens/start positions must have equal length")
        if tuple(seed_hiddens.shape[:2]) != (batch, 1):
            raise ValueError("MTP draft graph seed hidden must have shape [B, 1, H]")
        seed_tokens_host, slots_host, start_positions_host = self._host_inputs[batch]
        seed_tokens_view, slots_view, start_positions_view = self._host_input_views[batch]
        slots_view[:] = slots
        start_positions_view[:] = start_positions
        if isinstance(seed_tokens, torch.Tensor) or (
            isinstance(seed_tokens, (list, tuple))
            and seed_tokens
            and all(isinstance(token, torch.Tensor) for token in seed_tokens)
        ):
            if isinstance(seed_tokens, (list, tuple)):
                seed_tokens = torch.cat([token.reshape(1) for token in seed_tokens])
            if tuple(seed_tokens.shape) != (batch,):
                raise ValueError(
                    "MTP draft graph device seed tokens must have shape [B]"
                    f", got {tuple(seed_tokens.shape)}"
                )
            if seed_tokens.dtype != torch.long or seed_tokens.device.type != self.device.type:
                raise ValueError(
                    "MTP draft graph device seed tokens must be torch.long on "
                    f"{self.device}, got dtype={seed_tokens.dtype} device={seed_tokens.device}"
                )
            input_ids[:, 0].copy_(seed_tokens, non_blocking=True)
        else:
            seed_tokens_view[:] = seed_tokens
            input_ids[:, 0].copy_(seed_tokens_host, non_blocking=True)
        prev_hidden.copy_(seed_hiddens)
        slot_buf.copy_(slots_host, non_blocking=True)
        start_pos.copy_(start_positions_host, non_blocking=True)
        driver = self._drivers[batch]
        slot_key = _page_table_slot_key(self.engine.backend.pool, slots)
        if self._page_table_slots[batch] != slot_key:
            for row, slot in enumerate(slots):
                driver.page_table[row].copy_(self._page_table_by_slot[slot])
            self._page_table_slots[batch] = slot_key
        # Re-chunk the draft attention split-KV for this round's live KV
        # length; without it every replay keeps the worst-case chunking
        # captured at load (~1.1 ms vs ~0.4 ms per draft attention call at
        # 128K, measured 2026-08-06).
        driver.cache_seqlens.copy_(start_pos, non_blocking=True)
        driver.update_replay_metadata()

    def _forward_all_steps(self, batch: int) -> torch.Tensor:
        cos_sin_cache = self.engine.model.model.cos_sin_cache
        page_size = self.engine.mtp_page_size
        pages_per_slot = self.engine.mtp_pages_per_slot
        next_input, prev_hidden, slot_buf, start_pos = self._inputs[batch]
        driver = self._drivers[batch]
        attn_output = self._attn_outputs[batch]
        draft_tokens: list[torch.Tensor] = []
        driver.cache_seqlens.copy_(start_pos)
        dynamic = getattr(self.engine.backend.pool, "dynamic_arena", False)
        for step in range(self.steps):
            pos = start_pos + step
            embeds = self.engine.model.model.embed_tokens(next_input)
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
            if dynamic:
                # Phase 2 dynamic arena: the physical bundle for each logical
                # page comes from the shared page table (same bundle mapping
                # as the backbone). ``driver.page_table`` is [B, pages_per_slot]
                # and was already filled by ``_fill`` for this round's slots;
                # gather is graph-capturable (a pure device tensor op).
                batch_idx = torch.arange(batch, device=self.device)
                logical_page = pos // page_size
                physical_page = driver.page_table[
                    batch_idx, logical_page.clamp(max=pages_per_slot - 1)
                ]
                write_index = physical_page * page_size + pos % page_size
            else:
                write_index = (slot_buf * pages_per_slot + pos // page_size) * page_size + (
                    pos % page_size
                )
            driver.cache_seqlens.add_(1)
            hidden = self.mtp_layer.self_attn.decode_batch(
                hidden,
                pos,
                cos_sin_cache,
                k_pool=self.engine.mtp_k_pool,
                v_pool=self.engine.mtp_v_pool,
                write_index=write_index,
                attn=driver,
                output=attn_output,
            )
            hidden = residual + hidden

            residual = hidden
            hidden = self.mtp_layer.post_attention_layernorm(hidden)
            hidden = self.mtp_layer.mlp(hidden)
            hidden = residual + hidden

            post_norm = self.mtp_head.norm(hidden)
            logits = self.engine.model.lm_head(post_norm)
            draft_token = logits[:, -1, :].argmax(dim=-1)
            draft_tokens.append(draft_token)

            next_input = draft_token.unsqueeze(1)
            prev_hidden = post_norm

        if not draft_tokens:
            return torch.empty(batch, 0, dtype=torch.long, device=self.device)
        return torch.stack(draft_tokens, dim=1)  # [B, K-1]

    def capture(self) -> None:
        if self._captured:
            return
        for batch in range(1, self.engine.backend.num_slots + 1):
            slots = list(range(batch))
            zeros = torch.zeros(
                batch,
                1,
                self.engine.model.model.hidden_size,
                dtype=self.dtype,
                device=self.device,
            )
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._fill(slots, [0] * batch, zeros, [0] * batch)
                    self._forward_all_steps(batch)
            torch.cuda.current_stream().wait_stream(side)

            self._fill(slots, [0] * batch, zeros, [0] * batch)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._graph_pool):
                self._draft_tokens[batch] = self._forward_all_steps(batch)
            if self._graph_pool is None:
                self._graph_pool = graph.pool()
            self._graphs[batch] = graph
        self._captured = True
        for cache in self.engine._caches:
            cache.seq_len = 0

    def replay_batch(
        self,
        slots: list[int],
        seed_tokens: list[int] | torch.Tensor,
        seed_hiddens: torch.Tensor,
        start_positions: list[int],
    ) -> dict[int, torch.Tensor]:
        """Replay one B-wide chained draft graph and advance each cache.

        ``seed_tokens`` may be a device tensor (D2D staged by :meth:`_fill`)
        or a host list (historical H2D staging).  Returns the K-1 tails per
        slot as device rows ``[K-1]`` -- the values are consumed only by the
        next verify fill and by the GPU-side accept comparison, neither of
        which needs a host round-trip.  The 2026-08-06 128K/c4 phase profile
        measured the old ``.tolist()`` here as ~7 ms/round of host time that
        serialised after the draft graph had already finished.
        """
        if not self._captured:
            raise RuntimeError("MTP draft CUDA Graph replay requested before capture")
        batch = len(slots)
        self._fill(slots, seed_tokens, seed_hiddens, start_positions)
        round_profile.phase("draft_fill")
        self._graphs[batch].replay()
        for slot, start_pos in zip(slots, start_positions, strict=True):
            self.engine._caches[slot].seq_len = start_pos + self.steps
        round_profile.phase("draft_gpu_wait")
        draft_rows = self._draft_tokens[batch]
        return {slot: draft_rows[index] for index, slot in enumerate(slots)}

    def replay(
        self, slot: int, seed_token: int, seed_hidden: torch.Tensor, start_pos: int
    ) -> list[int]:
        """Single-slot compatibility wrapper around :meth:`replay_batch`."""
        return self.replay_batch([slot], [seed_token], seed_hidden, [start_pos])[slot]


class Qwen36MTPBatchedSync:
    """B-wide teacher-forced MTP synchronisation over the pooled draft KV.

    Target verify has one fixed ``K+1`` shape, but each request accepts a
    different number of drafts.  The historical runner's batched step-0 path
    therefore padded every slot to one fixed ``max_q`` body, then chose each
    request's own last valid logits/hidden row and rewound the physical MTP
    cache lengths back to the real boundary.  This adapter mirrors that
    contract: it executes its one-layer MTP forward as ``q`` B-wide decode
    steps over the pooled KV ownership and reuses the same small ``(B, q)``
    CUDA-graph buckets for both equal-length and ragged suffixes.
    """

    def __init__(self, engine: Qwen36MTPEngine) -> None:
        self.engine = engine
        self.device = engine.device
        self.dtype = engine.dtype
        self.mtp_head = engine.model.mtp
        self.mtp_layer = engine.model.mtp.layers[0]
        attn = self.mtp_layer.self_attn
        if getattr(engine.backend.pool, "dynamic_arena", False):
            # Phase 2 dynamic arena: same shared-bundle page table as the
            # backbone (see Qwen36MTPDraftCudaGraph.__init__).
            self._page_table_by_slot = engine.backend.pool._global_page_table  # noqa: SLF001
        else:
            self._page_table_by_slot = torch.arange(
                (engine.backend.num_slots + 1) * engine.mtp_pages_per_slot,
                dtype=torch.int32,
                device=self.device,
            ).reshape(engine.backend.num_slots + 1, engine.mtp_pages_per_slot)
        self._inputs: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._host_inputs: dict[
            int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        self._host_input_views: dict[int, tuple[object, object, object, object]] = {}
        self._tokens: dict[int, torch.Tensor] = {}
        self._target_hidden: dict[int, torch.Tensor] = {}
        self._lengths: dict[int, torch.Tensor] = {}
        self._row_index: dict[int, torch.Tensor] = {}
        self._drivers: dict[int, Qwen36DecodeGraphAttention] = {}
        self._attn_outputs: dict[int, torch.Tensor] = {}
        self._decode_page_table_slots: dict[int, tuple[object, ...] | None] = {}
        self._graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self._verify_inputs: dict[
            tuple[int, int],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}
        self._verify_host_inputs: dict[
            tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        self._verify_host_input_views: dict[tuple[int, int], tuple[object, object, object]] = {}
        self._verify_drivers: dict[tuple[int, int], Qwen36VerifyGraphAttention] = {}
        self._verify_attn_outputs: dict[tuple[int, int], torch.Tensor] = {}
        self._verify_page_table_slots: dict[tuple[int, int], tuple[object, ...] | None] = {}
        self._verify_graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        #: One shared graph-private memory pool for ALL sync + sync-verify
        #: buckets in this family (plan §4.6 P0-M3, the 28->1 merge): every
        #: graph body here writes its results into the pre-allocated
        #: ``_step_tokens``/``_step_hidden`` buffers, so nothing a graph
        #: produces lives in the pool -- pool contents are pure
        #: intermediates, and only one bucket replays per round. Kept
        #: separate from the verify/draft families on purpose (their
        #: pool-owned outputs are consumed across family boundaries).
        self._graph_pool: object | None = None
        self._step_tokens: dict[int, torch.Tensor] = {}
        self._step_hidden: dict[int, torch.Tensor] = {}
        self._verify_supported = hasattr(self.mtp_layer.self_attn, "verify_batch")
        hidden_size = engine.model.model.hidden_size
        for batch in range(1, engine.backend.num_slots + 1):
            self._inputs[batch] = (
                torch.zeros(batch, 1, dtype=torch.long, device=self.device),
                torch.zeros(batch, 1, hidden_size, dtype=self.dtype, device=self.device),
                torch.zeros(batch, dtype=torch.long, device=self.device),
                torch.zeros(batch, dtype=torch.long, device=self.device),
            )
            self._tokens[batch] = torch.zeros(
                batch, engine.k + 1, dtype=torch.long, device=self.device
            )
            self._target_hidden[batch] = torch.zeros(
                batch,
                engine.k + 1,
                hidden_size,
                dtype=self.dtype,
                device=self.device,
            )
            self._lengths[batch] = torch.zeros(batch, dtype=torch.long, device=self.device)
            self._row_index[batch] = torch.arange(batch, dtype=torch.long, device=self.device)
            host_inputs = (
                torch.zeros(batch, engine.k + 1, dtype=torch.long, device="cpu", pin_memory=True),
                torch.zeros(batch, dtype=torch.long, device="cpu", pin_memory=True),
                torch.zeros(batch, dtype=torch.long, device="cpu", pin_memory=True),
                torch.zeros(batch, dtype=torch.long, device="cpu", pin_memory=True),
            )
            self._host_inputs[batch] = host_inputs
            self._host_input_views[batch] = tuple(tensor.numpy() for tensor in host_inputs)
            self._drivers[batch] = Qwen36DecodeGraphAttention(
                batch=batch,
                num_q_heads=attn.num_heads,
                num_kv_heads=attn.num_kv_heads,
                head_dim=attn.head_dim,
                page_size=engine.mtp_page_size,
                pages_per_slot=engine.mtp_pages_per_slot,
                num_cache_pages=engine.mtp_k_pool.shape[0],
                max_seq_len=engine.mtp_pages_per_slot * engine.mtp_page_size,
                dtype=self.dtype,
                kv_dtype=attn.kv_cache_dtype,
                device=self.device,
            )
            self._attn_outputs[batch] = torch.zeros(
                batch, attn.num_heads, attn.head_dim, dtype=self.dtype, device=self.device
            )
            self._decode_page_table_slots[batch] = None
            self._step_tokens[batch] = torch.zeros(
                batch, engine.k + 1, dtype=torch.long, device=self.device
            )
            self._step_hidden[batch] = torch.zeros(
                batch,
                engine.k + 1,
                hidden_size,
                dtype=self.dtype,
                device=self.device,
            )
            if self._verify_supported:
                # ``q=1`` is exactly the decode shape and shares sparkinfer's
                # workspace cache key with the decode driver above.  It has no
                # multi-token work to collapse, so keep that case on the
                # proven decode fallback instead of creating an incompatible
                # verify-mode workspace for the same geometry.
                for query_len in range(2, engine.k + 2):
                    key = (batch, query_len)
                    self._verify_inputs[key] = (
                        torch.zeros(batch, query_len, dtype=torch.long, device=self.device),
                        torch.zeros(batch * query_len, dtype=torch.long, device=self.device),
                        torch.zeros(batch * query_len, dtype=torch.long, device=self.device),
                        torch.zeros(
                            batch, engine.mtp_pages_per_slot, dtype=torch.int32, device=self.device
                        ),
                        torch.zeros(batch, dtype=torch.int32, device=self.device),
                    )
                    verify_host_inputs = (
                        torch.zeros(
                            batch * query_len, dtype=torch.long, device="cpu", pin_memory=True
                        ),
                        torch.zeros(
                            batch * query_len, dtype=torch.long, device="cpu", pin_memory=True
                        ),
                        torch.zeros(batch, dtype=torch.int32, device="cpu", pin_memory=True),
                    )
                    self._verify_host_inputs[key] = verify_host_inputs
                    self._verify_host_input_views[key] = tuple(
                        tensor.numpy() for tensor in verify_host_inputs
                    )
                    self._verify_drivers[key] = Qwen36VerifyGraphAttention(
                        batch=batch,
                        num_q_heads=attn.num_heads,
                        num_kv_heads=attn.num_kv_heads,
                        head_dim=attn.head_dim,
                        page_size=engine.mtp_page_size,
                        pages_per_slot=engine.mtp_pages_per_slot,
                        num_cache_pages=engine.mtp_k_pool.shape[0],
                        max_seq_len=engine.mtp_pages_per_slot * engine.mtp_page_size,
                        verify_tokens=query_len,
                        dtype=self.dtype,
                        kv_dtype=attn.kv_cache_dtype,
                        device=self.device,
                    )
                    self._verify_attn_outputs[key] = torch.zeros(
                        batch * query_len,
                        attn.num_heads,
                        attn.head_dim,
                        dtype=self.dtype,
                        device=self.device,
                    )
                    self._verify_page_table_slots[key] = None

        self._captured = False

    def _fill(
        self,
        slots: list[int],
        shifted_token_ids: list[list[int]],
        target_hidden: torch.Tensor,
        start_positions: list[int],
    ) -> int:
        batch = len(slots)
        if batch < 1 or batch not in self._inputs:
            raise ValueError("batched MTP sync requires 1..num_slots slots")
        if len(shifted_token_ids) != batch or len(start_positions) != batch:
            raise ValueError("MTP sync slots/tokens/start positions must have equal length")
        query_len = len(shifted_token_ids[0])
        if query_len <= 0 or query_len > self.engine.k + 1:
            raise ValueError(f"MTP sync query length must be in 1..{self.engine.k + 1}")
        if any(len(tokens) != query_len for tokens in shifted_token_ids):
            raise ValueError("batched MTP sync requires one uniform query length")
        if tuple(target_hidden.shape[:2]) != (batch, query_len):
            raise ValueError("batched MTP sync target hidden must have shape [B, q, H]")

        input_ids, prev_hidden, slot_buf, start_pos = self._inputs[batch]
        del input_ids, prev_hidden
        tokens_host, slots_host, starts_host, _ = self._host_inputs[batch]
        tokens_view, slots_view, starts_view, _ = self._host_input_views[batch]
        tokens_view[:, :query_len] = shifted_token_ids
        slots_view[:] = slots
        starts_view[:] = start_positions
        self._tokens[batch][:, :query_len].copy_(tokens_host[:, :query_len], non_blocking=True)
        self._target_hidden[batch][:, :query_len].copy_(target_hidden)
        slot_buf.copy_(slots_host, non_blocking=True)
        start_pos.copy_(starts_host, non_blocking=True)
        driver = self._drivers[batch]
        slot_key = _page_table_slot_key(self.engine.backend.pool, slots)
        if self._decode_page_table_slots[batch] != slot_key:
            for row, slot in enumerate(slots):
                driver.page_table[row].copy_(self._page_table_by_slot[slot])
            self._decode_page_table_slots[batch] = slot_key
        self._lengths[batch].fill_(query_len)
        return query_len

    def _fill_ragged(
        self,
        slots: list[int],
        shifted_token_ids: list[list[int]],
        target_hidden_rows: list[torch.Tensor],
        start_positions: list[int],
    ) -> int:
        batch = len(slots)
        if batch < 1 or batch not in self._inputs:
            raise ValueError("batched MTP sync requires 1..num_slots slots")
        if (
            len(shifted_token_ids) != batch
            or len(target_hidden_rows) != batch
            or len(start_positions) != batch
        ):
            raise ValueError("MTP sync slots/tokens/hidden/start positions must have equal length")
        lengths = [len(tokens) for tokens in shifted_token_ids]
        query_len = max(lengths)
        if query_len <= 0 or query_len > self.engine.k + 1:
            raise ValueError(f"MTP sync query length must be in 1..{self.engine.k + 1}")

        input_ids, prev_hidden, slot_buf, start_pos = self._inputs[batch]
        del input_ids, prev_hidden
        tokens_host, slots_host, starts_host, lengths_host = self._host_inputs[batch]
        tokens_view, slots_view, starts_view, lengths_view = self._host_input_views[batch]
        slots_view[:] = slots
        starts_view[:] = start_positions
        lengths_view[:] = lengths
        slot_buf.copy_(slots_host, non_blocking=True)
        start_pos.copy_(starts_host, non_blocking=True)
        self._lengths[batch].copy_(lengths_host, non_blocking=True)
        driver = self._drivers[batch]
        for row, slot in enumerate(slots):
            row_hidden = target_hidden_rows[row]
            if (
                row_hidden.dim() != 3
                or row_hidden.shape[0] != 1
                or row_hidden.shape[1] != lengths[row]
            ):
                raise ValueError("ragged MTP sync target hidden must have shape [1, q, H] per slot")
            row_tokens = shifted_token_ids[row]
            tokens_view[row, : lengths[row]] = row_tokens
            self._tokens[batch][row, : lengths[row]].copy_(
                tokens_host[row, : lengths[row]], non_blocking=True
            )
            self._target_hidden[batch][row, : lengths[row]].copy_(row_hidden[0])
            if lengths[row] < query_len:
                pad_len = query_len - lengths[row]
                self._tokens[batch][row, lengths[row] : query_len].fill_(row_tokens[-1])
                self._target_hidden[batch][row, lengths[row] : query_len].copy_(
                    row_hidden[0, -1:].expand(pad_len, -1)
                )
        slot_key = _page_table_slot_key(self.engine.backend.pool, slots)
        if self._decode_page_table_slots[batch] != slot_key:
            for row, slot in enumerate(slots):
                driver.page_table[row].copy_(self._page_table_by_slot[slot])
            self._decode_page_table_slots[batch] = slot_key
        return query_len

    def _fill_verify_ragged(
        self,
        slots: list[int],
        shifted_token_ids: list[list[int]],
        target_hidden_rows: list[torch.Tensor],
        start_positions: list[int],
    ) -> int:
        query_len = self._fill_ragged(slots, shifted_token_ids, target_hidden_rows, start_positions)
        key = (len(slots), query_len)
        input_ids, positions, write_index, page_table, cache_seqlens = self._verify_inputs[key]
        positions_host, write_index_host, cache_seqlens_host = self._verify_host_inputs[key]
        positions_view, write_index_view, cache_seqlens_view = self._verify_host_input_views[key]
        driver = self._verify_drivers[key]
        input_ids.copy_(self._tokens[len(slots)][:, :query_len])
        positions_view[:] = [
            position for start in start_positions for position in range(start, start + query_len)
        ]
        write_index_view[:] = [
            decode_write_index(
                slot,
                start + offset,
                self.engine.mtp_page_size,
                self.engine.mtp_pages_per_slot,
                page_row=(
                    self.engine.backend.pool._page_table_host[slot]  # noqa: SLF001
                    if getattr(self.engine.backend.pool, "dynamic_arena", False)
                    else None
                ),
            )
            for slot, start in zip(slots, start_positions, strict=True)
            for offset in range(query_len)
        ]
        cache_seqlens_view[:] = [start + query_len for start in start_positions]
        positions.copy_(positions_host, non_blocking=True)
        write_index.copy_(write_index_host, non_blocking=True)
        cache_seqlens.copy_(cache_seqlens_host, non_blocking=True)
        slot_key = _page_table_slot_key(self.engine.backend.pool, slots)
        if self._verify_page_table_slots[key] != slot_key:
            for row, slot in enumerate(slots):
                page_table[row].copy_(self._page_table_by_slot[slot])
            self._verify_page_table_slots[key] = slot_key
        driver.update_metadata(
            page_table,
            cache_seqlens,
            driver._cu_seqlens_q,  # noqa: SLF001 - graph-owned uniform-q metadata
        )
        return query_len

    def _forward_all_steps(self, batch: int, query_len: int) -> None:
        input_ids, prev_hidden, slot_buf, start_pos = self._inputs[batch]
        tokens = self._tokens[batch]
        target_hidden = self._target_hidden[batch]
        driver = self._drivers[batch]
        cos_sin_cache = self.engine.model.model.cos_sin_cache
        page_size = self.engine.mtp_page_size
        pages_per_slot = self.engine.mtp_pages_per_slot
        driver.cache_seqlens.copy_(start_pos)
        for step in range(query_len):
            input_ids.copy_(tokens[:, step : step + 1])
            prev_hidden.copy_(target_hidden[:, step : step + 1])
            positions = start_pos + step
            embeds = self.engine.model.model.embed_tokens(input_ids)
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
            if getattr(self.engine.backend.pool, "dynamic_arena", False):
                # Phase 2 dynamic arena: same shared-bundle page-table gather
                # as Qwen36MTPDraftCudaGraph._forward_all_steps.
                batch_idx = torch.arange(batch, device=self.device)
                logical_page = positions // page_size
                physical_page = driver.page_table[
                    batch_idx, logical_page.clamp(max=pages_per_slot - 1)
                ]
                write_index = physical_page * page_size + positions % page_size
            else:
                write_index = (slot_buf * pages_per_slot + positions // page_size) * page_size + (
                    positions % page_size
                )
            driver.cache_seqlens.add_(1)
            hidden = self.mtp_layer.self_attn.decode_batch(
                hidden,
                positions,
                cos_sin_cache,
                k_pool=self.engine.mtp_k_pool,
                v_pool=self.engine.mtp_v_pool,
                write_index=write_index,
                attn=driver,
                output=self._attn_outputs[batch],
            )
            hidden = residual + hidden
            residual = hidden
            hidden = self.mtp_layer.post_attention_layernorm(hidden)
            hidden = self.mtp_layer.mlp(hidden)
            post_norm = self.mtp_head.norm(residual + hidden)
            self._step_hidden[batch][:, step : step + 1].copy_(post_norm)
        # The shifted teacher-forced tokens are supplied by the target; only
        # its final MTP row is consumed as draft step 0.  Avoid projecting
        # the preceding rows through the 248k-vocabulary head.
        self._step_tokens[batch][:, 0].copy_(
            self.engine.model.lm_head(post_norm).argmax(dim=-1).reshape(batch)
        )

    def _forward_verify_body(self, batch: int, query_len: int) -> None:
        key = (batch, query_len)
        input_ids, positions, write_index, _, _ = self._verify_inputs[key]
        driver = self._verify_drivers[key]
        output = self._verify_attn_outputs[key]
        hidden_states = self._target_hidden[batch][:, :query_len]
        cos_sin_cache = self.engine.model.model.cos_sin_cache

        embeds = self.engine.model.model.embed_tokens(input_ids)
        fused = torch.cat(
            [
                self.mtp_head.pre_fc_norm_embedding(embeds),
                self.mtp_head.pre_fc_norm_hidden(hidden_states),
            ],
            dim=-1,
        )
        hidden = self.mtp_head.fc(fused)
        residual = hidden
        hidden = self.mtp_layer.input_layernorm(hidden)
        hidden = self.mtp_layer.self_attn.verify_batch(
            hidden,
            positions,
            cos_sin_cache,
            k_pool=self.engine.mtp_k_pool,
            v_pool=self.engine.mtp_v_pool,
            write_index=write_index,
            attn=driver,
            output=output,
        )
        hidden = residual + hidden
        residual = hidden
        hidden = self.mtp_layer.post_attention_layernorm(hidden)
        hidden = self.mtp_layer.mlp(hidden)
        post_norm = self.mtp_head.norm(residual + hidden)
        self._step_hidden[batch][:, :query_len].copy_(post_norm)
        # Ragged rows have different real last positions.  The static graph
        # still receives one fixed ``query_len`` body, but its only required
        # vocab projections are the B dynamically selected final rows.
        last_post_norm = post_norm[self._row_index[batch], self._lengths[batch] - 1]
        self._step_tokens[batch][:, 0].copy_(
            self.engine.model.lm_head(last_post_norm).argmax(dim=-1)
        )

    def _gather_last(self, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        last_index = self._lengths[batch] - 1
        row_index = self._row_index[batch]
        last_tokens = self._step_tokens[batch][:, 0]
        last_hidden = self._step_hidden[batch][row_index, last_index].unsqueeze(1)
        return last_tokens, last_hidden

    def capture(self) -> None:
        if self._captured:
            return
        try:
            for batch in self._inputs:
                slots = list(range(batch))
                zeros = torch.zeros(
                    batch,
                    self.engine.k + 1,
                    self.engine.model.model.hidden_size,
                    dtype=self.dtype,
                    device=self.device,
                )
                for query_len in range(1, self.engine.k + 2):
                    self._fill(
                        slots,
                        [[0] * query_len for _ in slots],
                        zeros[:, :query_len],
                        [0] * batch,
                    )
                    side = torch.cuda.Stream()
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        for _ in range(3):
                            self._forward_all_steps(batch, query_len)
                    torch.cuda.current_stream().wait_stream(side)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=self._graph_pool):
                        self._forward_all_steps(batch, query_len)
                    if self._graph_pool is None:
                        self._graph_pool = graph.pool()
                    self._graphs[(batch, query_len)] = graph
                    if self._verify_supported and query_len > 1:
                        zero_rows = [zeros[row : row + 1, :query_len] for row in range(batch)]
                        self._fill_verify_ragged(
                            slots,
                            [[0] * query_len for _ in slots],
                            zero_rows,
                            [0] * batch,
                        )
                        verify_graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(verify_graph, pool=self._graph_pool):
                            self._forward_verify_body(batch, query_len)
                        self._verify_graphs[(batch, query_len)] = verify_graph
        except Exception:
            # Do not replay an arbitrary successful prefix of the bucket
            # matrix after a later capture failed.  The caller retains this
            # object and uses its all-eager B-wide fallback instead.
            self._graphs.clear()
            self._verify_graphs.clear()
            raise
        finally:
            for cache in self.engine._caches:
                cache.seq_len = 0
        self._captured = True

    def replay(
        self,
        slots: list[int],
        shifted_token_ids: list[list[int]],
        target_hidden: torch.Tensor,
        start_positions: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Synchronise one equal-length group and return its last token/hidden.

        ``target_hidden`` is request-major ``[B, q, H]`` and every shifted
        token row has that same ``q``.  The caller owns cache-length and
        real-boundary bookkeeping after this returns.
        """
        query_len = self._fill(slots, shifted_token_ids, target_hidden, start_positions)
        graph = self._graphs.get((len(slots), query_len))
        if graph is None:
            self._forward_all_steps(len(slots), query_len)
            return self._gather_last(len(slots))
        graph.replay()
        return self._gather_last(len(slots))

    def replay_ragged(
        self,
        slots: list[int],
        shifted_token_ids: list[list[int]],
        target_hidden_rows: list[torch.Tensor],
        start_positions: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Synchronise one ragged all-slot body and gather each real last row."""
        batch = len(slots)
        max_query_len = max((len(tokens) for tokens in shifted_token_ids), default=0)
        # q=1 is the pre-existing decode geometry (and deliberately has no
        # verify workspace because sparkinfer caches that geometry by shape).
        # There is no multi-token work to fuse in this case.
        if self._verify_supported and max_query_len > 1:
            query_len = self._fill_verify_ragged(
                slots, shifted_token_ids, target_hidden_rows, start_positions
            )
            graph = self._verify_graphs.get((batch, query_len))
            if graph is None:
                self._forward_verify_body(batch, query_len)
            else:
                graph.replay()
            return self._gather_last(batch)
        query_len = self._fill_ragged(slots, shifted_token_ids, target_hidden_rows, start_positions)
        graph = self._graphs.get((batch, query_len))
        if graph is None:
            self._forward_all_steps(batch, query_len)
        else:
            graph.replay()
        return self._gather_last(batch)

    def replay_eager(
        self,
        slots: list[int],
        shifted_token_ids: list[list[int]],
        target_hidden: torch.Tensor,
        start_positions: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the same grouped sync body without a CUDA Graph.

        Capture failure must retain the existing B-wide eager path rather
        than silently turning a multi-slot group back into serial work.  The
        method is also the direct correctness reference for the graph smoke.
        """
        batch = len(slots)
        query_len = self._fill(slots, shifted_token_ids, target_hidden, start_positions)
        self._forward_all_steps(batch, query_len)
        return self._gather_last(batch)
