"""Fixed-slot resource pool for Qwen3.6 -- Track B / B2.

This is the module that makes "递归状态纳入槽位生命周期" concrete. B1's
:class:`runtime.model.qwen36_model.Qwen36GenerationState` is a *plain
per-sequence container a caller owns directly* (its own docstring says so);
serving needs the opposite: a fixed number of slots whose buffers are
allocated **once**, at a stable address, and handed to whichever request
currently owns that slot.

Three properties this pool exists to guarantee
----------------------------------------------

1. **Allocate once, never rebind** (B0-5, ``notes/2026-08-02-trackB-b0-gpu-
   facts.md``). Every recurrent-state buffer below is created in
   :meth:`__init__`, marked with ``torch._dynamo.mark_static_address``, and
   from then on only ever written through ``.copy_()`` /
   ``.index_copy_()`` / ``.zero_()``. The matching half of this rule lives
   in ``Qwen36GatedDeltaNet.forward``/``decode_batch``, which B2 changed
   from ``state.recurrent_state = last_state.to(dtype)`` (a rebind) to
   ``state.recurrent_state.copy_(last_state)``.

2. **A fresh slot's recurrent state is explicitly zeroed** -- the one
   operational requirement B0-5 attached to its "capture-safe" verdict.
   The recurrence is not idempotent: unlike a KV cache, whose stale bytes
   past ``kv_len`` are simply never read, a stale GDN state *is* read on
   the very first step of the next sequence and silently poisons it. So
   :meth:`reset_slot` zeroes, and does not merely mark, and
   ``has_previous_state`` is reset alongside so the first forward takes
   the ``initial_state=None`` branch.

   The KV side is deliberately *not* zeroed, matching
   ``LagunaBackend.reset_slot``: its bytes are the prefix cache.

3. **One allocation, two addressings.** Slot ``s``'s attention pages are
   ``k_pool[s * pages_per_slot : (s+1) * pages_per_slot]`` -- contiguous,
   so the same bytes can be reached either as a per-slot view with local
   page ids ``0..P-1`` (the single-sequence path: byte-identical to what
   B1 runs) or through a global page table (``s * P + local``, the batched
   decode path). Two views, one allocation, so they can never disagree
   about what is where.

Static per-slot page assignment, not a ``BlockPool``
----------------------------------------------------
Slot ``s`` owns pages ``[s*P, (s+1)*P)`` for the life of the process. This
matches what the runtime already does for its only production model
(``server/app.py:1248-1249``: "LagunaBackend uses static block allocation
(num_slots × blocks_per_slot), not a dynamic BlockPool") and it is what
``docs/a3-cache-coordinator-design.md`` §8 decision 5 explicitly decided
*not* to change as part of this work ("切到 ``BlockPool``是一个独立的、有
自己收益论证的性能特性"). Same-slot prefix reuse is exactly what static
assignment supports, and it is what the prefix-cache path below implements.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from runtime.model.qwen36_kv_arena import BlockKey
from runtime.model.qwen36_model import (
    _PAGED_ATTENTION_PAGE_SIZE,
    GdnLayerState,
    Qwen36BatchedDecodeAttention,
    Qwen36BatchedExtendAttention,
    Qwen36DecodeBatch,
    Qwen36DecodeGraphAttention,
    Qwen36ForCausalLMSelfBuilt,
    Qwen36GenerationState,
    Qwen36PagedAttentionCache,
    Qwen36PrefillBatch,
)


def _mark_static(tensor: torch.Tensor) -> torch.Tensor:
    """``mark_static_address`` where available, no-op where it is not.

    Copied in spirit from ``transformers/cache_utils.py``'s
    ``LinearAttentionLayer`` (B0-5's "现成可抄的模式"). It is a hint to
    dynamo/inductor that this buffer's address never changes; it is not
    what makes CUDA Graph capture correct (the ``.copy_()`` discipline
    is), so a torch build without it is not a correctness problem and
    must not be a hard dependency.
    """
    marker = getattr(torch._dynamo, "mark_static_address", None)
    if marker is not None:  # pragma: no branch - present in every supported torch
        marker(tensor)
    return tensor


@dataclass(frozen=True)
class SlotPoolGeometry:
    """Shapes this pool was built for -- reported, not re-derived.

    ``/metrics`` and the GPU verification scripts both need to state what
    was actually allocated; deriving it a second time from the model is
    how two numbers that must agree stop agreeing.
    """

    num_slots: int
    max_seq_len: int
    page_size: int
    pages_per_slot: int
    num_recurrent_layers: int
    num_paged_kv_layers: int
    #: Bytes of recurrent (conv + ssm) state for ONE slot, all layers. This
    #: is the number ``docs/a3-cache-coordinator-design.md`` §4 says must be
    #: recomputed from the real checkpoint rather than inherited from
    #: ``notes/prefix-cache-design.md``'s "~151MB/checkpoint".
    recurrent_bytes_per_slot: int
    kv_bytes_per_slot: int
    #: Physical KV rows actually allocated: ``num_slots`` real slots + one
    #: scratch row. ``kv_bytes_total`` is the formula equivalent of the
    #: actual tensor storage (per-layer ``total_rows * pages_per_slot *
    #: page_size * kv_heads * head_dim * element_size * 2`` summed over
    #: full-attention layers); :meth:`Qwen36SlotPool.kv_storage_bytes`
    #: reports the measured tensor storage, and the two must agree. This is
    #: the number Phase 0 of ``.omx/plans/qwen38-dynamic-context-vllm-plan.md``
    #: locks before any ownership change: current layout with the scratch row
    #: is 45 GiB-equivalent at 4×256K, and the dynamic arena's first win is
    #: the 9 GiB scratch row.
    total_rows: int
    kv_bytes_total: int


class Qwen36SlotPool:
    """Pre-allocated per-slot KV pages + recurrent state for one model.

    ``num_slots`` slots plus **one scratch row** (index ``num_slots``).
    The scratch row exists so a CUDA Graph captured at batch ``B`` can be
    replayed with fewer than ``B`` live sequences without a padded entry
    aliasing a real slot's recurrent state -- an alias there would not
    crash, it would let a padding row's write land in a live sequence's
    state, which is the silent-corruption failure mode
    ``docs/a3-cache-coordinator-design.md`` §2 INV-A3-1 describes.
    """

    def __init__(
        self,
        model: Qwen36ForCausalLMSelfBuilt,
        *,
        num_slots: int,
        max_seq_len: int,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        dynamic_arena: bool = False,
        pool_bundles: int | None = None,
        watermark_bundles: int = 0,
    ) -> None:
        if num_slots < 1:
            raise ValueError(f"num_slots must be >= 1, got {num_slots}")
        self.model = model
        self.num_slots = num_slots
        self.device = torch.device(device)
        self.dtype = dtype
        self.page_size = _PAGED_ATTENTION_PAGE_SIZE
        self.pages_per_slot = (max_seq_len + self.page_size - 1) // self.page_size
        self.max_seq_len = self.pages_per_slot * self.page_size
        #: Rows are ``num_slots`` real slots + one scratch row (class docstring).
        self._num_rows = num_slots + 1
        self.scratch_row = num_slots
        #: Phase 2 dynamic arena (`.omx/plans/qwen38-dynamic-context-vllm-plan.md`
        #: Phase 2, "接入全局物理 arena,先完成 strict 模式"): when enabled, KV
        #: tensors span a GLOBAL physical bundle pool sized by ``pool_bundles``
        #: instead of ``(num_slots + 1) * pages_per_slot`` fixed rows. Every
        #: logical slot row keeps its fixed ``pages_per_slot`` width but starts
        #: empty (all entries point at the null bundle 0); bundles are handed
        #: out on demand by :meth:`prepare_kv_writes` and returned by
        #: :meth:`reset_slot`. Legacy mode (default) is byte-identical to the
        #: historical fixed-row layout and remains the A/B baseline.
        self.dynamic_arena = dynamic_arena
        if dynamic_arena:
            if pool_bundles is None:
                # strict-mode default: full concurrent capacity (one full
                # sequence per slot) plus a small COW/emergency reserve.
                pool_bundles = num_slots * self.pages_per_slot + 8
            self.pool_bundles = pool_bundles
            self.watermark_bundles = watermark_bundles
            from runtime.model.qwen36_kv_arena import QwenPageBundlePool

            self._arena = QwenPageBundlePool(
                pool_bundles,
                watermark_bundles=watermark_bundles,
                assert_invariants=True,
            )
            #: slot -> set of physical bundle ids it currently references.
            #: Mirrors vLLM ``req_to_blocks``; ``reset_slot`` decrefs them all
            #: (plan §7 invariant 7: slot reset leaves no old-epoch ownership).
            self._slot_bundles: list[set[int]] = [set() for _ in range(self._num_rows)]
        else:
            self.pool_bundles = self._num_rows * self.pages_per_slot
            self.watermark_bundles = 0
            self._arena = None
            self._slot_bundles = [set() for _ in range(self._num_rows)]

        layers = model.model.layers
        self.num_layers = len(layers)
        self.k_pools: list[torch.Tensor | None] = [None] * self.num_layers
        self.v_pools: list[torch.Tensor | None] = [None] * self.num_layers
        self.conv_pools: list[torch.Tensor | None] = [None] * self.num_layers
        self.recurrent_pools: list[torch.Tensor | None] = [None] * self.num_layers
        self.attn_outputs: list[torch.Tensor | None] = [None] * self.num_layers
        #: batch size -> shared attention driver for a decode step of that
        #: size. Eager drivers are built on demand; a graph driver replaces
        #: the eager one for a batch size only once its graph is captured.
        self.decode_attn: dict[int, Qwen36BatchedDecodeAttention] = {}
        self.graph_attn: dict[int, Qwen36DecodeGraphAttention] = {}
        # Prefill has a variable Q axis and is not graph-replayed. Keep only
        # the most recently used exact-shape driver rather than accumulating
        # one large sparkinfer workspace per prompt length.
        self._prefill_attn: Qwen36BatchedExtendAttention | None = None
        self._attn_geometry: dict[str, int] | None = None
        #: The KV pools' own storage dtype -- read off each full-attention
        #: layer's ``kv_cache_dtype`` (FP8 KV, 2026-08-03 follow-up: BF16
        #: unless that layer was built with ``enable_fp8_kv=True``) rather
        #: than assumed to equal ``dtype`` (the pool's *compute* dtype,
        #: still used for conv/recurrent pools below, which never change).
        #: ``getattr(..., dtype)`` falls back to ``dtype`` for a minimal
        #: stub layer object that doesn't expose the attribute at all (e.g.
        #: ``tests/test_qwen36_slot_pool.py``'s ``SimpleNamespace`` stub) --
        #: same BF16-pool-dtype behavior this class always had before FP8
        #: KV existed.
        self._kv_dtype: torch.dtype | None = None

        recurrent_bytes = 0
        kv_bytes = 0
        num_recurrent = 0
        num_paged = 0
        total_pages = self.pool_bundles

        for layer in layers:
            i = layer.layer_idx
            if layer.layer_type == "linear_attention":
                num_recurrent += 1
                gdn = layer.linear_attn
                conv = torch.zeros(
                    self._num_rows,
                    gdn.conv_dim,
                    gdn.conv_kernel_size,
                    device=self.device,
                    dtype=dtype,
                )
                recurrent = torch.zeros(
                    self._num_rows,
                    gdn.num_v_heads,
                    gdn.head_k_dim,
                    gdn.head_v_dim,
                    device=self.device,
                    dtype=dtype,
                )
                self.conv_pools[i] = _mark_static(conv)
                self.recurrent_pools[i] = _mark_static(recurrent)
                recurrent_bytes += (
                    conv[0].numel() * conv.element_size()
                    + recurrent[0].numel() * recurrent.element_size()
                )
            else:
                num_paged += 1
                attn = layer.self_attn
                kv_dtype = getattr(attn, "kv_cache_dtype", dtype)
                if self._kv_dtype is None:
                    self._kv_dtype = kv_dtype
                elif self._kv_dtype != kv_dtype:
                    # One shared decode driver per batch size (below) needs
                    # one KV dtype for the whole step -- same reasoning as
                    # the _attn_geometry uniformity check just below this
                    # loop, for a different attribute.
                    raise ValueError(
                        "Qwen36SlotPool assumes every full-attention layer shares one KV "
                        f"cache dtype; layer {i} has {kv_dtype}, earlier layers have "
                        f"{self._kv_dtype}"
                    )
                shape = (total_pages, self.page_size, attn.num_kv_heads, attn.head_dim)
                k = torch.zeros(shape, device=self.device, dtype=kv_dtype)
                v = torch.zeros(shape, device=self.device, dtype=kv_dtype)
                self.k_pools[i] = _mark_static(k)
                self.v_pools[i] = _mark_static(v)
                kv_bytes += (
                    2
                    * self.pages_per_slot
                    * self.page_size
                    * (attn.num_kv_heads * attn.head_dim * k.element_size())
                )
                # Actual per-layer physical storage (K + V), which is what
                # ``kv_bytes_total`` must reproduce exactly -- even when the
                # dynamic arena's pool size is not a multiple of
                # ``pages_per_slot`` (strict 4x256K + reserve is).
                kv_bytes_actual = getattr(self, "_kv_bytes_actual", 0)
                kv_bytes_actual += 2 * total_pages * self.page_size * (
                    attn.num_kv_heads * attn.head_dim * k.element_size()
                )
                self._kv_bytes_actual = kv_bytes_actual
                self.attn_outputs[i] = _mark_static(
                    torch.zeros(
                        self._num_rows,
                        attn.num_heads,
                        attn.head_dim,
                        device=self.device,
                        dtype=dtype,
                    )
                )
                layer_max = getattr(attn, "max_seq_len", None)
                if layer_max is not None:
                    layer_pages = (layer_max + self.page_size - 1) // self.page_size
                    if layer_pages != self.pages_per_slot:
                        # Qwen36Attention caches ONE extend workspace per layer,
                        # sized from the first cache it is handed
                        # (``_workspace_for``). Load-time warmup hands it a
                        # cache built from the layer's own ``max_seq_len``. If
                        # the pool's per-slot page count disagrees, prefill
                        # later feeds a differently-sized page table into a
                        # workspace whose capacity was fixed for the other one
                        # -- a capacity error at best, and at worst a silent
                        # mis-plan. Refuse at construction, where the two
                        # numbers are both in scope.
                        raise ValueError(
                            f"slot pool wants {self.pages_per_slot} pages/slot but layer {i} "
                            f"was built for max_seq_len={layer_max} "
                            f"({layer_pages} pages); load the model with the same "
                            "max_seq_len the backend is given"
                        )
                geometry = {
                    "num_q_heads": attn.num_heads,
                    "num_kv_heads": attn.num_kv_heads,
                    "head_dim": attn.head_dim,
                }
                if self._attn_geometry is None:
                    self._attn_geometry = geometry
                elif self._attn_geometry != geometry:
                    # One shared driver per step is only correct while every
                    # full-attention layer has the same shape. This
                    # checkpoint's 16 do; a future one that does not must
                    # fail here rather than silently run 15 layers through a
                    # workspace built for the 16th.
                    raise ValueError(
                        "Qwen36SlotPool assumes every full-attention layer shares one "
                        f"geometry; layer {i} has {geometry}, earlier layers have "
                        f"{self._attn_geometry}"
                    )

        self.geometry = SlotPoolGeometry(
            num_slots=num_slots,
            max_seq_len=self.max_seq_len,
            page_size=self.page_size,
            pages_per_slot=self.pages_per_slot,
            num_recurrent_layers=num_recurrent,
            num_paged_kv_layers=num_paged,
            recurrent_bytes_per_slot=recurrent_bytes,
            kv_bytes_per_slot=kv_bytes,
            # Per-layer KV storage is ``physical_bundles * page_size *
            # kv_heads * head_dim * element_size`` per tensor (K and V), so
            # the whole-pool formula is per-slot bytes scaled by the physical
            # bundle count. Legacy: ``num_rows`` rows of ``pages_per_slot``.
            # Dynamic arena (Phase 2): the configured global pool size.
            total_rows=self._num_rows,
            kv_bytes_total=getattr(self, "_kv_bytes_actual", kv_bytes * self._num_rows),
        )

        # One logical-to-physical page-table row per slot.  The initial
        # mapping is the historical contiguous ownership, but *every* Qwen
        # attention path reads this table now: B=1, B×1 decode, B×Q prefill,
        # MTP graph metadata.  Future prefix-cache allocation can therefore
        # replace entries without changing the model's addressing contract.
        if self.dynamic_arena:
            # Phase 2 dynamic arena: every logical row starts EMPTY, all
            # entries pointing at the null bundle (0). Bundles are handed out
            # on demand by prepare_kv_writes. The device table is still
            # created once at a stable address so CUDA Graph capture sees a
            # fixed tensor; only its contents change.
            self._global_page_table = _mark_static(
                torch.zeros(
                    self._num_rows, self.pages_per_slot, dtype=torch.int32, device=self.device
                )
            )
        else:
            self._global_page_table = _mark_static(
                torch.arange(
                    self._num_rows * self.pages_per_slot, dtype=torch.int32, device=self.device
                ).view(self._num_rows, self.pages_per_slot)
            )
        # Host-side mirror for write-index construction.  Attention metadata
        # consumes the device table, while decode/prefill must know the same
        # physical page ids before their kernels run.  Keeping a small Python
        # mirror avoids a device-to-host synchronisation on every decode
        # round; all future remapping goes through ``set_page_table_row`` so
        # the two representations change atomically.
        if self.dynamic_arena:
            self._page_table_host = [
                [0] * self.pages_per_slot for _ in range(self._num_rows)
            ]
        else:
            self._page_table_host = [
                [slot * self.pages_per_slot + page for page in range(self.pages_per_slot)]
                for slot in range(self._num_rows)
            ]
        # Graph wrappers cache a copied page-table row by slot identity.  A
        # monotonic version makes that cache safe once a logical row is
        # remapped for prefix sharing without forcing every stable replay to
        # copy its whole table again.
        self._page_table_versions = [0] * self._num_rows
        if self.dynamic_arena:
            # Legacy refcount bookkeeping is unused in dynamic mode: the
            # arena owns ownership. Keep the fields at None so any legacy
            # code path that touches them fails loudly instead of silently
            # tracking a second truth.
            self._page_refcounts = None  # type: ignore[assignment]
            self._free_physical_pages = None  # type: ignore[assignment]
        else:
            # Every physical page starts owned by exactly one logical row.  A
            # cross-slot prefix restore can replace a target's prefix pages with
            # aliases of a retained source; the displaced pages become the small
            # free reserve used for copy-on-write before either sharer writes.
            # This is intentionally fixed-capacity: it reuses the allocation the
            # backend already owns and never asks a shared single GPU for more
            # KV memory at admission time.
            self._page_refcounts = [1] * (self._num_rows * self.pages_per_slot)
            self._free_physical_pages: set[int] = set()

        # -- per-slot single-sequence views (the B1-identical path) --------
        self._slot_states: list[Qwen36GenerationState] = [
            self._build_slot_state(row) for row in range(self._num_rows)
        ]
        #: Per-slot destination buffers for recurrent-state checkpoints.
        #: ``capture_recurrent_state`` copies into these (one
        #: ``torch._foreach_copy_`` launch) instead of issuing 96
        #: per-tensor ``clone()`` + allocator round trips on every block
        #: boundary.  A slot holds at most one live rolling checkpoint at a
        #: time (``_evict_checkpoint`` pops the previous one before the next
        #: capture), so reusing one buffer per slot is safe; the persistent
        #: prefix family clones again at store time and never aliases it.
        self._checkpoint_dest: dict[int, list[torch.Tensor]] = {}
        #: Populated exactly once, before CUDA Graph capture, when MTP is
        #: enabled.  Each entry has ``K + 1`` views per physical slot: column
        #: zero aliases the ordinary live row and columns ``1..K`` are the
        #: persistent speculative candidates.  Keeping the base rows in the
        #: same allocation is the historical ``spec_row`` contract; a copied
        #: private column zero would add both D2D work and one unnecessary
        #: GDN-state row per slot.
        self._mtp_gdn_columns: dict[int, list[list[GdnLayerState]]] | None = None
        self._mtp_num_speculative_tokens: int | None = None

        # -- persistent batched-decode buffers -----------------------------
        b = self._num_rows
        self._batch_input_ids = _mark_static(
            torch.zeros(b, 1, dtype=torch.long, device=self.device)
        )
        self._batch_positions = _mark_static(torch.zeros(b, dtype=torch.long, device=self.device))
        self._batch_write_index = _mark_static(torch.zeros(b, dtype=torch.long, device=self.device))
        self._batch_slot_index = _mark_static(torch.arange(b, dtype=torch.long, device=self.device))
        # Historical CUDA-graph replay used pinned CPU staging for every
        # per-round scalar/vector input.  Keep the decode path under the
        # same discipline: building a fresh CPU tensor for token ids,
        # positions, lengths, write rows, and slots costs host allocations
        # on precisely the one-token hot path graphs are meant to trim.
        # CPU-backed unit-test pools cannot allocate pinned memory, so pin
        # only when this pool actually feeds CUDA.
        pin_memory = self.device.type == "cuda"
        self._batch_input_ids_host = torch.zeros(
            b, 1, dtype=torch.long, device="cpu", pin_memory=pin_memory
        )
        self._batch_positions_host = torch.zeros(
            b, dtype=torch.long, device="cpu", pin_memory=pin_memory
        )
        self._batch_cache_seqlens_host = torch.zeros(
            b, dtype=torch.int32, device="cpu", pin_memory=pin_memory
        )
        self._batch_write_index_host = torch.zeros(
            b, dtype=torch.long, device="cpu", pin_memory=pin_memory
        )
        self._batch_slot_index_host = torch.zeros(
            b, dtype=torch.long, device="cpu", pin_memory=pin_memory
        )
        self._np_batch_input_ids = self._batch_input_ids_host.numpy()
        self._np_batch_positions = self._batch_positions_host.numpy()
        self._np_batch_cache_seqlens = self._batch_cache_seqlens_host.numpy()
        self._np_batch_write_index = self._batch_write_index_host.numpy()
        self._np_batch_slot_index = self._batch_slot_index_host.numpy()
        # Host mirrors: the scheduler reads these every round, and reading
        # them off the device would put a sync in the decode loop.
        self.slot_kv_len = [0] * self._num_rows
        self.slot_committed_tokens: list[list[int]] = [[] for _ in range(self._num_rows)]

    # -- Phase 0 capacity evidence (read-only) ------------------------------

    def kv_storage_bytes(self) -> int:
        """Measured device storage of every backbone KV tensor in the pool.

        Sums ``torch.Tensor.numel() * element_size`` over the K/V pools of
        all full-attention layers -- the ground truth that
        ``geometry.kv_bytes_total``'s formula must reproduce. Phase 0 locks
        ``formula == measured`` before any ownership change can move either
        number.
        """
        total = 0
        for k_pool, v_pool in zip(self.k_pools, self.v_pools, strict=True):
            if k_pool is None:
                assert v_pool is None
                continue
            assert v_pool is not None
            total += k_pool.numel() * k_pool.element_size()
            total += v_pool.numel() * v_pool.element_size()
        return total

    def capacity_snapshot(self) -> dict[str, int]:
        """Read-only geometry/capacity snapshot, reported not re-derived.

        ``/debug/stats`` and the GPU verification scripts both state what
        was actually allocated; deriving it a second time from the model is
        how two numbers that must agree stop agreeing. Keys are stable and
        unit-annotated so a startup log and a metrics gauge stay comparable.
        The ``qwen_kv_*`` keys are the Prometheus-facing names used by
        ``server/metrics.py``; the plain names remain for the startup log.
        """
        return {
            "num_slots": self.num_slots,
            "max_seq_len": self.max_seq_len,
            "page_size": self.page_size,
            "pages_per_slot": self.pages_per_slot,
            "total_rows": self._num_rows,
            "scratch_row": self.scratch_row,
            "num_full_attention_layers": self.geometry.num_paged_kv_layers,
            "num_gdn_layers": self.geometry.num_recurrent_layers,
            "kv_bytes_per_slot": self.geometry.kv_bytes_per_slot,
            "kv_bytes_total": self.geometry.kv_bytes_total,
            "kv_bytes_measured": self.kv_storage_bytes(),
            "kv_bytes_scratch_row": self.geometry.kv_bytes_per_slot,
            "recurrent_bytes_per_slot": self.geometry.recurrent_bytes_per_slot,
            # Prometheus-facing aliases (Phase 0 gauges).
            "qwen_kv_pool_bytes": self.geometry.kv_bytes_total,
            "qwen_kv_pool_bytes_measured": self.kv_storage_bytes(),
            "qwen_kv_scratch_row_bytes": self.geometry.kv_bytes_per_slot,
            "qwen_kv_total_bundles": self.pool_bundles,
            "qwen_kv_pages_per_slot": self.pages_per_slot,
            "qwen_kv_slots": self.num_slots,
            "qwen_kv_full_attention_layers": self.geometry.num_paged_kv_layers,
            "qwen_kv_gdn_layers": self.geometry.num_recurrent_layers,
            "qwen_kv_mode": 1 if self.dynamic_arena else 0,
        }

    def assert_kv_storage_consistent(self) -> None:
        """Refuse to run when the formula and the tensors disagree.

        The formula and the measured storage are computed from the same
        loops in ``__init__``, so a mismatch means either this method or the
        geometry is no longer derived from the actual allocation -- exactly
        the class of drift Phase 0 exists to catch before it becomes a
        memory claim. Mirrors the existing "reported, not re-derived"
        contract of :class:`SlotPoolGeometry`.
        """
        measured = self.kv_storage_bytes()
        if measured != self.geometry.kv_bytes_total:
            raise RuntimeError(
                "Qwen3.6 KV capacity formula disagrees with actual tensor storage: "
                f"formula={self.geometry.kv_bytes_total}, measured={measured}; "
                "fix the geometry derivation before trusting either number"
            )

    # -- per-slot single-sequence views (the B1-identical path) ------------
    # (init code: _slot_states is built in __init__ after the page tables)

    # -- construction helpers ---------------------------------------------

    def _build_slot_state(self, row: int) -> Qwen36GenerationState:
        gdn_states: list[GdnLayerState | None] = []
        attn_caches: list[Qwen36PagedAttentionCache | None] = []
        for i in range(self.num_layers):
            if self.conv_pools[i] is not None:
                gdn_states.append(
                    GdnLayerState(
                        conv_state=self.conv_pools[i][row : row + 1],
                        recurrent_state=self.recurrent_pools[i][row : row + 1],
                        has_previous_state=False,
                    )
                )
                attn_caches.append(None)
            else:
                gdn_states.append(None)
                attn_caches.append(
                    Qwen36PagedAttentionCache.wrap(
                        k_cache=self.k_pools[i],
                        v_cache=self.v_pools[i],
                        page_size=self.page_size,
                        page_table=self._global_page_table[row : row + 1],
                    )
                )
        return Qwen36GenerationState(gdn_states=gdn_states, attn_caches=attn_caches)

    # -- slot lifecycle ----------------------------------------------------

    def slot_state(self, slot: int) -> Qwen36GenerationState:
        """The persistent per-sequence state object for ``slot``.

        Always the *same* object for the same slot: callers may hold it
        across requests, because :meth:`reset_slot` clears it in place
        rather than replacing it.
        """
        return self._slot_states[slot]

    def set_page_table_row(self, slot: int, physical_pages: list[int]) -> None:
        """Replace one slot's logical-to-physical page row atomically.

        This is deliberately the sole remapping primitive.  The static
        allocator initially gives every slot a contiguous row, but prefix
        sharing/COW will not.  Batched attention reads the device table while
        decode/prefill construct write rows from the host mirror; updating
        only one of those two would produce a valid-looking but corrupted KV
        cache.

        Dynamic-arena mode: ownership bookkeeping belongs to the arena, so
        this method only syncs the two representations. Callers that change a
        row's ownership must go through :meth:`prepare_kv_writes` /
        :meth:`share_prefix_kv` / :meth:`reset_slot`, which update the arena
        refcounts and then call this for the device copy.
        """
        if not 0 <= slot < self._num_rows:
            raise ValueError(f"slot {slot} is outside the pool")
        if len(physical_pages) != self.pages_per_slot:
            raise ValueError(
                f"slot {slot} needs {self.pages_per_slot} physical pages, got {len(physical_pages)}"
            )
        total_pages = self.pool_bundles
        if any(page < 0 or page >= total_pages for page in physical_pages):
            raise ValueError("a page-table row must contain in-range physical pages")
        if not self.dynamic_arena:
            # Legacy: rows must be bijective (each slot owns distinct pages).
            if len(set(physical_pages)) != len(physical_pages):
                raise ValueError("a page-table row must contain distinct physical pages")
        else:
            # Dynamic: repeated null bundles (0) are legal -- an empty row is
            # all zeros. Non-null entries must still be distinct: two logical
            # pages cannot alias one physical bundle unless share_prefix_kv
            # incref'd it (which it does before remapping).
            non_null = [p for p in physical_pages if p != 0]
            if len(set(non_null)) != len(non_null):
                raise ValueError(
                    "a dynamic page-table row must not repeat non-null bundles"
                )
        if not self.dynamic_arena:
            old_pages = self._page_table_host[slot]
            for page in old_pages:
                self._page_refcounts[page] -= 1
                if self._page_refcounts[page] == 0:
                    self._free_physical_pages.add(page)
            for page in physical_pages:
                self._page_refcounts[page] += 1
                self._free_physical_pages.discard(page)
        self._page_table_host[slot] = list(physical_pages)
        self._global_page_table[slot].copy_(
            torch.tensor(physical_pages, dtype=torch.int32, device=self.device)
        )
        self._page_table_versions[slot] += 1

    def page_table_version(self, slot: int) -> int:
        """Return the generation for graph metadata cached for ``slot``."""
        return self._page_table_versions[slot]

    def write_index(self, slot: int, kv_len: int) -> int:
        """Return the physical flattened KV row for one logical token."""
        if not 0 <= kv_len < self.max_seq_len:
            raise ValueError(f"KV position {kv_len} is outside pool capacity {self.max_seq_len}")
        physical_page = self._page_table_host[slot][kv_len // self.page_size]
        return physical_page * self.page_size + kv_len % self.page_size

    def prepare_kv_writes(self, slot: int, start: int, length: int) -> None:
        """Make every page touched by a pending write private to ``slot``.

        Prefix aliases are read-only by construction.  The first target
        decode/prefill/verify write that reaches an aliased page clones the
        whole physical page into a previously displaced fixed-capacity page,
        remaps the logical entry, then lets the normal write proceed.  Whole
        page copies deliberately cover a 64-token GDN checkpoint that lies
        inside one 128-token attention page: the untouched prefix half stays
        byte-identical while the suffix overwrites its own page.
        """
        if length < 0 or start < 0 or start + length > self.max_seq_len:
            raise ValueError(
                f"KV write [{start}, {start + length}) is outside pool capacity {self.max_seq_len}"
            )
        if length == 0:
            return
        logical_pages = range(
            start // self.page_size,
            (start + length - 1) // self.page_size + 1,
        )
        row = list(self._page_table_host[slot])
        if self.dynamic_arena:
            # Phase 2 dynamic arena: hand out physical bundles on demand.
            # A null entry (0) gets a fresh private bundle; a shared entry
            # (arena refcnt > 1) is COW-cloned. Private entries (refcnt == 1)
            # are already writable and left alone. All bookkeeping stays in
            # the arena; the device table is synced once at the end (one
            # copy per chunk, not per token -- plan §6.6).
            replacements: list[tuple[int, int, int]] = []
            for logical_page in logical_pages:
                source_page = row[logical_page]
                if source_page == 0:
                    target_page = self._arena.allocate(1, owner=f"slot-{slot}")[0]
                    replacements.append((logical_page, source_page, target_page))
                    row[logical_page] = target_page
                    self._slot_bundles[slot].add(target_page)
                elif self._arena.bundles[source_page].ref_cnt > 1:
                    # COW detach: the slot transfers its reference from the
                    # shared source to a fresh private clone.
                    target_page = self._arena.ensure_writable(source_page)
                    self._arena.decref([source_page], owner=f"slot-{slot}")
                    self._slot_bundles[slot].discard(source_page)
                    replacements.append((logical_page, source_page, target_page))
                    row[logical_page] = target_page
                    self._slot_bundles[slot].add(target_page)
            if not replacements:
                return
            for _logical_page, source_page, target_page in replacements:
                for k_pool, v_pool in zip(self.k_pools, self.v_pools, strict=True):
                    if k_pool is None:
                        continue
                    assert v_pool is not None
                    if source_page != 0:
                        # COW clone: copy the shared page's bytes so the
                        # untouched prefix half stays valid.
                        k_pool[target_page].copy_(k_pool[source_page])
                        v_pool[target_page].copy_(v_pool[source_page])
            self._arena.drain_pending_cow()
            self.set_page_table_row(slot, row)
            return
        replacements: list[tuple[int, int, int]] = []
        for logical_page in logical_pages:
            source_page = row[logical_page]
            if self._page_refcounts[source_page] <= 1:
                continue
            if not self._free_physical_pages:
                raise RuntimeError(
                    "Qwen3.6 KV copy-on-write exhausted the fixed page pool; "
                    "a shared prefix still has no displaced page to privatize"
                )
            target_page = min(self._free_physical_pages)
            self._free_physical_pages.remove(target_page)
            replacements.append((logical_page, source_page, target_page))
            row[logical_page] = target_page
        if not replacements:
            return
        for _logical_page, source_page, target_page in replacements:
            for k_pool, v_pool in zip(self.k_pools, self.v_pools, strict=True):
                if k_pool is None:
                    continue
                assert v_pool is not None
                k_pool[target_page].copy_(k_pool[source_page])
                v_pool[target_page].copy_(v_pool[source_page])
        self.set_page_table_row(slot, row)

    def copy_prefix_kv(self, source_slot: int, target_slot: int, kv_len: int) -> None:
        """Copy a reusable paged-KV prefix between two real slots.

        The page-table indirection introduced for prefix caching means the
        source and destination need not own contiguous physical pages.  Copy
        whole logical pages so a block-aligned GDN checkpoint that falls in
        the middle of a 128-token attention page remains valid; the target's
        subsequent suffix prefill overwrites the unused tail before it can be
        read.

        This is intentionally a copy, not page-table aliasing.  The current
        fixed slot allocator has no page refcount/LRU ownership yet, and
        aliasing here would let a target request overwrite a retained source
        prefix.  Keeping the operation at this boundary gives the future
        BlockPool allocator one place to replace with retain/release logic.
        """
        if not 0 <= source_slot < self.num_slots:
            raise ValueError(f"source slot {source_slot} is not a live slot")
        if not 0 <= target_slot < self.num_slots:
            raise ValueError(f"target slot {target_slot} is not a live slot")
        if source_slot == target_slot or kv_len == 0:
            return
        if not 0 < kv_len <= self.max_seq_len:
            raise ValueError(f"prefix length {kv_len} is outside pool capacity {self.max_seq_len}")

        pages = (kv_len + self.page_size - 1) // self.page_size
        if self.dynamic_arena:
            # Copy, not alias (legacy semantics preserved): the target must
            # own its physical pages, so allocate fresh bundles for the
            # copied range first.
            for logical_page in range(pages):
                if self._page_table_host[target_slot][logical_page] == 0:
                    fresh = self._arena.allocate(1, owner=f"slot-{target_slot}")[0]
                    self._page_table_host[target_slot][logical_page] = fresh
                    self._slot_bundles[target_slot].add(fresh)
            self.set_page_table_row(target_slot, self._page_table_host[target_slot])
        source_pages = self._global_page_table[source_slot, :pages].to(dtype=torch.long)
        target_pages = self._global_page_table[target_slot, :pages].to(dtype=torch.long)
        for k_pool, v_pool in zip(self.k_pools, self.v_pools, strict=True):
            if k_pool is None:
                continue
            assert v_pool is not None
            # Advanced-index assignment is deliberately used instead of a
            # view slice: page tables may be non-contiguous once a dynamic
            # BlockPool starts recycling physical pages.
            k_pool[target_pages] = k_pool[source_pages]
            v_pool[target_pages] = v_pool[source_pages]

    def share_prefix_kv(self, source_slot: int, target_slot: int, kv_len: int) -> None:
        """Alias a retained prefix; later writes detach through COW.

        This replaces the old cross-slot D2D copy on the target backbone KV.
        It aliases whole 128-token attention pages, including a final partial
        page; :meth:`prepare_kv_writes` privatizes that final page before a
        suffix write.  GDN and MTP state retain their independent explicit
        restore paths because neither has page-composable recurrent state.
        """
        if not 0 <= source_slot < self.num_slots:
            raise ValueError(f"source slot {source_slot} is not a live slot")
        if not 0 <= target_slot < self.num_slots:
            raise ValueError(f"target slot {target_slot} is not a live slot")
        if source_slot == target_slot or kv_len == 0:
            return
        if not 0 < kv_len <= self.max_seq_len:
            raise ValueError(f"prefix length {kv_len} is outside pool capacity {self.max_seq_len}")
        shared_pages = (kv_len + self.page_size - 1) // self.page_size
        target_row = list(self._page_table_host[target_slot])
        if self.dynamic_arena:
            # Alias the source's prefix bundles and register the target's
            # references with the arena (vLLM ``touch``): each aliased entry
            # raises the bundle's refcnt, so the source stays alive while
            # either sharer holds it (INV9). ``prepare_kv_writes`` COW-detaches
            # before the target's first suffix write.
            for logical_page in range(shared_pages):
                src = self._page_table_host[source_slot][logical_page]
                if src == 0:
                    raise RuntimeError(
                        f"source slot {source_slot} has no bundle at page {logical_page} "
                        "to share; prefill must have written the prefix first"
                    )
                target_row[logical_page] = src
            self._arena.incref(
                self._page_table_host[source_slot][:shared_pages], owner=f"slot-{target_slot}"
            )
            self._slot_bundles[target_slot].update(
                self._page_table_host[source_slot][:shared_pages]
            )
        else:
            target_row[:shared_pages] = self._page_table_host[source_slot][:shared_pages]
        self.set_page_table_row(target_slot, target_row)

    def copy_prefix_to_scratch(
        self, source_slot: int, kv_len: int, *, scratch_pages: list[int] | None = None
    ) -> None:
        """Snapshot a reusable prefix into the pre-allocated scratch row.

        CUDA-graph capture needs the scratch row only while warming/capturing;
        production replay addresses real rows directly.  It is therefore the
        one bounded KV arena that can retain a prefix after its originating
        request is assigned a new prompt, without reserving extra GPU memory.
        The backend owns the corresponding token identity and GDN checkpoint;
        this method deliberately owns bytes only.
        """
        if not 0 <= source_slot < self.num_slots:
            raise ValueError(f"source slot {source_slot} is not a live slot")
        if not 0 < kv_len <= self.max_seq_len:
            raise ValueError(f"prefix length {kv_len} is outside pool capacity {self.max_seq_len}")
        pages = (kv_len + self.page_size - 1) // self.page_size
        if self.dynamic_arena:
            # Phase 2: the scratch row's prefix pages are arena bundles like
            # any other; allocate fresh ones so the snapshot owns its bytes
            # (a copy, not an alias -- the originating slot may be reused).
            scratch_pages = self._arena.allocate(pages, owner="scratch")
            self._slot_bundles[self.scratch_row].update(scratch_pages)
            row = list(self._page_table_host[self.scratch_row])
            row[:pages] = scratch_pages
            self._page_table_host[self.scratch_row] = row
        elif scratch_pages is None:
            scratch_pages = self._page_table_host[self.scratch_row][:pages]
        if len(scratch_pages) != pages or len(set(scratch_pages)) != pages:
            raise ValueError("scratch prefix needs one distinct physical page per logical page")
        if not self.dynamic_arena:
            scratch_lo = self.scratch_row * self.pages_per_slot
            scratch_hi = scratch_lo + self.pages_per_slot
            if any(page < scratch_lo or page >= scratch_hi for page in scratch_pages):
                raise ValueError("scratch prefix pages must belong to the scratch arena")
        source_pages = self._global_page_table[source_slot, :pages].to(dtype=torch.long)
        scratch_page_tensor = torch.tensor(scratch_pages, dtype=torch.long, device=self.device)
        for k_pool, v_pool in zip(self.k_pools, self.v_pools, strict=True):
            if k_pool is None:
                continue
            assert v_pool is not None
            k_pool[scratch_page_tensor] = k_pool[source_pages]
            v_pool[scratch_page_tensor] = v_pool[source_pages]
        if self.dynamic_arena:
            self.set_page_table_row(self.scratch_row, self._page_table_host[self.scratch_row])

    def share_scratch_prefix(
        self, target_slot: int, kv_len: int, *, scratch_pages: list[int] | None = None
    ) -> None:
        """Alias a scratch-arena prefix into a real slot.

        ``prepare_kv_writes`` already implements the required copy-on-write
        detach before the first suffix token is written, so a cached partial
        128-token page is safe even when the GDN checkpoint is at its 64-token
        midpoint.
        """
        if not 0 <= target_slot < self.num_slots:
            raise ValueError(f"target slot {target_slot} is not a live slot")
        if not 0 < kv_len <= self.max_seq_len:
            raise ValueError(f"prefix length {kv_len} is outside pool capacity {self.max_seq_len}")
        shared_pages = (kv_len + self.page_size - 1) // self.page_size
        if scratch_pages is None:
            scratch_pages = self._page_table_host[self.scratch_row][:shared_pages]
        if len(scratch_pages) != shared_pages or len(set(scratch_pages)) != shared_pages:
            raise ValueError("scratch prefix needs one distinct physical page per logical page")
        if not self.dynamic_arena:
            scratch_lo = self.scratch_row * self.pages_per_slot
            scratch_hi = scratch_lo + self.pages_per_slot
            if any(page < scratch_lo or page >= scratch_hi for page in scratch_pages):
                raise ValueError("scratch prefix pages must belong to the scratch arena")
        target_row = list(self._page_table_host[target_slot])
        target_row[:shared_pages] = scratch_pages
        if self.dynamic_arena:
            self._arena.incref(scratch_pages, owner=f"slot-{target_slot}")
            self._slot_bundles[target_slot].update(scratch_pages)
        self.set_page_table_row(target_slot, target_row)

    def detach_scratch_aliases(self, slot: int, kv_len: int, *, scratch_pages: set[int]) -> bool:
        """Copy an idle slot's aliased scratch pages back into its own row.

        A restored (full-prompt) request keeps its page-table alias on the
        persistent entry's scratch pages after ``reset_slot`` -- that alias
        is what makes same-slot reuse cheap, but it also pins the entry's
        refcount above 1 forever, so the persistent arena can never evict
        that entry to make room for a larger one (measured 2026-08-05: the
        five-context grid stored 4K..128K, then the 250K store silently
        failed because every candidate still had an idle-slot alias).
        Before evicting such an entry, copy its pages back into the idle
        slot's own row (the same per-page COW discipline
        :meth:`prepare_kv_writes` uses) and remap; the KV bytes stay valid
        for that slot's retained prefix, and the scratch pages return to
        refcount 1 where the eviction path can reclaim them.

        A LIVE slot is also allowed, but only its *committed* range is
        detached: pages below ``slot_kv_len`` were already written and are
        read-only for the rest of the sequence, so privatizing them changes
        nothing observable while releasing the scratch pin.  Pages at or
        beyond ``slot_kv_len`` are pages the slot will still write and are
        left aliased for ``prepare_kv_writes`` to COW-detach on the first
        write.  This is what lets ``_evict_persistent_until`` make room at
        a prefill-commit store even when the just-prefilled slot restored a
        shorter cached prefix (measured 2026-08-05: the 250K prefill slot
        still aliased the 128K entry, the detach loop skipped it as live,
        and the 250K store failed with the entry pinned at refcount 2).
        Returns True when at least one page was remapped.
        """
        if not 0 <= slot < self.num_slots:
            raise ValueError(f"slot {slot} is not a live slot")
        if not 0 < kv_len <= self.max_seq_len:
            raise ValueError(f"prefix length {kv_len} is outside pool capacity {self.max_seq_len}")
        if self.dynamic_arena:
            # Phase 2: scratch pages are arena bundles; detach by COW-cloning
            # each aliased committed page into a fresh private bundle and
            # releasing the scratch reference (same discipline as
            # prepare_kv_writes).
            logical_pages = range((kv_len + self.page_size - 1) // self.page_size)
            committed_pages = (self.slot_kv_len[slot] + self.page_size - 1) // self.page_size
            row = list(self._page_table_host[slot])
            replacements: list[tuple[int, int, int]] = []
            for logical_page in logical_pages:
                if committed_pages and logical_page >= committed_pages:
                    break
                source_page = row[logical_page]
                if source_page not in scratch_pages:
                    continue
                if self._arena.bundles[source_page].ref_cnt <= 1:
                    continue
                target_page = self._arena.ensure_writable(source_page)
                self._arena.decref([source_page], owner=f"slot-{slot}")
                self._slot_bundles[slot].discard(source_page)
                replacements.append((logical_page, source_page, target_page))
                row[logical_page] = target_page
                self._slot_bundles[slot].add(target_page)
            if not replacements:
                return False
            for _logical_page, source_page, target_page in replacements:
                for k_pool, v_pool in zip(self.k_pools, self.v_pools, strict=True):
                    if k_pool is None:
                        continue
                    assert v_pool is not None
                    k_pool[target_page].copy_(k_pool[source_page])
                    v_pool[target_page].copy_(v_pool[source_page])
            self._arena.drain_pending_cow()
            self.set_page_table_row(slot, row)
            return True
        scratch_lo = self.scratch_row * self.pages_per_slot
        scratch_hi = scratch_lo + self.pages_per_slot
        if any(not 0 <= page < self.pages_per_slot for page in scratch_pages):
            raise ValueError("scratch pages must be logical pages of the scratch arena")
        logical_pages = range((kv_len + self.page_size - 1) // self.page_size)
        committed_pages = (self.slot_kv_len[slot] + self.page_size - 1) // self.page_size
        row = list(self._page_table_host[slot])
        replacements: list[tuple[int, int, int]] = []
        for logical_page in logical_pages:
            if committed_pages and logical_page >= committed_pages:
                break
            source_page = row[logical_page]
            if not scratch_lo <= source_page < scratch_hi:
                continue
            if (source_page - scratch_lo) not in scratch_pages:
                continue
            if self._page_refcounts[source_page] <= 1:
                continue
            if not self._free_physical_pages:
                raise RuntimeError(
                    "Qwen3.6 scratch detach exhausted the fixed page pool; "
                    "no displaced page is available to privatize an alias"
                )
            target_page = min(self._free_physical_pages)
            self._free_physical_pages.remove(target_page)
            replacements.append((logical_page, source_page, target_page))
            row[logical_page] = target_page
        if not replacements:
            return False
        for _logical_page, source_page, target_page in replacements:
            for k_pool, v_pool in zip(self.k_pools, self.v_pools, strict=True):
                if k_pool is None:
                    continue
                assert v_pool is not None
                k_pool[target_page].copy_(k_pool[source_page])
                v_pool[target_page].copy_(v_pool[source_page])
        self.set_page_table_row(slot, row)
        return True

    def enable_mtp_gdn_rows(self, num_speculative_tokens: int) -> None:
        """Extend each GDN pool with MTP's ``K`` candidate rows.

        This is intentionally a one-time, pre-capture operation.  Column
        zero remains the ordinary per-slot row, while ``spec_row`` maps the
        candidate columns into the appended range.  Replacing a pool after a
        decode CUDA Graph has captured its address would make replay use stale
        state, so callers must configure MTP before graph capture.
        """
        if num_speculative_tokens < 1:
            raise ValueError("MTP requires at least one speculative token")
        if self._mtp_num_speculative_tokens is not None:
            if self._mtp_num_speculative_tokens != num_speculative_tokens:
                raise ValueError(
                    "MTP GDN rows already configured for "
                    f"K={self._mtp_num_speculative_tokens}, not K={num_speculative_tokens}"
                )
            return
        if any(self.slot_kv_len):
            raise RuntimeError("enable MTP before any Qwen3.6 slot has processed tokens")

        total_gdn_rows = self._num_rows * (num_speculative_tokens + 1)
        columns_by_layer: dict[int, list[list[GdnLayerState]]] = {}
        for layer in self.model.model.layers:
            if layer.layer_type != "linear_attention":
                continue
            layer_idx = layer.layer_idx
            old_conv = self.conv_pools[layer_idx]
            old_recurrent = self.recurrent_pools[layer_idx]
            assert old_conv is not None and old_recurrent is not None
            gdn = layer.linear_attn
            conv = _mark_static(
                torch.zeros(
                    total_gdn_rows,
                    gdn.conv_dim,
                    gdn.conv_kernel_size,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
            recurrent = _mark_static(
                torch.zeros(
                    total_gdn_rows,
                    gdn.num_v_heads,
                    gdn.head_k_dim,
                    gdn.head_v_dim,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
            conv[: self._num_rows].copy_(old_conv)
            recurrent[: self._num_rows].copy_(old_recurrent)
            self.conv_pools[layer_idx] = conv
            self.recurrent_pools[layer_idx] = recurrent
            columns_by_layer[layer_idx] = [
                [
                    GdnLayerState(
                        conv_state=conv[row : row + 1],
                        recurrent_state=recurrent[row : row + 1],
                        has_previous_state=False,
                    )
                    for row in (
                        physical_slot,
                        *range(
                            self._num_rows + physical_slot * num_speculative_tokens,
                            self._num_rows + (physical_slot + 1) * num_speculative_tokens,
                        ),
                    )
                ]
                for physical_slot in range(self._num_rows)
            ]

        self._mtp_gdn_columns = columns_by_layer
        self._mtp_num_speculative_tokens = num_speculative_tokens
        for slot in range(self._num_rows):
            self.activate_mtp_gdn_state(slot, 0)

    @property
    def mtp_num_speculative_tokens(self) -> int | None:
        return self._mtp_num_speculative_tokens

    def mtp_gdn_columns(self, layer_idx: int, slot: int) -> list[GdnLayerState]:
        if self._mtp_gdn_columns is None:
            raise RuntimeError("MTP GDN rows have not been configured")
        return self._mtp_gdn_columns[layer_idx][slot]

    def activate_mtp_gdn_state(self, slot: int, col: int) -> None:
        """Make a slot's live GDN views point at its selected MTP column."""
        if self._mtp_gdn_columns is None:
            raise RuntimeError("MTP GDN rows have not been configured")
        if col < 0 or col > self._mtp_num_speculative_tokens:
            raise ValueError(f"MTP GDN column {col} is out of range")
        state = self._slot_states[slot]
        for layer_idx, per_slot in self._mtp_gdn_columns.items():
            state.gdn_states[layer_idx] = per_slot[slot][col]

    def reset_slot(self, slot: int) -> None:
        """Return ``slot`` to the fresh state a new sequence needs.

        Zeroes the recurrent state (the B0-5 operational requirement --
        see the module docstring's point 2) and clears the length
        bookkeeping. Does **not** zero KV bytes: past ``kv_len`` they are
        never read, and below it they are the prefix cache.

        Dynamic-arena mode additionally returns every bundle this slot
        referenced to the arena (``decref``), so a slot reset/reuse leaks no
        old-epoch page ownership (plan §7 invariant 7). A bundle whose hash
        was published by :meth:`publish_committed_blocks` stays reachable at
        ``ref_cnt == 0`` (CACHED_REF0, LRU tail) -- that is the Phase 3
        prefix cache: the KV survives the slot, owned by the arena, evicted
        only on real pressure. The page-table row is reset to all-null so
        the next sequence starts from empty.
        """
        state = self._slot_states[slot]
        for gdn in state.gdn_states:
            if gdn is None:
                continue
            gdn.conv_state.zero_()
            gdn.recurrent_state.zero_()
            gdn.has_previous_state = False
        for cache in state.attn_caches:
            if cache is not None:
                cache.seq_len = 0
        state.num_tokens_seen = 0
        self.slot_kv_len[slot] = 0
        self.slot_committed_tokens[slot] = []
        if self.dynamic_arena:
            owned = self._slot_bundles[slot]
            if owned:
                self._arena.decref(sorted(owned), owner=f"slot-{slot}")
                owned.clear()
            if any(self._page_table_host[slot]):
                self.set_page_table_row(slot, [0] * self.pages_per_slot)

    # -- Phase 3: prefix cache as arena-owned cached blocks ----------------

    def publish_committed_blocks(
        self, slot: int, kv_len: int, keys: Sequence[BlockKey], block_size: int
    ) -> int:
        """Publish hashes for ``slot``'s committed blocks to the arena.

        Phase 3 (``.omx/plans/qwen38-dynamic-context-vllm-plan.md`` §6.4):
        the prefix KV survives a slot reset as an arena CACHED_REF0 block
        instead of a fixed scratch row. Call BEFORE ``reset_slot`` (while
        the slot still holds its bundles at ``ref_cnt > 0``) so the hashes
        are published against live ownership; the reset's ``decref`` then
        parks each hashed bundle at the LRU tail where prefix lookup can
        revive it (vLLM ``cache_full_blocks`` before ``free_blocks``).

        ``keys`` are the request's CHAINED BLOCK hashes at ``block_size``
        boundaries (vLLM ``hash_block_size``); one physical page holds
        ``page_size // block_size`` of them, all pointing at the same
        bundle (vLLM fine-grained lookup). Only blocks fully covered by
        committed tokens are published -- a final partial block stays
        private (plan §5.1C). Returns the number of blocks published.
        """
        if not self.dynamic_arena:
            return 0
        if not 0 <= slot < self.num_slots:
            raise ValueError(f"slot {slot} is not a live slot")
        if kv_len < 0 or kv_len > self.max_seq_len:
            raise ValueError(f"prefix length {kv_len} is outside pool capacity")
        if block_size <= 0 or self.page_size % block_size != 0:
            raise ValueError("block_size must divide the page size")
        row = self._page_table_host[slot]
        published = 0
        for key in keys:
            end = key.num_tokens
            if end > kv_len:
                break  # not committed yet
            if end % block_size != 0:
                continue  # not a block boundary
            logical_page = (end - 1) // self.page_size
            bundle_id = row[logical_page]
            if bundle_id == 0:
                raise RuntimeError(
                    f"slot {slot} has no bundle for block ending at {end}; "
                    "prefill must have written it first"
                )
            self._arena.publish_full_block(bundle_id, key)
            published += 1
        return published

    def restore_prefix_from_arena(
        self, slot: int, kv_len: int, keys: Sequence[BlockKey]
    ) -> tuple[int, list[int]]:
        """Revive the deepest arena-cached prefix into ``slot``.

        Phase 3 prefix restore: looks up the request's chained hashes in the
        arena and, on a hit, ``incref``s (vLLM ``touch``) the cached bundles
        so they are LIVE_SHARED again, then points ``slot``'s page-table row
        at them. Returns ``(published_tokens, bundle_ids)``; the caller must
        still restore the recurrent checkpoint at the matching boundary
        (plan §5.1D invariant 9 -- this method owns KV blocks only).
        """
        if not self.dynamic_arena:
            return 0, []
        if not 0 <= slot < self.num_slots:
            raise ValueError(f"slot {slot} is not a live slot")
        hit = self._arena.lookup_longest_prefix(keys)
        if not hit.hit or not hit.bundle_ids:
            return 0, []
        bundle_ids = list(hit.bundle_ids)
        # Multiple hash blocks (block_size boundaries) share one physical
        # page, so the hit's bundle list is denser than the page row. The
        # bundle is the physical ownership unit: dedup before incref, or a
        # two-block page would be referenced twice and never reach ref_cnt 0
        # on release (vLLM touches one KVCacheBlock per block; here one
        # bundle IS the page, one reference).
        unique_bundles = list(dict.fromkeys(bundle_ids))
        self._arena.incref(unique_bundles, owner=f"slot-{slot}")
        self._slot_bundles[slot].update(unique_bundles)
        row = list(self._page_table_host[slot])
        # Write each page's bundle once: the page covering hash-block ``i``
        # is ``(keys[i].num_tokens - 1) // page_size``.
        for i, bundle_id in enumerate(bundle_ids):
            logical_page = (keys[i].num_tokens - 1) // self.page_size
            row[logical_page] = bundle_id
        self.set_page_table_row(slot, row)
        return hit.effective_tokens, unique_bundles

    def reset_all(self) -> None:
        for row in range(self._num_rows):
            self.reset_slot(row)

    def rewind_slot(self, slot: int, kv_len: int) -> None:
        """Re-point ``slot``'s bookkeeping at an already-computed prefix.

        Used by the prefix-cache path: the KV bytes for ``[0, kv_len)`` are
        already resident (same slot, same tokens, same positions), so the
        only thing that has to change is where the next write goes.

        The recurrent state is **not** rewound here and cannot be -- there
        is no such thing as "the GDN state at position ``kv_len``" unless a
        checkpoint was taken at exactly that boundary. Handing this method
        a ``kv_len`` for which no recurrent checkpoint was restored is
        precisely the INV-A3-2 violation ``PrefixHit.effective`` exists to
        prevent, which is why the backend only ever calls this with
        ``PrefixHit.effective`` and never with ``kv_hit``.
        """
        if kv_len < 0 or kv_len > self.max_seq_len:
            raise ValueError(f"kv_len={kv_len} out of range for slot {slot}")
        state = self._slot_states[slot]
        for cache in state.attn_caches:
            if cache is not None:
                cache.seq_len = kv_len
        state.num_tokens_seen = kv_len
        self.slot_kv_len[slot] = kv_len

    # -- recurrent-state checkpoints (the second cache family) -------------

    def capture_recurrent_state(self, slot: int) -> list[torch.Tensor]:
        """Clone ``slot``'s recurrent state -- one checkpoint.

        Returns a flat list of tensors (conv, recurrent, conv, recurrent, ...
        in layer order). Copied, not viewed: the point of a checkpoint is to
        survive the slot being reused.  The destination buffers are
        pre-allocated per slot and reused, so the copy is a single
        ``torch._foreach_copy_`` launch instead of 96 small ``clone()``
        allocations on the hot path.
        """
        out: list[torch.Tensor] = []
        state = self._slot_states[slot]
        live: list[torch.Tensor] = []
        for gdn in state.gdn_states:
            if gdn is None:
                continue
            live.append(gdn.conv_state)
            live.append(gdn.recurrent_state)
        dest = self._checkpoint_dest.get(slot)
        if (
            dest is None
            or len(dest) != len(live)
            or any(
                a.shape != b.shape or a.dtype != b.dtype for a, b in zip(dest, live, strict=True)
            )
        ):
            dest = [t.clone() for t in live]
            self._checkpoint_dest[slot] = dest
        torch._foreach_copy_(dest, live)
        out = dest
        return out

    def restore_recurrent_state(self, slot: int, checkpoint: list[torch.Tensor]) -> None:
        """Copy a checkpoint back into ``slot``'s live buffers.

        ``torch._foreach_copy_`` for the same reason
        ``oracle/qwen36_vllm/gdn_state.py:213-219`` uses it: 96 separate
        small copies (48 layers x 2 tensors) is a launch-bound operation,
        and this is on the admission path. Copies -- never aliases -- so
        two slots restoring from the same checkpoint in one round cannot
        interfere (``docs/a3-cache-coordinator-design.md`` §6 pitfall 4's
        "structural exemption").
        """
        state = self._slot_states[slot]
        live: list[torch.Tensor] = []
        for gdn in state.gdn_states:
            if gdn is None:
                continue
            live.append(gdn.conv_state)
            live.append(gdn.recurrent_state)
        if len(live) != len(checkpoint):
            raise ValueError(
                f"checkpoint has {len(checkpoint)} tensors, slot {slot} needs {len(live)}"
            )
        torch._foreach_copy_(live, checkpoint)
        for gdn in state.gdn_states:
            if gdn is not None:
                gdn.has_previous_state = True

    def recurrent_checkpoint_nbytes(self) -> int:
        return self.geometry.recurrent_bytes_per_slot

    # -- batched decode ----------------------------------------------------

    def build_decode_batch(
        self, slots: list[int], token_ids: list[int]
    ) -> tuple[Qwen36DecodeBatch, int]:
        """Fill the persistent batch buffers for one decode round.

        Returns the batch descriptor plus the batch size actually used.
        Every tensor in the descriptor is a **narrowed view of a
        pre-allocated buffer**, so the addresses a CUDA Graph capture
        bakes in stay valid for every later replay at the same batch size.

        Advancing ``slot_kv_len`` happens here, on the host, *before* the
        forward: ``cache_seqlens`` must already include the token this step
        writes (sparkinfer's decode contract, same as B1's single-sequence
        path, which calls ``cache.append`` before the kernel).
        """
        b = len(slots)
        if b == 0:
            raise ValueError("build_decode_batch requires at least one slot")
        if b > self._num_rows:
            raise ValueError(f"batch of {b} exceeds pool capacity {self._num_rows}")

        write_rows: list[int] = []
        seqlens: list[int] = []
        for slot, token in zip(slots, token_ids):
            past = self.slot_kv_len[slot]
            if past >= self.max_seq_len:
                raise RuntimeError(
                    f"slot {slot} is at capacity ({self.max_seq_len} tokens); "
                    "the scheduler must retire it before decoding further"
                )
            self.prepare_kv_writes(slot, past, 1)
            write_rows.append(self.write_index(slot, past))
            seqlens.append(past + 1)
            self.slot_kv_len[slot] = past + 1
            self.slot_committed_tokens[slot].append(int(token))
            state = self._slot_states[slot]
            state.num_tokens_seen = past + 1
            for cache in state.attn_caches:
                if cache is not None:
                    cache.seq_len = past + 1

        attn = self.attention_driver(b)
        self._np_batch_input_ids[:b, 0] = token_ids
        self._np_batch_positions[:b] = [s - 1 for s in seqlens]
        self._np_batch_cache_seqlens[:b] = seqlens
        self._np_batch_write_index[:b] = write_rows
        self._np_batch_slot_index[:b] = slots
        self._batch_input_ids[:b].copy_(self._batch_input_ids_host[:b], non_blocking=True)
        self._batch_positions[:b].copy_(self._batch_positions_host[:b], non_blocking=True)
        attn.cache_seqlens.copy_(self._batch_cache_seqlens_host[:b], non_blocking=True)
        # Graph-mode driver: re-chunk split-KV for the live lengths (the
        # capture-static alternative freezes worst-case chunking and runs
        # ~3x slower at mid contexts -- see Qwen36DecodeGraphAttention.
        # update_replay_metadata). Eager drivers do not expose it.
        update = getattr(attn, "update_replay_metadata", None)
        if update is not None:
            update()
        self._batch_write_index[:b].copy_(self._batch_write_index_host[:b], non_blocking=True)
        self._batch_slot_index[:b].copy_(self._batch_slot_index_host[:b], non_blocking=True)
        # ``out=`` fills the graph-owned page table without the temporary
        # result allocation that ``index_select(...)`` creates.  The source
        # table itself is a fixed slot-id mapping, while the staged slot ids
        # remain replay inputs, so this also preserves arbitrary batch order.
        torch.index_select(
            self._global_page_table,
            0,
            self._batch_slot_index[:b],
            out=attn.page_table,
        )

        batch = Qwen36DecodeBatch(
            input_ids=self._batch_input_ids[:b],
            positions=self._batch_positions[:b],
            write_index=self._batch_write_index[:b],
            slot_index=self._batch_slot_index[:b],
            attn=attn,
            k_pools=self.k_pools,
            v_pools=self.v_pools,
            conv_pools=self.conv_pools,
            recurrent_pools=self.recurrent_pools,
            attn_outputs=[None if out is None else out[:b] for out in self.attn_outputs],
        )
        return batch, b

    def build_prefill_batch(
        self, slots: list[int], token_ids_per_slot: list[list[int]]
    ) -> Qwen36PrefillBatch:
        """Build a safe uniform multi-slot extend descriptor.

        This is intentionally narrower than the historical vLLM metadata
        builder: it accepts only equal Q lengths and identical entering GDN
        regimes.  The backend selects it only for that shape; other requests
        keep using the B1-compatible serial path.  Refusing a mixed state is
        essential because GDN's ``has_previous_state`` is part of the
        recurrence contract, not padding metadata.
        """
        b = len(slots)
        if b < 1 or b != len(token_ids_per_slot):
            raise ValueError("slots and token_ids_per_slot must be non-empty and equal length")
        if len(set(slots)) != b:
            raise ValueError("prefill batch slots must be distinct")
        lengths = {len(tokens) for tokens in token_ids_per_slot}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) < 1:
            raise ValueError("batched prefill requires one positive, uniform token count")
        q = next(iter(lengths))
        if any(slot < 0 or slot >= self.num_slots for slot in slots):
            raise ValueError("prefill batch contains an invalid real slot")

        prior_lens = [self.slot_kv_len[slot] for slot in slots]
        if len(set(prior_lens)) != 1:
            raise ValueError("batched prefill requires equal prior KV lengths")
        past = prior_lens[0]
        if past + q > self.max_seq_len:
            raise RuntimeError(
                f"batched prefill would reach {past + q} tokens, capacity is {self.max_seq_len}"
            )
        has_previous_values: set[bool] = set()
        for slot in slots:
            state = self._slot_states[slot]
            if state.num_tokens_seen != past:
                raise RuntimeError(
                    f"slot {slot} state length {state.num_tokens_seen} != pooled KV length {past}"
                )
            for layer_idx, gdn in enumerate(state.gdn_states):
                if gdn is None:
                    continue
                # A post-verify MTP slot can point at a candidate row instead
                # of its ordinary row.  It must not be gathered using the raw
                # slot id; the serial path remains exact for that uncommon
                # admission shape until generalized row ids are explicit.
                pool = self.conv_pools[layer_idx]
                assert pool is not None
                if gdn.conv_state.data_ptr() != pool[slot : slot + 1].data_ptr():
                    raise ValueError("batched prefill requires ordinary live GDN rows")
                has_previous_values.add(gdn.has_previous_state)
        if len(has_previous_values) > 1:
            raise ValueError("batched prefill requires matching GDN state regimes")
        has_previous_state = next(iter(has_previous_values), False)

        for slot in slots:
            self.prepare_kv_writes(slot, past, q)
        attn = self.prefill_attention_driver(b, q)
        slot_index = torch.tensor(slots, dtype=torch.long, device=self.device)
        input_ids = torch.tensor(token_ids_per_slot, dtype=torch.long, device=self.device)
        positions = (
            torch.arange(past, past + q, dtype=torch.long, device=self.device)
            .unsqueeze(0)
            .expand(b, q)
            .reshape(-1)
        )
        offsets = torch.arange(q, dtype=torch.long, device=self.device).unsqueeze(0)
        # Page ids are scheduler metadata, so construct the physical rows
        # from the host mirror directly.  Deriving ``logical_pages`` on the
        # device and calling ``.tolist()`` would synchronize every BxQ
        # prefill merely to reconstruct this same tiny list on the host.
        global_pages = torch.tensor(
            [
                [
                    self._page_table_host[slot][(past + offset) // self.page_size]
                    for offset in range(q)
                ]
                for slot in slots
            ],
            dtype=torch.long,
            device=self.device,
        )
        write_index = (global_pages * self.page_size + (past + offsets) % self.page_size).reshape(
            -1
        )
        attn.cache_seqlens.fill_(past + q)
        torch.index_select(self._global_page_table, 0, slot_index, out=attn.page_table)
        # All full-attention layers run serially and share this one scratch
        # buffer.  See Qwen36BatchedExtendAttention.output for the memory
        # accounting; allocating one output per layer would retain 16 large
        # BxQ tensors until the whole model forward returns.
        outputs: list[torch.Tensor | None] = [
            None if layer.layer_type == "linear_attention" else attn.output
            for layer in self.model.model.layers
        ]
        return Qwen36PrefillBatch(
            input_ids=input_ids,
            positions=positions,
            write_index=write_index,
            slot_index=slot_index,
            attn=attn,
            k_pools=self.k_pools,
            v_pools=self.v_pools,
            conv_pools=self.conv_pools,
            recurrent_pools=self.recurrent_pools,
            attn_outputs=outputs,
            has_previous_state=has_previous_state,
        )

    def commit_prefill_batch(self, slots: list[int], token_ids_per_slot: list[list[int]]) -> None:
        """Commit host-side lengths only after the batched forward succeeds."""
        if len(slots) != len(token_ids_per_slot):
            raise ValueError("slots and token_ids_per_slot must have equal length")
        for slot, tokens in zip(slots, token_ids_per_slot):
            new_len = self.slot_kv_len[slot] + len(tokens)
            state = self._slot_states[slot]
            self.slot_kv_len[slot] = new_len
            state.num_tokens_seen = new_len
            for gdn in state.gdn_states:
                if gdn is not None:
                    gdn.has_previous_state = True
            for cache in state.attn_caches:
                if cache is not None:
                    cache.seq_len = new_len

    def prefill_attention_driver(
        self, batch: int, tokens_per_slot: int
    ) -> Qwen36BatchedExtendAttention:
        driver = self._prefill_attn
        if driver is None or driver.batch != batch or driver.tokens_per_slot != tokens_per_slot:
            self._prefill_attn = Qwen36BatchedExtendAttention(
                batch=batch, tokens_per_slot=tokens_per_slot, **self._driver_kwargs()
            )
        return self._prefill_attn

    def attention_driver(self, batch: int):
        """The shared attention driver for a decode step of size ``batch``.

        A captured graph's driver wins once it exists: the graph was
        captured against *its* metadata buffers, so replaying it while the
        eager driver's buffers are the ones being written would replay last
        capture's schedule forever -- correct-looking output computed from
        the wrong context lengths.
        """
        graph = self.graph_attn.get(batch)
        if graph is not None:
            return graph
        existing = self.decode_attn.get(batch)
        if existing is None:
            existing = Qwen36BatchedDecodeAttention(batch=batch, **self._driver_kwargs())
            self.decode_attn[batch] = existing
        return existing

    def build_graph_attention_driver(self, batch: int) -> Qwen36DecodeGraphAttention:
        """Build (but do not install) a graph-replay driver for ``batch``."""
        return Qwen36DecodeGraphAttention(
            batch=batch, max_seq_len=self.max_seq_len, **self._driver_kwargs()
        )

    def _driver_kwargs(self) -> dict:
        if self._attn_geometry is None:
            raise RuntimeError("this model has no full-attention layers to drive")
        return {
            **self._attn_geometry,
            "page_size": self.page_size,
            "pages_per_slot": self.pages_per_slot,
            "num_cache_pages": self._num_rows * self.pages_per_slot,
            "dtype": self.dtype,
            # The KV pools' own dtype (BF16 unless FP8 KV was enabled --
            # see __init__'s _kv_dtype uniformity check), NOT necessarily
            # self.dtype (the compute dtype) now that the two can diverge.
            "kv_dtype": self._kv_dtype,
            "device": self.device,
        }

    def ensure_decode_workspaces(self, max_batch: int) -> None:
        """Pre-build the eager decode driver for every batch size in use.

        Sized for the *whole* page pool, not one slot's worth: batched
        decode addresses pages globally, so ``num_cache_pages`` must cover
        every slot's range or sparkinfer's own capacity check rejects the
        page table.
        """
        for batch in range(1, max_batch + 1):
            self.attention_driver(batch)
