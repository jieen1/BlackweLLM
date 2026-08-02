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

from dataclasses import dataclass

import torch

from runtime.model.qwen36_model import (
    _PAGED_ATTENTION_PAGE_SIZE,
    GdnLayerState,
    Qwen36BatchedDecodeAttention,
    Qwen36DecodeBatch,
    Qwen36DecodeGraphAttention,
    Qwen36ForCausalLMSelfBuilt,
    Qwen36GenerationState,
    Qwen36PagedAttentionCache,
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
        self._attn_geometry: dict[str, int] | None = None

        recurrent_bytes = 0
        kv_bytes = 0
        num_recurrent = 0
        num_paged = 0
        total_pages = self._num_rows * self.pages_per_slot

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
                shape = (total_pages, self.page_size, attn.num_kv_heads, attn.head_dim)
                k = torch.zeros(shape, device=self.device, dtype=dtype)
                v = torch.zeros(shape, device=self.device, dtype=dtype)
                self.k_pools[i] = _mark_static(k)
                self.v_pools[i] = _mark_static(v)
                kv_bytes += 2 * self.pages_per_slot * self.page_size * (
                    attn.num_kv_heads * attn.head_dim * k.element_size()
                )
                self.attn_outputs[i] = _mark_static(
                    torch.zeros(
                        self._num_rows, attn.num_heads, attn.head_dim,
                        device=self.device, dtype=dtype,
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
        )

        # -- per-slot single-sequence views (the B1-identical path) --------
        self._slot_states: list[Qwen36GenerationState] = [
            self._build_slot_state(row) for row in range(self._num_rows)
        ]

        # -- persistent batched-decode buffers -----------------------------
        b = self._num_rows
        self._batch_input_ids = _mark_static(
            torch.zeros(b, 1, dtype=torch.long, device=self.device)
        )
        self._batch_positions = _mark_static(
            torch.zeros(b, dtype=torch.long, device=self.device)
        )
        self._batch_write_index = _mark_static(
            torch.zeros(b, dtype=torch.long, device=self.device)
        )
        self._batch_slot_index = _mark_static(
            torch.arange(b, dtype=torch.long, device=self.device)
        )
        # Global page table rows are a pure function of the slot id and never
        # change (static per-slot assignment), so fill them once here rather
        # than rebuilding them every step.
        self._global_page_table = _mark_static(
            torch.arange(
                self._num_rows * self.pages_per_slot, dtype=torch.int32, device=self.device
            ).view(self._num_rows, self.pages_per_slot)
        )

        # Host mirrors: the scheduler reads these every round, and reading
        # them off the device would put a sync in the decode loop.
        self.slot_kv_len = [0] * self._num_rows
        self.slot_committed_tokens: list[list[int]] = [[] for _ in range(self._num_rows)]

    # -- construction helpers ---------------------------------------------

    def _build_slot_state(self, row: int) -> Qwen36GenerationState:
        gdn_states: list[GdnLayerState | None] = []
        attn_caches: list[Qwen36PagedAttentionCache | None] = []
        lo = row * self.pages_per_slot
        hi = lo + self.pages_per_slot
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
                        k_cache=self.k_pools[i][lo:hi],
                        v_cache=self.v_pools[i][lo:hi],
                        page_size=self.page_size,
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

    def reset_slot(self, slot: int) -> None:
        """Return ``slot`` to the fresh state a new sequence needs.

        Zeroes the recurrent state (the B0-5 operational requirement --
        see the module docstring's point 2) and clears the length
        bookkeeping. Does **not** zero KV bytes: past ``kv_len`` they are
        never read, and below it they are the prefix cache.
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

        Returns a flat list of freshly allocated tensors (conv, recurrent,
        conv, recurrent, ... in layer order). Cloned, not viewed: the point
        of a checkpoint is to survive the slot being reused.
        """
        out: list[torch.Tensor] = []
        state = self._slot_states[slot]
        for gdn in state.gdn_states:
            if gdn is None:
                continue
            out.append(gdn.conv_state.clone())
            out.append(gdn.recurrent_state.clone())
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
            global_page = slot * self.pages_per_slot + past // self.page_size
            write_rows.append(global_page * self.page_size + past % self.page_size)
            seqlens.append(past + 1)
            self.slot_kv_len[slot] = past + 1
            self.slot_committed_tokens[slot].append(int(token))
            state = self._slot_states[slot]
            state.num_tokens_seen = past + 1
            for cache in state.attn_caches:
                if cache is not None:
                    cache.seq_len = past + 1

        attn = self.attention_driver(b)
        self._batch_input_ids[:b, 0].copy_(
            torch.tensor(token_ids, dtype=torch.long, device="cpu"), non_blocking=True
        )
        self._batch_positions[:b].copy_(
            torch.tensor([s - 1 for s in seqlens], dtype=torch.long, device="cpu"),
            non_blocking=True,
        )
        attn.cache_seqlens.copy_(
            torch.tensor(seqlens, dtype=torch.int32, device="cpu"), non_blocking=True
        )
        self._batch_write_index[:b].copy_(
            torch.tensor(write_rows, dtype=torch.long, device="cpu"), non_blocking=True
        )
        self._batch_slot_index[:b].copy_(
            torch.tensor(slots, dtype=torch.long, device="cpu"), non_blocking=True
        )
        attn.page_table.copy_(
            self._global_page_table.index_select(
                0, torch.tensor(slots, dtype=torch.long, device=self.device)
            )
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
            attn_outputs=[
                None if out is None else out[:b] for out in self.attn_outputs
            ],
        )
        return batch, b

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
            existing = Qwen36BatchedDecodeAttention(
                batch=batch, **self._driver_kwargs()
            )
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
            "kv_dtype": self.dtype,
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
