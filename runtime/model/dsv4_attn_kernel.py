"""Kernel-path DSV4 attention: FP8 packed KV pages + sparkinfer compressed_mla.

The eager ``Dsv4Attention`` keeps its bf16 round-trip cache and its torch
sparse-gather attention -- it is the executable definition of the official
reference and stays untouched as the parity oracle. This layer is the
production path Phase 3 is building: the same projections, the same
compressor/indexer math, but latent KV stored in the fork's compressed-MLA
FP8 page layout (written by ``pack_latent_kv``, bit-exact against the
eager round-trip) and the attention math run by the fork's
``compressed_mla_decode_forward`` kernel (64 heads verified on SM120).

Both prefill and decode go through the decode kernel: the layer packs the
window ring (position p -> slot p % 128) and the compressed entries into
their page buffers, then runs one kernel call with per-row gather indices
(the eager graph's window_topk_idxs / compress_topk_idxs / indexer
semantics, mapped from eager's concatenated-cache offsets to the two
separate flat id spaces of the kernel contract).

Attribute names mirror ``Dsv4Attention`` (wq_a/q_norm_weight/wq_b/wkv/
kv_norm_weight/wo_a/wo_b/attn_sink/compressor/indexer) so the GGUF loader
(``store_gguf_tensor``) writes into this class unchanged.

Self-contained per layer (own page buffers and compressor state, like the
eager module); the slot-pool/backend wiring in Phase 3 step 4 replaces the
page buffers with pool views.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from runtime.kernels.dsv4_kv_pack import (
    DSV4_PAGE_SIZE,
    pack_latent_kv,
    page_nbytes,
)
from runtime.model.dsv4_attention import (
    apply_rotary_emb,
    compress_topk_idxs,
    precompute_freqs_cis,
    window_topk_idxs,
)
from runtime.model.dsv4_config import Dsv4Config
from runtime.model.dsv4_model import (
    Dsv4Compressor,
    Dsv4Indexer,
    PackedQ8_0Linear,
    rms_norm,
)

#: SGLang/DSV4 kernel contract page sizes for the two compressed streams
#: (compressed_reference.py in the fork).
C4_PAGE_SIZE = 64
C128_PAGE_SIZE = 2

# Preserve one BI=64 chunk per split for every native decode batch.  The
# widest ratio-128 contract has 2 SWA + 16 indexed chunks; smaller contracts
# clamp this value to their own chunk count inside b12x.  Using the same chunk
# partition for B=1/2/4 is both bitwise-stable and faster than B4's generic
# wave-balanced five-split plan on the production SM120 card.
DSV4_DECODE_NUM_SPLITS = 18
DSV4_INDEX_CHUNK = 64


def _forced_dsv4_h16(ratio: int, seqlen: int) -> bool | None:
    """Force H16 only for DSV4's measured ratio-4 decode regime."""
    return ratio == 4 if seqlen == 1 else None


def _pad_prefill_index_width(indices: torch.Tensor, max_entries: int) -> torch.Tensor:
    """Pad compressed prefill indices to one native BI=64 chunk bucket.

    b12x specializes on the indexed tensor width.  Letting the exact live
    width grow every prefill chunk causes a new JIT specialization even when
    both widths occupy the same kernel chunk.  Trailing ``-1`` entries are
    already the native invalid-id contract, so chunk bucketing changes neither
    the valid candidates nor their order.
    """
    if indices.ndim != 2:
        raise ValueError(f"prefill indices must be rank 2, got {tuple(indices.shape)}")
    width = int(indices.shape[1])
    max_entries = int(max_entries)
    if width > max_entries:
        raise ValueError(f"prefill index width {width} exceeds capacity {max_entries}")
    if width == 0:
        return indices
    bucket = min(max_entries, math.ceil(width / DSV4_INDEX_CHUNK) * DSV4_INDEX_CHUNK)
    if bucket == width:
        return indices
    return torch.nn.functional.pad(indices, (0, bucket - width), value=-1)


class Dsv4AttnKernelLayer(nn.Module):
    """Loader-compatible attention layer running the fork MLA kernel."""

    def __init__(
        self,
        config: Dsv4Config,
        layer_id: int,
        *,
        num_slots: int = 1,
        max_seq_len: int = 4096,
        max_q_rows: int = 1,
        device: torch.device | str | None = None,
        shared_from: Any = None,
        allocate_mla_scratch: bool = True,
    ) -> None:
        super().__init__()
        if num_slots < 1:
            raise ValueError(f"num_slots must be >= 1, got {num_slots}")
        self.layer_id = layer_id
        self.num_slots = num_slots
        self.ratio = config.layer_ratio(layer_id)
        self.window = config.window_size
        self.n_heads = config.num_heads
        self.head_dim = config.head_dim
        self.rope_head_dim = config.rope_head_dim
        self.n_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.eps = config.norm_eps
        self.softmax_scale = self.head_dim**-0.5
        self.max_q_rows = max_q_rows

        if shared_from is not None:
            # Reuse the eager layer's weight modules (Phase 3 step 5 gate and
            # the Phase 4 backend run one model, two paths; duplicating the
            # packed projections would cost another ~5 GB). Buffers are
            # shared by reference too.
            self.wq_a = shared_from.wq_a
            self.wq_b = shared_from.wq_b
            self.wkv = shared_from.wkv
            self.wo_a = shared_from.wo_a
            self.wo_b = shared_from.wo_b
            self.q_norm_weight = shared_from.q_norm_weight
            self.kv_norm_weight = shared_from.kv_norm_weight
            self.attn_sink = shared_from.attn_sink
            self.compressor = (
                Dsv4Compressor(config, layer_id, num_slots=num_slots, quantize=False, device=device)
                if self.ratio
                else None
            )
            if self.compressor is not None:
                self.compressor.wkv = shared_from.compressor.wkv
                self.compressor.wgate = shared_from.compressor.wgate
                self.compressor.ape = shared_from.compressor.ape
                self.compressor.norm_weight = shared_from.compressor.norm_weight
            self.indexer = (
                Dsv4Indexer(
                    config,
                    layer_id,
                    num_slots=num_slots,
                    max_seq_len=max_seq_len,
                    device=device,
                )
                if self.ratio == 4
                else None
            )
            if self.indexer is not None:
                self.indexer.wq_b = shared_from.indexer.wq_b
                self.indexer.weights_proj = shared_from.indexer.weights_proj
                self.indexer.compressor.wkv = shared_from.indexer.compressor.wkv
                self.indexer.compressor.wgate = shared_from.indexer.compressor.wgate
                self.indexer.compressor.ape = shared_from.indexer.compressor.ape
                self.indexer.compressor.norm_weight = shared_from.indexer.compressor.norm_weight
        else:
            self.wq_a = PackedQ8_0Linear(
                config.q_lora_rank,
                config.hidden_size,
                weight_dtype=torch.bfloat16,
                device=device,
            )
            self.register_buffer(
                "q_norm_weight",
                torch.empty(config.q_lora_rank, dtype=torch.float32, device=device),
            )
            self.wq_b = PackedQ8_0Linear(
                self.n_heads * self.head_dim,
                config.q_lora_rank,
                weight_dtype=torch.bfloat16,
                device=device,
            )
            self.wkv = PackedQ8_0Linear(
                self.head_dim, config.hidden_size, weight_dtype=torch.bfloat16, device=device
            )
            self.register_buffer(
                "kv_norm_weight", torch.empty(self.head_dim, dtype=torch.float32, device=device)
            )
            self.wo_a = PackedQ8_0Linear(
                self.n_groups * self.o_lora_rank,
                self.n_heads * self.head_dim // self.n_groups,
                weight_dtype=torch.bfloat16,
                device=device,
            )
            self.wo_b = PackedQ8_0Linear(
                config.hidden_size,
                self.n_groups * self.o_lora_rank,
                weight_dtype=torch.bfloat16,
                device=device,
            )
            self.register_buffer(
                "attn_sink", torch.empty(self.n_heads, dtype=torch.float32, device=device)
            )

            self.compressor = (
                Dsv4Compressor(config, layer_id, num_slots=num_slots, quantize=False, device=device)
                if self.ratio
                else None
            )
            self.indexer = (
                Dsv4Indexer(
                    config,
                    layer_id,
                    num_slots=num_slots,
                    max_seq_len=max_seq_len,
                    device=device,
                )
                if self.ratio == 4
                else None
            )

        if self.ratio:
            freqs = precompute_freqs_cis(
                self.rope_head_dim,
                max_seq_len,
                original_seq_len=config.rope_original_seq_len,
                base=config.compress_rope_theta,
                factor=config.rope_factor,
                beta_fast=config.beta_fast,
                beta_slow=config.beta_slow,
            )
        else:
            # window-only layers: base theta, no YaRN (verified reference behavior)
            freqs = precompute_freqs_cis(
                self.rope_head_dim,
                max_seq_len,
                original_seq_len=0,
                base=config.rope_theta,
                factor=config.rope_factor,
                beta_fast=config.beta_fast,
                beta_slow=config.beta_slow,
            )
        if device is not None:
            freqs = freqs.to(device)
        self.register_buffer("freqs_cis", freqs)

        # -- page buffers (self-contained; slot-pool views replace them later) --
        self.register_buffer(
            "window_pages",
            torch.empty(
                num_slots,
                1,
                page_nbytes(DSV4_PAGE_SIZE),
                dtype=torch.uint8,
                device=device,
            ),
        )
        # Prefill attention reads the FULL current sequence (the eager path
        # attends its in-flight kv, not the ring), which can exceed the ring's
        # 128 slots; this second page area holds the current prefill.
        prefill_cap = max_q_rows
        n_prefill_pages = max(1, math.ceil(prefill_cap / DSV4_PAGE_SIZE))
        self.register_buffer(
            "prefill_pages",
            torch.empty(
                num_slots,
                n_prefill_pages,
                page_nbytes(DSV4_PAGE_SIZE),
                dtype=torch.uint8,
                device=device,
            ),
        )
        if self.ratio == 4:
            n_pages = max(1, math.ceil((max_seq_len // 4) / C4_PAGE_SIZE))
            self.register_buffer(
                "csa_pages",
                torch.empty(
                    num_slots,
                    n_pages,
                    page_nbytes(C4_PAGE_SIZE),
                    dtype=torch.uint8,
                    device=device,
                ),
            )
            self.register_buffer(
                "hca_pages",
                torch.empty(num_slots, 0, 0, dtype=torch.uint8, device=device),
            )
        elif self.ratio == 128:
            n_pages = max(1, math.ceil((max_seq_len // 128) / C128_PAGE_SIZE))
            self.register_buffer(
                "hca_pages",
                torch.empty(
                    num_slots,
                    n_pages,
                    page_nbytes(C128_PAGE_SIZE),
                    dtype=torch.uint8,
                    device=device,
                ),
            )
            self.register_buffer(
                "csa_pages",
                torch.empty(num_slots, 0, 0, dtype=torch.uint8, device=device),
            )
        else:
            self.register_buffer(
                "csa_pages",
                torch.empty(num_slots, 0, 0, dtype=torch.uint8, device=device),
            )
            self.register_buffer(
                "hca_pages",
                torch.empty(num_slots, 0, 0, dtype=torch.uint8, device=device),
            )

        # compressor writes its raw entries here (quantize=False); the layer
        # packs them into the FP8 pages. Sized for the whole sequence so the
        # prefill emit (one big write) is in bounds.
        if self.compressor is not None:
            self.compressor.kv_cache = torch.empty(
                num_slots,
                max_seq_len // self.ratio,
                self.head_dim,
                dtype=torch.bfloat16,
                device=device,
            )
            self.compressor.freqs_cis = self.freqs_cis
            if self.indexer is not None:
                self.indexer.freqs_cis = self.freqs_cis

        # -- fork MLA plan (fixed shapes, one opaque scratch, bind per step) --
        if self.ratio == 4:
            comp_width = min(config.index_topk, max_seq_len // 4)
        elif self.ratio == 128:
            comp_width = max_seq_len // 128
        else:
            comp_width = 0
        self._width = self.window + comp_width
        self._init_mla_plan(device, allocate_scratch=allocate_mla_scratch)
        # CUDA-Graph capture buffers: pre-allocated ids slot so the decode
        # pack path never allocates during capture (torch.arange in the
        # capture region is graph-pool-allocated and legal, but the
        # pack_latent_kv bounds check's .item() is a GPU->CPU sync that is
        # NOT; the capture path writes this fixed slot and skips the check).
        self._capture_ids = torch.empty(1, dtype=torch.int64, device=device)

    def _init_mla_plan(self, device, *, allocate_scratch: bool) -> None:
        from b12x.attention.compressed_mla import Caps, plan, split_chunks_for_contract

        width = self._width
        chunks = split_chunks_for_contract(rows=self.max_q_rows, width=width)
        self._mla_plan = plan(
            Caps(
                device=str(device) if device is not None else "cuda",
                dtype=torch.bfloat16,
                kv_dtype=torch.uint8,
                num_q_heads=self.n_heads,
                head_dim=self.head_dim,
                v_head_dim=self.head_dim,
                max_width=width,
                max_page_table_width=width,
                max_q_rows=self.max_q_rows,
                max_batch=self.max_q_rows,
                max_kv_rows=self.max_q_rows * width,
                max_chunks_per_row=chunks,
            )
        )
        self._mla_scratch: torch.Tensor | None = None
        if allocate_scratch:
            self._allocate_mla_scratch(device)

    def mla_scratch_spec(self):
        """Expose the MLA scratch contract for backend-owned arena sharing."""
        (spec,) = self._mla_plan.scratch_specs()
        if len(spec.shape) != 1:
            raise RuntimeError(f"{spec.name} scratch must be one-dimensional, got {spec.shape}")
        return spec

    def _allocate_mla_scratch(self, device) -> None:
        spec = self.mla_scratch_spec()
        self._mla_scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)

    def set_mla_scratch(self, scratch: torch.Tensor) -> None:
        """Bind a caller-owned scratch arena prefix to this layer's MLA plan."""
        spec = self.mla_scratch_spec()
        if scratch.dtype != spec.dtype:
            raise TypeError(
                f"{spec.name} scratch must have dtype {spec.dtype}, got {scratch.dtype}"
            )
        if scratch.device != spec.device:
            raise ValueError(
                f"{spec.name} scratch device {scratch.device} does not match {spec.device}"
            )
        if not scratch.is_contiguous():
            raise ValueError(f"{spec.name} scratch must be contiguous")
        scratch_1d = scratch.reshape(-1)
        required = int(spec.shape[0])
        if int(scratch_1d.numel()) < required:
            raise ValueError(
                f"{spec.name} scratch has {int(scratch_1d.numel())} bytes, requires {required}"
            )
        # Every layer binds a fixed prefix view into the backend-owned arena so
        # CUDA Graph capture sees stable addresses while serial forwards share
        # the same underlying storage.
        self._mla_scratch = scratch_1d.narrow(0, 0, required)

    def _require_mla_scratch(self) -> torch.Tensor:
        if self._mla_scratch is None:
            raise RuntimeError("DSV4 MLA scratch was not bound before forward")
        return self._mla_scratch

    # -- per-step pieces ----------------------------------------------------

    def _require_slot(self, slot: int) -> None:
        if not 0 <= slot < self.num_slots:
            raise IndexError(f"slot {slot} out of range ({self.num_slots} slots)")

    @staticmethod
    def _flat_pages(pages: torch.Tensor) -> torch.Tensor:
        return pages.reshape(-1, pages.shape[-1])

    @staticmethod
    def _offset_valid_ids(ids: torch.Tensor, base: int) -> torch.Tensor:
        if base == 0:
            return ids
        return torch.where(ids >= 0, ids + base, ids)

    @staticmethod
    def _offset_valid_ids_batch(
        ids: torch.Tensor,
        slot_ids: torch.Tensor,
        ids_per_slot: int,
    ) -> torch.Tensor:
        """Offset batched local ids without widening MLA's int32 contract."""
        offsets = slot_ids.to(ids.dtype).reshape(-1, 1) * ids_per_slot
        return torch.where(ids >= 0, ids + offsets, -1).to(torch.int32).contiguous()

    @staticmethod
    def _slot_raw_base(slot: int, pages: torch.Tensor, page_size: int) -> int:
        return slot * int(pages.shape[1]) * page_size

    def _comp_pages(self) -> tuple[torch.Tensor, int]:
        if self.ratio == 4:
            return self.csa_pages, C4_PAGE_SIZE
        return self.hca_pages, C128_PAGE_SIZE

    def _pack_window(
        self,
        slot: int,
        kv_row: torch.Tensor,
        seqlen: int,
        start_pos: int,
        *,
        capture: bool = False,
        pos_tensor: torch.Tensor | None = None,
    ) -> None:
        self._require_slot(slot)
        win = self.window
        window_pages = self._flat_pages(self.window_pages)
        window_base = self._slot_raw_base(slot, self.window_pages, DSV4_PAGE_SIZE)
        prefill_pages = self._flat_pages(self.prefill_pages)
        prefill_base = self._slot_raw_base(slot, self.prefill_pages, DSV4_PAGE_SIZE)
        if start_pos == 0 and not capture:
            # the ring keeps the last win tokens (ring layout, p % win), so
            # decode continuity holds; the full current sequence goes to the
            # prefill page area, which the attention of this step reads
            tail = min(seqlen, win)
            ids = (torch.arange(seqlen, device=kv_row.device)[-tail:] % win).to(torch.int64)
            pack_latent_kv(
                kv_row[-tail:],
                window_pages,
                ids + window_base,
                page_size=DSV4_PAGE_SIZE,
                validate_ids=False,
            )
            pack_latent_kv(
                kv_row,
                prefill_pages,
                torch.arange(seqlen, dtype=torch.int64, device=kv_row.device) + prefill_base,
                page_size=DSV4_PAGE_SIZE,
                validate_ids=False,
            )
        else:
            # mid-sequence prefill chunk OR single-token decode: ring slots
            # p % win for every token in the chunk (decode is a 1-token chunk)
            if capture:
                # Capture-safe decode: fixed 1-token ids slot, no allocation,
                # no .item() bounds check (capacity is pre-validated).
                assert pos_tensor is not None
                self._capture_ids.copy_((pos_tensor % win).to(torch.int64) + window_base)
                pack_latent_kv(
                    kv_row,
                    window_pages,
                    self._capture_ids,
                    page_size=DSV4_PAGE_SIZE,
                    validate_ids=False,
                )
                return
            ids = (torch.arange(start_pos, start_pos + seqlen, device=kv_row.device) % win).to(
                torch.int64
            )
            pack_latent_kv(
                kv_row,
                window_pages,
                ids + window_base,
                page_size=DSV4_PAGE_SIZE,
                validate_ids=False,
            )

    def _pack_compressed(
        self,
        slot: int,
        entry: torch.Tensor,
        start_pos: int,
        seqlen: int,
        *,
        capture: bool = False,
        pos_tensor: torch.Tensor | None = None,
    ) -> None:
        self._require_slot(slot)
        pages, page_size = self._comp_pages()
        flat_pages = self._flat_pages(pages)
        base = self._slot_raw_base(slot, pages, page_size)
        n = entry.shape[1]
        if capture:
            # forward_graph always returns the CURRENT cache slot: a newly
            # emitted entry on a compression boundary, otherwise the inert
            # next slot (which is not visible to attention yet).  The eager
            # emission formula would address the previous slot on a
            # non-boundary step and overwrite valid compressed KV.
            assert pos_tensor is not None
            first = pos_tensor // self.ratio  # [1] GPU
            ids = torch.arange(n, dtype=torch.int64, device=entry.device) + first + base
            pack_latent_kv(
                entry.reshape(n, self.head_dim),
                flat_pages,
                ids,
                page_size=page_size,
                validate_ids=False,
            )
            return
        first = (start_pos + seqlen) // self.ratio - n
        ids = torch.arange(first, first + n, dtype=torch.int64, device=entry.device) + base
        pack_latent_kv(
            entry.reshape(n, self.head_dim),
            flat_pages,
            ids,
            page_size=page_size,
            validate_ids=False,
        )

    def _attn_indices(
        self,
        slot: int,
        seqlen: int,
        start_pos: int,
        qr: torch.Tensor,
        x: torch.Tensor,
        *,
        pos_tensor: torch.Tensor | None = None,
        graph_max_index_entries: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """swa idx/len, compressed idx/len (kernel flat-id spaces).

        Mid-sequence prefill chunks (start_pos > 0, seqlen > 1) run through
        the same helpers; the kernel's compressed flat-id space is absolute
        compressed position (offset=0).

        ``pos_tensor``: when given (CUDA-Graph decode), the indices are
        generated on-device from a GPU scalar position -- no Python int,
        so the same captured graph can advance position between replays.
        """
        self._require_slot(slot)
        win = self.window
        swa_base = self._slot_raw_base(
            slot,
            self.prefill_pages if start_pos == 0 and pos_tensor is None else self.window_pages,
            DSV4_PAGE_SIZE,
        )
        comp_base = 0
        if self.ratio:
            comp_pages, comp_page_size = self._comp_pages()
            comp_base = self._slot_raw_base(slot, comp_pages, comp_page_size)
        if pos_tensor is not None:
            from runtime.kernels.dsv4_decode_indices import (
                decode_comp_indices,
                decode_swa_indices,
            )

            swa = decode_swa_indices(pos_tensor, win, device=x.device)  # [1, win]
            swa = self._offset_valid_ids(swa, swa_base)
            swa_len = (swa >= 0).sum(dim=-1)
            comp: torch.Tensor | None = None
            comp_len: torch.Tensor | None = None
            if self.ratio:
                if self.indexer is not None:
                    # ratio-4 layer: top-k indexed selection on device
                    comp = self.indexer.forward_graph(
                        x,
                        qr,
                        pos_tensor,
                        slot=slot,
                        max_entries=graph_max_index_entries,
                    )  # [1,1,index_topk]
                    comp = comp.reshape(seqlen, -1)
                    comp_len = (comp >= 0).sum(dim=-1)
                else:
                    max_comp = self.freqs_cis.shape[0] // self.ratio
                    comp, comp_len = decode_comp_indices(
                        pos_tensor, self.ratio, max_comp, device=x.device
                    )
                    comp_len = comp_len.reshape(1)
                if comp is not None:
                    comp = self._offset_valid_ids(comp, comp_base)
            return (
                swa.contiguous(),
                swa_len.to(torch.int32),
                comp,
                comp_len.to(torch.int32) if comp is not None else None,
            )
        swa = window_topk_idxs(win, 1, seqlen, start_pos, x.device)[0]
        if swa.shape[1] < win:
            # prefill under the window: eager's matrix is [s, min(s, win)];
            # pad to the fixed kernel width with the -1 sentinel
            swa = torch.nn.functional.pad(swa, (0, win - swa.shape[1]), value=-1)
        swa = self._offset_valid_ids(swa, swa_base)
        swa_len = (swa >= 0).sum(dim=-1)

        comp: torch.Tensor | None = None
        comp_len: torch.Tensor | None = None
        if self.ratio:
            if self.indexer is not None:
                comp = self.indexer(x, qr, start_pos, offset=0, slot=slot).int()
                comp = comp.reshape(seqlen, -1)
                comp = _pad_prefill_index_width(comp, self.indexer.index_topk)
            else:
                comp = compress_topk_idxs(
                    self.ratio, 1, seqlen, start_pos, offset=0, device=x.device
                ).reshape(seqlen, -1)
                comp = _pad_prefill_index_width(
                    comp,
                    self.freqs_cis.shape[0] // self.ratio,
                )
            comp = self._offset_valid_ids(comp, comp_base)
            comp_len = (comp >= 0).sum(dim=-1)
        return (
            swa.contiguous(),
            swa_len.to(torch.int32),
            comp,
            comp_len.to(torch.int32) if comp is not None else None,
        )

    def kv_norm(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.kv_norm_weight, self.eps)

    def reset_caches(self, slot: int = 0) -> None:
        """Zero recursive state while retaining prefix-addressable KV bytes."""
        self._require_slot(slot)
        if self.compressor is not None:
            self.compressor.reset_slot(slot)
        if self.indexer is not None:
            self.indexer.reset_state(slot)

    def clear_pages(self, slot: int = 0) -> None:
        self._require_slot(slot)
        self.window_pages[slot].zero_()
        self.prefill_pages[slot].zero_()
        if self.ratio == 4:
            self.csa_pages[slot].zero_()
        elif self.ratio == 128:
            self.hca_pages[slot].zero_()

    def hard_clear_slot(self, slot: int = 0) -> None:
        """Clear all causal bytes after a failed restore or rollback."""
        self.reset_caches(slot)
        self.clear_pages(slot)
        if self.compressor is not None and self.compressor.kv_cache is not None:
            self.compressor.kv_cache[slot].zero_()
        if self.indexer is not None:
            self.indexer.clear_cache(slot)

    def copy_prefix(self, source_slot: int, destination_slot: int, length: int) -> None:
        """Copy one complete 256-token-boundary prefix between slot arenas."""
        self._require_slot(source_slot)
        self._require_slot(destination_slot)
        if source_slot == destination_slot:
            return
        if length <= 0 or length % DSV4_PAGE_SIZE:
            raise ValueError(f"prefix length must be a positive multiple of 256, got {length}")
        self.window_pages[destination_slot].copy_(self.window_pages[source_slot])
        page_count = length // DSV4_PAGE_SIZE
        if self.ratio == 4:
            self.csa_pages[destination_slot, :page_count].copy_(
                self.csa_pages[source_slot, :page_count]
            )
        elif self.ratio == 128:
            self.hca_pages[destination_slot, :page_count].copy_(
                self.hca_pages[source_slot, :page_count]
            )
        if self.compressor is not None and self.compressor.kv_cache is not None:
            entries = length // self.ratio
            self.compressor.kv_cache[destination_slot, :entries].copy_(
                self.compressor.kv_cache[source_slot, :entries]
            )
        if self.indexer is not None:
            entries = length // self.ratio
            self.indexer.kv_cache[destination_slot, :entries].copy_(
                self.indexer.kv_cache[source_slot, :entries]
            )

    def clear_after_prefix(self, slot: int, length: int) -> None:
        """Remove stale compressed/indexer tail beyond a restored boundary."""
        self._require_slot(slot)
        if length <= 0 or length % DSV4_PAGE_SIZE:
            raise ValueError(f"prefix length must be a positive multiple of 256, got {length}")
        page_count = length // DSV4_PAGE_SIZE
        if self.ratio == 4:
            self.csa_pages[slot, page_count:].zero_()
        elif self.ratio == 128:
            self.hca_pages[slot, page_count:].zero_()
        if self.compressor is not None and self.compressor.kv_cache is not None:
            entries = length // self.ratio
            self.compressor.kv_cache[slot, entries:].zero_()
        if self.indexer is not None:
            entries = length // self.ratio
            self.indexer.kv_cache[slot, entries:].zero_()

    # -- forward -------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        *,
        slot: int = 0,
        capture: bool = False,
        pos_tensor: torch.Tensor | None = None,
        graph_max_index_entries: int | None = None,
    ) -> torch.Tensor:
        self._require_slot(slot)
        bsz, seqlen, _ = x.shape
        assert bsz == 1, "kernel-path layer is batch-1 until the backend wiring"
        if pos_tensor is None:
            if start_pos < 0 or start_pos + seqlen > self.freqs_cis.shape[0]:
                raise IndexError(
                    f"attention rows [{start_pos}, {start_pos + seqlen}) exceed "
                    f"capacity {self.freqs_cis.shape[0]}"
                )
            if seqlen > self.max_q_rows:
                raise ValueError(
                    f"attention prefill rows {seqlen} exceed max_q_rows={self.max_q_rows}"
                )
        ratio, rd = self.ratio, self.rope_head_dim
        if pos_tensor is not None:
            # CUDA-Graph capture path: position is a GPU scalar, read by
            # advanced indexing (capture-safe) instead of a Python slice.
            pos = pos_tensor  # [1] int64 on device
            freqs = self.freqs_cis[pos]  # [1, rd] -- matches [pos:pos+1]
            start_pos = int(pos.item()) if not capture else -1
        else:
            freqs = self.freqs_cis[start_pos : start_pos + seqlen]

        # q path (identical math to the eager layer)
        qr = rms_norm(self.wq_a(x), self.q_norm_weight, self.eps)
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        apply_rotary_emb(q[..., -rd:], freqs)
        q_kernel = q.reshape(seqlen, self.n_heads, self.head_dim).to(torch.bfloat16)

        # kv path (identical math), stored packed
        kv = self.kv_norm(self.wkv(x))
        apply_rotary_emb(kv[..., -rd:], freqs)
        kv_row = kv.reshape(seqlen, self.head_dim)
        self._pack_window(
            slot,
            kv_row,
            seqlen,
            start_pos,
            capture=capture,
            pos_tensor=pos_tensor,
        )
        if ratio:
            if capture:
                entry = self.compressor.forward_graph(x, pos, slot=slot)
            else:
                entry = self.compressor(x, start_pos, slot=slot)
            if entry is not None:
                self._pack_compressed(
                    slot,
                    entry,
                    start_pos,
                    seqlen,
                    capture=capture,
                    pos_tensor=pos_tensor,
                )

        swa_idx, swa_len, comp_idx, comp_len = self._attn_indices(
            slot,
            seqlen,
            start_pos,
            qr,
            x,
            pos_tensor=pos_tensor,
            graph_max_index_entries=graph_max_index_entries,
        )
        if comp_idx is not None and comp_idx.shape[1] == 0:
            # no compressed entries exist yet (early prefill / empty slot):
            # a zero-width stream is rejected by the kernel, drop it entirely
            comp_idx, comp_len = None, None

        from b12x.attention.compressed_mla import run

        binding = self._mla_plan.bind(
            scratch=self._require_mla_scratch(),
            q=q_kernel.contiguous(),
            swa_indices=swa_idx,
            swa_lengths=swa_len,
            indexed_indices=comp_idx,
            indexed_lengths=comp_len,
            indexed_page_table=None,
        )
        swa_cache = self._flat_pages(self.prefill_pages if start_pos == 0 else self.window_pages)
        if ratio and comp_idx is not None:
            comp_pages, comp_page_size = self._comp_pages()
            comp_pages = self._flat_pages(comp_pages)
        else:
            comp_pages, comp_page_size = None, None
        out = run(
            swa_k_cache=swa_cache,
            binding=binding,
            swa_page_size=DSV4_PAGE_SIZE,
            indexed_k_cache=comp_pages,
            indexed_page_size=comp_page_size,
            attn_sink=self.attn_sink,
            sm_scale=self.softmax_scale,
            forced_num_splits=(DSV4_DECODE_NUM_SPLITS if seqlen == 1 else None),
            forced_dsv4_h16=_forced_dsv4_h16(self.ratio, seqlen),
        )
        o = out.reshape(bsz, seqlen, self.n_heads, self.head_dim)

        # output path (identical to the eager layer)
        apply_rotary_emb(o[..., -rd:], freqs, inverse=True)
        o = o.view(bsz, seqlen, self.n_groups, -1)
        if self.wo_a.fused_q8:
            from runtime.kernels.dsv4_q8_gemm import q8_0_grouped_dequant_gemm

            # Group-major rows: [G, bs*seq, d] flattened to [G*bs*seq, d],
            # the wo_a einsum contraction per group, no dequant material.
            d_k = o.shape[-1]
            x2 = o.permute(2, 0, 1, 3).reshape(self.n_groups * bsz * seqlen, d_k)
            res = q8_0_grouped_dequant_gemm(
                x2,
                self.wo_a.packed,
                num_groups=self.n_groups,
                group_size=self.o_lora_rank,
                in_features=d_k,
                rows_per_group=bsz * seqlen,
            )
            o = res.reshape(self.n_groups, bsz, seqlen, self.o_lora_rank).permute(1, 2, 0, 3)
        else:
            wo_a = self.wo_a.dequantized().to(x.dtype).view(self.n_groups, self.o_lora_rank, -1)
            o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        return self.wo_b(o.flatten(2))

    def forward_graph_prefill(
        self,
        x: torch.Tensor,
        pos_tensor: torch.Tensor,
        *,
        slot: int = 0,
        graph_max_index_entries: int | None = None,
    ) -> torch.Tensor:
        """Capture-safe prefill tile (mid-sequence): ``seqlen`` rows at GPU
        position.  Replaces the eager per-token compressor Python loop with
        the batched-GEMM fused-kernel path; must be bit-exact with ``forward``
        for start_pos>0.  The first cold-prefill tile (pos 0) must still go
        through ``forward`` (its prefill-page packing differs)."""
        self._require_slot(slot)
        bsz, seqlen, _ = x.shape
        assert bsz == 1
        ratio, rd = self.ratio, self.rope_head_dim
        win = self.window
        pos_idx = pos_tensor + torch.arange(seqlen, dtype=torch.int64, device=x.device)
        freqs = self.freqs_cis[pos_idx]

        qr = rms_norm(self.wq_a(x), self.q_norm_weight, self.eps)
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        apply_rotary_emb(q[..., -rd:], freqs)
        q_kernel = q.reshape(seqlen, self.n_heads, self.head_dim).to(torch.bfloat16)

        kv = self.kv_norm(self.wkv(x))
        apply_rotary_emb(kv[..., -rd:], freqs)
        kv_row = kv.reshape(seqlen, self.head_dim)

        window_base = self._slot_raw_base(slot, self.window_pages, DSV4_PAGE_SIZE)
        pack_latent_kv(
            kv_row,
            self._flat_pages(self.window_pages),
            (pos_idx % win) + window_base,
            page_size=DSV4_PAGE_SIZE,
            validate_ids=False,
        )

        comp_pages: torch.Tensor | None = None
        comp_page_size: int | None = None
        if ratio:
            entry = self.compressor.forward_graph_prefill(x, pos_tensor, slot=slot)
            comp_arena, comp_page_size = self._comp_pages()
            comp_base = self._slot_raw_base(slot, comp_arena, comp_page_size)
            n_comp = entry.shape[1]
            first = (pos_tensor + seqlen) // ratio - n_comp
            ids = torch.arange(n_comp, dtype=torch.int64, device=x.device) + first + comp_base
            pack_latent_kv(
                entry.reshape(n_comp, self.head_dim),
                self._flat_pages(comp_arena),
                ids,
                page_size=comp_page_size,
                validate_ids=False,
            )
            comp_pages = self._flat_pages(comp_arena)

        rows = pos_idx.unsqueeze(1)
        cols = torch.arange(win, device=x.device).unsqueeze(0)
        first = (rows - win + 1).clamp_min(0)
        swa = first + cols
        swa = torch.where(swa <= rows, swa, torch.full_like(swa, -1))
        swa = torch.where(swa >= 0, swa % win, swa)
        swa_idx = swa.int()
        swa_idx = self._offset_valid_ids(swa_idx, window_base)
        swa_len = (swa_idx >= 0).sum(dim=-1).to(torch.int32)

        comp_idx: torch.Tensor | None = None
        comp_len: torch.Tensor | None = None
        if ratio:
            if self.indexer is not None:
                comp_idx = self.indexer.forward_graph_prefill(
                    x, qr, pos_tensor, slot=slot, max_entries=graph_max_index_entries or 512
                ).reshape(seqlen, -1)
                comp_arena2, comp_page_sz2 = self._comp_pages()
                comp_idx = self._offset_valid_ids(
                    comp_idx, self._slot_raw_base(slot, comp_arena2, comp_page_sz2)
                )
                comp_len = (comp_idx >= 0).sum(dim=-1).to(torch.int32)
            else:
                # ratio-128: every compressed entry up to each row's position.
                # Width must be the LIVE entry count (n_entries), padded to a
                # 64-entry bucket -- matching eager's _pad_prefill_index_width
                # -- not the full capacity, or the MLA run width drifts.
                if graph_max_index_entries is None:
                    n_entries = self.freqs_cis.shape[0] // ratio
                else:
                    n_entries = graph_max_index_entries
                max_comp = n_entries
                n_valid = (pos_idx + 1) // ratio
                col = torch.arange(max_comp, device=x.device).unsqueeze(0)
                comp = col.repeat(seqlen, 1)
                comp = torch.where(comp < n_valid.unsqueeze(1), comp, torch.full_like(comp, -1))
                comp_idx = comp.int()
                bucket = min(
                    self.freqs_cis.shape[0] // ratio,
                    ((max_comp + 63) // 64) * 64,
                )
                if max_comp < bucket:
                    comp_idx = torch.nn.functional.pad(comp_idx, (0, bucket - max_comp), value=-1)
                comp_arena2, comp_page_sz2 = self._comp_pages()
                comp_idx = self._offset_valid_ids(
                    comp_idx, self._slot_raw_base(slot, comp_arena2, comp_page_sz2)
                )
                comp_len = (comp_idx >= 0).sum(dim=-1).to(torch.int32)

        from b12x.attention.compressed_mla import run

        binding = self._mla_plan.bind(
            scratch=self._require_mla_scratch(),
            q=q_kernel.contiguous(),
            swa_indices=swa_idx,
            swa_lengths=swa_len,
            indexed_indices=comp_idx,
            indexed_lengths=comp_len,
            indexed_page_table=None,
        )
        out = run(
            swa_k_cache=self._flat_pages(self.window_pages),
            binding=binding,
            swa_page_size=DSV4_PAGE_SIZE,
            indexed_k_cache=comp_pages,
            indexed_page_size=comp_page_size,
            attn_sink=self.attn_sink,
            sm_scale=self.softmax_scale,
            forced_num_splits=None,
            forced_dsv4_h16=None,
        )
        o = out.reshape(bsz, seqlen, self.n_heads, self.head_dim)
        apply_rotary_emb(o[..., -rd:], freqs, inverse=True)
        o = o.view(bsz, seqlen, self.n_groups, -1)
        if self.wo_a.fused_q8:
            from runtime.kernels.dsv4_q8_gemm import q8_0_grouped_dequant_gemm

            d_k = o.shape[-1]
            x2 = o.permute(2, 0, 1, 3).reshape(self.n_groups * bsz * seqlen, d_k)
            res = q8_0_grouped_dequant_gemm(
                x2,
                self.wo_a.packed,
                num_groups=self.n_groups,
                group_size=self.o_lora_rank,
                in_features=d_k,
                rows_per_group=bsz * seqlen,
            )
            o = res.reshape(self.n_groups, bsz, seqlen, self.o_lora_rank).permute(1, 2, 0, 3)
        else:
            wo_a = self.wo_a.dequantized().to(x.dtype).view(self.n_groups, self.o_lora_rank, -1)
            o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        return self.wo_b(o.flatten(2))

    def forward_decode_batch(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        slot_ids: torch.Tensor,
        *,
        graph_max_index_entries: int | None = None,
    ) -> torch.Tensor:
        """Native heterogeneous B=1/2/4 decode over the slot arena.

        The backend validates distinct slot ids and host-side bounds before
        filling the persistent graph inputs; this graph body performs no
        device-to-host validation or Python position branching.
        """
        bsz, seqlen, _ = x.shape
        if bsz not in (1, 2, 4) or seqlen != 1 or bsz > self.max_q_rows:
            raise ValueError(
                "forward_decode_batch requires [B, 1, hidden], B in (1, 2, 4) "
                f"and B <= max_q_rows={self.max_q_rows}, got {tuple(x.shape)}"
            )
        if positions.shape != (bsz,) or positions.dtype != torch.int64:
            raise ValueError(
                f"positions must be int64 [{bsz}], got {tuple(positions.shape)} {positions.dtype}"
            )
        if slot_ids.shape != (bsz,) or slot_ids.dtype != torch.int64:
            raise ValueError(
                f"slot_ids must be int64 [{bsz}], got {tuple(slot_ids.shape)} {slot_ids.dtype}"
            )

        ratio, rd = self.ratio, self.rope_head_dim
        freqs = self.freqs_cis[positions]
        qr = rms_norm(self.wq_a(x), self.q_norm_weight, self.eps)
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        apply_rotary_emb(q[..., -rd:], freqs)
        q_kernel = q.reshape(bsz, self.n_heads, self.head_dim).to(torch.bfloat16)

        kv = self.kv_norm(self.wkv(x))
        apply_rotary_emb(kv[..., -rd:], freqs)
        window_base = slot_ids * (int(self.window_pages.shape[1]) * DSV4_PAGE_SIZE)
        window_ids = window_base + positions % self.window
        pack_latent_kv(
            kv.reshape(bsz, self.head_dim),
            self._flat_pages(self.window_pages),
            window_ids,
            page_size=DSV4_PAGE_SIZE,
            validate_ids=False,
        )

        comp_pages: torch.Tensor | None = None
        comp_page_size: int | None = None
        if ratio:
            entry = self.compressor.forward_graph_batch(x, positions, slot_ids)
            comp_arena, comp_page_size = self._comp_pages()
            comp_base = slot_ids * (int(comp_arena.shape[1]) * comp_page_size)
            comp_write_ids = comp_base + positions // ratio
            pack_latent_kv(
                entry.reshape(bsz, self.head_dim),
                self._flat_pages(comp_arena),
                comp_write_ids,
                page_size=comp_page_size,
                validate_ids=False,
            )
            comp_pages = self._flat_pages(comp_arena)

        from runtime.kernels.dsv4_decode_indices import (
            decode_comp_indices,
            decode_swa_indices,
        )

        swa_idx, swa_len = decode_swa_indices(
            positions,
            self.window,
            device=x.device,
            slot_ids=slot_ids,
            pages_per_slot=int(self.window_pages.shape[1]),
            page_size=DSV4_PAGE_SIZE,
            return_lengths=True,
        )
        comp_idx: torch.Tensor | None = None
        comp_len: torch.Tensor | None = None
        if ratio == 4:
            comp_idx = self.indexer.forward_graph_batch(
                x,
                qr,
                positions,
                slot_ids,
                max_entries=graph_max_index_entries,
            ).reshape(bsz, -1)
            assert comp_page_size is not None
            comp_idx = self._offset_valid_ids_batch(
                comp_idx,
                slot_ids,
                int(self.csa_pages.shape[1]) * comp_page_size,
            )
            comp_len = (comp_idx >= 0).sum(dim=-1).to(torch.int32)
        elif ratio == 128:
            max_comp = self.freqs_cis.shape[0] // ratio
            comp_idx, comp_len = decode_comp_indices(
                positions,
                ratio,
                max_comp,
                device=x.device,
                slot_ids=slot_ids,
                pages_per_slot=int(self.hca_pages.shape[1]),
                page_size=C128_PAGE_SIZE,
            )

        from b12x.attention.compressed_mla import run

        binding = self._mla_plan.bind(
            scratch=self._require_mla_scratch(),
            q=q_kernel.contiguous(),
            swa_indices=swa_idx,
            swa_lengths=swa_len,
            indexed_indices=comp_idx,
            indexed_lengths=comp_len,
            indexed_page_table=None,
        )
        out = run(
            swa_k_cache=self._flat_pages(self.window_pages),
            binding=binding,
            swa_page_size=DSV4_PAGE_SIZE,
            indexed_k_cache=comp_pages,
            indexed_page_size=comp_page_size,
            attn_sink=self.attn_sink,
            sm_scale=self.softmax_scale,
            forced_num_splits=DSV4_DECODE_NUM_SPLITS,
            forced_dsv4_h16=_forced_dsv4_h16(self.ratio, seqlen),
        )
        o = out.reshape(bsz, 1, self.n_heads, self.head_dim)
        apply_rotary_emb(o[..., -rd:], freqs, inverse=True)
        o = o.view(bsz, 1, self.n_groups, -1)
        if self.wo_a.fused_q8:
            from runtime.kernels.dsv4_q8_gemm import q8_0_grouped_dequant_gemm

            d_k = o.shape[-1]
            x2 = o.permute(2, 0, 1, 3).reshape(self.n_groups * bsz, d_k)
            res = q8_0_grouped_dequant_gemm(
                x2,
                self.wo_a.packed,
                num_groups=self.n_groups,
                group_size=self.o_lora_rank,
                in_features=d_k,
                rows_per_group=bsz,
            )
            o = res.reshape(self.n_groups, bsz, 1, self.o_lora_rank).permute(1, 2, 0, 3)
        else:
            wo_a = self.wo_a.dequantized().to(x.dtype).view(self.n_groups, self.o_lora_rank, -1)
            o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        return self.wo_b(o.flatten(2))
