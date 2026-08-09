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


class Dsv4AttnKernelLayer(nn.Module):
    """Loader-compatible attention layer running the fork MLA kernel."""

    def __init__(
        self,
        config: Dsv4Config,
        layer_id: int,
        *,
        max_seq_len: int = 4096,
        max_q_rows: int = 1,
        device: torch.device | str | None = None,
        shared_from: Any = None,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
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
                Dsv4Compressor(config, layer_id, quantize=False, device=device)
                if self.ratio
                else None
            )
            if self.compressor is not None:
                self.compressor.wkv = shared_from.compressor.wkv
                self.compressor.wgate = shared_from.compressor.wgate
                self.compressor.ape = shared_from.compressor.ape
                self.compressor.norm_weight = shared_from.compressor.norm_weight
            self.indexer = (
                Dsv4Indexer(config, layer_id, max_seq_len=max_seq_len, device=device)
                if self.ratio == 4
                else None
            )
            if self.indexer is not None:
                self.indexer.wq_b = shared_from.indexer.wq_b
                self.indexer.weights_proj = shared_from.indexer.weights_proj
                self.indexer.compressor.wkv = shared_from.indexer.compressor.wkv
                self.indexer.compressor.wgate = shared_from.indexer.compressor.wgate
                self.indexer.compressor.ape = shared_from.indexer.compressor.ape
                self.indexer.compressor.norm_weight = (
                    shared_from.indexer.compressor.norm_weight
                )
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
                Dsv4Compressor(config, layer_id, quantize=False, device=device)
                if self.ratio
                else None
            )
            self.indexer = (
                Dsv4Indexer(config, layer_id, max_seq_len=max_seq_len, device=device)
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
            torch.empty(1, page_nbytes(DSV4_PAGE_SIZE), dtype=torch.uint8, device=device),
        )
        # Prefill attention reads the FULL current sequence (the eager path
        # attends its in-flight kv, not the ring), which can exceed the ring's
        # 128 slots; this second page area holds the current prefill.
        prefill_cap = max_q_rows
        n_prefill_pages = max(1, math.ceil(prefill_cap / DSV4_PAGE_SIZE))
        self.register_buffer(
            "prefill_pages",
            torch.empty(
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
                torch.empty(n_pages, page_nbytes(C4_PAGE_SIZE), dtype=torch.uint8, device=device),
            )
        elif self.ratio == 128:
            n_pages = max(1, math.ceil((max_seq_len // 128) / C128_PAGE_SIZE))
            self.register_buffer(
                "hca_pages",
                torch.empty(n_pages, page_nbytes(C128_PAGE_SIZE), dtype=torch.uint8, device=device),
            )
        else:
            self.register_buffer("csa_pages", torch.empty(0, 0, dtype=torch.uint8, device=device))

        # compressor writes its raw entries here (quantize=False); the layer
        # packs them into the FP8 pages. Sized for the whole sequence so the
        # prefill emit (one big write) is in bounds.
        if self.compressor is not None:
            self.compressor.kv_cache = torch.empty(
                1, max_seq_len // self.ratio, self.head_dim, dtype=torch.bfloat16, device=device
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
        self._init_mla_plan(device)

    def _init_mla_plan(self, device) -> None:
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
        (spec,) = self._mla_plan.scratch_specs()
        self._mla_scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)

    # -- per-step pieces ----------------------------------------------------

    def _comp_pages(self) -> tuple[torch.Tensor, int]:
        if self.ratio == 4:
            return self.csa_pages, C4_PAGE_SIZE
        return self.hca_pages, C128_PAGE_SIZE

    def _pack_window(self, kv_row: torch.Tensor, seqlen: int, start_pos: int) -> None:
        win = self.window
        if start_pos == 0:
            # the ring keeps the last win tokens (ring layout, p % win), so
            # decode continuity holds; the full current sequence goes to the
            # prefill page area, which the attention of this step reads
            tail = min(seqlen, win)
            ids = (torch.arange(seqlen, device=kv_row.device)[-tail:] % win).to(torch.int64)
            pack_latent_kv(kv_row[-tail:], self.window_pages, ids, page_size=DSV4_PAGE_SIZE)
            pack_latent_kv(
                kv_row,
                self.prefill_pages,
                torch.arange(seqlen, dtype=torch.int64, device=kv_row.device),
                page_size=DSV4_PAGE_SIZE,
            )
        else:
            # mid-sequence prefill chunk OR single-token decode: ring slots
            # p % win for every token in the chunk (decode is a 1-token chunk)
            ids = (
                torch.arange(start_pos, start_pos + seqlen, device=kv_row.device) % win
            ).to(torch.int64)
            pack_latent_kv(kv_row, self.window_pages, ids, page_size=DSV4_PAGE_SIZE)

    def _pack_compressed(self, entry: torch.Tensor, start_pos: int, seqlen: int) -> None:
        pages, page_size = self._comp_pages()
        n = entry.shape[1]
        first = (start_pos + seqlen) // self.ratio - n
        ids = torch.arange(first, first + n, dtype=torch.int64, device=entry.device)
        pack_latent_kv(entry.reshape(n, self.head_dim), pages, ids, page_size=page_size)

    def _attn_indices(
        self, seqlen: int, start_pos: int, qr: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """swa idx/len, compressed idx/len (kernel flat-id spaces).

        Mid-sequence prefill chunks (start_pos > 0, seqlen > 1) run through
        the same helpers; the kernel's compressed flat-id space is absolute
        compressed position (offset=0).
        """
        win = self.window
        swa = window_topk_idxs(win, 1, seqlen, start_pos, x.device)[0]
        if swa.shape[1] < win:
            # prefill under the window: eager's matrix is [s, min(s, win)];
            # pad to the fixed kernel width with the -1 sentinel
            swa = torch.nn.functional.pad(swa, (0, win - swa.shape[1]), value=-1)
        swa_len = (swa >= 0).sum(dim=-1)

        comp: torch.Tensor | None = None
        comp_len: torch.Tensor | None = None
        if self.ratio:
            if self.indexer is not None:
                comp = self.indexer(x, qr, start_pos, offset=0).int()
                comp = comp.reshape(seqlen, -1)
            else:
                comp = compress_topk_idxs(
                    self.ratio, 1, seqlen, start_pos, offset=0, device=x.device
                ).reshape(seqlen, -1)
            comp_len = (comp >= 0).sum(dim=-1)
        return (
            swa.contiguous(),
            swa_len.to(torch.int32),
            comp,
            comp_len.to(torch.int32) if comp is not None else None,
        )

    def kv_norm(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.kv_norm_weight, self.eps)

    def reset_caches(self) -> None:
        """Zero the recursive compressor state (the pool reset rule)."""
        if self.compressor is not None:
            self.compressor.kv_state.zero_()
            self.compressor.score_state.fill_(float("-inf"))
        if self.indexer is not None:
            self.indexer.compressor.kv_state.zero_()
            self.indexer.compressor.score_state.fill_(float("-inf"))
            self.indexer.kv_cache.zero_()

    # -- forward -------------------------------------------------------------

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        assert bsz == 1, "kernel-path layer is batch-1 until the backend wiring"
        ratio, rd = self.ratio, self.rope_head_dim
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
        self._pack_window(kv_row, seqlen, start_pos)
        if ratio:
            entry = self.compressor(x, start_pos)
            if entry is not None:
                self._pack_compressed(entry, start_pos, seqlen)

        swa_idx, swa_len, comp_idx, comp_len = self._attn_indices(seqlen, start_pos, qr, x)
        if comp_idx is not None and comp_idx.shape[1] == 0:
            # no compressed entries exist yet (early prefill / empty slot):
            # a zero-width stream is rejected by the kernel, drop it entirely
            comp_idx, comp_len = None, None

        from b12x.attention.compressed_mla import run

        binding = self._mla_plan.bind(
            scratch=self._mla_scratch,
            q=q_kernel.contiguous(),
            swa_indices=swa_idx,
            swa_lengths=swa_len,
            indexed_indices=comp_idx,
            indexed_lengths=comp_len,
            indexed_page_table=None,
        )
        swa_cache = self.prefill_pages if start_pos == 0 else self.window_pages
        if ratio and comp_idx is not None:
            comp_pages, comp_page_size = self._comp_pages()
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
