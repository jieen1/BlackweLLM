# SPDX-License-Identifier: Apache-2.0
"""Opt-in vLLM attention backend wrapping the sm120-flash-attention project's
hand-written BF16 paged-KV GQA FlashAttention CUDA kernel.

Project: /home/bot/project/sm120-flash-attention/ (kernel/csrc/flash_attn_sm120.cu,
Python bindings in kernel/flash_attn_sm120.py, pip-installed editable in this
venv as the `flash_attn_sm120` package). Built for Qwen3.6-27B's full-attention
layers: head_dim=256, GQA 24:4, causal, on sm120 (RTX PRO 6000 Blackwell /
compute capability 12.0). See that project's
notes/phase-5-vllm-integration.md for the full design writeup, and
02-执行计划.md / CLAUDE.md Phase 5 for the project goal (make THIS machine's
vLLM actually call this kernel -- upstreaming to vLLM/FlashInfer proper is
explicitly NOT a goal).

THIS FILE IS PURELY ADDITIVE. It is not imported by any existing vLLM code
path and has ZERO effect on default behavior. It only takes effect when a
caller explicitly does both of:

    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum, register_backend)
    register_backend(
        AttentionBackendEnum.CUSTOM,
        "vllm.v1.attention.backends.sm120_gqa.SM120GQABackend")

...and then launches vLLM with `--attention-backend CUSTOM`. See
/home/bot/project/sm120-flash-attention/vllm_integration/ for the
registration + test-launcher scripts that do this (kept outside this repo
entirely -- registration is a one-line call from a launcher script, not
something baked into vLLM's own startup path).

Known scope / limitations (see the design notes for the full rationale):
  - BF16 (default), FP8-KV (--kv-cache-dtype fp8_e4m3, verified under real
    serving: 2.10x KV-cache capacity vs BF16, commits 6c1b044/f4c6ff3 in the
    project repo), and NVFP4-KV (--kv-cache-dtype nvfp4, wired in via this
    backend's own combined-tensor read/write kernels -- see get_kv_cache_shape
    and do_kv_cache_update below for why vLLM's native nvfp4 write kernel
    can't be reused) are all wired into this backend. NVFP4-KV's expected
    capacity win is close to 4x vs BF16 (e2m1 is 4 bits/element vs bf16's 16),
    not yet cross-checked against a real serving run at the time of writing --
    see notes/phase-5-vllm-integration.md for whichever number was last
    actually measured.
  - The kernel hardcodes softmax scale = 1/sqrt(head_size); this backend
    refuses to load (raises NotImplementedError) if the model requests a
    different scale.
  - No sliding window / ALiBi / logits soft cap / attention sinks / MLA /
    encoder attention -- decoder self-attention only, matching the kernel's
    actual scope.
  - CUDA Graph support: AttentionCGSupport.UNIFORM_BATCH (see the class
    attribute below) -- flipped from the original NEVER after root-causing a
    real-serving `illegal memory access` to SM120GQAMetadataBuilder.build()
    returning freshly-allocated tensors every call while the captured graph's
    kernel launches recorded fixed memory addresses; fixed by adopting
    FlashInfer's own persistent-buffer pattern (pre-allocate once at
    __init__, write in-place via .copy_() every call) and re-verified correct
    under the exact real-serving scenario that crashed before (single +
    concurrent varlen requests spanning multiple MTP verification steps).
    decode's kv_split_size/max_num_splits were made call-invariant (fixed
    once at builder-init from max_model_len; see _CUDAGRAPH_SAFE_MAX_NUM_SPLITS
    above) as a prerequisite for this. See notes/phase-5-vllm-integration.md's
    CUDA Graph sub-phase sections for the full history (this was NOT a quick
    fix -- several rounds of root-causing were needed).
  - The general (ragged/paged) kernel path handles EVERY case correctly
    (plain prefill, chunked-prefill continuation, single-token decode,
    multi-token decode from speculative/MTP verification, and arbitrary
    mixed prefill+decode batches) since it's fundamentally a causal
    varlen-batch kernel -- see flash_attn_sm120_paged's own docstring and
    notes/phase-4-paged-kv.md's "varlen-decode-like" test cases. The
    decode-specialized split-KV kernel (flash_attn_sm120_decode_paged) is
    used as a fast-path whenever every request in the current step shares
    the SAME query length (uniform decode -- either plain single-token
    decode, or MTP/speculative verification with every request drafting the
    same number of tokens this step, the normal case since
    num_speculative_tokens is a global engine config, not per-request), up
    to _MAX_DECODE_QO_LEN tokens/request (Phase 5 round 7 -- see
    notes/phase-5-vllm-integration.md sec 14 for why the general kernel
    wastes 3/4 of a CTA's warpgroups at the real qo_len=4 MTP shape, and the
    ~1.4-1.8x kernel-level speedup this closes). A batch with ANY request
    still prefilling/chunk-extending alongside decode/verify requests (mixed
    lengths) always falls back to the general kernel -- it is never
    necessary for correctness, only for decode throughput.
"""

import os
from dataclasses import dataclass
from typing import ClassVar

import torch
import triton
import triton.language as tl

from vllm._custom_ops import reshape_and_cache_flash
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.utils.torch_utils import nvfp4_kv_cache_full_dim
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

# Escape hatch to force-disable the decode-specialized split-KV kernel path
# for debugging/A-B testing without touching code (default: use it).
_USE_DECODE_KERNEL = os.environ.get("SM120_GQA_USE_DECODE_KERNEL", "1") != "0"

# 2026-07-15, "v2 Decode Kernel接入SM120GQABackend" section: opt-in switch to
# route FP8-KV decode calls to flash_attn_sm120_fwd_v2_decode_fp8kv_paged
# (from-scratch M-tile-packing design, ~1.45-1.46x vs native at kernel level,
# vs the existing flash_attn_sm120_fp8_kv_decode_paged's 6.027x) instead of
# the scalar kernel below. Default OFF -- this is the FIRST real-serving
# verification of the v2 kernel, not yet validated under CUDA Graph capture/
# replay or a full end-to-end benchmark, so it must not silently change
# default production behavior. Only takes effect when is_fp8_kv and
# use_mma_kernel is False (v2 does not yet cover the BF16 24:4 MMA path).
_USE_V2_DECODE_KERNEL = os.environ.get("SM120_GQA_USE_V2_DECODE_KERNEL", "0") == "1"

# 2026-07-15, "Decode v2 Q-amax热路径开销修复" section: opt-in switch to route
# FP8-KV decode calls to flash_attn_sm120_fwd_v2_decode_fp8kv_paged_nativefp8
# instead of the bf16-dequant v2 kernel above -- same M-tile-packing/split-KV
# architecture, but QK/PV now use mma.sync...m16n8k32.e4m3 directly (the same
# port prefill v2 did, commit b5b78a8), with Q's e4m3 scale computed IN-KERNEL
# (not via a separate host-side Q.abs().amax() reduction -- that pattern,
# fine for prefill's much larger per-call kernel, cost 16.6% of this smaller
# kernel's whole call and erased its entire kernel-level win at decode's
# call granularity; see the roadmap note for the full diagnosis). Verified:
# kernel-level -33.7% (ncu), wrapper-level +32.7% (do_bench) at the isolated
# W2-shape call. Only takes effect when _USE_V2_DECODE_KERNEL is ALSO set
# (this is a QK/PV-mechanism upgrade layered on top of v2's grid, not an
# independent path) -- default OFF, first real-serving verification, not yet
# validated end-to-end under real MTP-verify traffic.
_USE_V2_DECODE_NATIVEFP8_KERNEL = os.environ.get("SM120_GQA_USE_V2_DECODE_NATIVEFP8_KERNEL", "0") == "1"

# 2026-07-21, "qo_len==1 long-context native-FP8 routing" section: the native
# FP8 v2 decode kernel (flash_attn_sm120_fwd_v2_decode_fp8kv_paged_nativefp8)
# is validated for qo_len in [1,4] but the routing below historically sent
# qo_len==1 (pure decode + the 3 MTP draft steps) to the scalar CUDA-core
# kernel flash_attn_sm120_fp8_kv_decode_paged purely because q_decode is 3D
# [BS,QH,D] there (the v2 kernel wants 4D). Micro-benchmarks (c=4, paged FP8
# KV, split=4096) show the native-FP8 kernel is FASTER at long context
# (128K: 1.803ms vs 2.131ms scalar = 1.18x; 64K: 1.12x) while the scalar
# kernel is marginally faster at short context (32K: 0.95x). So route
# qo_len==1 FP8-KV decode to native-FP8 when the per-request KV length is at
# least _QO1_NATIVEFP8_MIN_KV (estimated cheaply from max_num_splits *
# kv_split_size, no host sync), keeping the scalar kernel below that. Default
# 49152 (48K) sits between the measured 32K/64K crossover. Set the env var to
# 0 to disable this routing entirely (always scalar for qo_len==1).
_QO1_NATIVEFP8_MIN_KV = int(os.environ.get("SM120_GQA_QO1_NATIVEFP8_MIN_KV", "49152"))

# 2026-07-15, "Prefill v2产品化:paged KV" section: opt-in switch to route
# FP8-KV prefill calls (pure prefill, chunked-prefill continuation, mixed
# prefill+decode batches) to flash_attn_sm120_fwd_prefill_v2_fp8kv_paged
# (packed-M grid + native FP8 QK+PV MMA, verified 11.7-15.1% faster than
# native FlashInfer at the dense/fixed-shape vertical slice, commit
# b5b78a8) instead of flash_attn_sm120_fp8_kv_paged below. Default OFF --
# same reasoning as _USE_V2_DECODE_KERNEL: first real-serving verification
# of the paged/ragged-Q port, not yet validated end-to-end. Unlike decode
# v2, this path is never CUDA-graph-captured by this backend (prefill
# shapes vary per call; only decode/MTP-verify is captured under
# AttentionCGSupport.UNIFORM_BATCH), so there is no CUDA-graph-safety
# concern to gate on here.
_USE_V2_PREFILL_KERNEL = os.environ.get("SM120_GQA_USE_V2_PREFILL_KERNEL", "0") == "1"

# Host-side heuristic for the decode kernel's kv_split_size (the kernel
# itself takes this as an explicit argument, not an auto-tuned one -- see
# notes/phase-4-paged-kv.md's "known limitations" #3). Targets roughly this
# many splits per request; that file's own benchmark found diminishing
# returns past ~8-16 splits/request for its tested shape -- but that finding
# was for splits computed from the REAL per-call kv_len, a different
# quantity from this constant (see the CUDA-graph-safety comment below):
# this one sizes kv_split_size from max_model_len, so at any real kv_len
# well below max_model_len the actual split count collapses far below this
# target. Phase 5 notes/phase-5-vllm-integration.md section 19 quantified
# that gap directly (kv_split_size derived from 16 here gave 1.6-2.4x
# slower decode than a well-tuned split size at realistic kv_len in
# 5000-16000): 16 was fine for the old dynamic-per-call scheme but far too
# coarse once fixed to max_model_len. 64 was swept against 16/32/128/256
# across kv_len 2000-131072 and was the best broad compromise -- strong
# gains through the realistic mid-range without 128/256's merge-overhead
# regression at the largest lengths (see that section for the full sweep).
_DECODE_TARGET_SPLITS_PER_REQ = 64

# CUDA Graph sub-phase (notes/phase-5-vllm-integration.md, "stage 2"): kv_split_size
# used to be derived from the CURRENT call's max_seq_len, which grows every decode
# step -- fine for eager mode, but fatal for CUDA graph capture/replay, since a
# captured kernel launch's scalar arguments (kv_split_size is a plain int64, not a
# device tensor) freeze to whatever value was live at capture time; replaying that
# same launch on a later step with a different real kv_len would use a stale,
# wrong split boundary.
#
# Rather than bucket-capture-per-kv_len-range (vLLM's own batch-size bucketing
# pattern, but a second, orthogonal bucketing axis is real added complexity), this
# derives kv_split_size ONCE from max_model_len (a build-time-fixed engine config,
# not a per-call value) instead of the live max_seq_len. Proof this stays correct
# for every real kv_len from 1 up to max_model_len, not just the value it was
# derived from: for split_size s = ceil(L/16) (L = max_model_len), and any real
# kv_len k <= L, num_splits(k) = ceil(k/s) <= ceil(L/s) <= ceil(16) = 16 (s >= L/16
# by construction of the ceiling). So a single fixed (kv_split_size,
# max_num_splits_override=16) pair -- computed once at metadata-builder init, never
# recomputed per call -- is a valid upper bound for the ENTIRE decode lifetime of
# any request, not just one snapshot of it. Cost: for short sequences this
# kv_split_size is much larger than "ideal" (their ceil(k/s) collapses toward 1),
# so they get little/no split-KV parallelism benefit -- a quantifiable perf
# tradeoff, not a correctness one (kernel/tests/test_correctness_decode_qolen.py's
# stage-1 override sweep already covers "oversized override, undersized effective
# split count" as a correct, just-wasteful-of-buffer-capacity case).
_CUDAGRAPH_SAFE_MAX_NUM_SPLITS = _DECODE_TARGET_SPLITS_PER_REQ

# Phase 5 round 7: upper bound on the decode-specialized kernel's qo_len
# (MTP/speculative-verification new-tokens-per-step) fast path. Matches the
# kernel's own tested range (kernel/tests/test_correctness_decode_qolen.py
# sweeps up to 16) -- not an architectural limit of the kernel itself (grid.z
# scales with BS*qo_len with no per-thread register cost, see
# flash_attn_decode_partial_kernel's header comment), just how far this round
# actually verified. A batch whose uniform query length exceeds this falls
# back to the general kernel, same as an unverified shape always would.
_MAX_DECODE_QO_LEN = 16


@triton.jit
def _sm120_compact_page_indices_kernel(
    page_indices,
    block_table,
    block_table_stride,
    cu_num_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    # Full-decode-step profiling (notes/phase-5-vllm-integration.md sec 16)
    # found the real 2x-ish gap: NOT slower attention math (GPU compute time
    # within 11% of FlashInfer's) but ~2x the kernel-launch count, GPU idle
    # ~60% of wall time waiting on the CPU. The single biggest contributor was
    # build()'s `cm.block_table_tensor[valid_mask]` boolean-mask-select --
    # dense-to-ragged page-table compaction done with generic torch ops
    # (arange/broadcast-compare/masked-select), which internally lowers to
    # several kernels (mask compute, compaction-index derivation, gather)
    # instead of one. This kernel replaces that whole chain with a single
    # launch, one CTA per request, copying exactly this request's valid
    # page-table entries into their CSR-compacted destination slice --
    # verbatim the same algorithm/kernel vllm/v1/attention/backends/
    # flashinfer.py's own `_copy_page_indices_kernel` already uses (same
    # proven approach, not a novel one) for the identical dense-block-table
    # -> ragged-CSR-page-indices problem.
    req_idx = tl.program_id(0)
    row_ptr = block_table + req_idx * block_table_stride
    start_idx = tl.load(cu_num_blocks + req_idx)
    end_idx = tl.load(cu_num_blocks + req_idx + 1)
    num_blocks = end_idx - start_idx

    offset = tl.arange(0, BLOCK_SIZE)
    for i in tl.range(0, num_blocks, BLOCK_SIZE):
        block_ids = tl.load(row_ptr + i + offset, mask=i + offset < num_blocks)
        tl.store(
            page_indices + start_idx + i + offset,
            block_ids,
            mask=i + offset < num_blocks,
        )


@triton.jit
def _sm120_compute_csr_scalars_kernel(
    seq_lens,
    kv_page_indptr,
    kv_last_page_len,
    num_reqs,
    block_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Continuation of the sec-16 fusion above _sm120_compact_page_indices_kernel:
    # that kernel fused the page_indices gather, but build() still computed
    # num_blocks_per_req/kv_page_indptr/kv_last_page_len with four separate
    # generic torch ops (torch.div, torch.zeros, torch.cumsum, elementwise
    # sub/mul + two .to() casts) every single call -- each one its own kernel
    # launch even though num_reqs is tiny (bounded by max_num_seqs, e.g. 4 in
    # the real production config). One CTA, one launch: load all seq_lens
    # for this step, compute num_blocks_per_req and kv_last_page_len
    # elementwise, and get kv_page_indptr via tl.cumsum -- exact same
    # inclusive-prefix-sum semantics as torch.cumsum, just done in-register
    # instead of via a dedicated cumsum kernel + a separate write.
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < num_reqs
    lens = tl.load(seq_lens + offs, mask=mask, other=0).to(tl.int32)
    num_blocks = (lens + block_size - 1) // block_size
    last_page_len = lens - (num_blocks - 1) * block_size
    tl.store(kv_last_page_len + offs, last_page_len, mask=mask)

    cumsum = tl.cumsum(tl.where(mask, num_blocks, 0), axis=0)
    tl.store(kv_page_indptr + 1 + offs, cumsum, mask=mask)
    tl.store(kv_page_indptr + offs, 0, mask=offs == 0)


def _kernel():
    """Lazy import of the compiled kernel bindings so that importing this
    module (e.g. were it ever imported speculatively/accidentally) doesn't
    hard-fail a process that never actually selects this backend."""
    import flash_attn_sm120

    return flash_attn_sm120


@dataclass
class SM120GQAMetadata:
    num_actual_tokens: int
    num_reqs: int

    # FlashInfer-paged_kv_t-style CSR page table -- see
    # kernel/flash_attn_sm120.py's flash_attn_sm120_paged docstring for the
    # exact convention (this is a straight re-derivation of it from vLLM's
    # own CommonAttentionMetadata, done entirely with GPU tensor ops, no
    # host sync beyond what CommonAttentionMetadata already resolved on CPU).
    qo_indptr: torch.Tensor
    kv_page_indptr: torch.Tensor
    kv_page_indices: torch.Tensor
    kv_last_page_len: torch.Tensor
    page_size: int

    # True only when EVERY request this step has exactly 1 new query token
    # (plain decode, no speculative/MTP draft tokens this step). Kept for
    # backward compat / readability; decode_qo_len (below) is the general
    # form SM120GQAImpl.forward() actually branches on.
    is_pure_decode: bool
    kv_split_size: int
    # CUDA Graph sub-phase stage 2: fixed at builder-init time (see
    # SM120GQAMetadataBuilder.__init__), NOT derived from this call's data --
    # threaded through per-instance metadata (rather than a module global)
    # since it's still logically a property of this step's attention config,
    # matching how kv_split_size itself is exposed.
    max_num_splits: int

    # Phase 5 round 7: uniform per-request query length for this step IF
    # every request shares it (0 if the batch is mixed -- e.g. some requests
    # still prefilling while others decode/verify -- in which case the
    # general kernel is the only correct option and this must stay 0).
    # is_pure_decode is exactly decode_qo_len==1; MTP/speculative verification
    # with every request drafting the same number of tokens this step (the
    # normal case -- num_speculative_tokens is a global engine config, not
    # per-request) makes decode_qo_len the (>1) draft-plus-bonus count
    # instead, gating the SAME split-KV kernel generalized to qo_len>1 (see
    # kernel/csrc/flash_attn_sm120.cu's flash_attn_decode_partial_kernel
    # header comment) rather than always falling back to the general kernel
    # (notes/phase-5-vllm-integration.md sec 10-13's diagnosed 2.1-2.2x gap).
    decode_qo_len: int


class SM120GQAMetadataBuilder(AttentionMetadataBuilder[SM120GQAMetadata]):
    # CUDA Graph sub-phase: kv_split_size/max_num_splits are now call-invariant
    # (see _CUDAGRAPH_SAFE_MAX_NUM_SPLITS above) and were verified correct under
    # a REAL, isolated torch.cuda.graph() capture/replay test sweeping kv_len
    # from 17 to max_model_len against ONE captured graph (kernel/tests/
    # test_cudagraph_decode_fixed_sizing.py). Flipping this to UNIFORM_BATCH
    # and serving real chat-completion traffic under MTP (num_spec_tokens=3,
    # qo_len=4) DID hit a real `CUDA error: an illegal memory access` a few
    # requests into a live server on the first attempt -- root cause was this
    # builder's SM120GQAMetadata.build() returning a freshly-allocated
    # dataclass/tensors every call while the CUDA-graph-captured kernel
    # launches had recorded fixed memory addresses from capture time. Fixed
    # by adopting FlashInfer's own persistent-buffer pattern (pre-allocate
    # once at __init__, write in-place via .copy_() every call instead of
    # allocating fresh tensors) and re-verified correct under the exact
    # real-serving scenario that crashed before: single request (6/6 signal-
    # probe markers correct) + 4 concurrent varlen requests (1.4K-15K tokens,
    # 4/4 probes correct), both spanning multiple MTP verification steps,
    # server survived the whole run with zero crashes. See
    # notes/phase-5-vllm-integration.md secs 13-15 (root-cause + fix +
    # capture/replay verification) for the full history -- this was NOT a
    # quick fix, several rounds of root-causing were needed, and the module
    # docstring above has the condensed version. UNIFORM_BATCH is the
    # current, kept (not reverted) state; do not flip back to NEVER without
    # a new real-serving-crash repro to justify it.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = kv_cache_spec.block_size
        self.device = device

        # CUDA Graph sub-phase stage 2: fixed once at init from a build-time
        # engine config (never recomputed per call, unlike the old
        # max_seq_len-derived heuristic) -- see _CUDAGRAPH_SAFE_MAX_NUM_SPLITS's
        # comment for the correctness proof that this bounds num_splits for
        # every real kv_len up to max_model_len, not just one snapshot value.
        max_model_len = max(1, vllm_config.model_config.max_model_len)
        self.fixed_kv_split_size = max(
            1, (max_model_len + _DECODE_TARGET_SPLITS_PER_REQ - 1) // _DECODE_TARGET_SPLITS_PER_REQ
        )
        self.fixed_max_num_splits = _CUDAGRAPH_SAFE_MAX_NUM_SPLITS

        # CUDA Graph sub-phase, root-causing the real-serving illegal-memory-
        # access: build() used to allocate FOUR fresh tensors every call
        # (torch.zeros(...)/boolean-mask-select/.to(...) all return new GPU
        # memory each time). That's fine in eager mode, but fatal once a
        # downstream kernel launch consuming them gets captured into a CUDA
        # graph -- capture freezes the launch's tensor-pointer arguments to
        # whatever addresses were live at capture time; a later replay reusing
        # those addresses reads stale/invalid memory once build() has since
        # handed out a *different* freshly-allocated tensor. (Confirmed via an
        # isolated repro: the boolean-mask-select itself throws
        # cudaErrorStreamCaptureUnsupported if attempted directly inside a
        # capture region, proving build() runs eager -- not captured -- and
        # the bug is really about the ADDRESS instability of what it returns,
        # not the select operation itself.) Fix (matches vllm/v1/attention/
        # backends/flashinfer.py's own paged_kv_indices/paged_kv_indptr/
        # paged_kv_last_page_len pattern at __init__ time, ~line 794-799):
        # pre-allocate persistent, worst-case-sized GPU buffers ONCE here;
        # build() below now writes each call's real data into a slice of
        # these buffers in-place (.copy_()) and returns a view of the
        # persistent buffer, never a new tensor.
        #
        # qo_indptr/kv_page_indptr/kv_last_page_len are all sized by num_reqs,
        # which is fixed for the lifetime of a given captured graph (vLLM's
        # own batch-size-bucketing invariant -- Q's shape, also fixed per
        # bucket, is what the C++ binding actually derives BS from; see
        # flash_attn_sm120_fwd_decode_paged's `const int BS = Q.size(0)`), so
        # a [:num_reqs+1]/[:num_reqs] slice of a max_num_reqs-sized buffer is
        # the SAME size (and, since it always starts at offset 0, the SAME
        # base address) on every call within one graph's replay lifetime.
        #
        # kv_page_indices' *logical* content length (total valid pages across
        # the batch) is genuinely data-dependent and grows every decode step
        # -- unlike the three CSR-shape tensors above, its "used portion" size
        # is NOT bucket-invariant. But flash_attn_sm120_fwd_decode_paged's
        # TORCH_CHECKs never assert kv_page_indices.size(0) against anything
        # (only kv_page_indptr/kv_last_page_len are size-checked against BS)
        # -- the kernel indexes into it purely via kv_page_indptr's per-
        # request offsets, so passing the FULL fixed-size worst-case buffer
        # (never a data-dependent slice of it) is both sufficient and
        # graph-safe: capture always sees the same full-buffer pointer, and
        # trailing unused slots are simply never read (kv_page_indptr bounds
        # every real read to the portion build() actually wrote this step).
        max_num_reqs = max(1, vllm_config.scheduler_config.max_num_seqs)
        max_blocks_per_req = max(1, (max_model_len + self.block_size - 1) // self.block_size)
        max_num_pages = max_num_reqs * max_blocks_per_req

        self.max_num_reqs = max_num_reqs
        self._qo_indptr_buf = torch.zeros(max_num_reqs + 1, dtype=torch.int32, device=device)
        self._kv_page_indptr_buf = torch.zeros(max_num_reqs + 1, dtype=torch.int32, device=device)
        self._kv_last_page_len_buf = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self._kv_page_indices_buf = torch.zeros(max_num_pages, dtype=torch.int32, device=device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> SM120GQAMetadata:
        cm = common_attn_metadata
        causal = cm.causal
        if causal is not True:
            raise NotImplementedError(
                "SM120_GQA backend only supports uniform causal=True attention "
                f"(got {causal!r}); dynamic per-request causal masks and "
                "non-causal attention are out of scope for the underlying "
                "kernel."
            )

        block_size = self.block_size
        num_reqs = cm.num_reqs
        TORCH_CHECK_msg = (
            f"num_reqs={num_reqs} exceeds this builder's worst-case "
            f"max_num_reqs={self.max_num_reqs} (derived from "
            "scheduler_config.max_num_seqs at builder-init time) -- the "
            "persistent CUDA-graph-safe buffers below are sized for that "
            "bound and would overflow."
        )
        assert num_reqs <= self.max_num_reqs, TORCH_CHECK_msg

        # ---- vLLM's dense block_table_tensor [num_reqs, max_blocks_per_req]
        # (padded with NULL_BLOCK_ID=0, not -1) -> our kernel's ragged CSR
        # page table (kv_page_indptr/kv_page_indices/kv_last_page_len,
        # verbatim FlashInfer paged_kv_t convention). Done entirely with GPU
        # tensor ops (no .item()/host sync) -- seq_lens is already a GPU
        # tensor, block_table_tensor is already a GPU tensor.
        #
        # Full-decode-step profiling (sec 16) found this dense->ragged
        # compaction step -- previously a boolean-mask-select
        # (cm.block_table_tensor[valid_mask]) built from arange/broadcast-
        # compare -- was the dominant contributor to a ~2x kernel-launch-
        # count gap vs FlashInfer (GPU compute time itself was within 11%).
        # Replaced with _sm120_compact_page_indices_kernel, one Triton launch
        # doing the same dense-block-table -> ragged-CSR-indices copy that
        # flashinfer.py's own _copy_page_indices_kernel uses for the
        # identical problem -- not a novel algorithm, the proven one already
        # in this codebase. It writes directly into the persistent
        # _kv_page_indices_buf (no intermediate fresh tensor, no separate
        # .copy_() step), and needs no num_actual_pages host read: the kernel
        # takes kv_page_indptr itself (a GPU tensor) and each program derives
        # its own start/end via tl.load, so there is no new host<->device
        # sync where the old code had none either.
        # kv_page_indptr/kv_last_page_len/qo_indptr must be the buffer-backed,
        # fixed-address views (never a freshly-allocated tensor) BEFORE
        # launching either kernel below -- both write into them in place, and
        # downstream attention calls need the same addresses across replays.
        kv_page_indptr = self._kv_page_indptr_buf[: num_reqs + 1]
        kv_last_page_len = self._kv_last_page_len_buf[:num_reqs]
        qo_indptr = self._qo_indptr_buf[: num_reqs + 1]

        # query_start_loc is already computed by vLLM itself (not derived
        # here) -- just cast+copy into our persistent buffer.
        self._qo_indptr_buf[: num_reqs + 1].copy_(cm.query_start_loc.to(torch.int32))

        # Sec-16 fusion, continued: num_blocks_per_req/kv_page_indptr/
        # kv_last_page_len used to be four separate torch ops (torch.div,
        # torch.zeros, torch.cumsum, elementwise sub/mul) every call -- see
        # _sm120_compute_csr_scalars_kernel's docstring. One CTA is enough
        # since num_reqs is tiny (bounded by max_num_seqs).
        _sm120_compute_csr_scalars_kernel[(1,)](
            cm.seq_lens,
            kv_page_indptr,
            kv_last_page_len,
            num_reqs,
            block_size,
            BLOCK_SIZE=triton.next_power_of_2(max(num_reqs, 1)),
        )

        max_blocks_this_call = cm.block_table_tensor.shape[1]
        _sm120_compact_page_indices_kernel[(max(num_reqs, 1),)](
            self._kv_page_indices_buf,
            cm.block_table_tensor,
            cm.block_table_tensor.stride(0),
            kv_page_indptr,
            BLOCK_SIZE=min(1024, triton.next_power_of_2(max(max_blocks_this_call, 1))),
        )

        # kv_page_indices: pass the FULL fixed-size persistent buffer, not a
        # num_actual_pages-sized slice of it -- see __init__'s comment for why
        # this is both correct (the kernel bounds every real read via
        # kv_page_indptr, never via kv_page_indices.size(0)) and required for
        # graph-safety (a data-dependent slice length would itself vary call
        # to call, unlike the three CSR-shape tensors below whose slice
        # length is pinned to the graph-bucket-invariant num_reqs).
        kv_page_indices = self._kv_page_indices_buf

        # max_query_len is a plain Python int on CommonAttentionMetadata
        # (already resolved CPU-side by the model runner) -- this check is
        # zero-cost, no additional host<->device sync. kv_split_size is no
        # longer derived from the per-call max_seq_len (see
        # _CUDAGRAPH_SAFE_MAX_NUM_SPLITS's comment) -- self.fixed_kv_split_size
        # was computed once at builder-init time from max_model_len, so it (and
        # the max_num_splits_override passed to the kernel below) are the same
        # value on every call, capture or replay alike.
        is_pure_decode = cm.max_query_len == 1
        kv_split_size = self.fixed_kv_split_size

        # Phase 5 round 7: is every request's query length THIS step exactly
        # cm.max_query_len (uniform -- no request still prefilling/chunk-
        # extending alongside decode/verify requests)? query_start_loc_cpu is
        # the SAME already-CPU-resolved tensor query_start_loc's GPU version
        # mirrors (no new host<->device sync, same zero-cost reasoning as
        # max_query_len/max_seq_len above -- just a CPU-side diff+compare on
        # a tensor that already exists).
        query_lens_cpu = cm.query_start_loc_cpu[1:] - cm.query_start_loc_cpu[:-1]
        is_uniform_qo_len = bool((query_lens_cpu == cm.max_query_len).all())
        decode_qo_len = cm.max_query_len if (is_uniform_qo_len and cm.max_query_len <= _MAX_DECODE_QO_LEN) else 0

        return SM120GQAMetadata(
            num_actual_tokens=cm.num_actual_tokens,
            num_reqs=num_reqs,
            qo_indptr=qo_indptr,
            kv_page_indptr=kv_page_indptr,
            kv_page_indices=kv_page_indices,
            kv_last_page_len=kv_last_page_len,
            page_size=block_size,
            is_pure_decode=is_pure_decode,
            kv_split_size=kv_split_size,
            max_num_splits=self.fixed_max_num_splits,
            decode_qo_len=decode_qo_len,
        )


class SM120GQABackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    # "fp8_e4m3": the FP8-KV paged+decode kernels (kernel/csrc/flash_attn_sm120.cu,
    # commits 5b6d333/6377013) expect k_cache/v_cache as [max_num_pages, page_size,
    # KVH, D] uint8 e4m3 with a SINGLE GLOBAL per-tensor scale read back as
    # `stored_e4m3 * scale == true_value` -- verified (by reading
    # csrc/libtorch_stable/quantization/w8a8/fp8/nvidia/quant_utils.cuh's
    # scaled_vec_conversion) to be exactly vLLM's own native
    # reshape_and_cache_flash's "fp8"/"fp8_e4m3" convention (write side divides by
    # scale, read side multiplies), same NHD [num_blocks, block_size, num_heads,
    # head_size] layout as get_kv_cache_shape below -- so do_kv_cache_update needs
    # NO changes, it already threads self.kv_cache_dtype through generically.
    # "nvfp4": the byte-level SHAPE convention (last dim = data_dim + scale_dim,
    # data first then scale, same nvfp4_kv_cache_full_dim(head_size) =
    # head_size//2 + head_size//16 formula vLLM's own FlashInfer nvfp4 path
    # uses -- see vllm.utils.torch_utils) happens to match this project's own
    # NVFP4 paged layout ([max_num_pages, page_size, KVH, D/2] e2m1 + [.., D/16]
    # UE4M3, packed into one combined [.., D/2+D/16] tensor -- kernel/csrc/
    # flash_attn_sm120.cu's *_combined kernels, commits c27a5fc/0dbbfd0/07cef1d).
    # BUT vLLM's own reshape_and_cache_nvfp4 WRITE kernel (csrc/libtorch_stable/
    # nvfp4_kv_cache_kernels.cu) additionally SWIZZLES the scale region for
    # SM100's trtllm-gen consumer, which this project's own (unswizzled) read
    # kernels can't parse -- so do_kv_cache_update below calls this project's
    # OWN nvfp4_kv_paged_cache_write_combined kernel instead of vLLM's native
    # op whenever kv_cache_dtype == "nvfp4", while still reusing vLLM's shape
    # convention (get_kv_cache_shape below) so the cache TENSOR ITSELF is
    # allocated identically to how FlashInfer's nvfp4 path would.
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16", "fp8_e4m3", "nvfp4"]
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        # Must equal the AttentionBackendEnum MEMBER NAME this backend is
        # registered under, not just a descriptive label -- vLLM round-trips
        # it back through `AttentionBackendEnum[self.attn_backend.get_name()]`
        # (see vllm/model_executor/layers/attention/attention.py's
        # `self.backend = AttentionBackendEnum[self.attn_backend.get_name()]`)
        # to recover the enum member. Since this backend is registered under
        # the CUSTOM placeholder slot (see register_sm120_backend.py), it
        # must return "CUSTOM" here, not a made-up name -- confirmed by a
        # real launch failure ("Unknown attention backend: 'SM120_GQA'")
        # before this was fixed; see notes/phase-5-vllm-integration.md.
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["SM120GQAImpl"]:
        return SM120GQAImpl

    @staticmethod
    def get_builder_cls() -> type[SM120GQAMetadataBuilder]:
        return SM120GQAMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # The kernel is hard-specialized for head_dim=256 (Qwen3.6-27B's
        # full-attention layers) -- not a generic-head-dim kernel.
        return [256]

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 12

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # Same (num_blocks, 2, block_size, num_kv_heads, head_size) NHD
        # convention as FlashAttentionBackend/TritonAttentionBackend --
        # deliberately NOT (2, num_blocks, ...) (some other vLLM code paths,
        # e.g. vllm/v1/worker/gpu/attn_utils.py, only support block_dim==0).
        # kernel/csrc/flash_attn_sm120.cu was fixed (see git history) to read
        # the real per-page stride off the tensor at call time instead of
        # assuming a standalone tensor, specifically so that the
        # kv_cache.unbind(1) views this shape produces address correctly.
        if cache_dtype_str == "nvfp4":
            # Combined [data|scale] last dim -- see supported_kv_cache_dtypes'
            # comment above for why this reuses vLLM's own formula/convention
            # even though the write path is entirely this project's own.
            last_dim = nvfp4_kv_cache_full_dim(head_size)
            return (num_blocks, 2, block_size, num_kv_heads, last_dim)
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @classmethod
    def get_required_kv_cache_layout(cls):
        return "NHD"


class SM120GQAImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ) -> None:
        if alibi_slopes is not None:
            raise NotImplementedError("SM120_GQA backend does not support alibi_slopes")
        if sliding_window is not None:
            raise NotImplementedError("SM120_GQA backend does not support sliding_window")
        if logits_soft_cap:
            raise NotImplementedError("SM120_GQA backend does not support logits_soft_cap")
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("SM120_GQA backend only supports decoder self-attention")
        if head_size != 256:
            raise NotImplementedError(f"SM120_GQA backend is specialized for head_size=256, got {head_size}")

        # The kernel hardcodes sm_scale = (1/sqrt(D)) * log2(e) internally --
        # it does not accept a runtime scale argument. Refuse to silently
        # miscompute if a model ever requests a non-default scale.
        expected_scale = head_size**-0.5
        if abs(float(scale) - expected_scale) > 1e-5:
            raise NotImplementedError(
                "SM120_GQA backend's kernel hardcodes softmax scale = "
                f"1/sqrt(head_size) ({expected_scale:.6g}); got a different "
                f"scale ({scale!r}), which the kernel cannot apply."
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("SM120_GQA backend requires num_heads % num_kv_heads == 0 (GQA)")
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.attn_type = attn_type

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.kv_sharing_target_layer_name is not None:
            return
        if kv_cache.numel() == 0:
            return
        key_cache, value_cache = kv_cache.unbind(1)
        if key_cache.dtype == torch.uint8 and key_cache.shape[-1] != self.head_size:
            # NVFP4-KV: the cache tensor's last dim is the COMBINED
            # data+scale width (get_kv_cache_shape's nvfp4_kv_cache_full_dim
            # branch above), not head_size -- this is how we tell it apart
            # from FP8-KV's cache (also uint8, but last dim == head_size,
            # natural e4m3 orientation, no packing). vLLM's own native
            # reshape_and_cache_nvfp4 CANNOT be used here (see
            # supported_kv_cache_dtypes' comment for why its scale swizzling
            # is incompatible with this project's own read kernels) -- this
            # project's own combined-tensor write kernel is used instead.
            kernel = _kernel()
            kernel.flash_attn_sm120_nvfp4_kv_paged_cache_write_combined(
                key,
                value,
                slot_mapping,
                key_cache,
                value_cache,
                key_cache.shape[1],
            )
            return
        reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: SM120GQAMetadata | None,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("SM120_GQA backend does not support fused output quantization")

        if attn_metadata is None:
            # Profiling / memory-usage dry run: no real metadata yet.
            return output.fill_(0)

        num_actual_tokens = attn_metadata.num_actual_tokens
        if num_actual_tokens == 0 or attn_metadata.num_reqs == 0:
            return output.fill_(0)

        kernel = _kernel()
        key_cache, value_cache = kv_cache.unbind(1)
        q = query[:num_actual_tokens]

        # FP8-KV dispatch: kv_cache.dtype is torch.uint8 (raw bytes, e4m3
        # content) whenever the engine was launched with --kv-cache-dtype
        # fp8_e4m3/fp8 (vLLM allocates the cache tensor as uint8 for any FP8
        # cache dtype -- see get_kv_cache_shape's shape comment above, dtype
        # itself comes from vLLM's own cache-dtype-to-torch-dtype mapping,
        # not this backend). do_kv_cache_update needs no branch here -- it
        # already writes the correct format generically via
        # reshape_and_cache_flash(self.kv_cache_dtype, ...).
        #
        # ⚠️ NOT verified under real end-to-end serving this round (single-shot
        # fork execution, no time for the multi-round real-serving verification
        # this project's other kernel work required -- see notes/phase-5-vllm-
        # integration.md). Wiring below is a direct, careful reading of both
        # kernels' Python-binding docstrings (k_scale/v_scale convention
        # cross-checked byte-for-byte against vLLM's own
        # csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh scaled_vec_conversion
        # -- write side divides by scale, read side multiplies, identical to
        # this project's own kernel docstring's `bf16.abs().amax()/448`
        # convention), NOT a real running server test.
        #
        # ⚠️⚠️ KNOWN, DOCUMENTED, NOT-YET-MITIGATED RISK (CLAUDE.md's own
        # validation-methodology section): vLLM issue #37554 -- Qwen3.5/3.6
        # hybrid (GDN + full-attention) + FP8 KV cache + --calculate-kv-scales
        # can silently compute WRONG k_scale/v_scale from an uninitialized GDN
        # dummy-forward-pass state, with no crash/warning. This backend does
        # NOT compute or calibrate _k_scale/_v_scale itself -- it reads
        # whatever vLLM's own engine-level calibration machinery already put
        # into layer._k_scale/_v_scale, so if that machinery hits #37554 on
        # this exact hybrid model, this backend would silently consume a wrong
        # scale with no way to detect it locally. DO NOT enable fp8_e4m3
        # kv-cache-dtype with --calculate-kv-scales on this model without
        # first re-verifying #37554 is fixed/inapplicable, or supplying a
        # manually-verified static k_scale/v_scale instead of the
        # auto-calibrated one.
        is_fp8_kv = key_cache.dtype == torch.uint8 and key_cache.shape[-1] == self.head_size
        # NVFP4-KV: same uint8 storage dtype as FP8-KV, but the last dim is
        # the COMBINED data+scale width (get_kv_cache_shape's
        # nvfp4_kv_cache_full_dim branch), not head_size -- see
        # do_kv_cache_update's identical check above for why this
        # distinguishes the two uint8 cache dtypes. No k_scale/v_scale here:
        # NVFP4 scale lives per-16-element-group INSIDE key_cache/value_cache
        # itself (not a single global scalar like FP8-KV's layer._k_scale),
        # so the kernel needs no separate scale argument.
        is_nvfp4_kv = key_cache.dtype == torch.uint8 and key_cache.shape[-1] != self.head_size

        if _USE_DECODE_KERNEL and attn_metadata.decode_qo_len > 0:
            qo_len = attn_metadata.decode_qo_len
            if qo_len == 1:
                # Every request has exactly 1 new query token this step -- q
                # is already dense [num_reqs, QH, D] (num_actual_tokens ==
                # num_reqs in this case), exactly flash_attn_sm120_decode_
                # paged's original expected shape, no reshape needed.
                q_decode = q
            else:
                # Phase 5 round 7: MTP/speculative verification, every
                # request drafting the SAME number of tokens this step (the
                # normal case). q's rows are contiguous per request in
                # query_start_loc order with EQUAL qo_len-sized chunks (that
                # is exactly what decode_qo_len>0/is_uniform_qo_len asserted
                # in the metadata builder), so this reshape is a free view,
                # not a data-dependent gather -- see
                # flash_attn_sm120_decode_paged's [BS, qo_len, QH, D] contract.
                q_decode = q.reshape(attn_metadata.num_reqs, qo_len, q.shape[-2], q.shape[-1])
            # Round 34: qo_len 2-4 MTP-verify traffic on the production BF16
            # 24:4 GQA ratio gets the tensor-core MMA kernel (round 32/33 --
            # this is the real comparison target for FlashInfer's tensor-core
            # prefill.cuh path on this hardware, not decode.cuh; measured
            # 1.6-2x faster than the CUDA-core kernel below at real
            # production shape). qo_len==1 (M too small for MMA -- round 27)
            # and any non-BF16/non-24:4-GQA case keep using the CUDA-core
            # kernel, same reasoning as flash_attn_sm120_decode_paged_mtp_mma's
            # own docstring scope limits -- no silent fallback inside the
            # kernel itself, so the branch must be gated here.
            use_mma_kernel = (
                1 < qo_len <= 4
                and not is_fp8_kv
                and not is_nvfp4_kv
                and self.num_heads // self.num_kv_heads == 6
            )
            if use_mma_kernel:
                if not getattr(SM120GQAImpl, "_mma_logged", False):
                    logger.info("SM120_GQA: MMA decode kernel path HIT (qo_len=%d)", qo_len)
                    SM120GQAImpl._mma_logged = True
                out = kernel.flash_attn_sm120_decode_paged_mtp_mma(
                    q_decode,
                    key_cache,
                    value_cache,
                    attn_metadata.kv_page_indptr,
                    attn_metadata.kv_page_indices,
                    attn_metadata.kv_last_page_len,
                    attn_metadata.page_size,
                    attn_metadata.kv_split_size,
                    max_num_splits_override=attn_metadata.max_num_splits,
                )
                out = out.reshape(num_actual_tokens, out.shape[-2], out.shape[-1])
                output[:num_actual_tokens].copy_(out)
                return output
            elif qo_len > 1 and not getattr(SM120GQAImpl, "_mma_skip_logged", False):
                logger.info(
                    "SM120_GQA: MMA kernel SKIPPED for qo_len=%d (is_fp8_kv=%s is_nvfp4_kv=%s ratio=%d)",
                    qo_len, is_fp8_kv, is_nvfp4_kv, self.num_heads // self.num_kv_heads,
                )
                SM120GQAImpl._mma_skip_logged = True
            use_v2_decode_kernel = (
                _USE_V2_DECODE_KERNEL
                and is_fp8_kv
                # qo_len==1 excluded: q_decode is 3D [BS,QH,D] in that case
                # (see q_decode's own branch above), but the v2 kernel's
                # TORCH_CHECK requires 4D [BS,qo_len,QH,D] unconditionally --
                # same "M too small for MMA" reasoning use_mma_kernel below
                # already excludes qo_len==1 for, not a new restriction.
                and 1 < qo_len <= 4
                and self.num_heads // self.num_kv_heads == 6
                and self.head_size == 256
            )
            if use_v2_decode_kernel:
                # 2026-07-15, "Decode v2 Q-amax热路径开销修复" section: the
                # native-FP8-QK+PV-MMA kernel is a QK/PV-mechanism upgrade
                # layered on top of v2's grid/split-KV architecture, not an
                # independent path -- only takes effect when the base v2
                # switch is also on (use_v2_decode_kernel already true here).
                use_nativefp8 = _USE_V2_DECODE_NATIVEFP8_KERNEL
                if use_nativefp8:
                    if not getattr(SM120GQAImpl, "_v2_decode_nativefp8_logged", False):
                        logger.info("SM120_GQA: v2 decode NATIVE-FP8 kernel path HIT (qo_len=%d)", qo_len)
                        SM120GQAImpl._v2_decode_nativefp8_logged = True
                    out = kernel.flash_attn_sm120_fwd_v2_decode_fp8kv_paged_nativefp8(
                        q_decode,
                        key_cache,
                        value_cache,
                        attn_metadata.kv_page_indptr,
                        attn_metadata.kv_page_indices,
                        attn_metadata.kv_last_page_len,
                        attn_metadata.page_size,
                        layer._k_scale,
                        layer._v_scale,
                        attn_metadata.kv_split_size,
                        max_num_splits_override=attn_metadata.max_num_splits,
                    )
                else:
                    if not getattr(SM120GQAImpl, "_v2_decode_logged", False):
                        logger.info("SM120_GQA: v2 decode kernel path HIT (qo_len=%d)", qo_len)
                        SM120GQAImpl._v2_decode_logged = True
                    out = kernel.flash_attn_sm120_fwd_v2_decode_fp8kv_paged(
                        q_decode,
                        key_cache,
                        value_cache,
                        attn_metadata.kv_page_indptr,
                        attn_metadata.kv_page_indices,
                        attn_metadata.kv_last_page_len,
                        attn_metadata.page_size,
                        layer._k_scale,
                        layer._v_scale,
                        attn_metadata.kv_split_size,
                        max_num_splits_override=attn_metadata.max_num_splits,
                    )
                out = out.reshape(num_actual_tokens, out.shape[-2], out.shape[-1])
                output[:num_actual_tokens].copy_(out)
                return output
            if is_fp8_kv:
                # qo_len==1 long-context native-FP8 routing (see
                # _QO1_NATIVEFP8_MIN_KV's module comment): the native-FP8 v2
                # decode kernel beats the scalar kernel at long KV. q_decode is
                # 3D [BS,QH,D] here; unsqueeze(1) gives the 4D [BS,1,QH,D] the
                # v2 kernel requires (a free view, no copy).
                use_nativefp8_qo1 = (
                    qo_len == 1
                    and _USE_V2_DECODE_KERNEL
                    and _USE_V2_DECODE_NATIVEFP8_KERNEL
                    and _QO1_NATIVEFP8_MIN_KV >= 0
                    and self.num_heads // self.num_kv_heads == 6
                    and self.head_size == 256
                    and attn_metadata.kv_split_size > 0
                    and (attn_metadata.max_num_splits * attn_metadata.kv_split_size)
                    >= _QO1_NATIVEFP8_MIN_KV
                )
                if use_nativefp8_qo1:
                    if not getattr(SM120GQAImpl, "_v2_decode_nativefp8_qo1_logged", False):
                        logger.info(
                            "SM120_GQA: v2 decode NATIVE-FP8 kernel path HIT (qo_len=1 long-context, min_kv=%d)",
                            _QO1_NATIVEFP8_MIN_KV,
                        )
                        SM120GQAImpl._v2_decode_nativefp8_qo1_logged = True
                    out = kernel.flash_attn_sm120_fwd_v2_decode_fp8kv_paged_nativefp8(
                        q_decode.unsqueeze(1),
                        key_cache,
                        value_cache,
                        attn_metadata.kv_page_indptr,
                        attn_metadata.kv_page_indices,
                        attn_metadata.kv_last_page_len,
                        attn_metadata.page_size,
                        layer._k_scale,
                        layer._v_scale,
                        attn_metadata.kv_split_size,
                        max_num_splits_override=attn_metadata.max_num_splits,
                    )
                    out = out.reshape(num_actual_tokens, out.shape[-2], out.shape[-1])
                    output[:num_actual_tokens].copy_(out)
                    return output
                out = kernel.flash_attn_sm120_fp8_kv_decode_paged(
                    q_decode,
                    key_cache,
                    value_cache,
                    layer._k_scale,
                    layer._v_scale,
                    attn_metadata.kv_page_indptr,
                    attn_metadata.kv_page_indices,
                    attn_metadata.kv_last_page_len,
                    attn_metadata.page_size,
                    attn_metadata.kv_split_size,
                    max_num_splits_override=attn_metadata.max_num_splits,
                )
            elif is_nvfp4_kv:
                out = kernel.flash_attn_sm120_nvfp4_kv_decode_paged_combined(
                    q_decode,
                    key_cache,
                    value_cache,
                    attn_metadata.kv_page_indptr,
                    attn_metadata.kv_page_indices,
                    attn_metadata.kv_last_page_len,
                    attn_metadata.page_size,
                    attn_metadata.kv_split_size,
                    max_num_splits_override=attn_metadata.max_num_splits,
                )
            else:
                out = kernel.flash_attn_sm120_decode_paged(
                    q_decode,
                    key_cache,
                    value_cache,
                    attn_metadata.kv_page_indptr,
                    attn_metadata.kv_page_indices,
                    attn_metadata.kv_last_page_len,
                    attn_metadata.page_size,
                    attn_metadata.kv_split_size,
                    max_num_splits_override=attn_metadata.max_num_splits,
                )
            if qo_len > 1:
                out = out.reshape(num_actual_tokens, out.shape[-2], out.shape[-1])
        elif (
            is_fp8_kv
            and _USE_V2_PREFILL_KERNEL
            # same fixed-shape scope as the v2 decode kernel above -- this
            # kernel is a compile-time GQA_GROUP=6/head_dim=256 specialization,
            # not a general-purpose replacement.
            and self.num_heads // self.num_kv_heads == 6
            and self.head_size == 256
        ):
            # Prefill v2 (2026-07-15, "Prefill v2产品化:paged KV" section) --
            # see _USE_V2_PREFILL_KERNEL's module-level comment. Same
            # (q, key_cache, value_cache, k_scale, v_scale, qo_indptr,
            # kv_page_indptr, kv_page_indices, kv_last_page_len, page_size,
            # causal) argument shape as flash_attn_sm120_fp8_kv_paged below --
            # drop-in swap, not a new contract.
            if not getattr(SM120GQAImpl, "_v2_prefill_logged", False):
                logger.info("SM120_GQA: v2 prefill kernel path HIT")
                SM120GQAImpl._v2_prefill_logged = True
            out = kernel.flash_attn_sm120_fwd_prefill_v2_fp8kv_paged(
                q,
                key_cache,
                value_cache,
                layer._k_scale,
                layer._v_scale,
                attn_metadata.qo_indptr,
                attn_metadata.kv_page_indptr,
                attn_metadata.kv_page_indices,
                attn_metadata.kv_last_page_len,
                attn_metadata.page_size,
                True,
            )
        elif is_fp8_kv:
            # FP8-KV general/prefill kernel (commit 5b6d333) -- correct for
            # pure prefill, chunked-prefill continuation, and arbitrary mixed
            # prefill+decode batches, same scope as the BF16 general kernel
            # below, but reading an already-quantized e4m3 paged cache.
            out = kernel.flash_attn_sm120_fp8_kv_paged(
                q,
                key_cache,
                value_cache,
                layer._k_scale,
                layer._v_scale,
                attn_metadata.qo_indptr,
                attn_metadata.kv_page_indptr,
                attn_metadata.kv_page_indices,
                attn_metadata.kv_last_page_len,
                attn_metadata.page_size,
                True,
            )
        elif is_nvfp4_kv:
            # NVFP4-KV general/prefill kernel, combined-tensor variant
            # (commit 0dbbfd0's predecessors c27a5fc/07cef1d) -- correct for
            # pure prefill, chunked-prefill continuation, and arbitrary mixed
            # prefill+decode batches, same scope as the BF16/FP8-KV general
            # kernels, reading an already-quantized e2m1+UE4M3 paged cache.
            # use_level1 stays at its default True (Q's own on-the-fly
            # two-level quantization) -- unrelated to K/V's storage format.
            out = kernel.flash_attn_sm120_nvfp4_kv_paged_combined(
                q,
                key_cache,
                value_cache,
                attn_metadata.qo_indptr,
                attn_metadata.kv_page_indptr,
                attn_metadata.kv_page_indices,
                attn_metadata.kv_last_page_len,
                attn_metadata.page_size,
                True,
            )
        else:
            # General ragged/paged causal kernel: correct for pure prefill,
            # chunked-prefill continuation, single- or multi-token decode
            # (e.g. MTP/speculative verification), and arbitrary mixed
            # prefill+decode batches -- see this module's docstring.
            out = kernel.flash_attn_sm120_paged(
                q,
                key_cache,
                value_cache,
                attn_metadata.qo_indptr,
                attn_metadata.kv_page_indptr,
                attn_metadata.kv_page_indices,
                attn_metadata.kv_last_page_len,
                attn_metadata.page_size,
                True,
            )

        output[:num_actual_tokens].copy_(out)
        return output
