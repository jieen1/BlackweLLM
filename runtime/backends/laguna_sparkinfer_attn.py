"""SparkInfer paged attention — full replacement for FlashInfer in Laguna.

Handles both prefill (extend mode) and decode (CG mode) for all layer groups:
- Full attention: window_left=-1, 48 Q heads, 8 KV heads (gqa_group_size=6), head_dim=128
- SWA: window_left=511, 72 Q heads, 8 KV heads (gqa_group_size=9), head_dim=128

These are the real unsharded weight shapes (verified against the checkpoint's
safetensors tensors directly, not the config). Production runs TP=1 (no tensor
parallelism implemented), so these are also the shapes seen at runtime -- not
the TP=2-sharded num_kv_heads=4 that some upstream sparkinfer kernel
specializations are tuned for.

KV cache layout: vLLM stores [2, num_blocks, block_size, num_kv_heads, head_dim].
After unbind(0): k_cache/v_cache = [num_blocks, block_size, num_kv_heads, head_dim]
which is exactly sparkinfer's expected [num_pages, page_size, num_kv_heads, head_dim].

Integration: monkey-patch each Attention layer's impl after model load.
"""

from __future__ import annotations

import os

# Enable native FP8 attention MMA (turbo attention).  The KV cache is
# already FP8; this avoids dequantizing to BF16 before the QK/PV matmuls,
# halving memory traffic for KV reads.  Measured +6.2% at 64K context
# (310 → 330 tok/s) with acceptance rate improving from 0.96 to 1.0.
# SPARKINFER_TURBO_ATTN: Native FP8 attention MMA.  Enabled by default.
# +6.2% at 64K (310→330 tok/s), acceptance 0.96→1.0 on fox-64K.
# Known limitation: code-4K acceptance regression (0.978→0.586) due to
# per-tensor FP8 QK scale precision loss at short range.
# Fix path: per-head K/V descale (kernel supports 2D [batch,heads]
# descale at forward_paged.py:5429, needs runtime wiring) + per-row
# Q scale before e4m3 conversion (forward_paged.py mxfp8 helpers).
os.environ.setdefault("SPARKINFER_TURBO_ATTN", "0")

import logging
import time
from typing import Any

import torch

from runtime.backends._sparkinfer_import import ensure_sparkinfer_path
from runtime.kernels.fused_kv_scatter import fused_kv_scatter

logger = logging.getLogger("qwen_sm120_runtime.sparkinfer_attn")

# Threshold separating "genuine cold CuTe compile just happened" from "hit
# sparkinfer's on-disk compile cache" in the SparkinferPrefillWorkspace
# first-call diagnostic below. Compiles measured ~30s; disk-cache hits
# measured single-digit milliseconds -- there is no realistic middle ground,
# so this default has wide margin either way.
_ATTN_COMPILE_WARN_S = float(os.environ.get("QSR_ATTN_COMPILE_WARN_S", "5.0"))

# Route through the single controlled resolver (see
# runtime/backends/_sparkinfer_import.py) instead of inserting BF_SPARKINFER_PATH
# into sys.path directly here -- laguna.py's _patch_moe_sparkinfer touches
# `sparkinfer` before this module is even imported on the real Laguna startup
# path, so a local-only sys.path.insert here was consistently too late.
ensure_sparkinfer_path()

PAGE_SIZE = 64  # Default for SparkinferDecodeWorkspace.page_size; callers should
# pass the real LagunaBackend.block_size explicitly (64 or 128 both supported).


def _paged_descale(
    scale: torch.Tensor,
    *,
    batch_size: int,
    num_kv_heads: int,
) -> torch.Tensor:
    """Normalize vLLM KV scales to sparkinfer's per-request contract.

    vLLM attention layers expose scalar, per-head, or singleton-expanded
    scales depending on whether the checkpoint stores FP8 KV scale tensors.
    SparkInfer requires a rank-1 ``[batch]`` or rank-2
    ``[batch, num_kv_heads]`` descale tensor.  In particular, DFlash's BF16
    checkpoint-backed draft layers expose a rank-0 default scale.
    """
    scale = scale.detach().to(dtype=torch.float32)
    count = scale.numel()
    if count == 1:
        return scale.reshape(1).expand(batch_size).contiguous()
    if count == batch_size:
        return scale.reshape(batch_size).contiguous()
    if count == num_kv_heads:
        return scale.reshape(1, num_kv_heads).expand(batch_size, -1).contiguous()
    if count == batch_size * num_kv_heads:
        return scale.reshape(batch_size, num_kv_heads).contiguous()
    raise ValueError(
        "KV descale must be scalar, per-request, per-head, or per-request/per-head; "
        f"got shape {tuple(scale.shape)} for batch_size={batch_size}, "
        f"num_kv_heads={num_kv_heads}."
    )


class SparkinferAttnMetadata:
    """Lightweight metadata passed through forward context for sparkinfer."""

    __slots__ = (
        "mode",
        "page_table",
        "cache_seqlens",
        "cu_seqlens_q",
        "num_actual_tokens",
        "window_left",
    )

    def __init__(
        self,
        mode: str,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        num_actual_tokens: int,
        window_left: int = -1,
    ):
        self.mode = mode
        self.page_table = page_table
        self.cache_seqlens = cache_seqlens
        self.cu_seqlens_q = cu_seqlens_q
        self.num_actual_tokens = num_actual_tokens
        self.window_left = window_left


class SparkinferPrefillWorkspace:
    """Manages sparkinfer extend/verify-mode workspaces for prefill (eager, no CG).

    One instance can be shared by all layers in an attention group. The
    immutable planner metadata is identical for those layers within one model
    forward, so it is prepared once and reused by the remaining layers. A
    different metadata object always rebuilds the plan and uploads its runtime
    metadata; this cache never crosses requests.

    Root cause this class exists to avoid (see
    notes/2026-08-01-prefill-shape-buckets-root-cause.md): sparkinfer's
    ``paged_attention_forward`` JIT-compiles its CuTe launch wrapper keyed on
    a snapshot of several tensors' shapes (``forward_cache_key`` in
    sparkinfer's ``attention/paged/_forward.py``). ``q``'s own token
    dimension is already masked dynamic there, but ``page_table``'s block-
    table width is not -- it is a literal function of ``kv_len + qo_len``.
    Building a *fresh* ``PagedAttentionWorkspace`` (via ``for_tensors``) for
    every distinct request shape, as this class used to do, means every
    previously-unseen ``(kv_len, qo_len)`` combination pays a ~30s CuTe
    recompile -- and real multi-turn traffic almost never repeats a shape,
    so every turn pays it. Building ONE persistent workspace at a fixed
    capacity via ``PagedAttentionWorkspace.for_fixed_capacity`` instead means
    ``page_table`` (and every other capacity-bound buffer) keeps the SAME
    object and the SAME shape across calls; only its *contents* change per
    call. That makes the compile cache key stable, so the compile happens
    once per (mode, window_left) and every request within the declared
    capacity reuses it -- confirmed empirically: identical trailing calls
    against a fixed-capacity workspace measure ~4ms regardless of how much
    the real ``qo_len``/``kv_len`` differs from the previous call, vs ~32s
    for the very first call against any new workspace.

    **That is necessary but was not sufficient, and the gap was found on
    Qwen3.6 first (2026-08-02).** Pinning every *buffer* shape does not pin
    everything in sparkinfer's compile key: the key also contains
    ``_traits_compile_key(traits)``, and ``traits`` comes from
    ``select_paged_forward_traits_from_plan(plan)`` -- so ``plan.cta_tile_q``
    is itself part of the key. For an eager plan the planner derives it from
    the LIVE query length (``planner.py``, ``create_paged_plan``'s
    ``enable_cuda_graph=False`` branch), via
    ``_paged_determine_cta_tile_q(packed_qo_len=qo_len * gqa_group_size, ...)``.
    At Laguna's own geometry (``gqa_group_size=6``, ``head_dim=128``) that
    yields THREE tile buckets across query length -- measured on GPU, and
    identical for both layer groups (``window_left=-1`` and the SWA group's
    ``window_left=511``): ``cta_tile_q=16`` for ``qo_len <= 5``, 64 for
    6..10, 128 for ``qo_len >= 11``. So prefill had three distinct compiles,
    not one (25-37s each on this machine with a cold compile cache). Worse,
    ``LagunaBackend.warmup_paged_attention_shapes`` warms with
    ``dummy_qo=8``, which lands in the MIDDLE bucket -- the 128 bucket every
    ordinary-length prompt actually uses was never warmed, so it compiled
    inside the first real request on any machine with a cold
    ``~/.cache/sparkinfer``.

    Fix: pass sparkinfer's own ``PagedPlanBudget`` for ``mode="extend"``, so
    ``cta_tile_q`` is derived from this workspace's declared capacity instead
    of the live shape and all three buckets collapse into one. The planner
    has a branch specifically for this (``if mode in ("extend", "verify") and
    plan_budget.max_total_q is not None: avg_packed_qo_len = max(...)``); no
    sparkinfer source is modified. See
    ``scripts/laguna_probe_extend_jit_buckets.py`` for the measured before/
    after and for the bitwise-equality check that this changes no arithmetic
    (extend can never split KV -- ``create_paged_plan`` hard-sets
    ``split_kv=False, disable_split_kv=True`` for ``mode == "extend"``
    unconditionally -- so ``cta_tile_q`` only regroups query rows into CTAs
    and each row's softmax still runs over the whole KV span in one CTA).

    ``mode="verify"`` is deliberately EXCLUDED from that budget, and this is
    load-bearing rather than conservatism: ``_paged_determine_cta_tile_q``
    selects Laguna's M64 verifier by an exact match on
    ``packed_qo_len == 48`` (its q=8/GQA6 window), and several downstream
    kernel-policy flags (``use_laguna_verify_kernel``,
    ``laguna_verify_two_wave_b1``, the FP8 PV MMA path) are gated on
    ``plan.cta_tile_q == 64``. A capacity-derived ``packed_qo_len`` would
    miss that exact match and silently drop verify onto ``cta_tile_q=16``.
    Verify does not have extend's problem anyway: its query length is a
    fixed ``NUM_QUERY_PER_REQ`` window, so it is single-bucket already.
    ``mode="decode"`` never consults the budget branch at all
    (``_paged_determine_cta_tile_q`` returns a hard-coded 16 for decode
    before it looks at ``packed_qo_len``).

    A second, distinct bug lived in how ``forward()`` sized that fixed
    capacity (see notes/2026-08-01-c1-c2-gpu-investigation.md §C-1): it
    always called ``PagedAttentionWorkspace.eager_extend_work_items_capacity``
    -- an estimator whose name and design are for ``mode="extend"`` (its
    ``max_work_items`` scales with ``max_total_q``, which is what extend's
    real work-item count tracks) -- regardless of the ``mode`` actually being
    requested. DFlash's eager verify fallback (``_forward_verify_with_aux``)
    shares this same per-``(window_left, num_heads, num_kv_heads, head_size)``
    workspace object with ordinary prefill, and calls it with ``mode="verify"``,
    whose real work-item count does NOT scale with ``max_total_q`` the same
    way (verify's query is a fixed, tiny 1-16 token window; its work items
    scale with how many KV chunks the context needs, driven by kv_len/window,
    not by query length) -- so the extend-shaped estimate silently
    under-provisions it. Confirmed on real GPU: a direct call to
    ``DFlashEngine._forward_verify_with_aux`` at an ordinary shape
    (kv_len~2016, 16-token verify window) raised sparkinfer's
    ``_ensure_capacity``: ``ValueError: fixed-capacity paged workspace
    exceeded``, immediately, before any attention math ran.

    Fix: ``forward()`` now dispatches the capacity estimate by ``mode``.
    ``extend``/``decode`` keep ``eager_extend_work_items_capacity`` (the API
    sparkinfer itself names and designs for that case). ``verify`` runs
    sparkinfer's own real eager planner (``planner.create_paged_plan`` with
    ``enable_cuda_graph=False`` -- the exact function every real verify call
    below will use) once, up front, against a synthetic worst-case call at
    this group's declared max capacity, and reads its actual
    ``new_batch_size``/``total_num_partial_rows`` -- not a new,
    independently-invented number, and not sparkinfer's OTHER (graph-mode)
    capacity planner either.

    An earlier version of this fix tried ``planner.plan_verify_graph_capacity``
    on the theory that it was "the same capacity math LagunaCudaGraphVerify
    already trusts" -- measured wrong on real GPU (see ``_work_item_capacity``'s
    docstring): that planner computes a schedule for CUDA-Graph replay, a
    different (and, empirically, smaller) chunking policy than the eager
    path's own per-call schedule, so it under-provisioned the exact same way
    the original bug did, just by a different amount. Both attempts, and why
    the second one is right, are recorded in
    notes/2026-08-01-c1-c2-gpu-investigation.md's follow-up section.
    Deliberately NOT "call the extend estimator and multiply by a safety
    factor" either: a fudge factor only moves the hard failure to some
    larger shape nobody has tried yet, and there is no principled way to
    know if any given factor is enough (see notes for why this project has
    hit exactly that trap before, with a real per-shape kv_len+qo_len bound
    rather than a coefficient).
    """

    def __init__(self, device: torch.device, *, max_total_q: int, max_page_table_width: int):
        # Lazy, like every other sparkinfer import in this module: importing
        # sparkinfer at module scope would make this file unimportable in the
        # sparkinfer-free environments some tests run in.
        from b12x.attention.paged.planner import PagedPlanBudget

        self.device = device
        self._descale = torch.ones(1, dtype=torch.float32, device=device)
        self._workspace: Any | None = None
        self._workspace_key: tuple[Any, ...] | None = None
        self._prepared_metadata: object | None = None
        # Capacity for the persistent PagedAttentionWorkspace this instance
        # builds lazily (see forward()). Sized by the caller (LagunaBackend)
        # to bound every real qo_len / (kv_len+qo_len) this group can see, so
        # _ensure_capacity() below never needs to grow (and raises loudly --
        # rather than silently recompiling -- if the caller under-sized it).
        self._max_total_q = max_total_q
        self._max_page_table_width = max_page_table_width
        # One request per extend/verify call everywhere in this runtime
        # (prefill is always single-slot; DFlash verify is always
        # single-slot -- see laguna.py and laguna_dflash.py call sites).
        self._max_batch = 1
        # Declares this capacity to the planner so plan policy that is part
        # of sparkinfer's compile key -- above all ``cta_tile_q`` -- is
        # derived from capacity rather than from the live request's query
        # length. mode="extend" ONLY; see the class docstring for why
        # mode="verify" must not get this and why mode="decode" ignores it.
        self._extend_plan_budget = PagedPlanBudget(
            max_total_q=max_total_q,
            max_batch=self._max_batch,
            max_page_table_width=max_page_table_width,
        )
        # Upper bound on the query length any mode="verify" call will ever
        # use against this instance. Zero (unset) means "no caller has
        # declared verify traffic for this (window_left, heads, head_size)
        # group yet" -- forward() raises loudly rather than guessing if
        # mode="verify" shows up before declare_verify_capacity() is called.
        # Set via declare_verify_capacity(), monotonically (max of every
        # call), so multiple independent callers (main model's own verify
        # users, if any is ever added, plus DFlash) can't shrink an
        # already-declared bound.
        self._max_verify_query_len = 0

    def declare_verify_capacity(self, max_query_len: int) -> None:
        """Declare that this workspace's ``mode="verify"`` calls never use a
        query length above ``max_query_len``.

        Must be called before the first real ``mode="verify"`` call reaches
        ``forward()`` -- see the class docstring's second bug for why
        skipping this is not "safe by default": there is no sound default
        capacity for a mode this workspace has no other way to bound. Callers
        (e.g. ``DFlashEngine.__init__`` via ``LagunaBackend.
        declare_verify_capacity``) should pass the true maximum verify window
        (``NUM_QUERY_PER_REQ``), not a per-call value -- this is a fixed
        capacity contract, the same as ``max_total_q``/``max_page_table_width``
        above, not a per-call hint.
        """
        max_query_len = int(max_query_len)
        if max_query_len <= 1:
            raise ValueError(
                f"declare_verify_capacity requires max_query_len > 1, got {max_query_len} "
                "(a single-token verify call is not a real contract this workspace serves)"
            )
        self._max_verify_query_len = max(self._max_verify_query_len, max_query_len)

    @staticmethod
    def _key(
        *,
        mode: str,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        window_left: int,
    ) -> tuple[Any, ...]:
        """Return the static contract which permits workspace reuse.

        Deliberately excludes ``q.shape[0]`` (total query tokens) and
        ``k_cache.shape[0]``/``v_cache.shape[0]`` (page/block count): those
        are exactly the per-call-varying dimensions the fixed-capacity
        workspace built in ``forward()`` is designed to absorb without a
        rebuild. Runtime metadata (the actual per-call values) is guarded
        separately by object identity in ``_prepared_metadata``.
        """
        return (
            mode,
            q.device,
            q.dtype,
            int(q.shape[1]),
            int(q.shape[2]),
            k_cache.dtype,
            int(k_cache.shape[1]),
            int(k_cache.shape[2]),
            int(k_cache.shape[3]),
            v_cache.dtype,
            int(v_cache.shape[3]),
            int(window_left),
        )

    def _work_item_capacity(
        self,
        *,
        mode: str,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        num_q_heads: int,
        num_kv_heads: int,
        window_left: int,
    ) -> tuple[int, int]:
        """Return ``(max_work_items, max_partial_rows)`` for a fixed-capacity
        workspace serving ``mode`` at this group's declared capacity.

        ``extend``/``decode``: sparkinfer's own eager-mode estimator, scaled
        off ``max_total_q`` -- correct because extend/decode's real
        work-item count tracks total query tokens (a plain one-token decode
        call is comfortably inside a budget sized for up to
        ``max_total_q`` extend tokens). ``max_partial_rows`` is always 0
        here: matches ``PagedExtendGraphCapacity``, which has no
        ``max_partial_rows`` field at all (no split-KV merge buffer for
        these contracts).

        ``verify``: ``max_total_q`` is not a valid proxy -- see the class
        docstring's second bug.

        First attempt at the real fix (superseded, kept here as a documented
        dead end -- see notes/2026-08-01-c1-c2-gpu-investigation.md's
        follow-up): reuse ``planner.plan_verify_graph_capacity``, on the
        theory that it is "the same capacity math LagunaCudaGraphVerify
        already trusts". Measured wrong on real GPU: at a perfectly ordinary
        shape (kv_len~2000, 16-token window) the REAL eager plan
        (``create_paged_plan(enable_cuda_graph=False, mode="verify", ...)``)
        needed ``work_items=96, partial_rows=256``, while
        ``plan_verify_graph_capacity`` predicted only ``47``/``112`` for the
        same group at its declared max capacity. Root cause:
        ``plan_verify_graph_capacity`` computes a schedule for the OTHER
        execution mode -- one fixed, capture-static chunking policy that
        must stay valid for every possible replay length under CUDA Graph
        capture. The eager path computes a fresh, shape-specific schedule
        per call (that is the whole point of not being graph-captured), and
        that schedule can legitimately need MORE work items for the same
        bounds than the graph policy's "worst case" -- the two modes are not
        interchangeable capacity sources despite both being sparkinfer's own
        code.

        Actual fix: run the real eager planner itself
        (``create_paged_plan(enable_cuda_graph=False, mode="verify", ...)``,
        the exact function every real call below will use) once, up front,
        against a synthetic worst-case call at this group's own declared
        max capacity (``num_cache_pages`` full pages, ``query_len`` at the
        caller-declared ``declare_verify_capacity()`` bound) -- the same
        "build the real max-capacity plan, then trust its numbers" recipe
        ``LagunaCudaGraphVerify``/``DFlashDraftCudaGraph`` already use
        (``max_kv = max_pages * block_size - 1``), just read directly
        instead of discovered via ``_ensure_capacity``'s auto-grow (which
        eager's ``for_fixed_capacity`` workspace does not get, by design --
        it must hard-fail on any later underestimate, not silently grow).
        Confirmed monotonically increasing with kv_len on real GPU (full
        attention: 6 work items at kv_len=0 -> 12288 at max kv_len=262127),
        so sizing at the max bound is a genuine upper bound, not another
        guess.
        """
        from b12x.attention.paged.workspace import PagedAttentionWorkspace

        if mode in ("extend", "decode"):
            max_work_items = PagedAttentionWorkspace.eager_extend_work_items_capacity(
                max_total_q=self._max_total_q,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
            )
            return max_work_items, 0

        if mode == "verify":
            if self._max_verify_query_len <= 0:
                raise RuntimeError(
                    "SparkinferPrefillWorkspace: mode='verify' requested but no "
                    "verify capacity was declared for this "
                    f"(window_left={window_left}, num_q_heads={num_q_heads}, "
                    f"num_kv_heads={num_kv_heads}) group. Call "
                    "declare_verify_capacity(max_query_len) -- e.g. via "
                    "LagunaBackend.declare_verify_capacity(), which DFlashEngine."
                    "__init__ must call before any mode='verify' traffic can "
                    "reach this workspace -- guessing a capacity here would "
                    "repeat the exact under-provisioning bug this check exists "
                    "to prevent (see notes/2026-08-01-c1-c2-gpu-investigation.md)."
                )
            from b12x.attention.paged.planner import create_paged_plan

            num_cache_pages = int(k_cache.shape[0])
            page_size = int(k_cache.shape[1])
            max_kv = max(num_cache_pages * page_size - 1, 1)
            worst_page_table = torch.arange(
                num_cache_pages, dtype=torch.int32, device=q.device
            ).unsqueeze(0)
            worst_cache_seqlens = torch.tensor([max_kv], dtype=torch.int32, device=q.device)
            worst_cu_seqlens_q = torch.tensor(
                [0, self._max_verify_query_len], dtype=torch.int32, device=q.device
            )
            worst_q = torch.empty(
                self._max_verify_query_len,
                num_q_heads,
                int(q.shape[2]),
                dtype=q.dtype,
                device=q.device,
            )
            worst_plan = create_paged_plan(
                worst_q,
                k_cache,
                v_cache,
                worst_page_table,
                worst_cache_seqlens,
                worst_cu_seqlens_q,
                mode="verify",
                enable_cuda_graph=False,
                window_left=window_left,
            )
            max_partial_rows = int(worst_plan.total_num_partial_rows) if worst_plan.split_kv else 0
            return int(worst_plan.new_batch_size), max_partial_rows

        raise ValueError(f"SparkinferPrefillWorkspace: unknown mode {mode!r}")

    def forward(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        output: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        window_left: int = -1,
        k_descale: torch.Tensor | None = None,
        v_descale: torch.Tensor | None = None,
        mode: str = "extend",
        plan_cache_key: object | None = None,
    ) -> None:
        """Run extend/verify-mode attention (prefill or speculative verify)."""
        from b12x.attention.paged._forward import paged_attention_forward
        from b12x.attention.paged._scratch import build_paged_attention_binding
        from b12x.attention.paged.planner import create_paged_plan
        from b12x.attention.paged.workspace import PagedAttentionWorkspace

        if k_descale is None:
            k_descale = self._descale
        if v_descale is None:
            v_descale = self._descale

        workspace_key = self._key(
            mode=mode,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            window_left=window_left,
        )
        is_new_workspace = self._workspace_key != workspace_key
        if is_new_workspace:
            num_q_heads = int(q.shape[1])
            num_kv_heads = int(k_cache.shape[2])
            max_work_items, max_partial_rows = self._work_item_capacity(
                mode=mode,
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                window_left=window_left,
            )
            logger.info(
                "SparkinferPrefillWorkspace: new fixed-capacity contract "
                "mode=%s window_left=%d q_heads=%d kv_heads=%d "
                "max_total_q=%d max_page_table_width=%d max_work_items=%d "
                "max_partial_rows=%d -- the *next* paged_attention_forward "
                "call below pays sparkinfer's one-time CuTe compile for this "
                "(mode, window_left); every later call at any shape within "
                "this capacity reuses it (and it stays warm across process "
                "restarts via sparkinfer's own on-disk cache). See "
                "notes/2026-08-01-prefill-shape-buckets-root-cause.md and "
                "notes/2026-08-01-c1-c2-gpu-investigation.md.",
                mode,
                window_left,
                num_q_heads,
                num_kv_heads,
                self._max_total_q,
                self._max_page_table_width,
                max_work_items,
                max_partial_rows,
            )
            self._workspace = PagedAttentionWorkspace.for_fixed_capacity(
                mode=mode,
                device=q.device,
                dtype=q.dtype,
                kv_dtype=k_cache.dtype,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=int(q.shape[2]),
                head_dim_vo=int(v_cache.shape[3]),
                page_size=int(k_cache.shape[1]),
                max_total_q=self._max_total_q,
                max_batch=self._max_batch,
                max_page_table_width=self._max_page_table_width,
                max_work_items=max_work_items,
                max_partial_rows=max_partial_rows,
                num_cache_pages=int(k_cache.shape[0]),
                use_cuda_graph=False,
            )
            self._workspace_key = workspace_key
            self._prepared_metadata = None

        ws = self._workspace
        assert ws is not None
        if plan_cache_key is None or self._prepared_metadata is not plan_cache_key:
            plan = create_paged_plan(
                q,
                k_cache,
                v_cache,
                page_table,
                cache_seqlens,
                cu_seqlens_q,
                mode=mode,
                enable_cuda_graph=False,
                window_left=window_left,
                plan_budget=self._extend_plan_budget if mode == "extend" else None,
            )
            ws._ensure_capacity(plan)
            ws._copy_runtime_metadata(page_table, cache_seqlens, cu_seqlens_q)
            ws._copy_plan_metadata(plan)
            ws._plan = plan
            self._prepared_metadata = plan_cache_key

        binding = build_paged_attention_binding(
            scratch=ws,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            output=output,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        if is_new_workspace:
            t0 = time.perf_counter()
            paged_attention_forward(binding=binding)
            if q.device.type == "cuda":
                torch.cuda.synchronize(q.device)
            elapsed = time.perf_counter() - t0
            log = logger.warning if elapsed >= _ATTN_COMPILE_WARN_S else logger.info
            log(
                "SparkinferPrefillWorkspace: first paged_attention_forward "
                "for mode=%s window_left=%d took %.3fs (>=%.1fs means a "
                "genuine cold CuTe compile just happened; well under that "
                "means sparkinfer's on-disk cache at ~/.cache/sparkinfer "
                "already had this contract from a prior process). Every "
                "later call against this workspace reuses it with no "
                "further compiles, regardless of real qo_len/kv_len.",
                mode,
                window_left,
                elapsed,
                _ATTN_COMPILE_WARN_S,
            )
        else:
            paged_attention_forward(binding=binding)


class SparkinferDecodeWorkspace:
    """Manages sparkinfer decode-mode workspace for CG replay.

    One per layer group. Captured in CUDA graph. Per-step update is just
    writing cache_seqlens (GPU int32 write) + graph.replay().
    """

    def __init__(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_pages: int,
        window_left: int = -1,
        device: str = "cuda",
        page_size: int = PAGE_SIZE,
    ):
        from b12x.attention.paged.workspace import PagedAttentionWorkspace

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_pages = max_pages
        self.window_left = window_left
        self.device = torch.device(device)
        self.page_size = page_size

        # The planner only consumes shape, dtype, and device for this static
        # contract. ``for_contract`` keeps those K/V views zero-stride until
        # real cache storage is bound immediately before graph capture.
        self._workspace = PagedAttentionWorkspace.for_contract(
            mode="decode",
            device=self.device,
            dtype=torch.bfloat16,
            kv_dtype=torch.float8_e4m3fn,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            head_dim_vo=head_dim,
            page_size=page_size,
            max_total_q=1,
            num_cache_pages=max_pages,
            use_cuda_graph=True,
        )
        assert self._workspace._plan_q is not None
        assert self._workspace._plan_k_cache is not None
        assert self._workspace._plan_v_cache is not None
        assert self._workspace._plan_output is not None
        self._q = self._workspace._plan_q
        self._k_cache = self._workspace._plan_k_cache
        self._v_cache = self._workspace._plan_v_cache
        self._output = self._workspace._plan_output
        self._descale = torch.ones(1, dtype=torch.float32, device=self.device)
        self._k_descale = self._descale
        self._v_descale = self._descale

        # Create graph-mode workspace with prepare_decode_graph_replay_state.
        # Requires sparkinfer commit 0a7b143+ (fixes capacity underestimation
        # for windowed attention with small page counts).
        self._workspace.prepare_decode_graph_replay_state(
            batch=1, max_page_table_width=max_pages, window_left=window_left
        )

        # Bind runtime metadata at max context for capture
        capture_page_table = torch.arange(
            max_pages, dtype=torch.int32, device=self.device
        ).unsqueeze(0)
        capture_cache_seqlens = torch.tensor(
            [max_pages * page_size - 1], dtype=torch.int32, device=self.device
        )
        cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=self.device)
        self._workspace._copy_runtime_metadata(
            capture_page_table, capture_cache_seqlens, cu_seqlens_q
        )

        self._cu_seqlens_q = cu_seqlens_q
        logger.info(
            "SparkinferDecodeWorkspace: q_heads=%d kv_heads=%d head_dim=%d "
            "max_pages=%d window_left=%d",
            num_q_heads,
            num_kv_heads,
            head_dim,
            max_pages,
            window_left,
        )

    def bind_kv(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Bind real tensors (called before CG capture)."""
        self._q = q
        self._k_cache = k_cache
        self._v_cache = v_cache
        self._output = output

    def forward(self) -> torch.Tensor:
        """Run attention (captured in CUDA graph)."""
        from b12x.attention.paged._forward import paged_attention_forward
        from b12x.attention.paged._scratch import build_paged_attention_binding

        binding = build_paged_attention_binding(
            scratch=self._workspace,
            q=self._q,
            k_cache=self._k_cache,
            v_cache=self._v_cache,
            output=self._output,
            k_descale=self._k_descale,
            v_descale=self._v_descale,
        )
        paged_attention_forward(binding=binding)
        return self._output

    @property
    def cache_seqlens(self) -> torch.Tensor:
        return self._workspace.cache_seqlens

    @property
    def page_table(self) -> torch.Tensor:
        return self._workspace.page_table


class SparkinferAttentionImpl:
    """Drop-in replacement for FlashInferImpl on attention layers.

    Reads SparkinferAttnMetadata from forward context and dispatches to
    sparkinfer paged attention. Handles both prefill (extend) and decode.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        window_left: int = -1,
        prefill_workspace: SparkinferPrefillWorkspace | None = None,
        **kwargs,
    ):
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.window_left = window_left
        self.kv_cache_dtype = "fp8_e4m3"
        self.supports_quant_query_input = False
        self._prefill_ws = prefill_workspace

    def _get_prefill_ws(self, device: torch.device) -> SparkinferPrefillWorkspace:
        if self._prefill_ws is None:
            self._prefill_ws = SparkinferPrefillWorkspace(device)
        return self._prefill_ws

    def process_weights_after_loading(self, act_dtype):
        pass

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Write K/V into paged cache (self-contained, zero vLLM dependency).

        kv_cache: [2, num_blocks, block_size, num_kv_heads, head_dim] (uint8/FP8)
        key/value: [num_tokens, num_kv_heads, head_dim] (bf16)
        slot_mapping: [num_tokens] (int64, flat index = block_idx * block_size + block_off)
        """
        k_cache = kv_cache[0].view(torch.float8_e4m3fn)
        v_cache = kv_cache[1].view(torch.float8_e4m3fn)
        fused_kv_scatter(key, value, k_cache, v_cache, slot_mapping, layer._k_scale, layer._v_scale)

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: Any,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None:
            return output.fill_(0)

        num_actual_tokens = attn_metadata.num_actual_tokens
        if num_actual_tokens == 0:
            return output.fill_(0)

        q = query[:num_actual_tokens]
        # KV cache: [2, num_blocks, block_size, num_kv_heads, head_dim]
        key_cache, value_cache = kv_cache.unbind(0)
        # vLLM stores FP8 as uint8; sparkinfer expects float8_e4m3fn
        if key_cache.dtype == torch.uint8:
            key_cache = key_cache.view(torch.float8_e4m3fn)
            value_cache = value_cache.view(torch.float8_e4m3fn)
        # Now: [num_blocks, block_size, num_kv_heads, head_dim] — sparkinfer layout

        # CG decode path: metadata has a workspace attribute
        if hasattr(attn_metadata, "workspace"):
            ws = attn_metadata.workspace
            ws._q = q
            ws._k_cache = key_cache
            ws._v_cache = value_cache
            ws._output = output[:num_actual_tokens]
            ws.forward()
            return output

        batch_size = int(attn_metadata.cache_seqlens.numel())
        num_kv_heads = int(key_cache.shape[2])
        k_descale = _paged_descale(
            layer._k_scale,
            batch_size=batch_size,
            num_kv_heads=num_kv_heads,
        )
        v_descale = _paged_descale(
            layer._v_scale,
            batch_size=batch_size,
            num_kv_heads=num_kv_heads,
        )

        # Prefill/extend path: create ephemeral workspace
        ws = self._get_prefill_ws(q.device)
        ws.forward(
            q=q,
            k_cache=key_cache,
            v_cache=value_cache,
            output=output[:num_actual_tokens],
            page_table=attn_metadata.page_table,
            cache_seqlens=attn_metadata.cache_seqlens,
            cu_seqlens_q=attn_metadata.cu_seqlens_q,
            window_left=attn_metadata.window_left,
            k_descale=k_descale,
            v_descale=v_descale,
            mode=getattr(attn_metadata, "mode", "extend"),
            plan_cache_key=attn_metadata,
        )
        return output
