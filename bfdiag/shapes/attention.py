"""bfdiag.shapes.attention -- attention shapes, ``block_size`` (page_size) explicit.

Every formula in this module is an *independent* re-derivation of the real
decode/verify/prefill code paths -- it does not import
``runtime.backends.laguna`` or ``runtime.backends.laguna_cuda_graph`` (that
would make the cross-check tests in ``tests/test_bfdiag_shapes_attention.py``
tautological). Sources, for when the real code changes and this drifts:

- ``runtime/backends/laguna.py:48-49`` (``_ring_blocks_for_window``):
  ``cdiv(window - 1 + qo_max, block_size) + 1``.
- ``runtime/backends/laguna_cuda_graph.py``
  (``LagunaCudaGraphDecode._fill_buffers_b1``, decode SWA ring, M=1):
  ``new_kv = kv_len + 1``; ``window_start = max(0, kv_len - window + 1)``;
  ``aligned_start = (window_start // ps) * ps``;
  ``aligned_len = new_kv - aligned_start``; ``n_ring = cdiv(aligned_len, ps)``.
- ``runtime/backends/laguna_cuda_graph.py``
  (``LagunaCudaGraphVerify._fill_buffers``, DFlash M=16 verify): same
  alignment math with ``nt=16`` substituted for the "+1", and
  ``n_ring = min(cdiv(aligned_len, bs), ring_blocks_per_slot)`` (capped by the
  ring's static capacity, since it's a fixed-address CUDA Graph buffer).
- ``runtime/backends/laguna_dflash.py`` (draft KV ring sizing):
  ``_ring_blocks_for_window(DRAFT_WINDOW, block_size, NUM_QUERY_PER_REQ)`` --
  same formula, same ``qo_max``, because ``DRAFT_WINDOW == sliding_window ==
  512`` and ``NUM_QUERY_PER_REQ == SWA_QO_MAX == 16`` happen to coincide.

These files are read-only references for this module -- nothing here edits
``runtime/``.
"""

from __future__ import annotations

from dataclasses import dataclass

from bfdiag.shapes.model import cdiv

SWA_QO_MAX = 16
"""Mirrors ``runtime/backends/laguna.py``'s ``SWA_QO_MAX = 16`` (the DFlash
verify qo_max the main model's SWA ring is sized for)."""


def ring_blocks_for_window(window: int, block_size: int, qo_max: int = SWA_QO_MAX) -> int:
    """Re-derivation of ``runtime/backends/laguna.py:_ring_blocks_for_window``.

    Deliberately NOT imported from there -- see module docstring and
    ``tests/test_bfdiag_shapes_attention.py::test_ring_blocks_matches_real_formula``,
    which recomputes the same formula a *third* time, inline, to catch drift
    between the two independent copies.
    """
    return cdiv(window - 1 + qo_max, block_size) + 1


@dataclass(frozen=True)
class SwaAlignment:
    """The SWA ring-buffer alignment arithmetic for one attention call.

    ``aligned_len`` / ``n_ring`` are exactly the values ``bf shapes --diff``
    exists to compare across block_size -- they do not scale simply (halving
    block_size does not halve them) because ``aligned_start`` rounds *down*
    to a block boundary, and how much "slack" that throws away depends on
    where ``window_start`` falls relative to the block grid.
    """

    kv_len: int
    qo_len: int
    window: int
    block_size: int
    new_kv_len: int
    window_start: int
    aligned_start: int
    aligned_len: int
    n_ring: int
    ring_blocks_per_slot: int | None


def swa_alignment(
    *,
    kv_len: int,
    qo_len: int,
    window: int,
    block_size: int,
    ring_blocks_per_slot: int | None = None,
) -> SwaAlignment:
    """Independent re-derivation of the SWA ring alignment math shared by
    ``LagunaCudaGraphDecode._fill_buffers_b1`` (qo_len=1, uncapped) and
    ``LagunaCudaGraphVerify._fill_buffers`` (qo_len=16, capped by
    ``ring_blocks_per_slot``). Pass ``ring_blocks_per_slot=None`` to match the
    decode path (no cap); pass the static ring capacity to match verify.
    """
    if window <= 0:
        raise ValueError(f"swa_alignment requires window > 0, got {window}")
    new_kv_len = kv_len + qo_len
    window_start = max(0, kv_len - window + 1)
    aligned_start = (window_start // block_size) * block_size
    aligned_len = new_kv_len - aligned_start
    n_ring = cdiv(aligned_len, block_size)
    if ring_blocks_per_slot is not None:
        n_ring = min(n_ring, ring_blocks_per_slot)
    return SwaAlignment(
        kv_len=kv_len,
        qo_len=qo_len,
        window=window,
        block_size=block_size,
        new_kv_len=new_kv_len,
        window_start=window_start,
        aligned_start=aligned_start,
        aligned_len=aligned_len,
        n_ring=n_ring,
        ring_blocks_per_slot=ring_blocks_per_slot,
    )


def full_attention_pages(*, kv_len: int, qo_len: int, block_size: int) -> int:
    """Page count for a full-attention call: ``cdiv(kv_len + qo_len, block_size)``
    (``LagunaCudaGraphDecode``/``LagunaCudaGraphVerify``'s ``n_blocks``/``n_blocks_full``)."""
    return cdiv(kv_len + qo_len, block_size)


@dataclass(frozen=True)
class AttentionCallShape:
    """All tensor shapes for one paged-attention kernel call.

    ``k``/``v`` are the *paged KV cache* tensors (``[max_pages, block_size,
    num_kv_heads, head_dim]``), matching the real SparkInfer kernel calling
    convention (see ``LagunaCudaGraphVerify._init_workspaces``:
    ``k_cache = torch.zeros(max_pages, block_size, nkvh, 128, ...)``) -- not
    just the newly-written KV slice. This is what an isolated kernel
    microbench actually wants to construct and feed to
    ``create_paged_plan``/the paged attention kernel directly.
    """

    label: str  # e.g. "decode/full", "decode/sliding", "verify/sliding", "prefill/full"
    is_swa: bool
    block_size: int
    batch_size: int
    qo_len: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    max_pages: int
    cache_seqlen: int
    window: int | None = None
    swa: SwaAlignment | None = None

    def shapes(self) -> dict[str, tuple[int, ...]]:
        qo_total = self.batch_size * self.qo_len
        return {
            "q": (qo_total, self.num_qo_heads, self.head_dim),
            "k_cache": (self.max_pages, self.block_size, self.num_kv_heads, self.head_dim),
            "v_cache": (self.max_pages, self.block_size, self.num_kv_heads, self.head_dim),
            "page_table": (self.batch_size, self.max_pages),
            "cache_seqlens": (self.batch_size,),
        }

    def empty_tensors(self, *, dtype=None, kv_dtype=None, device: str = "cpu"):
        """Build (q, k_cache, v_cache, page_table, cache_seqlens) as real
        ``torch.empty`` tensors -- CPU only, see ``bfdiag.shapes.harness``.
        """
        import torch

        from bfdiag.shapes.harness import make_empty

        dtype = dtype if dtype is not None else torch.bfloat16
        kv_dtype = kv_dtype if kv_dtype is not None else torch.uint8
        shapes = self.shapes()
        q = make_empty(shapes["q"], dtype=dtype, device=device)
        k = make_empty(shapes["k_cache"], dtype=kv_dtype, device=device)
        v = make_empty(shapes["v_cache"], dtype=kv_dtype, device=device)
        page_table = make_empty(shapes["page_table"], dtype=torch.int32, device=device)
        cache_seqlens = make_empty(shapes["cache_seqlens"], dtype=torch.int32, device=device)
        return q, k, v, page_table, cache_seqlens


def full_attention_call(
    *,
    label: str,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    kv_len: int,
    qo_len: int = 1,
    batch_size: int = 1,
) -> AttentionCallShape:
    max_pages = full_attention_pages(kv_len=kv_len, qo_len=qo_len, block_size=block_size)
    return AttentionCallShape(
        label=label,
        is_swa=False,
        block_size=block_size,
        batch_size=batch_size,
        qo_len=qo_len,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_pages=max_pages,
        cache_seqlen=kv_len + qo_len,
        window=None,
        swa=None,
    )


def swa_attention_call(
    *,
    label: str,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    kv_len: int,
    window: int,
    qo_len: int = 1,
    batch_size: int = 1,
    ring_blocks_per_slot: int | None = None,
) -> AttentionCallShape:
    swa = swa_alignment(
        kv_len=kv_len,
        qo_len=qo_len,
        window=window,
        block_size=block_size,
        ring_blocks_per_slot=ring_blocks_per_slot,
    )
    return AttentionCallShape(
        label=label,
        is_swa=True,
        block_size=block_size,
        batch_size=batch_size,
        qo_len=qo_len,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_pages=swa.n_ring,
        cache_seqlen=swa.aligned_len,
        window=window,
        swa=swa,
    )


@dataclass(frozen=True)
class PrefillSwaScratch:
    """The SWA prefill scratch buffer (``LagunaBackend._swa_scratch``):
    ``shape = (2, swa_scratch_blocks, block_size, num_kv_heads, head_dim)``
    with ``swa_scratch_blocks = min(blocks_per_slot, cdiv(window + chunk_tokens,
    block_size))`` (``runtime/backends/laguna.py:305-312``). Persistent,
    allocated once, reused across slots and chunks."""

    block_size: int
    num_kv_heads: int
    head_dim: int
    window: int
    chunk_tokens: int
    blocks_per_slot_cap: int | None
    scratch_blocks: int

    def shape(self) -> tuple[int, int, int, int, int]:
        return (2, self.scratch_blocks, self.block_size, self.num_kv_heads, self.head_dim)


def prefill_swa_scratch(
    *,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    window: int,
    chunk_tokens: int,
    blocks_per_slot_cap: int | None = None,
) -> PrefillSwaScratch:
    needed = cdiv(window + chunk_tokens, block_size)
    scratch_blocks = needed if blocks_per_slot_cap is None else min(blocks_per_slot_cap, needed)
    return PrefillSwaScratch(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        window=window,
        chunk_tokens=chunk_tokens,
        blocks_per_slot_cap=blocks_per_slot_cap,
        scratch_blocks=scratch_blocks,
    )


def kv_cache_shape(
    *,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    num_blocks: int,
) -> tuple[int, int, int, int, int]:
    """The persistent, statically-allocated KV cache tensor:
    ``[2, num_blocks, block_size, num_kv_heads, head_dim]`` (dim 0: 0=K, 1=V).
    ``num_blocks`` is ``num_phys_slots * blocks_per_slot`` (full attention) or
    ``num_phys_slots * ring_blocks_for_window(...)`` (SWA) -- a *capacity*
    the caller must supply explicitly (this module does not invent a default
    context budget; see ``bfdiag.shapes.cli`` for the illustrative default
    used only for ``bf shapes``' printed table).
    """
    return (2, num_blocks, block_size, num_kv_heads, head_dim)
