"""Tests for bfdiag.shapes.attention.

Acceptance criterion #1: ``ring_blocks_for_window``'s output must match
``runtime/backends/laguna.py``'s ``_ring_blocks_for_window`` formula
value-for-value, for (window=512, block_size in {64, 128}). Per the task
brief, the real function is deliberately NOT imported here -- doing so would
make the comparison tautological (both sides would literally be the same
code object). Instead the formula is written out a *third* time, inline,
using plain ``math.ceil`` so this test is independent of both
``bfdiag.shapes.attention.ring_blocks_for_window`` (re-derivation #1) and
``runtime.backends.laguna._ring_blocks_for_window`` (the original).
"""

from __future__ import annotations

import math

import pytest

from bfdiag.shapes.attention import (
    full_attention_pages,
    prefill_swa_scratch,
    ring_blocks_for_window,
    swa_alignment,
)


def _independent_ring_blocks_for_window(window: int, block_size: int, qo_max: int) -> int:
    """Written from scratch using math.ceil, not the ``-(-a // b)`` cdiv idiom,
    so a copy-paste-with-refactor bug in either implementation would show up
    as a mismatch here."""
    return math.ceil((window - 1 + qo_max) / block_size) + 1


@pytest.mark.parametrize("block_size", [64, 128])
@pytest.mark.parametrize("qo_max", [1, 16])
def test_ring_blocks_matches_real_formula(block_size, qo_max):
    window = 512
    expected = _independent_ring_blocks_for_window(window, block_size, qo_max)
    actual = ring_blocks_for_window(window, block_size, qo_max=qo_max)
    assert actual == expected


def test_ring_blocks_for_window_512_qo_max_16_known_values():
    """Pin the two values the task brief calls out explicitly as a
    regression guard (the module docstring's "qo_max=1 -> 33, qo_max=16 ->
    34" note in runtime/backends/laguna.py refers to block_size=16; these are
    the real deployment block sizes 64/128)."""
    assert ring_blocks_for_window(512, 64, qo_max=16) == 10
    assert ring_blocks_for_window(512, 128, qo_max=16) == 6


def test_full_attention_pages_cdiv():
    assert full_attention_pages(kv_len=65536, qo_len=1, block_size=64) == math.ceil(65537 / 64)
    assert full_attention_pages(kv_len=65536, qo_len=1, block_size=128) == math.ceil(65537 / 128)
    assert full_attention_pages(kv_len=0, qo_len=1, block_size=64) == 1


@pytest.mark.parametrize(
    "block_size,expected_aligned_start,expected_aligned_len,expected_n_ring",
    [
        (64, 65024, 513, 9),
        (128, 65024, 513, 5),
    ],
)
def test_swa_alignment_decode_known_values(
    block_size, expected_aligned_start, expected_aligned_len, expected_n_ring
):
    """Pinned against an independent hand/CPU computation at kv_len=65536,
    window=512, qo_len=1 (decode) -- see notes/2026-07-27-bfdiag-shape-derivation.md
    for the full derivation. This is also acceptance criterion #3's raw
    material: aligned_len/n_ring do NOT simply halve between block_size=64
    and 128."""
    result = swa_alignment(kv_len=65536, qo_len=1, window=512, block_size=block_size)
    assert result.new_kv_len == 65537
    assert result.window_start == 65025
    assert result.aligned_start == expected_aligned_start
    assert result.aligned_len == expected_aligned_len
    assert result.n_ring == expected_n_ring


def test_swa_alignment_diverges_nontrivially_across_block_size():
    """At kv_len=65600 the two block sizes give genuinely different
    aligned_len (513 vs 577), not just a clean halving -- proof the
    alignment truncation is the actual mechanism (not a red herring)."""
    a64 = swa_alignment(kv_len=65600, qo_len=1, window=512, block_size=64)
    a128 = swa_alignment(kv_len=65600, qo_len=1, window=512, block_size=128)
    assert a64.aligned_len == 513
    assert a128.aligned_len == 577
    assert a64.aligned_len != a128.aligned_len
    # and neither is a simple function of the other (not exactly 2x or 0.5x)
    assert a128.aligned_len != 2 * a64.aligned_len
    assert a128.aligned_len != a64.aligned_len // 2


def test_swa_alignment_verify_caps_with_ring_blocks_per_slot():
    """Mirrors LagunaCudaGraphVerify._fill_buffers's
    n_ring = min(cdiv(aligned_len, bs), ring_blocks_per_slot)."""
    uncapped = swa_alignment(kv_len=65536, qo_len=16, window=512, block_size=64)
    assert uncapped.n_ring == 9  # not capped in this case (cap is 10)
    capped = swa_alignment(
        kv_len=65536, qo_len=16, window=512, block_size=64, ring_blocks_per_slot=3
    )
    assert capped.n_ring == 3
    assert capped.ring_blocks_per_slot == 3


def test_swa_alignment_rejects_nonpositive_window():
    with pytest.raises(ValueError, match="window"):
        swa_alignment(kv_len=10, qo_len=1, window=0, block_size=64)


def test_prefill_swa_scratch_matches_laguna_formula():
    """runtime/backends/laguna.py:305-312:
    swa_scratch_blocks = min(blocks_per_slot, cdiv(window + chunk_tokens, block_size))."""
    scratch = prefill_swa_scratch(
        block_size=64, num_kv_heads=8, head_dim=128, window=512, chunk_tokens=8192
    )
    assert scratch.scratch_blocks == math.ceil((512 + 8192) / 64)
    assert scratch.shape() == (2, scratch.scratch_blocks, 64, 8, 128)

    capped = prefill_swa_scratch(
        block_size=64,
        num_kv_heads=8,
        head_dim=128,
        window=512,
        chunk_tokens=8192,
        blocks_per_slot_cap=10,
    )
    assert capped.scratch_blocks == 10


# ---------------------------------------------------------------------------
# AttentionCallShape / kernel-call shapes
# ---------------------------------------------------------------------------


def test_full_attention_call_shapes():
    from bfdiag.shapes.attention import full_attention_call

    call = full_attention_call(
        label="decode/full",
        num_qo_heads=48,
        num_kv_heads=8,
        head_dim=128,
        block_size=64,
        kv_len=65536,
        qo_len=1,
        batch_size=1,
    )
    shapes = call.shapes()
    assert shapes["q"] == (1, 48, 128)
    assert shapes["k_cache"] == (call.max_pages, 64, 8, 128)
    assert shapes["v_cache"] == (call.max_pages, 64, 8, 128)
    assert shapes["page_table"] == (1, call.max_pages)
    assert shapes["cache_seqlens"] == (1,)


def test_swa_attention_call_shapes_and_diagnostics():
    from bfdiag.shapes.attention import swa_attention_call

    call = swa_attention_call(
        label="decode/sliding",
        num_qo_heads=72,
        num_kv_heads=8,
        head_dim=128,
        block_size=128,
        kv_len=65536,
        window=512,
        qo_len=1,
        batch_size=1,
    )
    assert call.is_swa
    assert call.max_pages == call.swa.n_ring
    assert call.cache_seqlen == call.swa.aligned_len
    shapes = call.shapes()
    assert shapes["q"] == (1, 72, 128)
    assert shapes["k_cache"][1:] == (128, 8, 128)


def test_empty_tensors_shapes_and_dtypes():
    # Everything above this point is pure arithmetic; only empty_tensors()
    # actually allocates, so torch is required from here on but not for the
    # rest of the module.
    torch = pytest.importorskip("torch")

    from bfdiag.shapes.attention import full_attention_call

    call = full_attention_call(
        label="decode/full",
        num_qo_heads=48,
        num_kv_heads=8,
        head_dim=128,
        block_size=64,
        kv_len=100,
        qo_len=1,
        batch_size=2,
    )
    q, k, v, page_table, cache_seqlens = call.empty_tensors()
    assert q.shape == (2, 48, 128)
    assert q.dtype == torch.bfloat16
    assert k.shape == (call.max_pages, 64, 8, 128)
    assert k.dtype == torch.uint8
    assert v.shape == k.shape
    assert page_table.shape == (2, call.max_pages)
    assert page_table.dtype == torch.int32
    assert cache_seqlens.shape == (2,)
    assert cache_seqlens.dtype == torch.int32
    assert q.device.type == "cpu"


def test_empty_tensors_refuses_cuda():
    pytest.importorskip("torch")

    from bfdiag.shapes.attention import full_attention_call

    call = full_attention_call(
        label="decode/full",
        num_qo_heads=48,
        num_kv_heads=8,
        head_dim=128,
        block_size=64,
        kv_len=10,
        qo_len=1,
    )
    with pytest.raises(RuntimeError, match="refuses to allocate"):
        call.empty_tensors(device="cuda")
