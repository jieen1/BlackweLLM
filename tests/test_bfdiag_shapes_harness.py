"""Tests for bfdiag.shapes.harness: real torch tensor construction, CPU-only.

Hard safety rule under test: this box has one GPU under active use right
now for a real debugging session, so any attempt to allocate on CUDA must
raise immediately rather than silently allocating (or silently falling back
to CPU, which would hide the mistake instead of surfacing it).
"""

from __future__ import annotations

# Optional torch is intentionally a collection-time skip in CPU-only CI.
# ruff: noqa: E402, I001

import pytest

torch = pytest.importorskip("torch")

from bfdiag.shapes.harness import (
    empty_from_shapes,
    make_empty,
    make_randn,
    make_tensor,
    make_zeros,
)


def test_make_empty_default_cpu_bfloat16():
    t = make_empty((2, 3))
    assert t.shape == (2, 3)
    assert t.dtype == torch.bfloat16
    assert t.device.type == "cpu"


def test_make_empty_dtype_override():
    t = make_empty((4,), dtype=torch.int32)
    assert t.dtype == torch.int32


def test_make_randn_uint8_and_int32_fall_back_to_randint():
    t = make_randn((8,), dtype=torch.uint8)
    assert t.dtype == torch.uint8
    assert (t >= 0).all() and (t <= 255).all()
    t2 = make_randn((8,), dtype=torch.int32)
    assert t2.dtype == torch.int32


def test_make_randn_float8_falls_back_via_float32():
    t = make_randn((4,), dtype=torch.float8_e4m3fn)
    assert t.dtype == torch.float8_e4m3fn


def test_make_zeros():
    t = make_zeros((2, 2), dtype=torch.bfloat16)
    assert torch.equal(t, torch.zeros(2, 2, dtype=torch.bfloat16))


def test_make_tensor_dispatch():
    assert make_tensor((2,), fill="empty").shape == (2,)
    assert make_tensor((2,), fill="zeros").shape == (2,)
    assert make_tensor((2,), fill="randn").shape == (2,)
    with pytest.raises(ValueError, match="fill must be one of"):
        make_tensor((2,), fill="bogus")


def test_empty_from_shapes_with_dtype_overrides():
    shapes = {"q": (1, 2, 3), "page_table": (1, 4), "cache_seqlens": (1,)}
    tensors = empty_from_shapes(
        shapes,
        dtype=torch.bfloat16,
        dtype_overrides={"page_table": torch.int32, "cache_seqlens": torch.int32},
    )
    assert tensors["q"].dtype == torch.bfloat16
    assert tensors["page_table"].dtype == torch.int32
    assert tensors["cache_seqlens"].dtype == torch.int32
    assert tensors["q"].shape == (1, 2, 3)


@pytest.mark.parametrize("fn", [make_empty, make_randn, make_zeros])
def test_cuda_device_refused_by_default(fn):
    with pytest.raises(RuntimeError, match="refuses to allocate"):
        fn((2, 2), device="cuda")


def test_cuda_allowed_only_with_explicit_env_opt_in(monkeypatch):
    """Confirms the guard is opt-in via env var, not something this module
    would ever set itself -- and that we don't actually need a GPU for this
    test to run (the guard check happens before any CUDA allocation call)."""
    monkeypatch.setenv("BF_SHAPES_ALLOW_CUDA", "1")
    # We assert only that the RuntimeError guard is bypassed; whether torch
    # can actually allocate on "cuda" depends on real hardware/driver state,
    # which this test suite must not depend on. So we use a device string
    # that resolves fine on CPU-only torch semantics for the guard check.
    from bfdiag.shapes.harness import _resolve_device

    # No RuntimeError should be raised now that the opt-in is set.
    dev = _resolve_device("cuda")
    assert dev.type == "cuda"
