"""Unit tests for bfprobe/reduce.py -- CPU-only.

Per the task's hard constraints, this repo's current environment has no GPU
access: ``_signature_reduce_kernel``/``reduce_gpu`` are never invoked here.
What *is* covered:

1. ``reduce_reference`` correctness against hand-computed numpy expectations
   (NaN/Inf/all-zero/extreme-value/bf16-precision cases).
2. The pure-Python shape/index helpers (``choose_block_size``,
   ``num_chunks``, ``next_power_of_2``) that the Triton kernel's grid/loop
   derivation depends on -- these stand in for "the kernel's shape/index
   derivation has test coverage" since the kernel body itself cannot be run.
3. ``finalize_signature``, the pure function that turns GPU raw accumulators
   into a ``Signature`` -- exercised directly with synthetic accumulator
   tuples, no device involved.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bfprobe.reduce import (  # noqa: E402
    Signature,
    choose_block_size,
    finalize_signature,
    next_power_of_2,
    num_chunks,
    reduce_reference,
)


def _numpy_reference(values: list[float]) -> Signature:
    """Independent hand-rolled numpy computation, deliberately not sharing
    any code path with ``reduce_reference``, to cross-check it against."""
    arr = np.asarray(values, dtype=np.float64)
    numel = arr.size
    if numel == 0:
        return Signature(0.0, 0.0, 0.0, 0, 0, 0)
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    # No NaN-skipping: this matches reduce_reference's naive full-tensor
    # reduction (see reduce.py's module docstring on why NaN/Inf are left
    # unmasked in absmax/l2/mean).
    absmax = float(np.abs(arr).max())
    l2 = float(np.sqrt(np.sum(arr * arr)))
    mean = float(np.mean(arr))
    return Signature(absmax, l2, mean, nan_count, inf_count, numel)


class TestReduceReferenceCorrectness:
    def test_simple_values_match_numpy(self):
        values = [1.0, -2.0, 3.0, -4.0, 5.0]
        tensor = torch.tensor(values, dtype=torch.float32)
        got = reduce_reference(tensor)
        want = _numpy_reference(values)
        assert got.absmax == pytest.approx(want.absmax, rel=1e-5)
        assert got.l2 == pytest.approx(want.l2, rel=1e-5)
        assert got.mean == pytest.approx(want.mean, rel=1e-5)
        assert got.nan_count == 0
        assert got.inf_count == 0
        assert got.numel == 5

    def test_multi_dim_tensor_is_flattened(self):
        tensor = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        got = reduce_reference(tensor)
        flat = list(range(24))
        want = _numpy_reference([float(v) for v in flat])
        assert got.absmax == pytest.approx(want.absmax)
        assert got.l2 == pytest.approx(want.l2, rel=1e-5)
        assert got.mean == pytest.approx(want.mean)
        assert got.numel == 24

    def test_all_zero(self):
        tensor = torch.zeros(128, dtype=torch.float32)
        got = reduce_reference(tensor)
        assert got.absmax == 0.0
        assert got.l2 == 0.0
        assert got.mean == 0.0
        assert got.nan_count == 0
        assert got.inf_count == 0
        assert got.numel == 128

    def test_empty_tensor(self):
        tensor = torch.zeros(0, dtype=torch.float32)
        got = reduce_reference(tensor)
        assert got == Signature(0.0, 0.0, 0.0, 0, 0, 0)

    def test_nan_is_counted_and_poisons_reductions(self):
        tensor = torch.tensor([1.0, float("nan"), 3.0])
        got = reduce_reference(tensor)
        assert got.nan_count == 1
        assert got.inf_count == 0
        # NaN propagates through max/sum/mean under IEEE-754 semantics --
        # this is intended (see reduce.py's module docstring): the verdict
        # in bfprobe.baseline.judge short-circuits on nan_count > 0 before
        # ever looking at these poisoned values.
        assert math.isnan(got.absmax)
        assert math.isnan(got.l2)
        assert math.isnan(got.mean)
        assert got.numel == 3

    def test_inf_is_counted_and_poisons_reductions(self):
        tensor = torch.tensor([1.0, float("inf"), -2.0])
        got = reduce_reference(tensor)
        assert got.nan_count == 0
        assert got.inf_count == 1
        assert math.isinf(got.absmax)
        assert got.numel == 3

    def test_nan_and_inf_together(self):
        tensor = torch.tensor([float("nan"), float("inf"), float("-inf"), 1.0])
        got = reduce_reference(tensor)
        assert got.nan_count == 1
        assert got.inf_count == 2
        assert got.numel == 4

    def test_extreme_magnitude_values(self):
        # Large enough to be a meaningful "extreme value" case, but chosen
        # so squaring (for L2) stays within float32's ~3.4e38 max -- see
        # test_fp32_accumulator_overflow_produces_inf_without_inf_count
        # below for what happens past that point.
        values = [1e18, -1e18, 1e-18]
        tensor = torch.tensor(values, dtype=torch.float32)
        got = reduce_reference(tensor)
        want = _numpy_reference(values)
        assert got.absmax == pytest.approx(want.absmax, rel=1e-5)
        assert got.l2 == pytest.approx(want.l2, rel=1e-5)
        assert got.numel == 3

    def test_fp32_accumulator_overflow_produces_inf_without_inf_count(self):
        # A real, worth-documenting edge case: no *element* is Inf (so
        # inf_count stays 0), but squaring 1e30 for the L2 accumulator
        # overflows float32's ~3.4e38 max, so l2 itself comes out `inf`.
        # reduce_reference intentionally accumulates in float32 (matching
        # what the Triton kernel does when loading e.g. bf16/fp16 input --
        # see this module's docstring), so this overflow is expected,
        # reproducible behavior, not a bug -- recorded here so it is never
        # mistaken for one. See notes/2026-07-27-bfprobe-t1-signatures.md.
        tensor = torch.tensor([1e30, -1e30], dtype=torch.float32)
        got = reduce_reference(tensor)
        assert got.inf_count == 0
        assert got.nan_count == 0
        assert math.isinf(got.l2)
        assert got.absmax == pytest.approx(1e30, rel=1e-5)

    def test_bf16_precision_boundary(self):
        # bf16 has a 7-bit mantissa: values that are distinct in fp32 collapse
        # to the same bf16 value. reduce_reference upcasts bf16 -> fp32 before
        # reducing (matching the Triton kernel's `.to(tl.float32)` load), so
        # its output should reflect the *already-rounded* bf16 values, not
        # additional precision loss from the reduction itself.
        raw_fp32 = torch.tensor([1.0, 1.0001, 1.0002, 100000.0], dtype=torch.float32)
        bf16 = raw_fp32.to(torch.bfloat16)
        got = reduce_reference(bf16)
        # Recompute the expected signature from the *rounded* bf16 values
        # (upcast back to fp32, matching what the reference does internally).
        rounded = bf16.to(torch.float32).tolist()
        want = _numpy_reference(rounded)
        assert got.absmax == pytest.approx(want.absmax, rel=1e-6)
        assert got.l2 == pytest.approx(want.l2, rel=1e-6)
        assert got.mean == pytest.approx(want.mean, rel=1e-6)
        assert got.numel == 4
        # bf16 rounding actually happened (values 1.0001/1.0002 are not
        # distinguishable from 1.0 at this exponent) -- confirms the test is
        # exercising real precision loss, not a no-op cast.
        assert rounded[1] == rounded[0] or rounded[2] == rounded[0]

    def test_single_element(self):
        tensor = torch.tensor([-7.5])
        got = reduce_reference(tensor)
        assert got.absmax == pytest.approx(7.5)
        assert got.l2 == pytest.approx(7.5)
        assert got.mean == pytest.approx(-7.5)
        assert got.numel == 1


class TestShapeIndexHelpers:
    """Kernel launch parameters derived in pure Python -- the closest thing
    to "kernel shape/index test coverage" available without a GPU."""

    @pytest.mark.parametrize(
        "value,expected",
        [(0, 1), (1, 1), (2, 2), (3, 4), (4, 4), (5, 8), (4096, 4096), (4097, 8192)],
    )
    def test_next_power_of_2(self, value, expected):
        assert next_power_of_2(value) == expected

    def test_choose_block_size_caps_at_max(self):
        assert choose_block_size(1_000_000, max_block_size=4096) == 4096

    def test_choose_block_size_small_tensor(self):
        # A router-logits tap (16 tokens x 256 experts = 4096) fits one block.
        assert choose_block_size(4096) == 4096
        # A hidden-state tap (16 tokens x 3072 = 49152) needs the cap.
        assert choose_block_size(49152) == 4096

    def test_choose_block_size_non_positive(self):
        assert choose_block_size(0) == 1

    @pytest.mark.parametrize(
        "numel,block_size,expected",
        [(0, 4096, 0), (1, 4096, 1), (4096, 4096, 1), (4097, 4096, 2), (49152, 4096, 12)],
    )
    def test_num_chunks(self, numel, block_size, expected):
        assert num_chunks(numel, block_size) == expected

    def test_block_size_and_num_chunks_cover_whole_tensor(self):
        # The loop `for off in range(0, numel, block_size)` must reach every
        # element exactly once (modulo the final block's mask) -- i.e. the
        # last chunk's start must be < numel and the one after must not exist.
        for numel in (1, 4095, 4096, 4097, 49152, 100_003):
            block_size = choose_block_size(numel)
            chunks = num_chunks(numel, block_size)
            last_chunk_start = (chunks - 1) * block_size
            assert last_chunk_start < numel
            assert chunks * block_size >= numel


class TestFinalizeSignature:
    def test_matches_reference_for_the_same_data(self):
        values = [1.0, -2.0, 3.0, -4.0, 5.0]
        tensor = torch.tensor(values, dtype=torch.float32)
        reference = reduce_reference(tensor)

        raw_sum = float(sum(values))
        raw_sum_sq = float(sum(v * v for v in values))
        raw_absmax = float(max(abs(v) for v in values))
        raw = (raw_sum, raw_sum_sq, raw_absmax, 0.0, 0.0)
        got = finalize_signature(raw, numel=len(values))

        assert got.absmax == pytest.approx(reference.absmax)
        assert got.l2 == pytest.approx(reference.l2)
        assert got.mean == pytest.approx(reference.mean)
        assert got.nan_count == 0
        assert got.inf_count == 0
        assert got.numel == 5

    def test_zero_numel(self):
        got = finalize_signature((0.0, 0.0, 0.0, 0.0, 0.0), numel=0)
        assert got == Signature(0.0, 0.0, 0.0, 0, 0, 0)

    def test_nan_inf_counts_pass_through(self):
        got = finalize_signature((1.0, 1.0, 1.0, 2.0, 3.0), numel=10)
        assert got.nan_count == 2
        assert got.inf_count == 3
