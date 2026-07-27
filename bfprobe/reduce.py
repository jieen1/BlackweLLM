"""T1 signature reduction: one 32-byte fingerprint per intermediate tensor.

A T1 signature answers "did *this* tensor look normal" for ~0 cost: absmax,
L2 norm, mean, a NaN count, and an Inf count, plus the element count needed
to interpret ``mean``/``l2``. See
notes/2026-07-27-bfprobe-t1-signatures.md for the full design writeup; the
short version:

Why NaN/Inf are separate counters, not folded into absmax/l2/mean
------------------------------------------------------------------
``absmax``/``l2``/``mean`` are computed the naive way over *every* element,
with no masking pass. If the tensor contains a NaN, IEEE-754 propagation
means ``absmax``/``l2``/``mean`` will themselves come out NaN -- and that is
fine, because ``bfprobe.baseline.judge`` treats ``nan_count > 0`` (or
``inf_count > 0``) as immediately, unconditionally out-of-band, before it
ever looks at the other three fields. Masking NaN/Inf out of the other three
reductions would need a second pass (or a fused compare-and-select per
element, which is what this module actually does inside the kernel/reference
loop -- see below); simply *not* masking is both simpler and just as
informative, since the verdict never depends on the poisoned values anyway.

Why this must be exactly one kernel launch per tensor
--------------------------------------------------------
A round of DFlash touches 48 layers x 4 taps = 192 tensors needing a
signature (see notes/2026-07-27-probe-system-design-and-plan.md section 4).
Computing the five reduced quantities (absmax, sum-of-squares, sum, NaN
count, Inf count) as five separate ``torch`` reduction calls would be five
kernel launches each, i.e. 960 launches/round -- a measurable fraction of a
44.16ms round. This module instead does the whole reduction, all five
quantities, in a single pass:

* ``reduce_reference`` -- a CPU-only, torch-CPU reference implementation.
  Multiple ``torch`` ops are used here (correctness reference; CPU kernel
  launch counting is not the concern this file's design constraint is about
  -- see the module docstring's "why one kernel" section, which is about the
  *GPU* path only).
* ``_signature_reduce_kernel`` -- a Triton kernel that does the reduction in
  one launch: a single program instance loops over the flattened tensor in
  ``BLOCK_SIZE`` chunks, accumulating sum, sum-of-squares, running absmax,
  NaN count, and Inf count together in registers, and writes all of them out
  at the end. This mirrors the existing single-pass style already used in
  this repo (``runtime/triton_norm_ops.py``'s ``_rms_norm_triton_kernel``,
  ``runtime/kernels/fused_rms_norm.py``'s ``_fused_add_rms_norm_kernel``):
  one program, one or more serial loops over the row/tensor, values kept in
  registers across the whole pass.

Hard constraint honored throughout this file: nothing here may call
``.item()``, ``.cpu()``, or ``torch.cuda.synchronize()`` on a live reduction
path. ``reduce_gpu`` returns a GPU-resident tensor of raw accumulators;
turning that into a host-side ``Signature`` (``finalize_signature``) is a
separate, explicitly off-hot-path step, done only at ring-dump time.

GPU-untested code path
-----------------------
This whole repository is being developed with zero GPU access (see the
task's hard constraints). ``_signature_reduce_kernel`` and ``reduce_gpu`` are
written to the best of the author's understanding of Triton and this repo's
existing kernel style, but have **never been run**. Only ``reduce_reference``
(pure CPU) and the pure-Python shape/index helpers below are covered by
tests. See notes/2026-07-27-bfprobe-t1-signatures.md's GPU-verification
checklist.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from torch import Tensor

#: Raw accumulator field order written by the GPU kernel and consumed by
#: ``finalize_signature``: ``[sum, sum_sq, absmax, nan_count, inf_count]``.
#: ``mean``/``l2``/``numel`` are derived off-device from these plus the
#: (statically known, no-launch-required) element count.
RAW_ACCUMULATOR_FIELDS: tuple[str, str, str, str, str] = (
    "sum",
    "sum_sq",
    "absmax",
    "nan_count",
    "inf_count",
)
NUM_RAW_ACCUMULATORS = len(RAW_ACCUMULATOR_FIELDS)


@dataclass(frozen=True)
class Signature:
    """A 32-byte-class fingerprint of one tensor.

    Field layout matches the task's contract exactly (6 fields; the actual
    in-memory footprint in the ring is float64 absmax/l2/mean + int64
    nan_count/inf_count/numel = 48 bytes per the host backend used today, or
    the packed 32-byte layout once a GPU-resident backend replaces it --
    see bfprobe/signature.py).
    """

    absmax: float
    l2: float
    mean: float
    nan_count: int
    inf_count: int
    numel: int


# ---------------------------------------------------------------------------
# CPU reference implementation -- the correctness ground truth.
# ---------------------------------------------------------------------------


def reduce_reference(tensor: Tensor) -> Signature:
    """Pure CPU reference reduction. Ground truth for the GPU kernel's semantics.

    Accumulates in float32 regardless of input dtype (matching what the GPU
    kernel does when loading e.g. bf16 elements -- see
    ``_signature_reduce_kernel``), so a bf16 input's signature reflects
    float32-accumulated statistics, not raw bf16 arithmetic. NaN/Inf are
    counted separately but are *not* masked out of absmax/l2/mean -- see this
    module's docstring for why that is the intended, single-pass-friendly
    behavior.

    Never touches a GPU: callers are responsible for only ever passing a
    CPU-resident tensor to this function in this repo's current environment
    (no CUDA access). ``tensor`` may be any shape; it is flattened first.
    """
    flat = tensor.reshape(-1).to(torch.float32)
    numel = int(flat.numel())
    if numel == 0:
        return Signature(absmax=0.0, l2=0.0, mean=0.0, nan_count=0, inf_count=0, numel=0)

    nan_count = int(torch.isnan(flat).sum().item())
    inf_count = int(torch.isinf(flat).sum().item())
    absmax = float(flat.abs().max().item())
    l2 = float(flat.pow(2).sum().sqrt().item())
    mean = float(flat.mean().item())
    return Signature(
        absmax=absmax, l2=l2, mean=mean, nan_count=nan_count, inf_count=inf_count, numel=numel
    )


def finalize_signature(raw: tuple[float, float, float, float, float], numel: int) -> Signature:
    """Turn GPU raw accumulators ``(sum, sum_sq, absmax, nan_count,
    inf_count)`` (already read back to host floats -- the one place a
    ``.item()``-equivalent is expected, and only at dump time) into a
    ``Signature``. Pure function, no I/O, no device access; the only reason
    this is split out from ``reduce_gpu`` is so it's testable without a GPU.
    """
    total, sum_sq, absmax, nan_count, inf_count = raw
    if numel == 0:
        return Signature(absmax=0.0, l2=0.0, mean=0.0, nan_count=0, inf_count=0, numel=0)
    mean = total / numel
    l2 = sum_sq**0.5
    return Signature(
        absmax=absmax,
        l2=l2,
        mean=mean,
        nan_count=int(nan_count),
        inf_count=int(inf_count),
        numel=numel,
    )


# ---------------------------------------------------------------------------
# Pure-Python shape/index derivation -- unit-testable without running the
# Triton kernel itself (see this module's "GPU-untested code path" note).
# ---------------------------------------------------------------------------

#: Cap chosen to match the existing repo convention in
#: ``runtime/triton_norm_ops.py`` (``min(triton.next_power_of_2(n_cols),
#: 4096)``): large enough to cover one decode-round tap in a single block
#: (hidden state row 3072, router logits 256, up to a few thousand elements),
#: small enough to keep register pressure per program bounded for the
#: single-program design (see this module's docstring).
MAX_BLOCK_SIZE = 4096


def next_power_of_2(value: int) -> int:
    """Smallest power of 2 that is >= ``value`` (``value >= 1``)."""
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def choose_block_size(numel: int, *, max_block_size: int = MAX_BLOCK_SIZE) -> int:
    """BLOCK_SIZE for ``_signature_reduce_kernel``'s single program.

    Mirrors ``runtime/triton_norm_ops.py``'s ``_triton_rms_norm`` sizing
    policy: round up to the next power of 2, capped at ``max_block_size`` so
    a single tensor's reduction never demands an unreasonable number of
    registers/shared memory per program.
    """
    if numel <= 0:
        return 1
    return min(next_power_of_2(numel), max_block_size)


def num_chunks(numel: int, block_size: int) -> int:
    """Number of ``BLOCK_SIZE``-sized serial loop iterations the single
    program in ``_signature_reduce_kernel`` will execute to cover ``numel``
    elements. Exposed standalone so the loop trip count can be asserted by a
    test without launching the kernel."""
    if numel <= 0 or block_size <= 0:
        return 0
    return (numel + block_size - 1) // block_size


# ---------------------------------------------------------------------------
# Triton kernel -- single program, single launch, single pass. Never
# executed in this environment (no GPU access); see module docstring.
# ---------------------------------------------------------------------------


@triton.jit
def _signature_reduce_kernel(
    x_ptr,
    out_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    """Reduce a flattened 1D tensor to ``[sum, sum_sq, absmax, nan_count,
    inf_count]`` in one program, one launch (``grid=(1,)`` -- see
    ``reduce_gpu``). All five accumulators are carried in registers across
    every chunk of the serial loop and written to ``out_ptr`` exactly once,
    at the very end -- there is nothing to synchronize across programs
    because there is exactly one program.

    This intentionally trades cross-SM parallelism for launch-count
    simplicity: correct for the decode-round tap sizes this package targets
    (thousands of elements per tap, see
    notes/2026-07-27-probe-system-design-and-plan.md section 4's per-tensor
    byte counts), but not appropriate as-is for a prefill-scale tensor
    (tens/hundreds of millions of elements) -- a multi-program,
    atomic-accumulation variant would be needed there. See
    notes/2026-07-27-bfprobe-t1-signatures.md's GPU-verification checklist.
    """
    sum_acc = tl.zeros([1], dtype=tl.float32)
    sum_sq_acc = tl.zeros([1], dtype=tl.float32)
    absmax_acc = tl.zeros([1], dtype=tl.float32)
    nan_acc = tl.zeros([1], dtype=tl.float32)
    inf_acc = tl.zeros([1], dtype=tl.float32)

    for off in range(0, numel, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < numel
        # `other=0.0` is safe for sum/sum_sq/absmax (a masked-out lane
        # contributes nothing to any of them) but must not be mistaken for
        # a real zero when checking is-nan/is-inf below -- those checks are
        # gated by `mask` explicitly via `tl.where`, not via `other`.
        x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)

        is_nan = x != x  # NaN is the only float that compares unequal to itself.
        is_inf = (x == float("inf")) | (x == float("-inf"))

        sum_acc += tl.sum(tl.where(mask, x, 0.0))
        sum_sq_acc += tl.sum(tl.where(mask, x * x, 0.0))
        # A NaN/Inf lane must not corrupt the running absmax via `tl.maximum`
        # propagation, so it is excluded from the absmax candidate the same
        # way a masked-out lane is (both fall back to 0.0).
        finite_abs = tl.where(mask & ~(is_nan | is_inf), tl.abs(x), 0.0)
        absmax_acc = tl.maximum(absmax_acc, tl.max(finite_abs))
        nan_acc += tl.sum(tl.where(mask & is_nan, 1.0, 0.0))
        inf_acc += tl.sum(tl.where(mask & is_inf, 1.0, 0.0))

    tl.store(out_ptr + 0, sum_acc)
    tl.store(out_ptr + 1, sum_sq_acc)
    tl.store(out_ptr + 2, absmax_acc)
    tl.store(out_ptr + 3, nan_acc)
    tl.store(out_ptr + 4, inf_acc)


def reduce_gpu(tensor: Tensor, out: Tensor | None = None) -> Tensor:
    """Launch ``_signature_reduce_kernel`` once and return the raw
    accumulator tensor ``[sum, sum_sq, absmax, nan_count, inf_count]``,
    left resident on ``tensor.device``.

    Deliberately does not call ``.item()``/``.cpu()``/
    ``torch.cuda.synchronize()`` -- the result is meant to be written
    straight into a GPU-resident ring slot (a future backend for
    ``bfprobe.signature.SignatureRing``); turning it into a host-side
    ``Signature`` is ``finalize_signature``'s job, done only at dump time.

    Never called in this repo's current environment (no GPU access) -- see
    this module's docstring.
    """
    flat = tensor.reshape(-1)
    numel = int(flat.numel())
    if out is None:
        out = torch.zeros(NUM_RAW_ACCUMULATORS, dtype=torch.float32, device=tensor.device)
    block_size = choose_block_size(numel)
    _signature_reduce_kernel[(1,)](flat, out, numel, BLOCK_SIZE=block_size)
    return out


if __name__ == "__main__":
    demo = torch.tensor([1.0, -2.0, 3.0, float("nan"), float("inf")])
    print(reduce_reference(demo))
