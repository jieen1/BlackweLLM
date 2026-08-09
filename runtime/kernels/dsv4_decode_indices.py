"""CUDA-Graph-safe decode index generation for the DSV4 attention layers.

The eager ``window_topk_idxs`` / ``compress_topk_idxs`` build the per-step
attention index tensors from a Python ``start_pos`` using torch.arange /
cat / pad -- all legal inside CUDA Graph capture (graph-pool allocation),
but the value depends on the position, which the graph must read from a
GPU scalar.  These kernels regenerate the same index layouts from a GPU
``pos`` tensor, so a captured decode graph can advance its position
between replays without recapturing.

swa (window ring): [0..pos] when pos < win-1, else the ring order
    [pos%win+1 .. win-1, 0 .. pos%win], padded with -1 to win.
    Matches ``window_topk_idxs(win, 1, 1, pos)[0]``.
comp (compressed): [0 .. (pos+1)//ratio - 1], length (pos+1)//ratio,
    absolute compressed position (offset=0).  Matches
    ``compress_topk_idxs(ratio, 1, 1, pos, offset=0)``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_swa_indices_kernel(
    pos_ptr,  # [1] int64
    swa_ptr,  # [win] int32 out
    win: tl.constexpr,
):
    i = tl.program_id(0)
    pos = tl.load(pos_ptr)
    # pos < win-1: prefix 0..pos (absolute), then -1.
    # pos >= win-1: ring order v = pos-win+1+i (all >= 0), v % win.
    full = pos >= (win - 1)
    v = pos - win + 1 + i
    val = tl.where(full, v % win, tl.where(i <= pos, i, -1))
    tl.store(swa_ptr + i, val.to(tl.int32))


@triton.jit
def _decode_comp_indices_kernel(
    pos_ptr,  # [1] int64
    comp_ptr,  # [max_comp] int32 out
    n_ptr,  # [1] int32 out (valid count)
    ratio: tl.constexpr,
):
    i = tl.program_id(0)
    pos = tl.load(pos_ptr)
    n = (pos + 1) // ratio
    tl.store(comp_ptr + i, tl.where(i < n, i.to(tl.int32), -1))
    if i == 0:
        tl.store(n_ptr, n.to(tl.int32))


def decode_swa_indices(
    pos: torch.Tensor, window: int, *, device
) -> torch.Tensor:
    """Ring-ordered window indices for decode at ``pos``, [1, window] int32."""
    out = torch.empty((window,), dtype=torch.int32, device=device)
    _decode_swa_indices_kernel[(window,)](pos, out, win=window)
    return out.unsqueeze(0)


def decode_comp_indices(
    pos: torch.Tensor, ratio: int, max_comp: int, *, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compressed-entry indices for decode at ``pos``.

    Returns ([1, max_comp] int32, [1] int32 count); values past the valid
    prefix are -1.
    """
    comp = torch.empty((max_comp,), dtype=torch.int32, device=device)
    n = torch.zeros((1,), dtype=torch.int32, device=device)
    _decode_comp_indices_kernel[(max_comp,)](pos, comp, n, ratio=ratio)
    return comp.unsqueeze(0), n
