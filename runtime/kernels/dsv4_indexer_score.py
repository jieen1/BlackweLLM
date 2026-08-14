"""Standalone Triton scorer for the DSV4 ratio-4 indexer long-context scan.

This scorer covers decode and prefill under one narrow contract:

- q: [1, R, 64, 128] bf16 on CUDA, where ``1 <= R <= 32``
- kv: [N, 128] bf16 on the same CUDA device
- weights: [1, R, 64] bf16 on the same CUDA device
- out: [1, R, N] bf16 on the same CUDA device

Semantics:

``score[r, n] = sum_h(relu(dot(q[r, h], kv[n])) * weights[r, h])``

Production round points matter here: the eager expression returns bf16 from the
einsum, keeps bf16 through ReLU and the weight multiply, then returns bf16 from
the cross-head reduction. This is only the scorer: it does not perform top-k
selection.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

_BATCH = 1
_SEQ = 1
_HEADS = 64
_HEAD_DIM = 128
_MAX_ROWS = 32
_ALLOWED_BATCH_SIZES = (1, 2, 4)
_DEFAULT_BLOCK_N = 64
_DEFAULT_NUM_WARPS = 4
_DEFAULT_NUM_STAGES = 3


@triton.jit
def _dsv4_indexer_score_kernel(
    q_ptr,
    kv_ptr,
    weights_ptr,
    out_ptr,
    R,
    q_row_stride,
    weights_row_stride,
    out_row_stride,
    N,
    BLOCK_N: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """One CTA scores one query row against ``BLOCK_N`` compressed-KV rows."""
    pid_r = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_r >= R:
        return
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_h = tl.arange(0, HEADS)
    n_mask = offs_n < N

    q_base = pid_r * q_row_stride
    q_tile = tl.load(q_ptr + q_base + offs_h[:, None] * HEAD_DIM + offs_d[None, :]).to(
        tl.float32
    )
    kv_tile = tl.load(
        kv_ptr + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
        mask=n_mask[:, None],
        other=0,
    ).to(tl.float32)
    dots = tl.dot(
        q_tile,
        tl.trans(kv_tile),
        input_precision="ieee",
        out_dtype=tl.float32,
    )
    weights_base = pid_r * weights_row_stride
    weights = tl.load(weights_ptr + weights_base + offs_h).to(tl.float32)
    # Match the reference einsum exactly: relu, weight multiply, and the
    # head sum all stay in fp32; only the stored score rounds to bf16.
    relu_dots = tl.maximum(dots, 0)
    weighted = relu_dots * weights[:, None]
    scores = tl.sum(weighted, axis=0)
    out_base = pid_r * out_row_stride
    tl.store(out_ptr + out_base + offs_n, scores, mask=n_mask)


@triton.jit
def _dsv4_indexer_score_batch_kernel(
    q_ptr,
    kv_ptr,
    weights_ptr,
    slot_ids_ptr,
    out_ptr,
    N,
    q_batch_stride,
    kv_slot_stride,
    weights_batch_stride,
    out_batch_stride,
    BLOCK_N: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """One CTA scores one batch row against ``BLOCK_N`` rows of its selected slot."""
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_h = tl.arange(0, HEADS)
    n_mask = offs_n < N

    q_base = pid_b * q_batch_stride
    q_tile = tl.load(q_ptr + q_base + offs_h[:, None] * HEAD_DIM + offs_d[None, :]).to(
        tl.float32
    )
    slot = tl.load(slot_ids_ptr + pid_b)
    kv_base = slot * kv_slot_stride
    kv_tile = tl.load(
        kv_ptr + kv_base + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
        mask=n_mask[:, None],
        other=0,
    ).to(tl.float32)
    dots = tl.dot(
        q_tile,
        tl.trans(kv_tile),
        input_precision="ieee",
        out_dtype=tl.float32,
    )
    weights_base = pid_b * weights_batch_stride
    weights = tl.load(weights_ptr + weights_base + offs_h).to(tl.float32)
    # Match the reference einsum exactly: relu, weight multiply, and the head
    # sum stay in fp32; only the stored score rounds to bf16.
    relu_dots = tl.maximum(dots, 0)
    weighted = relu_dots * weights[:, None]
    scores = tl.sum(weighted, axis=0).to(tl.bfloat16)
    out_base = pid_b * out_batch_stride
    tl.store(out_ptr + out_base + offs_n, scores, mask=n_mask)


def _is_contiguous_exact(tensor: torch.Tensor) -> bool:
    # A slot view into the shared arena can be contiguous with a non-zero
    # storage offset.  Triton receives the already-offset data pointer, so
    # rejecting that valid view would break B1 graphs bound to slot > 0.
    return tensor.is_contiguous()


def _check_dsv4_indexer_score_contract(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    out: torch.Tensor | None = None,
) -> tuple[int, int]:
    """Validate the exact standalone scorer contract and return ``(R, N)``."""
    if q.device.type != "cuda" or kv.device.type != "cuda" or weights.device.type != "cuda":
        raise ValueError("dsv4_indexer_score requires CUDA q/kv/weights tensors")
    if q.device != kv.device or q.device != weights.device:
        raise ValueError("dsv4_indexer_score requires q/kv/weights on one CUDA device")
    if q.ndim != 4 or q.shape[0] != _BATCH or q.shape[2:] != (_HEADS, _HEAD_DIM):
        raise ValueError(
            f"dsv4_indexer_score requires q shaped [1, R, 64, 128], got {tuple(q.shape)}"
        )
    n_rows = int(q.shape[1])
    if not 1 <= n_rows <= _MAX_ROWS:
        raise ValueError(
            f"dsv4_indexer_score requires q rows R in [1, {_MAX_ROWS}], got {n_rows}"
        )
    if q.dtype != torch.bfloat16:
        raise ValueError(f"dsv4_indexer_score requires q bf16, got {q.dtype}")
    if not _is_contiguous_exact(q):
        raise ValueError("dsv4_indexer_score requires q to be contiguous")

    if kv.ndim != 2 or kv.shape[1] != _HEAD_DIM:
        raise ValueError(f"dsv4_indexer_score requires kv shaped [N, 128], got {tuple(kv.shape)}")
    if kv.dtype != torch.bfloat16:
        raise ValueError(f"dsv4_indexer_score requires kv bf16, got {kv.dtype}")
    if not _is_contiguous_exact(kv):
        raise ValueError("dsv4_indexer_score requires kv to be contiguous")
    n_entries = int(kv.shape[0])

    if weights.shape != (_BATCH, n_rows, _HEADS):
        raise ValueError(
            "dsv4_indexer_score requires weights shaped "
            f"[1, {n_rows}, 64], got {tuple(weights.shape)}"
        )
    if weights.dtype != torch.bfloat16:
        raise ValueError(f"dsv4_indexer_score requires weights bf16, got {weights.dtype}")
    if not _is_contiguous_exact(weights):
        raise ValueError("dsv4_indexer_score requires weights to be contiguous")

    if out is not None:
        if out.device.type != "cuda" or out.device != q.device:
            raise ValueError("dsv4_indexer_score requires out on the same CUDA device")
        if out.shape != (_BATCH, n_rows, n_entries):
            raise ValueError(
                f"dsv4_indexer_score requires out shaped [1, {n_rows}, {n_entries}], "
                f"got {tuple(out.shape)}"
            )
        if out.dtype != torch.float32:
            raise ValueError(f"dsv4_indexer_score requires out fp32, got {out.dtype}")
        if not _is_contiguous_exact(out):
            raise ValueError("dsv4_indexer_score requires out to be contiguous")
    return n_rows, n_entries


def _check_dsv4_indexer_score_batch_contract(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    slot_ids: torch.Tensor,
    out: torch.Tensor | None = None,
    n_entries: int | None = None,
) -> tuple[int, int]:
    """Validate the batched decode scorer contract and return ``(B, N)``."""
    if q.device.type != "cuda" or kv.device.type != "cuda" or weights.device.type != "cuda":
        raise ValueError("dsv4_indexer_score_batch requires CUDA q/kv/weights tensors")
    if slot_ids.device.type != "cuda":
        raise ValueError("dsv4_indexer_score_batch requires CUDA slot_ids")
    if q.device != kv.device or q.device != weights.device or q.device != slot_ids.device:
        raise ValueError(
            "dsv4_indexer_score_batch requires q/kv/weights/slot_ids on one CUDA device"
        )
    if q.ndim != 4 or q.shape[1:] != (_SEQ, _HEADS, _HEAD_DIM):
        raise ValueError(
            f"dsv4_indexer_score_batch requires q shaped [B, 1, 64, 128], got {tuple(q.shape)}"
        )
    batch_size = int(q.shape[0])
    if batch_size not in _ALLOWED_BATCH_SIZES:
        raise ValueError(
            f"dsv4_indexer_score_batch requires q batch B in (1, 2, 4), got {batch_size}"
        )
    if q.dtype != torch.bfloat16:
        raise ValueError(f"dsv4_indexer_score_batch requires q bf16, got {q.dtype}")
    if not _is_contiguous_exact(q):
        raise ValueError("dsv4_indexer_score_batch requires q to be contiguous")

    if kv.ndim != 3 or kv.shape[2] != _HEAD_DIM:
        raise ValueError(
            f"dsv4_indexer_score_batch requires kv shaped [S, N, 128], got {tuple(kv.shape)}"
        )
    if kv.dtype != torch.bfloat16:
        raise ValueError(f"dsv4_indexer_score_batch requires kv bf16, got {kv.dtype}")
    if not _is_contiguous_exact(kv):
        raise ValueError("dsv4_indexer_score_batch requires kv to be contiguous")
    n_slots = int(kv.shape[0])
    capacity = int(kv.shape[1])
    if n_entries is None:
        n_entries = capacity
    else:
        n_entries = int(n_entries)
        if not 0 <= n_entries <= capacity:
            raise ValueError(
                "dsv4_indexer_score_batch n_entries must be within the KV arena "
                f"capacity, got {n_entries} for {capacity}"
            )

    if weights.shape != (batch_size, _SEQ, _HEADS):
        raise ValueError(
            f"dsv4_indexer_score_batch requires weights shaped [{batch_size}, 1, 64], "
            f"got {tuple(weights.shape)}"
        )
    if weights.dtype != torch.bfloat16:
        raise ValueError(f"dsv4_indexer_score_batch requires weights bf16, got {weights.dtype}")
    if not _is_contiguous_exact(weights):
        raise ValueError("dsv4_indexer_score_batch requires weights to be contiguous")

    if slot_ids.shape != (batch_size,):
        raise ValueError(
            f"dsv4_indexer_score_batch requires slot_ids shaped [{batch_size}], "
            f"got {tuple(slot_ids.shape)}"
        )
    if slot_ids.dtype != torch.int64:
        raise ValueError(f"dsv4_indexer_score_batch requires slot_ids int64, got {slot_ids.dtype}")
    if not _is_contiguous_exact(slot_ids):
        raise ValueError("dsv4_indexer_score_batch requires slot_ids to be contiguous")
    if n_slots <= 0:
        raise ValueError("dsv4_indexer_score_batch requires kv to have at least one slot")
    # Do not inspect slot_ids values here: this wrapper is called during CUDA
    # Graph capture and any min/max/item/tolist would synchronize device to
    # host.  The backend validates bounds from its host-side slot list before
    # filling the persistent graph input tensor.  Duplicate ids are safe for
    # this scorer because it is read-only.

    if out is not None:
        if out.device.type != "cuda" or out.device != q.device:
            raise ValueError("dsv4_indexer_score_batch requires out on the same CUDA device")
        if out.shape != (batch_size, _SEQ, n_entries):
            raise ValueError(
                "dsv4_indexer_score_batch requires out shaped "
                f"[{batch_size}, 1, {n_entries}], got {tuple(out.shape)}"
            )
        if out.dtype != torch.float32:
            raise ValueError(f"dsv4_indexer_score_batch requires out fp32, got {out.dtype}")
        if not _is_contiguous_exact(out):
            raise ValueError("dsv4_indexer_score_batch requires out to be contiguous")
    return batch_size, n_entries


def dsv4_indexer_score(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    block_n: int = _DEFAULT_BLOCK_N,
    num_warps: int = _DEFAULT_NUM_WARPS,
    num_stages: int = _DEFAULT_NUM_STAGES,
) -> torch.Tensor:
    """Score every compressed-KV row for the ratio-4 DSV4 indexer scan."""
    if block_n <= 0:
        raise ValueError(f"dsv4_indexer_score requires block_n > 0, got {block_n}")
    n_rows, n_entries = _check_dsv4_indexer_score_contract(q, kv, weights, out)
    if out is None:
        out = torch.empty((_BATCH, n_rows, n_entries), dtype=torch.float32, device=q.device)
    if n_entries == 0:
        return out
    q_rows = q.view(n_rows, _HEADS, _HEAD_DIM)
    weight_rows = weights.view(n_rows, _HEADS)
    out_rows = out.view(n_rows, n_entries)
    grid = (n_rows, triton.cdiv(n_entries, block_n))
    _dsv4_indexer_score_kernel[grid](
        q_rows,
        kv,
        weight_rows,
        out_rows,
        n_rows,
        q_rows.stride(0),
        weight_rows.stride(0),
        out_rows.stride(0),
        n_entries,
        BLOCK_N=block_n,
        HEADS=_HEADS,
        HEAD_DIM=_HEAD_DIM,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def dsv4_indexer_score_batch(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    slot_ids: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    n_entries: int | None = None,
    block_n: int = _DEFAULT_BLOCK_N,
    num_warps: int = _DEFAULT_NUM_WARPS,
    num_stages: int = _DEFAULT_NUM_STAGES,
) -> torch.Tensor:
    """Score ``[B, 1, 64, 128]`` decode queries against a slot arena ``[S, N, 128]``.

    ``slot_ids`` selects which slot each batch row reads. Duplicate slot ids are
    allowed because this kernel is read-only: no per-slot state is mutated.
    """
    if block_n <= 0:
        raise ValueError(f"dsv4_indexer_score_batch requires block_n > 0, got {block_n}")
    batch_size, n_entries = _check_dsv4_indexer_score_batch_contract(
        q, kv, weights, slot_ids, out, n_entries
    )
    if out is None:
        out = torch.empty((batch_size, _SEQ, n_entries), dtype=torch.float32, device=q.device)
    if n_entries == 0:
        return out
    q_rows = q.view(batch_size, _HEADS, _HEAD_DIM)
    weight_rows = weights.view(batch_size, _HEADS)
    out_rows = out.view(batch_size, n_entries)
    grid = (batch_size, triton.cdiv(n_entries, block_n))
    _dsv4_indexer_score_batch_kernel[grid](
        q_rows,
        kv,
        weight_rows,
        slot_ids,
        out_rows,
        n_entries,
        q_rows.stride(0),
        kv.stride(0),
        weight_rows.stride(0),
        out_rows.stride(0),
        BLOCK_N=block_n,
        HEADS=_HEADS,
        HEAD_DIM=_HEAD_DIM,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def compile_dsv4_indexer_score_sm120(
    *,
    block_n: int = _DEFAULT_BLOCK_N,
    num_warps: int = _DEFAULT_NUM_WARPS,
    num_stages: int = _DEFAULT_NUM_STAGES,
):
    """Offline-compile the standalone scorer for SM120."""
    signature = {
        "q_ptr": "*bf16",
        "kv_ptr": "*bf16",
        "weights_ptr": "*bf16",
        "out_ptr": "*bf16",
        "R": "i32",
        "q_row_stride": "i64",
        "weights_row_stride": "i64",
        "out_row_stride": "i64",
        "N": "i32",
        "BLOCK_N": "constexpr",
        "HEADS": "constexpr",
        "HEAD_DIM": "constexpr",
    }
    src = ASTSource(
        fn=_dsv4_indexer_score_kernel,
        signature=signature,
        constexprs={
            "BLOCK_N": block_n,
            "HEADS": _HEADS,
            "HEAD_DIM": _HEAD_DIM,
        },
    )
    return triton.compile(
        src,
        target=GPUTarget("cuda", 120, 32),
        options={
            "num_warps": num_warps,
            "num_stages": num_stages,
        },
    )


def compile_dsv4_indexer_score_batch_sm120(
    *,
    block_n: int = _DEFAULT_BLOCK_N,
    num_warps: int = _DEFAULT_NUM_WARPS,
    num_stages: int = _DEFAULT_NUM_STAGES,
):
    """Offline-compile the batched slot-arena scorer for SM120."""
    signature = {
        "q_ptr": "*bf16",
        "kv_ptr": "*bf16",
        "weights_ptr": "*bf16",
        "slot_ids_ptr": "*i64",
        "out_ptr": "*bf16",
        "N": "i32",
        "q_batch_stride": "i64",
        "kv_slot_stride": "i64",
        "weights_batch_stride": "i64",
        "out_batch_stride": "i64",
        "BLOCK_N": "constexpr",
        "HEADS": "constexpr",
        "HEAD_DIM": "constexpr",
    }
    src = ASTSource(
        fn=_dsv4_indexer_score_batch_kernel,
        signature=signature,
        constexprs={
            "BLOCK_N": block_n,
            "HEADS": _HEADS,
            "HEAD_DIM": _HEAD_DIM,
        },
    )
    return triton.compile(
        src,
        target=GPUTarget("cuda", 120, 32),
        options={
            "num_warps": num_warps,
            "num_stages": num_stages,
        },
    )
