"""Device-side greedy acceptance for the Qwen DSpark verify epilogue.

SGLang's DSpark path does not materialize a Python decision for every
candidate position.  It keeps the verify predictions and candidates on the
device, computes the accepted prefix and recovery token in one small
epilogue, and exposes only the compact output row to the scheduler.

The target model still owns the expensive ``argmax`` over the vocabulary;
this module deliberately does not replace that operation.  It removes the
extra equality/cumulative-product/sum/concatenate launches that used to sit
between target verify and draft replay.  The Triton kernel has a compile-time
candidate width, so the inner prefix walk is unrolled for DSpark's fixed
``gamma`` and is safe to capture in a CUDA Graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

try:  # Triton is optional for the CPU-only test interpreter.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised by the torch-free gate
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _accept_greedy_kernel(
        predicted_ptr,
        candidates_ptr,
        accepted_ptr,
        committed_ptr,
        predicted_stride,
        candidate_stride,
        committed_stride,
        batch_size,
        GAMMA: tl.constexpr,
    ):
        row = tl.program_id(0)
        # The launch grid is exactly ``batch_size``.  Keeping this as a
        # scalar program (rather than a BLOCK-wide scan) avoids a second
        # reduction over the already tiny K dimension and makes the graph
        # shape independent of the active slot ids.
        accepted = 0
        alive = 1
        pred_base = predicted_ptr + row * predicted_stride
        candidate_base = candidates_ptr + row * candidate_stride
        out_base = committed_ptr + row * committed_stride
        for position in range(GAMMA):
            predicted = tl.load(pred_base + position)
            candidate = tl.load(candidate_base + position + 1)
            matches = predicted == candidate
            accepted += alive * matches
            alive *= matches

        # ``accepted`` is in [0, GAMMA].  The verifier's prediction at that
        # position is the recovery/bonus token.  Accepted positions are
        # exactly the candidate continuation tokens; later output cells are
        # zeroed so a host consumer can copy one fixed row without another
        # device-side gather.
        for position in range(GAMMA + 1):
            predicted = tl.load(pred_base + position)
            if position < GAMMA:
                candidate = tl.load(candidate_base + position + 1)
            else:
                candidate = tl.zeros((), dtype=tl.int64)
            value = tl.where(
                position < accepted,
                candidate,
                tl.where(position == accepted, predicted, 0),
            )
            tl.store(out_base + position, value)
        tl.store(accepted_ptr + row, accepted)

    @triton.jit
    def _accept_greedy_ragged_kernel(
        predicted_ptr,
        candidates_ptr,
        accepted_ptr,
        committed_ptr,
        q_indptr_ptr,
        verify_lens_ptr,
        predicted_stride,
        candidates_stride,
        committed_stride,
        MAX_GAMMA: tl.constexpr,
    ):
        request = tl.program_id(0)
        start = tl.load(q_indptr_ptr + request).to(tl.int64)
        verify_len = tl.load(verify_lens_ptr + request).to(tl.int32)
        gamma = verify_len - 1
        pred_base = predicted_ptr + start * predicted_stride
        candidate_base = candidates_ptr + start * candidates_stride
        out_base = committed_ptr + request * committed_stride

        accepted = 0
        alive = 1
        for position in range(MAX_GAMMA):
            valid = position < gamma
            predicted = tl.load(
                pred_base + position * predicted_stride,
                mask=position <= gamma,
                other=0,
            )
            candidate = tl.load(
                candidate_base + (position + 1) * candidates_stride,
                mask=valid,
                other=0,
            )
            matches = predicted == candidate
            accepted += alive * valid * matches
            alive *= tl.where(valid, matches, 1)

        for position in range(MAX_GAMMA + 1):
            predicted = tl.load(
                pred_base + position * predicted_stride,
                mask=position <= gamma,
                other=0,
            )
            candidate = tl.load(
                candidate_base + (position + 1) * candidates_stride,
                mask=position < gamma,
                other=0,
            )
            value = tl.where(
                position < accepted,
                candidate,
                tl.where(position == accepted, predicted, 0),
            )
            tl.store(out_base + position, value)
        tl.store(accepted_ptr + request, accepted)


def _validate_inputs(candidates: Tensor, target_logits: Tensor, gamma: int) -> tuple[int, int]:
    if candidates.ndim != 2 or candidates.shape[1] != gamma + 1:
        raise ValueError(
            "DSpark candidates must have shape [B, gamma+1], got "
            f"{tuple(candidates.shape)} for gamma={gamma}"
        )
    if target_logits.ndim == 3:
        if target_logits.shape[:2] != candidates.shape:
            raise ValueError(
                "DSpark target logits [B,Q,V] do not match candidates: "
                f"logits={tuple(target_logits.shape)}, candidates={tuple(candidates.shape)}"
            )
        batch_size, query_len = target_logits.shape[:2]
    elif target_logits.ndim == 2:
        expected_rows = candidates.shape[0] * (gamma + 1)
        if target_logits.shape[0] != expected_rows:
            raise ValueError(
                "DSpark target logits [B*Q,V] have the wrong row count: "
                f"got {target_logits.shape[0]}, expected {expected_rows}"
            )
        batch_size, query_len = candidates.shape
    else:
        raise ValueError(
            "DSpark target logits must be rank 2 or 3, "
            f"got rank {target_logits.ndim}"
        )
    if query_len != gamma + 1:
        raise ValueError(f"DSpark target query length must be gamma+1={gamma + 1}")
    return int(batch_size), int(query_len)


def greedy_accept_device(
    candidates: Tensor,
    target_logits: Tensor,
    gamma: int,
    *,
    predicted: Tensor | None = None,
    accepted_out: Tensor | None = None,
    committed_out: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return ``(accepted_draft_count, committed_tokens)`` on the device.

    ``candidates`` contains ``[anchor, draft_0, ..., draft_{K-1}]`` and
    ``target_logits`` contains the target prediction at each of the
    ``K+1`` verify positions.  ``committed_tokens`` is a fixed ``[B,K+1]``
    buffer; only cells ``0..accepted`` are valid for each row.

    Caller-owned output buffers are supported because CUDA Graph replay must
    not allocate.  The CPU/Torch implementation intentionally follows the
    same acceptance rule and is used for tests and environments without
    Triton.
    """

    batch_size, _query_len = _validate_inputs(candidates, target_logits, gamma)
    logits_3d = target_logits.view(batch_size, gamma + 1, -1)
    if predicted is None:
        predicted = torch.empty(
            (batch_size, gamma + 1), dtype=torch.long, device=target_logits.device
        )
    elif tuple(predicted.shape) != (batch_size, gamma + 1):
        raise ValueError("DSpark predicted buffer has the wrong shape")
    torch.argmax(logits_3d, dim=-1, out=predicted)

    if accepted_out is None:
        accepted_out = torch.empty(
            batch_size, dtype=torch.int32, device=target_logits.device
        )
    if committed_out is None:
        committed_out = torch.empty(
            (batch_size, gamma + 1), dtype=torch.long, device=target_logits.device
        )
    if tuple(accepted_out.shape) != (batch_size,) or accepted_out.dtype != torch.int32:
        raise ValueError("DSpark accepted output must be an int32 [B] tensor")
    if tuple(committed_out.shape) != (batch_size, gamma + 1):
        raise ValueError("DSpark committed output has the wrong shape")

    if (
        triton is not None
        and candidates.is_cuda
        and predicted.is_cuda
        and accepted_out.is_cuda
        and committed_out.is_cuda
    ):
        _accept_greedy_kernel[(batch_size,)](
            predicted,
            candidates,
            accepted_out,
            committed_out,
            predicted.stride(0),
            candidates.stride(0),
            committed_out.stride(0),
            batch_size,
            GAMMA=gamma,
        )
    else:
        matches = predicted[:, :gamma].eq(candidates[:, 1:])
        accepted = matches.to(torch.int32).cumprod(dim=1).sum(dim=1)
        accepted_out.copy_(accepted)
        positions = torch.arange(gamma + 1, device=predicted.device).view(1, -1)
        draft_continuations = torch.cat(
            [candidates[:, 1:], torch.zeros_like(candidates[:, :1])], dim=1
        )
        committed_out.copy_(
            torch.where(
                positions < accepted[:, None],
                draft_continuations,
                torch.where(positions == accepted[:, None], predicted, 0),
            )
        )
    return accepted_out, committed_out


def greedy_accept_ragged(
    candidates: Tensor,
    target_logits: Tensor,
    verify_lens: Tensor | list[int] | tuple[int, ...],
    max_gamma: int,
    *,
    q_indptr: Tensor | None = None,
    predicted: Tensor | None = None,
    accepted_out: Tensor | None = None,
    committed_out: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Accept one compact, request-major ragged verify batch on device.

    ``candidates`` and ``target_logits`` contain only the compact rows in
    request-major order.  Request ``i`` occupies
    ``q_indptr[i]:q_indptr[i + 1]`` and has ``verify_lens[i]`` rows, including
    the anchor and the bonus position.  The output remains a small fixed
    ``[B, max_gamma + 1]`` commit matrix so the scheduler can publish one
    row per request without padding the expensive target verify itself.

    ``verify_lens`` and ``q_indptr`` are device tensors in the CUDA-Graph
    path.  Python lists are accepted for the eager/CPU reference path and
    are converted once at the boundary.
    """

    if max_gamma < 0:
        raise ValueError(f"DSpark ragged max_gamma must be non-negative, got {max_gamma}")
    if candidates.ndim != 1:
        raise ValueError(
            "DSpark ragged candidates must be flat [capacity], "
            f"got {tuple(candidates.shape)}"
        )
    if target_logits.ndim != 2 or target_logits.shape[0] != candidates.shape[0]:
        raise ValueError(
            "DSpark ragged target logits must have shape [capacity, vocab], "
            f"got {tuple(target_logits.shape)} for candidates={tuple(candidates.shape)}"
        )
    if isinstance(verify_lens, torch.Tensor):
        lens = verify_lens
        if lens.ndim != 1:
            raise ValueError("DSpark ragged verify_lens must be rank 1")
        if lens.device != target_logits.device:
            raise ValueError("DSpark ragged verify_lens must share the logits device")
    else:
        lens = torch.tensor(verify_lens, dtype=torch.int32, device=target_logits.device)
    batch_size = int(lens.numel())
    if batch_size <= 0:
        raise ValueError("DSpark ragged acceptance requires at least one request")
    if lens.dtype not in (torch.int32, torch.int64):
        raise ValueError("DSpark ragged verify_lens must be int32 or int64")
    if bool(torch.any(lens < 1).item()) or bool(torch.any(lens > max_gamma + 1).item()):
        raise ValueError(
            f"DSpark ragged verify_lens must be in [1,{max_gamma + 1}], got {lens.tolist()}"
        )

    if q_indptr is None:
        q_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=target_logits.device)
        q_indptr[1:] = torch.cumsum(lens.to(torch.int32), dim=0)
    elif q_indptr.ndim != 1 or q_indptr.shape[0] != batch_size + 1:
        raise ValueError("DSpark ragged q_indptr must have shape [B+1]")
    if q_indptr.device != target_logits.device:
        raise ValueError("DSpark ragged q_indptr must share the logits device")
    if int(q_indptr[-1].item()) > int(candidates.shape[0]):
        raise ValueError("DSpark ragged q_indptr exceeds compact candidate capacity")

    if predicted is None:
        predicted = torch.empty(
            candidates.shape[0], dtype=torch.long, device=target_logits.device
        )
    elif tuple(predicted.shape) != (candidates.shape[0],):
        raise ValueError("DSpark ragged predicted buffer must have shape [capacity]")
    torch.argmax(target_logits, dim=-1, out=predicted)

    if accepted_out is None:
        accepted_out = torch.empty(batch_size, dtype=torch.int32, device=target_logits.device)
    if committed_out is None:
        committed_out = torch.empty(
            batch_size, max_gamma + 1, dtype=torch.long, device=target_logits.device
        )
    if tuple(accepted_out.shape) != (batch_size,) or accepted_out.dtype != torch.int32:
        raise ValueError("DSpark ragged accepted output must be int32 [B]")
    if tuple(committed_out.shape) != (batch_size, max_gamma + 1):
        raise ValueError("DSpark ragged committed output has the wrong shape")

    if (
        triton is not None
        and candidates.is_cuda
        and target_logits.is_cuda
        and predicted.is_cuda
        and accepted_out.is_cuda
        and committed_out.is_cuda
    ):
        _accept_greedy_ragged_kernel[(batch_size,)](
            predicted,
            candidates,
            accepted_out,
            committed_out,
            q_indptr,
            lens,
            predicted.stride(0),
            candidates.stride(0),
            committed_out.stride(0),
            MAX_GAMMA=max_gamma,
        )
    else:
        accepted_rows: list[int] = []
        committed_rows: list[list[int]] = []
        starts = q_indptr.tolist()
        lengths = lens.tolist()
        predicted_list = predicted.tolist()
        candidate_list = candidates.tolist()
        for start, length in zip(starts[:-1], lengths, strict=True):
            gamma = int(length) - 1
            accepted = 0
            while accepted < gamma and predicted_list[start + accepted] == candidate_list[
                start + accepted + 1
            ]:
                accepted += 1
            accepted_rows.append(accepted)
            row = [0] * (max_gamma + 1)
            for position in range(accepted):
                row[position] = candidate_list[start + position + 1]
            row[accepted] = predicted_list[start + accepted]
            committed_rows.append(row)
        accepted_out.copy_(
            torch.tensor(accepted_rows, dtype=torch.int32, device=accepted_out.device)
        )
        committed_out.copy_(
            torch.tensor(committed_rows, dtype=torch.long, device=committed_out.device)
        )
    return accepted_out, committed_out
