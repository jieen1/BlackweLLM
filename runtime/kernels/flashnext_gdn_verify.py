"""FP32-state Flash-Next GDN target-verify kernel.

This is the fixed-shape subset of SGLang's
``fused_sigmoid_gating_recurrent.py`` needed by the SM120 Flash-Next runtime:
one linear draft chain, FP32 persistent state, GQA heads, and a materialized
state after every candidate token.  Keeping the recurrence in one Triton
launch avoids replaying the surrounding projections for every draft row.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T"])
def _flashnext_gdn_verify_kernel(
    a_log,
    a,
    dt_bias,
    q,
    k,
    v,
    b,
    output,
    initial_state,
    intermediate_states,
    scale,
    T,
    stride_a,
    stride_q,
    stride_k,
    stride_v,
    stride_b,
    stride_state_batch,
    stride_intermediate_batch,
    stride_intermediate_step,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    block_k, block_v, batch_head = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )
    batch = batch_head // HV
    value_head = batch_head % HV
    key_head = value_head // (HV // H)

    offset_k = block_k * BK + tl.arange(0, BK)
    offset_v = block_v * BV + tl.arange(0, BV)
    mask_k = offset_k < K
    mask_v = offset_v < V
    mask_state = mask_k[:, None] & mask_v[None, :]

    token_offset = batch * T
    q_ptr = q + token_offset * stride_q + key_head * K + offset_k
    k_ptr = k + token_offset * stride_k + key_head * K + offset_k
    v_ptr = v + token_offset * stride_v + value_head * V + offset_v
    b_ptr = b + token_offset * stride_b + value_head
    a_ptr = a + token_offset * stride_a + value_head
    output_ptr = output + (token_offset * HV + value_head) * V + offset_v

    state = tl.load(
        initial_state
        + batch * stride_state_batch
        + value_head * K * V
        + offset_k[:, None] * V
        + offset_v[None, :],
        mask=mask_state,
        other=0.0,
    ).to(tl.float32)
    decay_scale = -tl.exp(tl.load(a_log + value_head).to(tl.float32))
    bias = tl.load(dt_bias + value_head).to(tl.float32)

    step = 0
    for _ in range(0, T):
        query = tl.load(q_ptr, mask=mask_k, other=0.0).to(tl.float32)
        key = tl.load(k_ptr, mask=mask_k, other=0.0).to(tl.float32)
        value = tl.load(v_ptr, mask=mask_v, other=0.0).to(tl.float32)
        raw_a = tl.load(a_ptr).to(tl.float32) + bias
        raw_b = tl.load(b_ptr).to(tl.float32)

        # Match SGLang's stable softplus and in-kernel sigmoid exactly.
        log_decay = decay_scale * tl.where(
            raw_a <= 20.0,
            tl.log(1.0 + tl.exp(raw_a)),
            raw_a,
        )
        beta = 1.0 / (1.0 + tl.exp(-raw_b))
        query = query / tl.sqrt(tl.sum(query * query) + 1e-6)
        key = key / tl.sqrt(tl.sum(key * key) + 1e-6)
        query *= scale

        state *= tl.exp(log_decay)
        value -= tl.sum(state * key[:, None], axis=0)
        value *= beta
        state += key[:, None] * value[None, :]
        result = tl.sum(state * query[:, None], axis=0)
        tl.store(output_ptr, result, mask=mask_v)

        tl.store(
            intermediate_states
            + batch * stride_intermediate_batch
            + step * stride_intermediate_step
            + value_head * K * V
            + offset_k[:, None] * V
            + offset_v[None, :],
            state,
            mask=mask_state,
        )

        q_ptr += stride_q
        k_ptr += stride_k
        v_ptr += stride_v
        b_ptr += stride_b
        a_ptr += stride_a
        output_ptr += HV * V
        step += 1


def flashnext_gdn_verify(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    intermediate_states: torch.Tensor,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Verify a linear draft chain and store its per-token FP32 states.

    ``q``/``k`` keep their original key-head count; the kernel maps each
    value head to its GQA key head. ``intermediate_states`` is
    ``[B, T, HV, K, V]`` and is written in place for rollback-free commit.
    """
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        raise ValueError("Flash-Next GDN verify expects q/k/v [B,T,H,D]")
    batch, steps, heads, key_dim = k.shape
    value_heads, value_dim = v.shape[-2:]
    expected_state = (batch, value_heads, key_dim, value_dim)
    expected_rows = (batch, steps, value_heads, key_dim, value_dim)
    if tuple(initial_state.shape) != expected_state:
        raise ValueError(
            f"initial_state must be {expected_state}, got {tuple(initial_state.shape)}"
        )
    if tuple(intermediate_states.shape) != expected_rows:
        raise ValueError(
            f"intermediate_states must be {expected_rows}, got {tuple(intermediate_states.shape)}"
        )
    if initial_state.dtype != torch.float32 or intermediate_states.dtype != torch.float32:
        raise TypeError("Flash-Next GDN verify requires FP32 persistent and candidate states")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise TypeError("Flash-Next GDN verify requires BF16 q/k/v")
    if value_heads % heads:
        raise ValueError(f"value heads {value_heads} must be divisible by key heads {heads}")
    if not all(tensor.is_contiguous() for tensor in (q, k, v, initial_state)):
        raise ValueError("Flash-Next GDN verify inputs must be contiguous")
    if not intermediate_states.is_contiguous():
        raise ValueError("Flash-Next GDN candidate states must be contiguous")

    block_k = triton.next_power_of_2(key_dim)
    block_v = min(triton.next_power_of_2(value_dim), 32)
    if triton.cdiv(key_dim, block_k) != 1:
        raise ValueError("Flash-Next GDN verify does not support split key tiles")
    if output is None:
        output = torch.empty_like(v)
    elif output.shape != v.shape or output.dtype != v.dtype or not output.is_contiguous():
        raise ValueError("Flash-Next GDN verify output must be contiguous and match v")
    grid = (1, triton.cdiv(value_dim, block_v), batch * value_heads)
    _flashnext_gdn_verify_kernel[grid](
        a_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        output,
        initial_state,
        intermediate_states,
        key_dim**-0.5,
        steps,
        a.stride(-2),
        q.stride(1),
        k.stride(1),
        v.stride(1),
        b.stride(-2),
        initial_state.stride(0),
        intermediate_states.stride(0),
        intermediate_states.stride(1),
        H=heads,
        HV=value_heads,
        K=key_dim,
        V=value_dim,
        BK=block_k,
        BV=block_v,
        num_warps=1,
        num_stages=3,
    )
    return output


def flashnext_gdn_commit(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    accepted_count: int,
    scratch_states: torch.Tensor,
    scratch_output: torch.Tensor,
) -> None:
    """Recompute only the accepted recurrence prefix into ``state``.

    Verification keeps the live FP32 state read-only.  Retaining all four
    candidate states costs 12 MiB per GDN layer, so the memory-capacity path
    stores the compact projected inputs and advances the chosen prefix after
    acceptance instead.  This is the identical kernel/math used by verify;
    only intermediate-state stores and the unused attention output are
    disabled.
    """
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        raise ValueError("Flash-Next GDN commit expects q/k/v [B,T,H,D]")
    batch, steps, heads, key_dim = k.shape
    value_heads, value_dim = v.shape[-2:]
    if not 1 <= accepted_count <= steps:
        raise ValueError(f"accepted_count must be in [1,{steps}], got {accepted_count}")
    if tuple(state.shape) != (batch, value_heads, key_dim, value_dim):
        raise ValueError("Flash-Next GDN commit state shape does not match q/k/v")
    if state.dtype != torch.float32:
        raise TypeError("Flash-Next GDN commit requires FP32 persistent state")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise TypeError("Flash-Next GDN commit requires BF16 q/k/v")
    if value_heads % heads:
        raise ValueError(f"value heads {value_heads} must be divisible by key heads {heads}")
    if not all(tensor.is_contiguous() for tensor in (q, k, v, state)):
        raise ValueError("Flash-Next GDN commit inputs must be contiguous")

    flashnext_gdn_verify(
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        a_log=a_log,
        dt_bias=dt_bias,
        initial_state=state,
        intermediate_states=scratch_states,
        output=scratch_output,
    )
    state.copy_(scratch_states[:, accepted_count - 1])
