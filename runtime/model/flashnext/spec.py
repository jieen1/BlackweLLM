"""Flash-Next fixed-width target verification and rollback-aware commit.

The target consumes ``anchor + K drafts`` in one model forward.  Every GDN
layer and the PLE convolution write one fixed-address state row per verify
position, while QSA writes candidate KV directly after the committed prefix.
Accept/reject then commits one state row and advances the logical QSA length;
rejected QSA rows and their compressed index groups are explicitly cleared
before the next round.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from runtime.model.flashnext.qsa import (
    _qsa_cache_is_quantized,
    qsa_cache_index_copy_,
    qsa_index_cache_rows,
    qsa_kv_cache_dtype,
    quantize_qsa_kv,
)
from runtime.model.qwen36_model import GdnLayerState
from runtime.mtp_accept import sample_accept_reject
from runtime.sampling import (
    SamplingNumericalError,
    SamplingParams,
    compute_sampling_distribution,
    make_generator,
    sample_from_distribution,
)

if TYPE_CHECKING:
    from runtime.model.flashnext.model import (
        FlashNextModel,
        FlashNextSession,
    )
    from runtime.model.flashnext.mtp import FlashNextMTP


logger = logging.getLogger("qwen_sm120_runtime.flashnext.spec")


def _sampling_tensor_stats(value: torch.Tensor | None) -> str:
    """Summarize a tensor for a rare sampled-MTP numerical failure.

    This intentionally performs device-to-host reductions only on the error
    path.  Calling it before every stochastic draw would serialize the hot
    decode loop, which is exactly what the sampled-MTP path was optimized to
    avoid.
    """
    if value is None:
        return "none"
    tensor = value.detach().float().reshape(-1)
    total = int(tensor.numel())
    finite = torch.isfinite(tensor)
    finite_count = int(finite.sum().item())
    nan_count = int(torch.isnan(tensor).sum().item())
    inf_count = int(torch.isinf(tensor).sum().item())
    if finite_count:
        finite_values = tensor[finite]
        minimum = float(finite_values.min().item())
        maximum = float(finite_values.max().item())
        range_text = f"min={minimum:.6g},max={maximum:.6g}"
    else:
        range_text = "min=none,max=none"
    return (
        f"shape={tuple(value.shape)},dtype={value.dtype},finite={finite_count}/{total},"
        f"nan={nan_count},inf={inf_count},{range_text}"
    )


def _new_gdn_row(state: GdnLayerState) -> GdnLayerState:
    return GdnLayerState(
        conv_state=torch.empty_like(state.conv_state),
        recurrent_state=torch.empty_like(state.recurrent_state),
        has_previous_state=True,
    )


def _pool_capacity(pool: object) -> int | None:
    """Return the first dimension of a fixed cache, when one is present."""
    if torch.is_tensor(pool) and pool.ndim > 0:
        return int(pool.shape[0])
    return None


def _validate_pool_range(
    pool: object,
    start: int,
    length: int,
    label: str,
) -> None:
    """Reject cache writes that would silently run past the fixed pool."""
    if start < 0:
        raise ValueError(f"{label} position must be non-negative, got {start}")
    if length < 0:
        raise ValueError(f"{label} length must be non-negative, got {length}")
    capacity = _pool_capacity(pool)
    if capacity is not None and start + length > capacity:
        raise ValueError(
            f"{label} exceeds cache capacity: "
            f"start={start}, length={length}, capacity={capacity}"
        )


@dataclass
class FlashNextVerifyBuffers:
    token_ids: torch.Tensor
    positions: torch.Tensor
    ple_embeddings: torch.Tensor
    gdn_rows: dict[int, list[GdnLayerState]]
    gdn_work: dict[int, GdnLayerState]
    gdn_recurrent_rows: dict[int, torch.Tensor]
    gdn_recompute_scratch: torch.Tensor | None
    gdn_recompute_output: torch.Tensor | None
    ple_rows: list[torch.Tensor]
    # Fixed-width scratch for quantized QSA candidate rows.  Advanced indexing
    # such as ``pool[positions]`` returns a copy, so passing it to
    # ``quantize_qsa_kv`` silently dropped verify writes.  These graph-owned
    # rows let us quantize first and commit with the dtype-aware indexed copy.
    qsa_k_rows: dict[int, torch.Tensor]
    qsa_v_rows: dict[int, torch.Tensor]
    qsa_k_scale_rows: dict[int, torch.Tensor]
    qsa_v_scale_rows: dict[int, torch.Tensor]


def allocate_verify_buffers(
    model: FlashNextModel,
    sess: FlashNextSession,
    *,
    qo_len: int,
    device: torch.device | str,
    allocate_sequential_work: bool = True,
    recompute_recurrent_state: bool = False,
) -> FlashNextVerifyBuffers:
    if qo_len < 2:
        raise ValueError(f"speculative verify requires anchor + drafts, got qo_len={qo_len}")
    gdn_rows: dict[int, list[GdnLayerState]] = {}
    gdn_work: dict[int, GdnLayerState] = {}
    gdn_recurrent_rows: dict[int, torch.Tensor] = {}
    gdn_recompute_scratch: torch.Tensor | None = None
    gdn_recompute_output: torch.Tensor | None = None
    qsa_k_rows: dict[int, torch.Tensor] = {}
    qsa_v_rows: dict[int, torch.Tensor] = {}
    qsa_k_scale_rows: dict[int, torch.Tensor] = {}
    qsa_v_scale_rows: dict[int, torch.Tensor] = {}
    for layer in model.layers:
        if layer.is_qsa:
            attn = layer.attn.attn
            kv_dtype = qsa_kv_cache_dtype()
            qsa_k_rows[layer.layer_idx] = torch.empty(
                qo_len,
                attn.num_kv_heads,
                attn.head_dim,
                dtype=kv_dtype,
                device=device,
            )
            qsa_v_rows[layer.layer_idx] = torch.empty_like(
                qsa_k_rows[layer.layer_idx]
            )
            qsa_k_scale_rows[layer.layer_idx] = torch.empty(
                qo_len,
                attn.num_kv_heads,
                dtype=torch.float16,
                device=device,
            )
            qsa_v_scale_rows[layer.layer_idx] = torch.empty_like(
                qsa_k_scale_rows[layer.layer_idx]
            )
            continue
        state = sess.gdn[f"gdn_{layer.layer_idx}"]
        if recompute_recurrent_state:
            expected_scratch_shape = (qo_len, *state.recurrent_state.shape[1:])
            if gdn_recompute_scratch is None:
                gdn_recompute_scratch = torch.empty(
                    expected_scratch_shape,
                    dtype=state.recurrent_state.dtype,
                    device=state.recurrent_state.device,
                )
                gdn_recompute_output = torch.empty(
                    1,
                    qo_len,
                    layer.attn.num_v_heads,
                    layer.attn.head_v_dim,
                    dtype=torch.bfloat16,
                    device=state.recurrent_state.device,
                )
            elif tuple(gdn_recompute_scratch.shape) != expected_scratch_shape:
                raise ValueError("all recomputed Flash-Next GDN layers must share a geometry")
            recurrent_rows = torch.empty(
                0,
                dtype=state.recurrent_state.dtype,
                device=state.recurrent_state.device,
            )
        else:
            recurrent_rows = torch.empty(
                qo_len,
                *state.recurrent_state.shape[1:],
                dtype=state.recurrent_state.dtype,
                device=state.recurrent_state.device,
            )
        gdn_recurrent_rows[layer.layer_idx] = recurrent_rows
        gdn_rows[layer.layer_idx] = [
            GdnLayerState(
                conv_state=torch.empty_like(state.conv_state),
                recurrent_state=recurrent_rows[row : row + 1],
                has_previous_state=True,
            )
            for row in range(qo_len)
        ]
        # The batched recurrent path writes all commit candidates directly
        # into ``recurrent_rows``.  A full extra recurrent state per layer was
        # nevertheless kept alive for the unused sequential compatibility
        # path, costing roughly one target GDN state per serving slot.  Only
        # allocate it when that path can actually execute.
        if allocate_sequential_work:
            gdn_work[layer.layer_idx] = _new_gdn_row(state)
    ple_rows = (
        [torch.empty_like(sess.ple_conv_state) for _ in range(qo_len)]
        if sess.ple_conv_state is not None
        else []
    )
    return FlashNextVerifyBuffers(
        token_ids=torch.zeros(qo_len, dtype=torch.long, device=device),
        positions=torch.zeros(qo_len, dtype=torch.long, device=device),
        ple_embeddings=torch.zeros(
            qo_len, model.cfg.hidden_size, dtype=torch.bfloat16, device=device
        ),
        gdn_rows=gdn_rows,
        gdn_work=gdn_work,
        gdn_recurrent_rows=gdn_recurrent_rows,
        gdn_recompute_scratch=gdn_recompute_scratch,
        gdn_recompute_output=gdn_recompute_output,
        ple_rows=ple_rows,
        qsa_k_rows=qsa_k_rows,
        qsa_v_rows=qsa_v_rows,
        qsa_k_scale_rows=qsa_k_scale_rows,
        qsa_v_scale_rows=qsa_v_scale_rows,
    )


def _write_qsa_verify_rows(
    buffers: FlashNextVerifyBuffers,
    layer_idx: int,
    positions: torch.Tensor,
    values: torch.Tensor,
    pool: torch.Tensor,
    scale_pool: torch.Tensor,
    *,
    key: str,
) -> None:
    """Write candidate QSA rows back to their fixed cache addresses.

    ``pool[positions]`` is an advanced-indexing copy, not a writable view.
    The old verify path therefore quantized into a temporary and discarded
    every candidate row.  Besides being incorrect for batched verify, that
    expression creates an avoidable gather/scatter sequence in CUDA Graph
    capture.  Quantized caches use graph-owned row scratch followed by one
    dtype-aware indexed copy; BF16 caches can copy the projected rows directly.
    """
    if _qsa_cache_is_quantized(pool.dtype):
        rows = getattr(buffers, f"qsa_{key}_rows", {}).get(layer_idx)
        scale_rows = getattr(buffers, f"qsa_{key}_scale_rows", {}).get(layer_idx)
        if rows is None or scale_rows is None:
            raise RuntimeError(
                "quantized QSA verify requires graph-owned row scratch; "
                f"layer={layer_idx}, key={key}"
            )
        quantize_qsa_kv(values, rows, scale_rows)
        qsa_cache_index_copy_(pool, positions, rows)
        scale_pool.index_copy_(0, positions, scale_rows)
    else:
        pool.index_copy_(0, positions, values)


@torch.no_grad()
def verify_body(
    model: FlashNextModel,
    sess: FlashNextSession,
    buffers: FlashNextVerifyBuffers,
    *,
    exact_row_math: bool = True,
    batch_lm_head: bool = False,
    batch_gdn_recurrence: bool = False,
    # BF16 large projections are intentionally sequential by default.  The
    # serving backend enables this only after checking the loaded projection
    # format against a validated batched-GEMM contract.
    batch_gdn_projections: bool = False,
    sequential_qsa: bool = False,
    gdn_commit_inputs: dict[int, dict[str, torch.Tensor]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one fixed-width ``anchor + drafts`` target verification.

    Returns ``(hc_hidden, logits)`` with one row per consumed token.  The live
    GDN and PLE states are read-only; QSA candidate KV is written into its
    fixed pool and rolled back when :meth:`FlashNextVerifyGraph.commit` accepts
    only a prefix of the verify rows.
    """
    cfg = model.cfg
    positions = buffers.positions
    x = model.embed_tokens(buffers.token_ids)
    x = torch.cat([x] * cfg.hc_count, dim=-1)

    def mix_rows(module, value):
        if not exact_row_math:
            return module.mix(value)
        mixed_rows = []
        raw_rows = []
        normed_rows = []
        for row in range(value.shape[0]):
            mixed_row, (raw_row, normed_row) = module.mix(value[row : row + 1])
            mixed_rows.append(mixed_row)
            raw_rows.append(raw_row)
            normed_rows.append(normed_row)
        return torch.cat(mixed_rows), (torch.cat(raw_rows), torch.cat(normed_rows))

    def combine_rows(module, value, residuals):
        if not exact_row_math:
            return module.combine(value, residuals)
        raw, normed = residuals
        return torch.cat(
            [
                module.combine(
                    value[row : row + 1],
                    (raw[row : row + 1], normed[row : row + 1]),
                )
                for row in range(value.shape[0])
            ]
        )

    # Temporary diagnosis seam for isolating CUDA-graph replay failures.  It
    # is deliberately opt-in and has no effect on the serving default; the
    # full model remains the normal path.  Keeping the limit at the verify
    # body boundary lets us capture/replay a valid prefix without duplicating
    # model construction in one-off probes.
    debug_layer_limit = int(os.environ.get("QSR_FLASHNEXT_DEBUG_LAYER_LIMIT", "0"))
    debug_skip_mlp = os.environ.get("QSR_FLASHNEXT_DEBUG_SKIP_MLP", "0") == "1"
    debug_skip_mlp_combine = (
        os.environ.get("QSR_FLASHNEXT_DEBUG_SKIP_MLP_COMBINE", "0") == "1"
    )
    debug_clone_mlp = os.environ.get("QSR_FLASHNEXT_DEBUG_CLONE_MLP", "0") == "1"
    debug_mlp_metadata = (
        os.environ.get("QSR_FLASHNEXT_DEBUG_MLP_METADATA", "0") == "1"
    )
    debug_router_stats = (
        os.environ.get("QSR_FLASHNEXT_DEBUG_ROUTER_STATS", "0") == "1"
    )
    layers = model.layers
    if debug_layer_limit > 0:
        layers = model.layers[:debug_layer_limit]

    for layer in layers:
        if layer.ple is not None:
            if exact_row_math:
                injected = [
                    layer.ple.inject(
                        buffers.ple_embeddings[row : row + 1],
                        x[row : row + 1],
                    )
                    for row in range(x.shape[0])
                ]
                gated_flat = torch.cat([pair[0] for pair in injected])
                normed_flat = torch.cat([pair[1] for pair in injected])
            else:
                gated_flat, normed_flat = layer.ple.inject(buffers.ple_embeddings, x)
            conv_out, _ = layer.ple.spec_conv(
                normed_flat,
                sess.ple_conv_state,
                buffers.ple_rows,
            )
            x = x + gated_flat + conv_out

        mixed, residuals = mix_rows(layer.attn_hc, x)
        if layer.is_qsa:
            bundle = layer.attn
            if sequential_qsa:
                # QSA's compressed index cache is a recurrent state: writing
                # all K+1 candidate rows before evaluating row 0 lets a
                # boundary group / sparse tail observe future draft keys.
                # M=1 decode is the numerical contract, so keep the cache
                # update, block selection, KV write, and attention in the
                # same row order for the correctness path.  Projections and
                # the rest of the layer remain fixed-width; this branch is
                # graph-safe because the row count is static.
                idx_pool = sess.qsa_idx_k_pool[layer.layer_idx]
                pooled = sess.qsa_pooled_k_pool[layer.layer_idx]
                rope_cache = (
                    sess.qsa_idx_rope_pool.get(layer.layer_idx)
                    if sess.qsa_idx_rope_pool is not None
                    else None
                )
                k_pool = sess.qsa_k_pool[layer.layer_idx]
                v_pool = sess.qsa_v_pool[layer.layer_idx]
                k_scales = sess.qsa_k_scale_pool[layer.layer_idx]
                v_scales = sess.qsa_v_scale_pool[layer.layer_idx]
                attn_rows = []
                for row in range(mixed.shape[0]):
                    row_mixed = mixed[row : row + 1]
                    row_pos = positions[row : row + 1]
                    row_qi, row_ki = bundle.indexer.project_qk(row_mixed, row_pos)
                    if rope_cache is None:
                        bundle.indexer.update_index_cache_fixed(
                            idx_pool,
                            pooled,
                            row_ki,
                            row_pos,
                        )
                    else:
                        bundle.indexer.update_index_cache_fixed(
                            idx_pool,
                            pooled,
                            row_ki,
                            row_pos,
                            rope_cache=rope_cache,
                            rope_positions=row_pos,
                        )
                    row_end = torch.clamp(
                        (row_pos - 3) // bundle.indexer.compress_ratio + 1,
                        min=0,
                    )
                    row_logits = bundle.indexer.score_blocks(row_qi, pooled, row_end)
                    row_blocks = bundle.indexer.select_blocks(row_logits, row_end)
                    row_q, row_k, row_v, row_gate = bundle.attn.project(row_mixed, row_pos)
                    if _qsa_cache_is_quantized(k_pool.dtype):
                        k_rows = getattr(buffers, "qsa_k_rows", {}).get(layer.layer_idx)
                        v_rows = getattr(buffers, "qsa_v_rows", {}).get(layer.layer_idx)
                        k_scale_rows = getattr(buffers, "qsa_k_scale_rows", {}).get(
                            layer.layer_idx
                        )
                        v_scale_rows = getattr(buffers, "qsa_v_scale_rows", {}).get(
                            layer.layer_idx
                        )
                        if any(
                            value is None
                            for value in (k_rows, v_rows, k_scale_rows, v_scale_rows)
                        ):
                            raise RuntimeError(
                                "quantized QSA sequential verify requires row scratch; "
                                f"layer={layer.layer_idx}"
                            )
                        quantize_qsa_kv(row_k, k_rows[row : row + 1], k_scale_rows[row : row + 1])
                        quantize_qsa_kv(row_v, v_rows[row : row + 1], v_scale_rows[row : row + 1])
                        qsa_cache_index_copy_(
                            k_pool, row_pos, k_rows[row : row + 1]
                        )
                        qsa_cache_index_copy_(
                            v_pool, row_pos, v_rows[row : row + 1]
                        )
                        k_scales.index_copy_(0, row_pos, k_scale_rows[row : row + 1])
                        v_scales.index_copy_(0, row_pos, v_scale_rows[row : row + 1])
                    else:
                        k_pool.index_copy_(0, row_pos, row_k)
                        v_pool.index_copy_(0, row_pos, row_v)
                    row_indices, row_valid = bundle.indexer.batch_decode_gather_indices(
                        row_blocks,
                        row_pos,
                        sess.qsa_pad,
                    )
                    row_selected_counts = row_valid.to(torch.int32).sum().reshape(1)
                    attn_rows.append(
                        sess.qsa_attn[layer.layer_idx](
                            row_q,
                            row_gate,
                            k_pool,
                            v_pool,
                            row_indices[0],
                            row_valid[0],
                            k_scales,
                            v_scales,
                            row_selected_counts,
                        )
                    )
                attn_out = torch.cat(attn_rows)
                # ``qi``/``ki``/``q``/``k``/``v``/``gate`` are intentionally
                # row-local above; no batched tensors are needed by this path.
            else:
                if exact_row_math:
                    qk_rows = [
                        bundle.indexer.project_qk(mixed[row : row + 1], positions[row : row + 1])
                        for row in range(mixed.shape[0])
                    ]
                    qi = torch.cat([pair[0] for pair in qk_rows])
                    ki = torch.cat([pair[1] for pair in qk_rows])
                else:
                    qi, ki = bundle.indexer.project_qk(mixed, positions)
                idx_pool = sess.qsa_idx_k_pool[layer.layer_idx]
                pooled = sess.qsa_pooled_k_pool[layer.layer_idx]
                rope_cache = (
                    sess.qsa_idx_rope_pool.get(layer.layer_idx)
                    if sess.qsa_idx_rope_pool is not None
                    else None
                )
                if rope_cache is None:
                    bundle.indexer.update_index_cache_fixed(
                        idx_pool,
                        pooled,
                        ki,
                        positions,
                    )
                else:
                    bundle.indexer.update_index_cache_fixed(
                        idx_pool,
                        pooled,
                        ki,
                        positions,
                        rope_cache=rope_cache,
                        rope_positions=positions,
                    )
                ends = torch.clamp(
                    (positions - 3) // bundle.indexer.compress_ratio + 1,
                    min=0,
                )
                if exact_row_math:
                    block_logits = torch.cat(
                        [
                            bundle.indexer.score_blocks(
                                qi[row : row + 1], pooled, ends[row : row + 1]
                            )
                            for row in range(qi.shape[0])
                        ]
                    )
                    blocks = torch.cat(
                        [
                            bundle.indexer.select_blocks(
                                block_logits[row : row + 1], ends[row : row + 1]
                            )
                            for row in range(qi.shape[0])
                        ]
                    )
                    projected = [
                        bundle.attn.project(mixed[row : row + 1], positions[row : row + 1])
                        for row in range(mixed.shape[0])
                    ]
                    q, k, v, gate = (
                        torch.cat([values[index] for values in projected]) for index in range(4)
                    )
                else:
                    block_logits = bundle.indexer.score_blocks(qi, pooled, ends)
                    blocks = bundle.indexer.select_blocks(block_logits, ends)
                    q, k, v, gate = bundle.attn.project(mixed, positions)
                layer_idx = layer.layer_idx
                k_pool = sess.qsa_k_pool[layer_idx]
                v_pool = sess.qsa_v_pool[layer_idx]
                k_scale_pool = sess.qsa_k_scale_pool[layer_idx]
                v_scale_pool = sess.qsa_v_scale_pool[layer_idx]
                _write_qsa_verify_rows(
                    buffers,
                    layer_idx,
                    positions,
                    k,
                    k_pool,
                    k_scale_pool,
                    key="k",
                )
                _write_qsa_verify_rows(
                    buffers,
                    layer_idx,
                    positions,
                    v,
                    v_pool,
                    v_scale_pool,
                    key="v",
                )
                indices, valid = bundle.indexer.batch_decode_gather_indices(
                    blocks,
                    positions,
                    sess.qsa_pad,
                )
                selected_counts = valid.to(torch.int32).sum(dim=1)
                if exact_row_math:
                    attn_out = torch.cat(
                        [
                            sess.qsa_attn[layer.layer_idx](
                                q[row : row + 1],
                                gate[row : row + 1],
                                k_pool,
                                v_pool,
                                indices[row],
                                valid[row],
                                k_scale_pool,
                                v_scale_pool,
                                selected_counts[row : row + 1],
                            )
                            for row in range(q.shape[0])
                        ]
                    )
                else:
                    attn_out = sess.qsa_attn[layer.layer_idx](
                        q,
                        gate,
                        k_pool,
                        v_pool,
                        indices,
                        valid,
                        k_scale_pool,
                        v_scale_pool,
                        selected_counts,
                    )
        else:
            state = sess.gdn[f"gdn_{layer.layer_idx}"]
            if exact_row_math and not batch_gdn_recurrence:
                # The compatibility oracle reproduces ordinary one-token
                # decode exactly.  The SGLang-style FP32 target-verify kernel
                # is selected separately by ``batch_gdn_recurrence`` because
                # it intentionally uses a different fused floating-point
                # path.  Advance a private work row here and persist every
                # commit candidate.
                work = buffers.gdn_work[layer.layer_idx]
                work.conv_state.copy_(state.conv_state)
                work.recurrent_state.copy_(state.recurrent_state)
                work.has_previous_state = state.has_previous_state
                output_rows = []
                for row in range(mixed.shape[0]):
                    output_rows.append(
                        layer.attn(mixed[row : row + 1].unsqueeze(1), work).squeeze(1)
                    )
                    candidate = buffers.gdn_rows[layer.layer_idx][row]
                    candidate.conv_state.copy_(work.conv_state)
                    candidate.recurrent_state.copy_(work.recurrent_state)
                    candidate.has_previous_state = True
                attn_out = torch.cat(output_rows)
            else:
                attn_out, snapshots = layer.attn.spec_forward(
                    mixed.unsqueeze(0),
                    state,
                    spec_state_rows=buffers.gdn_rows[layer.layer_idx],
                    # Keep the large BF16 projections on their qualified
                    # per-row path.  Only fuse the conv/recurrent portion;
                    # changing the GEMM M dimension is independently known
                    # to alter greedy output on this checkpoint.
                    batch_large_projections=(
                        batch_gdn_projections and not exact_row_math
                    ),
                    fp32_intermediate_states=(
                        buffers.gdn_recompute_scratch.unsqueeze(0)
                        if gdn_commit_inputs is not None
                        and layer.layer_idx in gdn_commit_inputs
                        and buffers.gdn_recompute_scratch is not None
                        else buffers.gdn_recurrent_rows[layer.layer_idx].unsqueeze(0)
                        if batch_gdn_recurrence
                        else None
                    ),
                    fp32_commit_inputs=(
                        gdn_commit_inputs.get(layer.layer_idx)
                        if gdn_commit_inputs is not None
                        else None
                    ),
                )
                if snapshots is not None:
                    raise RuntimeError(
                        "fixed-row GDN verify unexpectedly returned cloned snapshots"
                    )
                attn_out = attn_out.squeeze(0)

        x = combine_rows(layer.attn_hc, attn_out, residuals)
        if debug_skip_mlp:
            continue
        if os.environ.get("QSR_FLASHNEXT_DEBUG_SKIP_MLP_MIX", "0") == "1":
            # Diagnosis-only path: retain the post-attention stream while
            # avoiding the second HC mix.  The residual tuple preserves the
            # combine ABI for isolating graph failures.
            mixed2 = x.reshape(x.shape[0], model.cfg.hc_count, model.cfg.hidden_size).mean(
                dim=-2
            )
            res2 = (x, torch.zeros_like(x))
        else:
            mixed2, res2 = mix_rows(layer.mlp_hc, x)
        if os.environ.get("QSR_FLASHNEXT_DEBUG_SKIP_MLP_CALL", "0") == "1":
            continue
        if debug_mlp_metadata:
            logger.warning(
                "Flash-Next verify MLP metadata layer=%d rows=%d shape=%s stride=%s "
                "contiguous=%s ptr_mod256=%d router_cap=%s",
                layer.layer_idx,
                mixed2.shape[0],
                tuple(mixed2.shape),
                tuple(mixed2.stride()),
                mixed2.is_contiguous(),
                mixed2.data_ptr() % 256,
                getattr(layer.mlp._router_weights, "shape", None),
            )
        if (
            debug_router_stats
            and layer.layer_idx == 0
            and not torch.cuda.is_current_stream_capturing()
        ):
            logger.warning(
                "Flash-Next verify router input rows=%d finite=%s min=%s max=%s absmax=%s",
                mixed2.shape[0],
                bool(torch.isfinite(mixed2).all().item()),
                float(mixed2.float().amin().item()),
                float(mixed2.float().amax().item()),
                float(mixed2.float().abs().amax().item()),
            )
        if os.environ.get("QSR_FLASHNEXT_DEBUG_ZERO_MLP", "0") == "1":
            mlp_out = torch.zeros(
                mixed2.shape[0],
                model.cfg.hidden_size,
                dtype=mixed2.dtype,
                device=mixed2.device,
            )
        elif exact_row_math:
            mlp_out = torch.cat(
                [layer.mlp(mixed2[row : row + 1]).clone() for row in range(mixed2.shape[0])]
            )
        else:
            mlp_out = layer.mlp(mixed2)
        if debug_clone_mlp:
            mlp_out = mlp_out.clone()
        if debug_skip_mlp_combine:
            continue
        x = combine_rows(layer.mlp_hc, mlp_out, res2)

    hc_hidden = x
    mixed, _ = mix_rows(model.final_mixer, x)
    if exact_row_math and not batch_lm_head:
        logits = torch.cat(
            [model.lm_head(mixed[row : row + 1]).float() for row in range(mixed.shape[0])]
        )
    else:
        logits = model.lm_head(mixed).float()
    return hc_hidden, logits


class FlashNextVerifyGraph:
    """Captured K+1 target verify with O(number-of-state-layers) commit."""

    def __init__(
        self,
        model: FlashNextModel,
        sess: FlashNextSession,
        device: torch.device | str,
        *,
        k: int = 3,
        exact_row_math: bool = False,
        batch_lm_head: bool = False,
        batch_gdn_recurrence: bool = True,
        # Keep the BF16-safe per-row path unless the backend has performed the
        # format capability check before constructing the graph.
        batch_gdn_projections: bool = False,
        sequential_qsa: bool = False,
        recompute_recurrent_state: bool = False,
        buffers: FlashNextVerifyBuffers | None = None,
    ) -> None:
        if k < 1:
            raise ValueError(f"Flash-Next batch verify requires K>=1, got {k}")
        if sess.qsa_k_pool is None or sess.token_buf is None:
            raise ValueError("prepare_graph_buffers must run before verify graph allocation")
        self.model = model
        self.sess = sess
        self.device = device
        self.k = k
        self.qo_len = k + 1
        self.exact_row_math = exact_row_math
        self.batch_lm_head = batch_lm_head
        self.batch_gdn_recurrence = batch_gdn_recurrence
        self.batch_gdn_projections = batch_gdn_projections
        self.sequential_qsa = sequential_qsa
        self.recompute_recurrent_state = recompute_recurrent_state
        if recompute_recurrent_state and (
            exact_row_math or not batch_gdn_recurrence
        ):
            raise ValueError(
                "recomputed verify state requires batched FP32 GDN recurrence"
            )
        self._gdn_commit_inputs: dict[int, dict[str, torch.Tensor]] = (
            {
                layer.layer_idx: {}
                for layer in model.layers
                if not layer.is_qsa
            }
            if recompute_recurrent_state
            else {}
        )
        capacities = [
            capacity
            for capacity in (
                _pool_capacity(pool)
                for pool in sess.qsa_k_pool.values()
            )
            if capacity is not None
        ]
        self.max_seq = min(capacities) if capacities else None
        if buffers is None:
            buffers = allocate_verify_buffers(
                model,
                sess,
                qo_len=self.qo_len,
                device=device,
                allocate_sequential_work=(
                    self.exact_row_math and not self.batch_gdn_recurrence
                ),
                recompute_recurrent_state=self.recompute_recurrent_state,
            )
        elif buffers.token_ids.shape != (self.qo_len,):
            raise ValueError(
                "shared verify buffers have the wrong fixed row count: "
                f"expected {self.qo_len}, got {buffers.token_ids.shape[0]}"
            )
        self.buffers = buffers
        self.graph: torch.cuda.CUDAGraph | None = None
        self._hc_hidden: torch.Tensor | None = None
        self._logits: torch.Tensor | None = None
        self._last_tokens: tuple[int, ...] | None = None
        self._last_past_len: int | None = None
        self.last_ple_seconds = 0.0

    def _validate_past_len(self, past_len: int) -> None:
        if past_len < 0:
            raise ValueError(f"verify past length must be non-negative, got {past_len}")
        if self.max_seq is not None and past_len + self.qo_len > self.max_seq:
            raise ValueError(
                "verify exceeds target cache capacity: "
                f"past_len={past_len}, rows={self.qo_len}, capacity={self.max_seq}"
            )

    def _prepare_ple(self, token_ids: Sequence[int]) -> None:
        started = time.perf_counter()
        for layer in self.model.layers:
            if layer.ple is None:
                continue
            history = [*self.sess.window, *token_ids]
            # Hash on CPU: gather ultimately needs Python row ids for pread,
            # so putting this tiny window on CUDA only adds a synchronization.
            history_tensor = torch.tensor(history, dtype=torch.long)
            ids = layer.ple_hasher.sequence_ids(history_tensor)[-self.qo_len :]
            layer.ple.embed(ids, device=self.device, out=self.buffers.ple_embeddings)
            self.last_ple_seconds = time.perf_counter() - started
            return
        self.last_ple_seconds = time.perf_counter() - started

    def capture(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Flash-Next verify CUDA Graph capture requires CUDA")
        self._validate_past_len(0)
        self.buffers.positions.copy_(
            torch.arange(self.qo_len, dtype=torch.long, device=self.device)
        )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                verify_body(
                    self.model,
                    self.sess,
                    self.buffers,
                    exact_row_math=self.exact_row_math,
                    batch_lm_head=self.batch_lm_head,
                    batch_gdn_recurrence=self.batch_gdn_recurrence,
                    batch_gdn_projections=self.batch_gdn_projections,
                    sequential_qsa=self.sequential_qsa,
                    gdn_commit_inputs=self._gdn_commit_inputs,
                )
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._hc_hidden, self._logits = verify_body(
                self.model,
                self.sess,
                self.buffers,
                exact_row_math=self.exact_row_math,
                batch_lm_head=self.batch_lm_head,
                batch_gdn_recurrence=self.batch_gdn_recurrence,
                batch_gdn_projections=self.batch_gdn_projections,
                sequential_qsa=self.sequential_qsa,
                gdn_commit_inputs=self._gdn_commit_inputs,
            )
        self.graph = graph
        self.model._retain_graph_moe_allocations()

    def replay(
        self,
        token_ids: Sequence[int],
        *,
        past_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(token_ids) != self.qo_len:
            raise ValueError(
                f"verify requires anchor + {self.k} drafts ({self.qo_len} rows), "
                f"got {len(token_ids)}"
            )
        if self.graph is None:
            raise RuntimeError("capture must run before replay")
        if past_len is None:
            past_len = self.sess.pos
        self._validate_past_len(past_len)
        self._prepare_ple(token_ids)
        self.buffers.token_ids.copy_(
            torch.as_tensor(token_ids, dtype=torch.long, device=self.device)
        )
        self.buffers.positions.copy_(
            torch.arange(
                past_len,
                past_len + self.qo_len,
                dtype=torch.long,
                device=self.device,
            )
        )
        self.graph.replay()
        self._last_tokens = tuple(int(token) for token in token_ids)
        self._last_past_len = past_len
        return self._hc_hidden, self._logits

    def eager(
        self,
        token_ids: Sequence[int],
        *,
        past_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Uncaptured correctness oracle with the identical fixed-row ABI."""
        if len(token_ids) != self.qo_len:
            raise ValueError(f"expected {self.qo_len} verify tokens, got {len(token_ids)}")
        if past_len is None:
            past_len = self.sess.pos
        self._validate_past_len(past_len)
        self._prepare_ple(token_ids)
        self.buffers.token_ids.copy_(
            torch.as_tensor(token_ids, dtype=torch.long, device=self.device)
        )
        self.buffers.positions.copy_(
            torch.arange(
                past_len,
                past_len + self.qo_len,
                dtype=torch.long,
                device=self.device,
            )
        )
        # Keep graph-owned output handles intact.  ``eager`` is also used as
        # a correctness oracle while a graph is live; rebinding ``_logits``
        # here makes the next ``graph.replay()`` return the previous eager
        # allocation even though CUDA wrote the captured output address.
        hc_hidden, logits = verify_body(
            self.model,
            self.sess,
            self.buffers,
            exact_row_math=self.exact_row_math,
            batch_lm_head=self.batch_lm_head,
            batch_gdn_recurrence=self.batch_gdn_recurrence,
            batch_gdn_projections=self.batch_gdn_projections,
            sequential_qsa=getattr(self, "sequential_qsa", False),
            gdn_commit_inputs=self._gdn_commit_inputs,
        )
        self._last_tokens = tuple(int(token) for token in token_ids)
        self._last_past_len = past_len
        return hc_hidden, logits

    def _rollback_speculative_qsa(self, committed_end: int) -> None:
        """Remove target-QSA rows that were verified but not committed.

        Target verify writes all ``anchor + K`` rows into the fixed QSA pools
        before the accept/reject decision is known.  GDN and PLE already keep
        per-row snapshots, but QSA used to leave rejected rows in both the
        raw index cache and its pooled block keys.  A following decode could
        then score/select a block containing an uncommitted draft, changing
        the greedy target trajectory after the first rejection.

        Restore the rows after ``committed_end`` and rebuild every compression
        group touched by this verify.  Complete groups are reconstructed from
        the surviving raw keys; a trailing partial group is cleared because
        the next real decode write will rebuild it.  The operation is outside
        CUDA Graph replay and uses the same fixed pools as the captured body.
        """
        past_len = self._last_past_len
        if past_len is None:
            raise RuntimeError("verify must run before QSA rollback")
        verify_end = past_len + self.qo_len
        if not past_len <= committed_end <= verify_end:
            raise ValueError(
                "QSA rollback range is invalid: "
                f"past_len={past_len}, committed_end={committed_end}, verify_end={verify_end}"
            )
        if committed_end == verify_end:
            return

        for layer in self.model.layers:
            if not layer.is_qsa:
                continue
            layer_idx = layer.layer_idx
            idx_pool = self.sess.qsa_idx_k_pool[layer_idx]
            pooled_pool = self.sess.qsa_pooled_k_pool[layer_idx]
            capacity = idx_pool.shape[0]
            rejected = torch.arange(
                committed_end,
                verify_end,
                dtype=torch.long,
                device=idx_pool.device,
            ).remainder(capacity)
            idx_pool.index_fill_(0, rejected, 0)
            rope_pool = (
                self.sess.qsa_idx_rope_pool.get(layer_idx)
                if self.sess.qsa_idx_rope_pool is not None
                else None
            )
            if rope_pool is not None:
                rope_pool.index_fill_(0, rejected, 0)

            k_pool = self.sess.qsa_k_pool[layer_idx]
            v_pool = self.sess.qsa_v_pool[layer_idx]
            k_pool[committed_end:verify_end].zero_()
            v_pool[committed_end:verify_end].zero_()
            k_scale_pool = self.sess.qsa_k_scale_pool.get(layer_idx)
            v_scale_pool = self.sess.qsa_v_scale_pool.get(layer_idx)
            if k_scale_pool is not None:
                k_scale_pool[committed_end:verify_end].fill_(1.0)
            if v_scale_pool is not None:
                v_scale_pool[committed_end:verify_end].fill_(1.0)

            ratio = layer.attn.indexer.compress_ratio
            first_group = past_len // ratio
            last_group = (verify_end + ratio - 1) // ratio
            group_ids = torch.arange(
                first_group,
                last_group,
                dtype=torch.long,
                device=idx_pool.device,
            )
            if group_ids.numel() == 0:
                continue
            group_positions = group_ids * ratio
            complete = group_positions + ratio <= committed_end
            if complete.any():
                complete_ids = group_ids[complete]
                token_positions = (
                    complete_ids[:, None] * ratio
                    + layer.attn.indexer.group_offsets.to(idx_pool.device)[None, :]
                ).remainder(capacity)
                groups = idx_pool[token_positions]
                if rope_pool is None:
                    pooled = layer.attn.indexer.pool_key_groups_at_positions(
                        groups,
                        complete_ids * ratio,
                    )
                else:
                    group_rope = rope_pool[token_positions][:, 0, :]
                    pooled = layer.attn.indexer.pool_key_groups_at_positions(
                        groups,
                        group_rope,
                    )
                pooled_pool.index_copy_(0, complete_ids, pooled)
            incomplete_ids = group_ids[~complete]
            if incomplete_ids.numel() > 0:
                pooled_pool.index_fill_(0, incomplete_ids, 0)

    def commit(self, accepted_count: int) -> None:
        """Commit anchor plus accepted drafts without a target replay."""
        if self._last_tokens is None:
            raise RuntimeError("verify must run before commit")
        if not 1 <= accepted_count <= self.qo_len:
            raise ValueError(
                f"accepted_count must include the anchor and be in [1,{self.qo_len}], "
                f"got {accepted_count}"
            )
        if self._last_past_len is None:
            raise RuntimeError("verify past length is missing")
        self._rollback_speculative_qsa(self._last_past_len + accepted_count)
        row = accepted_count - 1
        for layer in self.model.layers:
            if layer.is_qsa:
                continue
            live = self.sess.gdn[f"gdn_{layer.layer_idx}"]
            chosen = self.buffers.gdn_rows[layer.layer_idx][row]
            live.conv_state.copy_(chosen.conv_state)
            commit_inputs = self._gdn_commit_inputs.get(layer.layer_idx)
            if commit_inputs is None:
                live.recurrent_state.copy_(chosen.recurrent_state)
            else:
                scratch = self.buffers.gdn_recompute_scratch
                scratch_output = self.buffers.gdn_recompute_output
                if scratch is None or scratch_output is None:
                    raise RuntimeError("recomputed GDN verify scratch is missing")
                required = {"q", "k", "v", "a", "b"}
                if commit_inputs.keys() != required:
                    raise RuntimeError("recomputed GDN verify inputs are incomplete")
                from runtime.kernels.flashnext_gdn_verify import flashnext_gdn_commit

                flashnext_gdn_commit(
                    **commit_inputs,
                    a_log=layer.attn.A_log.float(),
                    dt_bias=layer.attn.dt_bias,
                    state=live.recurrent_state,
                    accepted_count=accepted_count,
                    scratch_states=scratch.unsqueeze(0),
                    scratch_output=scratch_output,
                )
            live.has_previous_state = True
        if self.buffers.ple_rows:
            self.sess.ple_conv_state.copy_(self.buffers.ple_rows[row])
        committed = self._last_tokens[:accepted_count]
        self.sess.window.extend(committed)
        self.sess.window = self.sess.window[-self.model.cfg.ngram_size :]
        self.sess.pos += accepted_count
        self._last_tokens = None
        self._last_past_len = None


@dataclass
class FlashNextMtpSession:
    """Persistent QSA cache for one Flash-Next MTP slot."""

    mtp_k_pool: torch.Tensor
    mtp_v_pool: torch.Tensor
    mtp_idx_k_pool: torch.Tensor
    mtp_pooled_k_pool: torch.Tensor
    mtp_k_scale_pool: torch.Tensor | None = None
    mtp_v_scale_pool: torch.Tensor | None = None
    sync_len: int = 0
    pos: int = 0
    shared_sparse_indices: torch.Tensor | None = None
    shared_sparse_valid: torch.Tensor | None = None
    shared_sparse_captured_len: int = 0
    sparse_graph_buffers: FlashNextMtpSparseGraphBuffers | None = None
    # Candidate QSA rows use fixed graph-owned scratch.  ``mtp_k_pool[pos]``
    # is an advanced-indexing copy; passing that copy to ``quantize_qsa_kv``
    # silently drops the write and can leave a captured MTP graph reading
    # stale/invalid cache data.  Keep enough rows for the largest proposal
    # batch (K+1) and commit them with the dtype-aware indexed copy.
    mtp_k_rows: torch.Tensor | None = None
    mtp_v_rows: torch.Tensor | None = None
    mtp_k_scale_rows: torch.Tensor | None = None
    mtp_v_scale_rows: torch.Tensor | None = None


@dataclass
class FlashNextMtpSparseGraphBuffers:
    """Caller-owned fixed sparse-QSA scratch for graph-safe MTP replay."""

    pooled_source: torch.Tensor
    pooled_positions: torch.Tensor
    pooled_columns: torch.Tensor
    shared_indices: torch.Tensor
    shared_valid: torch.Tensor
    shared_captured_len: torch.Tensor
    row_block_ends: torch.Tensor
    block_logits: torch.Tensor
    block_indices: torch.Tensor
    gather_indices: torch.Tensor
    gather_valid: torch.Tensor
    reuse_indices: torch.Tensor
    reuse_valid: torch.Tensor
    tail_indices: torch.Tensor
    reuse_tail_indices: torch.Tensor


def prepare_mtp_sparse_graph_buffers(
    mtp: FlashNextMTP,
    sess: FlashNextMtpSession,
    *,
    max_rows: int,
    device: torch.device | str,
) -> None:
    if max_rows <= 0:
        raise ValueError(f"MTP sparse graph buffers require positive rows, got {max_rows}")
    pooled_rows = sess.mtp_pooled_k_pool.shape[0]
    extra_tail = max(max_rows - 2, 0)
    sess.sparse_graph_buffers = FlashNextMtpSparseGraphBuffers(
        # Raw index keys are a fixed-address ring.  Share its backing storage
        # with graph replay rather than retaining a second max-sequence copy.
        pooled_source=sess.mtp_idx_k_pool,
        pooled_positions=torch.arange(
            0,
            pooled_rows * mtp.indexer.compress_ratio,
            mtp.indexer.compress_ratio,
            dtype=torch.long,
            device=device,
        ),
        pooled_columns=torch.arange(
            pooled_rows,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0),
        shared_indices=torch.zeros(
            1,
            mtp.qsa_pad,
            dtype=torch.long,
            device=device,
        ),
        shared_valid=torch.zeros(
            1,
            mtp.qsa_pad,
            dtype=torch.bool,
            device=device,
        ),
        shared_captured_len=torch.zeros(1, dtype=torch.long, device=device),
        row_block_ends=torch.zeros(max_rows, dtype=torch.long, device=device),
        block_logits=torch.empty(
            max_rows,
            pooled_rows,
            dtype=torch.float32,
            device=device,
        ),
        block_indices=torch.empty(
            max_rows,
            mtp.indexer.block_topk,
            dtype=torch.long,
            device=device,
        ),
        gather_indices=torch.empty(
            max_rows,
            mtp.qsa_pad,
            dtype=torch.long,
            device=device,
        ),
        gather_valid=torch.empty(
            max_rows,
            mtp.qsa_pad,
            dtype=torch.bool,
            device=device,
        ),
        reuse_indices=torch.empty(
            max_rows,
            mtp.qsa_pad + extra_tail,
            dtype=torch.long,
            device=device,
        ),
        reuse_valid=torch.empty(
            max_rows,
            mtp.qsa_pad + extra_tail,
            dtype=torch.bool,
            device=device,
        ),
        tail_indices=torch.empty(
            max_rows,
            mtp.indexer.compress_ratio - 1,
            dtype=torch.long,
            device=device,
        ),
        reuse_tail_indices=torch.empty(
            max_rows,
            extra_tail,
            dtype=torch.long,
            device=device,
        ),
    )


class FlashNextMtpContinuationGraph:
    """Captured K-1 MTP continuation chain.

    Draft token ids stay on device between unrolled steps.  Replay crosses
    the host boundary once for the completed chain instead of once per token.
    """

    def __init__(
        self,
        model: FlashNextModel,
        mtp: FlashNextMTP,
        sess: FlashNextMtpSession,
        *,
        device: torch.device | str,
        graph_capacity: int,
        continuation_steps: int,
        sparse_qsa: bool = False,
    ) -> None:
        self.model = model
        self.mtp = mtp
        self.sess = sess
        self.graph_capacity = graph_capacity
        self.sparse_qsa = sparse_qsa
        _validate_pool_range(
            getattr(sess, "mtp_k_pool", None),
            0,
            graph_capacity,
            "MTP continuation graph capacity",
        )
        if continuation_steps <= 0:
            raise ValueError("MTP continuation graph requires at least one step")
        self.continuation_steps = continuation_steps
        cfg = model.cfg
        self.token = torch.zeros(1, dtype=torch.long, device=device)
        self.hidden = torch.zeros(
            1,
            cfg.hc_count * cfg.hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        self.position = torch.zeros(1, dtype=torch.long, device=device)
        self.graph: torch.cuda.CUDAGraph | None = None
        self._tokens: torch.Tensor | None = None
        self._next_hidden: torch.Tensor | None = None

    def _body(self) -> tuple[torch.Tensor, torch.Tensor]:
        token = self.token
        hidden = self.hidden
        next_tokens = []
        for step in range(self.continuation_steps):
            embeds = self.model.embed_tokens(token)
            position = self.position + step
            mixed, hidden = self.mtp.forward(
                embeds,
                hidden,
                position,
                self.sess,
                reuse_sparse_indices=self.sparse_qsa,
                graph_sparse_capacity=self.graph_capacity if self.sparse_qsa else None,
                graph_dense_capacity=None if self.sparse_qsa else self.graph_capacity,
            )
            token = self.model.lm_head(mixed).argmax(dim=-1)
            next_tokens.append(token)
        return torch.cat(next_tokens), hidden

    def capture(self) -> None:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._body()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._tokens, self._next_hidden = self._body()
        self.graph = graph
        self.sess.mtp_k_pool.zero_()
        self.sess.mtp_v_pool.zero_()
        self.sess.mtp_idx_k_pool.zero_()
        self.sess.mtp_pooled_k_pool.zero_()

    def replay(
        self,
        token: int,
        hidden: torch.Tensor,
        position: int,
    ) -> torch.Tensor:
        if self.graph is None:
            raise RuntimeError("capture must run before MTP continuation replay")
        if position + self.continuation_steps > self.graph_capacity:
            raise ValueError(
                "MTP graph continuation exceeds graph capacity: "
                f"position={position}, steps={self.continuation_steps}, "
                f"capacity={self.graph_capacity}"
            )
        _validate_pool_range(
            getattr(self.sess, "mtp_k_pool", None),
            position,
            self.continuation_steps,
            "MTP graph continuation",
        )
        self.token.fill_(token)
        self.hidden.copy_(hidden)
        self.position.fill_(position)
        self.graph.replay()
        return self._tokens


class FlashNextMtpSampledTeacherGraph:
    """Captured teacher-sync row block for sampled MTP.

    The sampled path cannot use :class:`FlashNextMtpProposalGraph` directly:
    that graph hard-wires argmax tokens into the continuation chain.  The
    teacher block itself is independent of the random draw, however, so it
    can stay captured and return the final draft logits/hidden state.  Keeping
    this part on the graph removes the expensive eager launch/allocator path
    for the common ``anchor + accepted drafts`` sync widths while preserving
    the exact request sampling distribution outside the graph.
    """

    def __init__(
        self,
        model: FlashNextModel,
        mtp: FlashNextMTP,
        sess: FlashNextMtpSession,
        *,
        device: torch.device | str,
        graph_capacity: int,
        query_len: int,
        sparse_qsa: bool = False,
    ) -> None:
        if query_len <= 0:
            raise ValueError("sampled MTP teacher graph requires query_len > 0")
        self.model = model
        self.mtp = mtp
        self.sess = sess
        self.graph_capacity = graph_capacity
        self.query_len = query_len
        self.sparse_qsa = sparse_qsa
        _validate_pool_range(
            getattr(sess, "mtp_k_pool", None),
            0,
            query_len,
            "sampled MTP teacher graph capacity",
        )
        cfg = model.cfg
        self.tokens = torch.zeros(query_len, dtype=torch.long, device=device)
        self.target_hidden = torch.zeros(
            query_len,
            cfg.hc_count * cfg.hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        self.teacher_embeds = torch.zeros(
            query_len,
            cfg.hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        self.position = torch.zeros(1, dtype=torch.long, device=device)
        self.position_offsets = torch.arange(query_len, dtype=torch.long, device=device)
        self.graph: torch.cuda.CUDAGraph | None = None
        self._hidden: torch.Tensor | None = None
        self._logits: torch.Tensor | None = None

    def _body(self) -> tuple[torch.Tensor, torch.Tensor]:
        positions = self.position + self.position_offsets
        mixed, own_hidden = self.mtp.forward(
            self.teacher_embeds,
            self.target_hidden,
            positions,
            self.sess,
            capture_sparse_indices=True,
            graph_sparse_capacity=self.graph_capacity if self.sparse_qsa else None,
            graph_dense_capacity=None if self.sparse_qsa else self.graph_capacity,
        )
        return own_hidden[-1:], self.model.lm_head(mixed[-1:]).float()

    def capture(self) -> None:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._body()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._hidden, self._logits = self._body()
        self.graph = graph
        self.sess.mtp_k_pool.zero_()
        self.sess.mtp_v_pool.zero_()
        self.sess.mtp_idx_k_pool.zero_()
        self.sess.mtp_pooled_k_pool.zero_()

    def replay(
        self,
        shifted_token_ids: Sequence[int],
        target_hidden: torch.Tensor,
        position: int,
        *,
        input_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.graph is None:
            raise RuntimeError("capture must run before sampled MTP teacher replay")
        if len(shifted_token_ids) != self.query_len:
            raise ValueError(
                f"sampled teacher graph requires {self.query_len} rows, "
                f"got {len(shifted_token_ids)}"
            )
        if position + self.query_len > self.graph_capacity:
            raise ValueError("sampled MTP teacher graph exceeds graph capacity")
        _validate_pool_range(
            getattr(self.sess, "mtp_k_pool", None),
            position,
            self.query_len,
            "sampled MTP teacher graph",
        )
        self.tokens.copy_(
            torch.as_tensor(shifted_token_ids, dtype=torch.long, device=self.tokens.device)
        )
        if input_embeds is None:
            input_embeds = self.model.embed_tokens(self.tokens)
        if tuple(input_embeds.shape) != tuple(self.teacher_embeds.shape):
            raise ValueError(
                "sampled MTP teacher input_embeds must have shape "
                f"{tuple(self.teacher_embeds.shape)}, got {tuple(input_embeds.shape)}"
            )
        self.teacher_embeds.copy_(
            input_embeds.to(
                device=self.teacher_embeds.device,
                dtype=self.teacher_embeds.dtype,
            )
        )
        if target_hidden.ndim == 3:
            target_hidden = target_hidden.squeeze(0)
        self.target_hidden.copy_(target_hidden)
        self.position.fill_(position)
        self.graph.replay()
        return self._hidden, self._logits


class FlashNextMtpSampledStepGraph:
    """Captured one-token MTP step with logits as an external sample seam.

    Random sampling remains outside CUDA Graph replay, but the recurrent MTP
    forward and shared LM head stay on a fixed graph.  The next sampled token
    is copied into the graph input buffer without a host ``.item()`` round
    trip, so a K=3 proposal performs three graph replays and one final token
    transfer instead of synchronizing once per draft row.
    """

    def __init__(
        self,
        model: FlashNextModel,
        mtp: FlashNextMTP,
        sess: FlashNextMtpSession,
        *,
        device: torch.device | str,
        graph_capacity: int,
        sparse_qsa: bool = False,
    ) -> None:
        self.model = model
        self.mtp = mtp
        self.sess = sess
        self.graph_capacity = graph_capacity
        self.sparse_qsa = sparse_qsa
        _validate_pool_range(
            getattr(sess, "mtp_k_pool", None),
            0,
            1,
            "sampled MTP step graph capacity",
        )
        cfg = model.cfg
        self.token = torch.zeros(1, dtype=torch.long, device=device)
        self.hidden = torch.zeros(
            1,
            cfg.hc_count * cfg.hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        self.position = torch.zeros(1, dtype=torch.long, device=device)
        self.graph: torch.cuda.CUDAGraph | None = None
        self._hidden: torch.Tensor | None = None
        self._logits: torch.Tensor | None = None

    def _body(self) -> tuple[torch.Tensor, torch.Tensor]:
        embeds = self.model.embed_tokens(self.token)
        mixed, own_hidden = self.mtp.forward(
            embeds,
            self.hidden,
            self.position,
            sess=self.sess,
            reuse_sparse_indices=self.sparse_qsa,
            graph_sparse_capacity=self.graph_capacity if self.sparse_qsa else None,
            graph_dense_capacity=None if self.sparse_qsa else self.graph_capacity,
        )
        return own_hidden, self.model.lm_head(mixed).float()

    def capture(self) -> None:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._body()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._hidden, self._logits = self._body()
        self.graph = graph
        self.sess.mtp_k_pool.zero_()
        self.sess.mtp_v_pool.zero_()
        self.sess.mtp_idx_k_pool.zero_()
        self.sess.mtp_pooled_k_pool.zero_()

    def replay(
        self,
        token: torch.Tensor,
        hidden: torch.Tensor,
        position: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.graph is None:
            raise RuntimeError("capture must run before sampled MTP step replay")
        if position + 1 > self.graph_capacity:
            raise ValueError("sampled MTP step graph exceeds graph capacity")
        _validate_pool_range(
            getattr(self.sess, "mtp_k_pool", None),
            position,
            1,
            "sampled MTP step graph",
        )
        self.token.copy_(token.reshape_as(self.token))
        self.hidden.copy_(hidden)
        self.position.fill_(position)
        self.graph.replay()
        return self._hidden, self._logits


class FlashNextMtpProposalGraph:
    """Teacher-sync plus K-token proposal in one graph for a fixed suffix width."""

    def __init__(
        self,
        model: FlashNextModel,
        mtp: FlashNextMTP,
        sess: FlashNextMtpSession,
        *,
        device: torch.device | str,
        graph_capacity: int,
        query_len: int,
        k: int,
        sparse_qsa: bool = False,
    ) -> None:
        if query_len <= 0 or k <= 0:
            raise ValueError("MTP proposal graph dimensions must be positive")
        self.model = model
        self.mtp = mtp
        self.sess = sess
        self.graph_capacity = graph_capacity
        self.query_len = query_len
        self.k = k
        self.sparse_qsa = sparse_qsa
        _validate_pool_range(
            getattr(sess, "mtp_k_pool", None),
            0,
            query_len + k - 1,
            "MTP proposal graph capacity",
        )
        cfg = model.cfg
        self.tokens = torch.zeros(query_len, dtype=torch.long, device=device)
        self.target_hidden = torch.zeros(
            query_len,
            cfg.hc_count * cfg.hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        # The target teacher rows may already contain vision features.  Keep
        # them in a graph-owned input buffer so the proposal graph can replay
        # the exact same MTP path instead of falling back to eager execution
        # whenever a multimodal prompt is admitted.  Text requests populate
        # this buffer from the ordinary token embedding lookup before replay.
        self.teacher_embeds = torch.zeros(
            query_len,
            cfg.hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        self.position = torch.zeros(1, dtype=torch.long, device=device)
        self.position_offsets = torch.arange(query_len, dtype=torch.long, device=device)
        self.graph: torch.cuda.CUDAGraph | None = None
        self._drafts: torch.Tensor | None = None

    def _body(self) -> torch.Tensor:
        positions = self.position + self.position_offsets
        embeds = self.teacher_embeds
        mixed, own_hidden = self.mtp.forward(
            embeds,
            self.target_hidden,
            positions,
            self.sess,
            capture_sparse_indices=self.sparse_qsa,
            graph_sparse_capacity=self.graph_capacity if self.sparse_qsa else None,
            graph_dense_capacity=None if self.sparse_qsa else self.graph_capacity,
        )
        token = self.model.lm_head(mixed[-1:]).argmax(dim=-1)
        hidden = own_hidden[-1:]
        drafts = [token]
        for step in range(self.k - 1):
            draft_position = self.position + self.query_len + step
            draft_embed = self.model.embed_tokens(token)
            mixed, hidden = self.mtp.forward(
                draft_embed,
                hidden,
                draft_position,
                self.sess,
                reuse_sparse_indices=self.sparse_qsa,
                graph_sparse_capacity=self.graph_capacity if self.sparse_qsa else None,
                graph_dense_capacity=None if self.sparse_qsa else self.graph_capacity,
            )
            token = self.model.lm_head(mixed).argmax(dim=-1)
            drafts.append(token)
        return torch.cat(drafts)

    def capture(self) -> None:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._body()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._drafts = self._body()
        self.graph = graph
        self.sess.mtp_k_pool.zero_()
        self.sess.mtp_v_pool.zero_()
        self.sess.mtp_idx_k_pool.zero_()
        self.sess.mtp_pooled_k_pool.zero_()

    def replay(
        self,
        shifted_token_ids: Sequence[int],
        target_hidden: torch.Tensor,
        position: int,
        *,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.graph is None:
            raise RuntimeError("capture must run before MTP proposal replay")
        if len(shifted_token_ids) != self.query_len:
            raise ValueError(
                f"proposal graph requires {self.query_len} teacher rows, "
                f"got {len(shifted_token_ids)}"
            )
        if position + self.query_len + self.k - 1 > self.graph_capacity:
            raise ValueError("MTP proposal graph exceeds graph capacity")
        _validate_pool_range(
            getattr(self.sess, "mtp_k_pool", None),
            position,
            self.query_len + self.k - 1,
            "MTP graph proposal",
        )
        token_tensor = torch.as_tensor(
            shifted_token_ids,
            dtype=torch.long,
            device=self.tokens.device,
        )
        self.tokens.copy_(token_tensor)
        if input_embeds is None:
            input_embeds = self.model.embed_tokens(token_tensor)
        if tuple(input_embeds.shape) != tuple(self.teacher_embeds.shape):
            raise ValueError(
                "MTP proposal input_embeds must have shape "
                f"{tuple(self.teacher_embeds.shape)}, got {tuple(input_embeds.shape)}"
            )
        self.teacher_embeds.copy_(
            input_embeds.to(
                device=self.teacher_embeds.device,
                dtype=self.teacher_embeds.dtype,
            )
        )
        self.target_hidden.copy_(target_hidden)
        self.position.fill_(position)
        self.graph.replay()
        return self._drafts


def new_mtp_session(
    mtp: FlashNextMTP,
    *,
    max_seq: int,
    device: torch.device | str,
    fixed_index_rows: int = 4,
) -> FlashNextMtpSession:
    if max_seq <= 0:
        raise ValueError(f"MTP session max_seq must be positive, got {max_seq}")
    attn = mtp.attn
    mtp.qsa_pad = (
        mtp.indexer.block_topk * mtp.indexer.compress_ratio + mtp.indexer.compress_ratio - 1
    )
    from runtime.model.flashnext.qsa import QsaDecodeAttention

    mtp.decode_attn = QsaDecodeAttention(attn, mtp.qsa_pad)
    kv_dtype = qsa_kv_cache_dtype()
    row_capacity = max(1, int(fixed_index_rows))
    k_pool = torch.zeros(
        max_seq,
        attn.num_kv_heads,
        attn.head_dim,
        dtype=kv_dtype,
        device=device,
    )
    scale_pool = torch.ones(max_seq, attn.num_kv_heads, dtype=torch.float16, device=device)
    return FlashNextMtpSession(
        mtp_k_pool=k_pool,
        mtp_v_pool=torch.zeros_like(k_pool),
        mtp_idx_k_pool=torch.zeros(
            qsa_index_cache_rows(
                max_seq,
                mtp.indexer.compress_ratio,
                fixed_rows=fixed_index_rows,
            ),
            mtp.indexer.head_dim,
            dtype=torch.bfloat16,
            device=device,
        ),
        mtp_pooled_k_pool=torch.zeros(
            (max_seq + mtp.indexer.compress_ratio - 1) // mtp.indexer.compress_ratio,
            mtp.indexer.head_dim,
            dtype=torch.bfloat16,
            device=device,
        ),
        mtp_k_scale_pool=scale_pool,
        mtp_v_scale_pool=torch.ones_like(scale_pool),
        mtp_k_rows=torch.empty(
            row_capacity,
            attn.num_kv_heads,
            attn.head_dim,
            dtype=kv_dtype,
            device=device,
        ),
        mtp_v_rows=torch.empty(
            row_capacity,
            attn.num_kv_heads,
            attn.head_dim,
            dtype=kv_dtype,
            device=device,
        ),
        mtp_k_scale_rows=torch.empty(
            row_capacity,
            attn.num_kv_heads,
            dtype=torch.float16,
            device=device,
        ),
        mtp_v_scale_rows=torch.empty(
            row_capacity,
            attn.num_kv_heads,
            dtype=torch.float16,
            device=device,
        ),
    )


class FlashNextSpecEngine:
    """K-token MTP driver over the fixed-row target verify engine.

    The MTP cache tracks only the teacher-forced real prefix in ``sync_len``;
    continuation rows after it are speculative and are overwritten on every
    round.  Greedy requests use the captured proposal/continuation graphs.
    Sampled requests use the same temperature/top-k/top-p distributions as
    ordinary decode and the exact rejection-sampling contract from
    :mod:`runtime.mtp_accept`; random draws stay outside the graph while the
    fixed teacher/step MTP kernels use sampled-specific CUDA Graph seams.
    """

    def __init__(
        self,
        model: FlashNextModel,
        mtp: FlashNextMTP,
        target_session: FlashNextSession,
        *,
        max_seq: int,
        device: torch.device | str,
        k: int = 3,
        exact_row_math: bool = False,
        batch_lm_head: bool = False,
        batch_gdn_recurrence: bool = True,
        # Keep the BF16-safe per-row path unless the backend has performed the
        # format capability check before constructing the MTP engine.
        batch_gdn_projections: bool = False,
        sequential_qsa: bool = False,
        recompute_recurrent_state: bool = False,
        mtp_continuation_graph: bool = False,
        mtp_sparse_graph: bool = False,
        verify_buffers: FlashNextVerifyBuffers | None = None,
    ) -> None:
        if max_seq <= 0:
            raise ValueError(f"MTP max_seq must be positive, got {max_seq}")
        self.model = model
        self.mtp = mtp
        self.target_session = target_session
        self.device = device
        self.max_seq = max_seq
        self.k = k
        self.verify = FlashNextVerifyGraph(
            model,
            target_session,
            device,
            k=k,
            exact_row_math=exact_row_math,
            batch_lm_head=batch_lm_head,
            batch_gdn_recurrence=batch_gdn_recurrence,
            batch_gdn_projections=batch_gdn_projections,
            sequential_qsa=sequential_qsa,
            recompute_recurrent_state=recompute_recurrent_state,
            buffers=verify_buffers,
        )
        self.mtp_session = new_mtp_session(
            mtp,
            max_seq=max_seq,
            device=device,
            fixed_index_rows=k + 1,
        )
        # The non-greedy proposer must retain q for the exact draft sequence
        # consumed by the next verify round.  This is one compact K x vocab
        # tensor (K=3 in production), not a second KV/state family.
        self.pending_draft_probs: torch.Tensor | None = None
        # A graph must keep its attention extent static.  Cover the complete
        # dense QSA budget so ordinary generations do not fall off the graph
        # cliff after position 512.  This is only the single MTP layer and its
        # grouped-GQA score tensor is small; the target's twelve QSA layers use
        # the fused sparse kernel instead.
        dense_capacity = min(
            max_seq,
            mtp.indexer.block_topk * mtp.indexer.compress_ratio,
        )
        graph_capacity = max_seq if mtp_sparse_graph else dense_capacity
        if mtp_sparse_graph:
            prepare_mtp_sparse_graph_buffers(
                mtp,
                self.mtp_session,
                max_rows=k + 1,
                device=device,
            )
        self.mtp_continuation_graph = (
            FlashNextMtpContinuationGraph(
                model,
                mtp,
                self.mtp_session,
                device=device,
                graph_capacity=graph_capacity,
                continuation_steps=k - 1,
                sparse_qsa=mtp_sparse_graph,
            )
            if mtp_continuation_graph and k > 1
            else None
        )
        # Sampled proposals cannot use the argmax-unrolled graphs below, but
        # their teacher-sync block and each stochastic one-token continuation
        # are still graph-safe.  Capture those fixed-width seams alongside
        # the greedy graphs so exact rejection sampling does not pay eager
        # allocator/launch overhead on every draft row.
        self.mtp_sampled_teacher_graphs = (
            {
                query_len: FlashNextMtpSampledTeacherGraph(
                    model,
                    mtp,
                    self.mtp_session,
                    device=device,
                    graph_capacity=graph_capacity,
                    query_len=query_len,
                    sparse_qsa=mtp_sparse_graph,
                )
                for query_len in range(1, k + 2)
            }
            if mtp_continuation_graph and k > 1
            else {}
        )
        self.mtp_sampled_step_graph = (
            FlashNextMtpSampledStepGraph(
                model,
                mtp,
                self.mtp_session,
                device=device,
                graph_capacity=graph_capacity,
                sparse_qsa=mtp_sparse_graph,
            )
            if mtp_continuation_graph and k > 1
            else None
        )
        # Acceptance is prompt-dependent: long generations commonly visit
        # every suffix width even when short benchmarks are dominated by full
        # acceptance.  Capture all ``anchor + accepted drafts`` widths so a
        # rejection never falls back to eager teacher sync plus a second graph
        # replay.  Each round then has exactly one proposal replay and one host
        # transfer regardless of its acceptance bucket.
        self.mtp_proposal_graphs = (
            {
                query_len: FlashNextMtpProposalGraph(
                    model,
                    mtp,
                    self.mtp_session,
                    device=device,
                    graph_capacity=graph_capacity,
                    query_len=query_len,
                    k=k,
                    sparse_qsa=mtp_sparse_graph,
                )
                for query_len in range(1, k + 2)
            }
            if mtp_continuation_graph and k > 1
            else {}
        )

    def capture_verify(self) -> None:
        self.verify.capture()
        if self.mtp_continuation_graph is not None:
            self.mtp_continuation_graph.capture()
        for graph in self.mtp_proposal_graphs.values():
            graph.capture()
        for graph in self.mtp_sampled_teacher_graphs.values():
            graph.capture()
        if self.mtp_sampled_step_graph is not None:
            self.mtp_sampled_step_graph.capture()

    def _validate_mtp_range(self, start: int, length: int, label: str) -> None:
        """Guard eager and graph MTP writes against the fixed cache extent."""
        for pool_name in ("mtp_k_pool", "mtp_v_pool"):
            _validate_pool_range(
                getattr(self.mtp_session, pool_name, None),
                start,
                length,
                f"{label} {pool_name}",
            )
        if start < 0 or length < 0:
            return
        if start + length > self.max_seq:
            raise ValueError(
                f"{label} exceeds configured max_seq: "
                f"start={start}, length={length}, max_seq={self.max_seq}"
            )

    @torch.no_grad()
    def sync_real_suffix(
        self,
        shifted_token_ids: Sequence[int],
        target_hc_hidden: torch.Tensor,
        *,
        input_embeds: torch.Tensor | None = None,
        params: SamplingParams | None = None,
        return_probs: bool = False,
        return_token_tensor: bool = False,
        return_first_token: bool = True,
    ) -> (
        tuple[int | torch.Tensor, torch.Tensor]
        | tuple[int | torch.Tensor, torch.Tensor, torch.Tensor | None]
    ):
        """Teacher-force real target rows and produce the first draft.

        ``shifted_token_ids[i]`` is the real token after the target position
        represented by ``target_hc_hidden[i]``.  This one-position shift is
        the NEXTN contract; feeding the same-position token was the cause of
        the earlier 0.40 acceptance ceiling in the bring-up probe.
        """
        query_len = len(shifted_token_ids)
        if query_len <= 0:
            raise ValueError("MTP sync requires at least one shifted real token")
        if target_hc_hidden.ndim == 3:
            if target_hc_hidden.shape[0] != 1:
                raise ValueError("MTP sync target hidden supports one sequence")
            target_hc_hidden = target_hc_hidden.squeeze(0)
        expected_shape = (query_len, self.model.cfg.hc_count * self.model.cfg.hidden_size)
        if tuple(target_hc_hidden.shape) != expected_shape:
            raise ValueError(
                f"MTP target hc hidden must have shape {expected_shape}, "
                f"got {tuple(target_hc_hidden.shape)}"
            )
        sess = self.mtp_session
        start = sess.sync_len
        self._validate_mtp_range(start, query_len, "MTP sync")
        sess.pos = start
        sampled = params is not None and not params.is_greedy
        sampled_graph = (
            getattr(self, "mtp_sampled_teacher_graphs", {}).get(query_len)
            if sampled
            else None
        )
        if (
            sampled_graph is not None
            and sampled_graph.graph is not None
            and start + query_len <= sampled_graph.graph_capacity
        ):
            own_hc, first_logits = sampled_graph.replay(
                shifted_token_ids,
                target_hc_hidden,
                start,
                input_embeds=input_embeds,
            )
        else:
            tokens = torch.as_tensor(
                shifted_token_ids,
                dtype=torch.long,
                device=self.device,
            )
            if input_embeds is None:
                embeds = self.model.embed_tokens(tokens)
            else:
                if tuple(input_embeds.shape) != (
                    query_len,
                    self.model.cfg.hidden_size,
                ):
                    raise ValueError(
                        "MTP sync input_embeds must have shape "
                        f"({query_len}, {self.model.cfg.hidden_size}), "
                        f"got {tuple(input_embeds.shape)}"
                    )
                embeds = input_embeds.to(device=self.device, dtype=torch.bfloat16)
            positions = torch.arange(
                start,
                start + query_len,
                dtype=torch.long,
                device=self.device,
            )
            mixed, own_hc = self.mtp.forward(
                embeds,
                target_hc_hidden,
                positions,
                sess,
                capture_sparse_indices=True,
            )
            first_logits = self.model.lm_head(mixed[-1:]) if return_first_token else None
        sess.sync_len = start + query_len
        sess.pos = sess.sync_len
        if not return_first_token:
            if params is not None:
                raise ValueError(
                    "MTP sync cannot sample when return_first_token is false"
                )
            first_probs = None
            # Intermediate chunk sync only needs to advance the recurrent and
            # QSA state.  Its first draft is intentionally discarded by the
            # caller, so avoid a vocabulary-wide lm_head and host argmax.
            first_draft = 0
        elif sampled:
            if first_logits is None:
                raise RuntimeError("MTP sync sampling requires first logits")
            try:
                first_probs = compute_sampling_distribution(first_logits, params)
                first_token = sample_from_distribution(
                    first_probs,
                    generator=make_generator(params.seed, str(self.device)),
                    output_device=first_logits.device,
                )
            except SamplingNumericalError as exc:
                raise SamplingNumericalError(
                    f"{exc}; Flash-Next MTP teacher sampling failed "
                    f"start={start},query_len={query_len},position={sess.pos},"
                    f"logits={_sampling_tensor_stats(first_logits)},"
                    f"target_hc_hidden={_sampling_tensor_stats(target_hc_hidden)}"
                ) from exc
            # Keep the sampled id on device while chaining the remaining MTP
            # rows.  ``.item()`` here used to force a CUDA sync before every
            # continuation, turning the exact sampled path into one host
            # round-trip per draft token.  The public/default ABI still
            # returns an int; the sampled driver opts into the tensor form.
            first_draft = (
                first_token.reshape(-1)
                if return_token_tensor
                else int(first_token.item())
            )
        else:
            if first_logits is None:
                raise RuntimeError("MTP sync requires first logits")
            first_probs = None
            first_draft = int(first_logits.argmax(dim=-1).item())
        if return_probs:
            return first_draft, own_hc[-1:], first_probs
        return first_draft, own_hc[-1:]

    @torch.no_grad()
    def continue_draft(
        self,
        first_draft: int | torch.Tensor,
        first_hidden: torch.Tensor,
        *,
        params: SamplingParams | None = None,
        first_probs: torch.Tensor | None = None,
        return_probs: bool = False,
    ) -> list[int] | tuple[list[int], torch.Tensor | None]:
        sampled = params is not None and not params.is_greedy
        if sampled and first_probs is None:
            raise RuntimeError("sampled MTP continuation is missing the first draft distribution")
        if torch.is_tensor(first_draft):
            if first_draft.numel() != 1:
                raise ValueError(
                    "sampled MTP continuation first draft must contain one token, "
                    f"got shape={tuple(first_draft.shape)}"
                )
            first_token = first_draft.reshape(-1).to(device=self.device, dtype=torch.long)
            first_id = None
        else:
            first_token = torch.tensor(
                [int(first_draft)], dtype=torch.long, device=self.device
            )
            first_id = int(first_draft)
        drafts = [first_id] if first_id is not None else []
        draft_token_tensors = [first_token] if sampled else []
        draft_probs: list[torch.Tensor] = []
        if sampled:
            assert first_probs is not None
            draft_probs.append(first_probs.squeeze(0))
        hidden = first_hidden
        sess = self.mtp_session
        self._validate_mtp_range(sess.pos, max(self.k - 1, 0), "MTP continuation")
        sampled_step_graph = (
            getattr(self, "mtp_sampled_step_graph", None) if sampled else None
        )
        if not sampled and self.mtp_continuation_graph is not None and (
            sess.pos + self.k - 1 <= self.mtp_continuation_graph.graph_capacity
        ):
            continuation = self.mtp_continuation_graph.replay(
                int(first_token.item()), hidden, sess.pos
            )
            sess.pos += self.k - 1
            result = [*drafts, *(int(token) for token in continuation.tolist())]
            return (result, None) if return_probs else result
        for draft_step in range(1, self.k):
            token_tensor = draft_token_tensors[-1] if sampled else torch.tensor(
                [drafts[-1]], dtype=torch.long, device=self.device
            )
            using_graph = (
                sampled_step_graph is not None
                and sampled_step_graph.graph is not None
                and sess.pos < sampled_step_graph.graph_capacity
            )
            if using_graph:
                hidden, logits = sampled_step_graph.replay(
                    token_tensor,
                    hidden,
                    sess.pos,
                )
            else:
                embeds = self.model.embed_tokens(token_tensor)
                positions = torch.tensor([sess.pos], dtype=torch.long, device=self.device)
                mixed, hidden = self.mtp.forward(
                    embeds,
                    hidden,
                    positions,
                    sess,
                    reuse_sparse_indices=True,
                )
                logits = self.model.lm_head(mixed)
            sess.pos += 1
            if sampled:
                try:
                    probs = compute_sampling_distribution(logits, params)
                    next_token = sample_from_distribution(
                        probs,
                        generator=make_generator(params.seed, str(self.device)),
                        output_device=logits.device,
                    ).reshape(-1)
                except SamplingNumericalError as exc:
                    raise SamplingNumericalError(
                        f"{exc}; Flash-Next MTP continuation sampling failed "
                        f"draft_step={draft_step},position={sess.pos - 1},"
                        f"using_graph={using_graph},graph_capacity="
                        f"{getattr(sampled_step_graph, 'graph_capacity', None)},"
                        f"sparse={sess.sparse_graph_buffers is not None},"
                        f"token={_sampling_tensor_stats(token_tensor)},"
                        f"logits={_sampling_tensor_stats(logits)},"
                        f"hidden={_sampling_tensor_stats(hidden)},"
                        f"first_hidden={_sampling_tensor_stats(first_hidden)}"
                    ) from exc
                draft_token_tensors.append(next_token)
                draft_probs.append(probs.squeeze(0))
            else:
                drafts.append(int(logits.argmax(dim=-1).item()))
        if sampled:
            # One and only one H2D synchronization for the complete proposal.
            # All K MTP forwards and random draws above stay on the CUDA
            # stream, so the host no longer serializes every continuation row.
            drafts = [int(token) for token in torch.cat(draft_token_tensors).tolist()]
        if return_probs:
            return drafts, torch.stack(draft_probs) if sampled else None
        return drafts

    def sync_and_propose(
        self,
        shifted_token_ids: Sequence[int],
        target_hc_hidden: torch.Tensor,
        *,
        input_embeds: torch.Tensor | None = None,
        params: SamplingParams | None = None,
    ) -> list[int]:
        sess = self.mtp_session
        sampled = params is not None and not params.is_greedy
        if sampled:
            # Sampling is deliberately outside the proposal graphs: the
            # graphs return argmax token ids and cannot carry a fresh RNG
            # draw.  The MTP forward itself remains the same, so target/MTP
            # state alignment and sparse-cache writes are unchanged.
            first, hidden, first_probs = self.sync_real_suffix(
                shifted_token_ids,
                target_hc_hidden,
                input_embeds=input_embeds,
                params=params,
                return_probs=True,
                return_token_tensor=True,
            )
            drafts, draft_probs = self.continue_draft(
                first,
                hidden,
                params=params,
                first_probs=first_probs,
                return_probs=True,
            )
            if draft_probs is None:
                raise RuntimeError("sampled MTP proposer did not produce draft distributions")
            self.pending_draft_probs = draft_probs.detach()
            return drafts
        graph = self.mtp_proposal_graphs.get(len(shifted_token_ids))
        if (
            graph is not None
            and len(shifted_token_ids) == graph.query_len
            and sess.sync_len + graph.query_len + self.k - 1 <= graph.graph_capacity
            and (
                not hasattr(graph, "graph")
                or graph.graph is not None
            )
        ):
            self._validate_mtp_range(
                sess.sync_len,
                graph.query_len + max(self.k - 1, 0),
                "MTP proposal",
            )
            replay_kwargs = {}
            if input_embeds is not None:
                replay_kwargs["input_embeds"] = input_embeds
            drafts = graph.replay(
                shifted_token_ids,
                target_hc_hidden,
                sess.sync_len,
                **replay_kwargs,
            )
            sess.sync_len += graph.query_len
            sess.pos = sess.sync_len + self.k - 1
            self.pending_draft_probs = None
            return [int(token) for token in drafts.tolist()]
        first, hidden = self.sync_real_suffix(
            shifted_token_ids,
            target_hc_hidden,
            input_embeds=input_embeds,
        )
        drafts = self.continue_draft(first, hidden)
        self.pending_draft_probs = None
        return drafts

    @torch.no_grad()
    def round(
        self,
        anchor_token: int,
        drafts: Sequence[int],
        *,
        use_graph: bool = True,
        return_verify_logits: bool = False,
        params: SamplingParams | None = None,
        thinking_force_position: int | None = None,
        thinking_force_token_id: int | None = None,
    ) -> dict[str, object]:
        """Verify, commit, teacher-sync and re-draft one MTP round.

        Greedy rounds retain the captured argmax path.  Non-greedy rounds
        reuse the same target verify forward, but compare the draft's stored
        ``q`` distribution with the target ``p`` distribution through exact
        rejection sampling before committing the accepted prefix.
        """
        if len(drafts) != self.k:
            raise ValueError(f"round requires K={self.k} drafts, got {len(drafts)}")
        past_len = self.target_session.pos
        if self.mtp_session.sync_len != past_len:
            raise RuntimeError(
                "target and MTP real prefixes diverged: "
                f"target={past_len}, mtp={self.mtp_session.sync_len}"
            )
        verify_tokens = [int(anchor_token), *(int(token) for token in drafts)]
        run = self.verify.replay if use_graph else self.verify.eager
        verify_started = time.perf_counter()
        hc_hidden, logits = run(verify_tokens, past_len=past_len)
        if (thinking_force_position is None) != (thinking_force_token_id is None):
            raise ValueError(
                "Flash-Next thinking force requires both a position and a token id"
            )
        if thinking_force_position is not None:
            if not 0 <= thinking_force_position < logits.shape[0]:
                raise ValueError(
                    "Flash-Next thinking force position is outside the verify block: "
                    f"position={thinking_force_position}, rows={logits.shape[0]}"
                )
            if not 0 <= thinking_force_token_id < logits.shape[-1]:
                raise ValueError(
                    "Flash-Next thinking force token is outside the vocabulary: "
                    f"token={thinking_force_token_id}, vocab={logits.shape[-1]}"
                )
            # Verify output can be a CUDA-Graph-owned tensor.  Clone only for
            # the rare thinking-budget boundary instead of mutating captured
            # output that the next replay must overwrite.
            logits = logits.clone()
            logits[thinking_force_position].fill_(-torch.inf)
            logits[thinking_force_position, thinking_force_token_id] = 0.0
        sampled = params is not None and not params.is_greedy
        if sampled:
            if self.pending_draft_probs is None:
                raise RuntimeError("sampled MTP verify is missing pending draft distributions")
            target_probs = compute_sampling_distribution(logits, params)
            decision = sample_accept_reject(
                [int(token) for token in drafts],
                self.pending_draft_probs,
                target_probs,
                generator=make_generator(params.seed, str(self.device)),
            )
            accepted_drafts = int(decision["num_accepted"])
            committed = [int(token) for token in decision["committed"]]
            prediction_ids = logits.argmax(dim=-1).tolist()
        else:
            predictions = logits.argmax(dim=-1)
            prediction_ids = predictions.tolist()
            accepted_drafts = 0
            for index, draft in enumerate(drafts):
                if int(prediction_ids[index]) != int(draft):
                    break
                accepted_drafts += 1
            bonus = int(prediction_ids[accepted_drafts])
            committed = [*(int(token) for token in drafts[:accepted_drafts]), bonus]
        verify_seconds = time.perf_counter() - verify_started
        bonus = int(committed[-1])

        consumed_count = accepted_drafts + 1  # anchor + accepted draft prefix
        self.verify.commit(consumed_count)

        shifted_real = [*(int(token) for token in drafts[:accepted_drafts]), bonus]
        mtp_started = time.perf_counter()
        if sampled:
            next_drafts = self.sync_and_propose(
                shifted_real,
                hc_hidden[:consumed_count],
                params=params,
            )
        else:
            next_drafts = self.sync_and_propose(
                shifted_real,
                hc_hidden[:consumed_count],
            )
        mtp_seconds = time.perf_counter() - mtp_started
        if os.environ.get("QSR_FLASHNEXT_DEBUG_ROUNDS", "0") == "1":
            logger.info(
                "Flash-Next MTP round past=%d verify=%s pred=%s accepted=%d "
                "committed=%s teacher=%s next_drafts=%s",
                past_len,
                verify_tokens,
                prediction_ids,
                accepted_drafts,
                committed,
                shifted_real,
                next_drafts,
            )
        result: dict[str, object] = {
            "committed": committed,
            "num_accepted": accepted_drafts,
            "reject_position": -1 if accepted_drafts == self.k else accepted_drafts,
            "verify_tokens": verify_tokens,
            "verify_prediction_ids": [
                int(token) for token in prediction_ids[:consumed_count]
            ],
            "teacher_tokens": shifted_real,
            "bonus_token": bonus,
            "next_anchor": bonus,
            "next_draft_tokens": next_drafts,
            "timing": {
                "ple": self.verify.last_ple_seconds,
                "verify": verify_seconds,
                "mtp": mtp_seconds,
            },
        }
        if return_verify_logits:
            result["verify_logits"] = logits[:consumed_count].clone()
        return result
