"""Flash-Next QSA (Qwen Sparse Attention) indexer -- plain-torch bring-up.

Semantics pinned from the reference implementation (sglang
``layers/attention/qsa/{qsa_indexer,mqa,kernel}.py``, read 2026-08-27):

* per QSA layer: ``index_qk_proj`` hidden -> (4 Q + 1 K) x 128; per-head
  Gemma-RMSNorm on Q and K; partial RoPE (rotary_dim 64 of 128, theta 1e7,
  interleaved pairs; text-only mrope reduces to plain RoPE on all axes);
* keys are average-pooled in FP32 over ``compress_ratio=4`` token blocks;
* block logits = relu(q . k) summed over the 4 query heads, divided by
  sqrt(128), masked to the causal window, top-512 blocks
  (budget 2048 tokens / ratio 4);
* block indices expand to token indices for the sparse gather.

The sparse attention itself gathers the selected KV rows and runs dense
attention over the compacted set (decode), mirroring this repo's DSV4
sparse path; kernels come in the optimization phase.
"""

from __future__ import annotations

import json
import math
import os
import pathlib

import torch
from torch import nn

from runtime.model.flashnext.qsa_kernels import (
    qsa_mqa_prefill,
    qsa_mqa_prefill_supported,
    qsa_prefill_gather_indices,
    qsa_prefill_gather_indices_supported,
    qsa_sparse_attention,
    qsa_sparse_attention_supported,
    qsa_sparse_prefill_attention,
)


def _normalize_rope_positions(
    positions: torch.Tensor,
    token_count: int,
) -> torch.Tensor:
    """Return MRoPE coordinates as ``[3, token_count]``.

    Text-only callers keep the historical ``[token_count]`` scalar layout;
    multimodal callers use the Qwen/sglang ``[3, token_count]`` axis-major
    layout.  Accepting ``[token_count, 3]`` as well makes the cache helpers
    less error-prone without changing the graph-facing ABI.
    """

    if positions.ndim == 1:
        if positions.numel() != token_count:
            raise ValueError(
                f"RoPE positions must contain {token_count} entries, "
                f"got {positions.numel()}"
            )
        return positions.reshape(1, token_count).expand(3, -1)
    if positions.ndim != 2:
        raise ValueError(f"RoPE positions must be 1-D or 2-D, got {positions.ndim}-D")
    if positions.shape == (3, token_count):
        return positions
    if positions.shape == (token_count, 3):
        return positions.transpose(0, 1)
    if positions.shape == (1, token_count):
        return positions.expand(3, -1)
    raise ValueError(
        "RoPE positions must have shape "
        f"[{token_count}], [3,{token_count}], or [{token_count},3], "
        f"got {tuple(positions.shape)}"
    )


def _mrope_axis_map(
    half_rotary_dim: int,
    section: tuple[int, ...] | list[int] | None,
    interleaved: bool,
    device: torch.device,
) -> torch.Tensor:
    """Build the frequency-pair -> temporal/height/width axis map.

    This is the exact interleaved composition used by sglang's
    ``MRotaryEmbedding`` and Qwen's official multimodal implementation.
    """

    axis_map = torch.zeros(half_rotary_dim, dtype=torch.long, device=device)
    if not section:
        return axis_map
    if len(section) != 3 or sum(int(value) for value in section) != half_rotary_dim:
        raise ValueError(
            "mrope_section must contain three values summing to rotary_dim/2, "
            f"got {section!r} for half_rotary_dim={half_rotary_dim}"
        )
    s0, s1, s2 = (int(value) for value in section)
    if interleaved:
        pair = torch.arange(half_rotary_dim, device=device)
        axis_map[((pair % 3) == 1) & (pair < s1 * 3)] = 1
        axis_map[((pair % 3) == 2) & (pair < s2 * 3)] = 2
    else:
        axis_map[s0 : s0 + s1] = 1
        axis_map[s0 + s1 :] = 2
    return axis_map


def _apply_partial_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    *,
    rotary_dim: int,
    rope_theta: float,
    mrope_section: tuple[int, ...] | list[int] | None,
    mrope_interleaved: bool,
    axis_map: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply NeoX-style partial RoPE with optional three-axis MRoPE."""

    if x.ndim < 2 or x.shape[0] == 0:
        return x
    if rotary_dim <= 0 or rotary_dim > x.shape[-1] or rotary_dim % 2:
        raise ValueError(
            f"rotary_dim must be even and fit the last dimension, got {rotary_dim}"
        )
    coords = _normalize_rope_positions(positions, int(x.shape[0])).to(
        device=x.device, dtype=torch.long
    )
    half = rotary_dim // 2
    inv = 1.0 / (
        rope_theta
        ** (torch.arange(half, device=x.device, dtype=torch.float32) / half)
    )
    if axis_map is None:
        axis_map = _mrope_axis_map(
            half,
            mrope_section,
            mrope_interleaved,
            x.device,
        )
    elif axis_map.device != x.device:
        # Module-owned buffers normally move with ``x``.  Keep the functional
        # helper usable for callers that pass a CPU buffer alongside CUDA
        # inputs without forcing every module call through a device copy.
        axis_map = axis_map.to(device=x.device)
    # ``coords.index_select(0, axis_map)`` gives [half, T]; transpose to the
    # row-major layout consumed by the broadcast below.
    angles = coords.index_select(0, axis_map).transpose(0, 1).float() * inv
    shape = (int(x.shape[0]),) + (1,) * (x.ndim - 2) + (half,)
    cos = torch.cos(angles).to(x.dtype).view(shape)
    sin = torch.sin(angles).to(x.dtype).view(shape)
    rot = x[..., :rotary_dim]
    x0 = rot[..., :half]
    x1 = rot[..., half:]
    rotated = torch.cat([x0 * cos - x1 * sin, x1 * cos + x0 * sin], dim=-1)
    return torch.cat([rotated, x[..., rotary_dim:]], dim=-1)


def qsa_kv_cache_dtype() -> torch.dtype:
    """Return the storage dtype for Flash-Next QSA K/V caches.

    Q/K indexer caches remain BF16 because the compressed-key score path is
    numerically sensitive.  Main-attention K/V uses row-scaled FP8 E4M3 by
    default so the validated Flash-Next service profile can keep multiple
    256K sessions on a 96-GiB card.  INT8 remains available as an explicit
    quality/capacity experiment, while BF16 remains available for an explicit
    reference run.
    """
    value = os.environ.get("QSR_FLASHNEXT_QSA_KV_DTYPE", "fp8").strip().lower()
    if value in {"bf16", "bfloat16", ""}:
        return torch.bfloat16
    if value in {"int8", "i8"}:
        return torch.int8
    if value in {"fp8", "fp8_e4m3", "fp8_e4m3fn", "e4m3", "float8_e4m3fn"}:
        dtype = getattr(torch, "float8_e4m3fn", None)
        if dtype is None:
            raise RuntimeError("QSA FP8 cache requested but torch.float8_e4m3fn is unavailable")
        return dtype
    raise ValueError(
        "QSR_FLASHNEXT_QSA_KV_DTYPE must be bf16, int8, or fp8_e4m3fn, "
        f"got {value!r}"
    )


def qsa_index_cache_rows(
    max_seq: int,
    compress_ratio: int,
    *,
    fixed_rows: int,
) -> int:
    """Return raw index-key rows for the selected capacity profile.

    Pooled index keys retain absolute group positions.  Raw per-token keys are
    needed only until their compression group is complete, so the 3x256K
    profile can retain them in a fixed-address ring instead of allocating one
    BF16 row for every token and QSA layer.
    """
    enabled = os.environ.get("QSR_FLASHNEXT_QSA_INDEX_RING", "0").strip().lower()
    if enabled in {"", "0", "false", "off"}:
        return max_seq
    if enabled not in {"1", "true", "on"}:
        raise ValueError(
            "QSR_FLASHNEXT_QSA_INDEX_RING must be 0 or 1, "
            f"got {enabled!r}"
        )
    if max_seq <= 0 or compress_ratio <= 0 or fixed_rows <= 0:
        raise ValueError(
            "QSA index ring dimensions must be positive: "
            f"max_seq={max_seq}, ratio={compress_ratio}, fixed_rows={fixed_rows}"
        )
    return min(max_seq, max(2 * compress_ratio, compress_ratio + fixed_rows))


def _qsa_cache_is_quantized(dtype: torch.dtype) -> bool:
    return dtype in {torch.int8, getattr(torch, "float8_e4m3fn", None)}


def qsa_cache_index_copy_(
    destination: torch.Tensor,
    index: torch.Tensor,
    source: torch.Tensor,
) -> torch.Tensor:
    """Copy rows into a QSA cache, including CUDA FP8 caches.

    PyTorch's CUDA ``index_copy_`` kernel does not implement Float8 E4M3,
    while ``index_put_`` does.  Keep the fast, established path for BF16 and
    INT8, and use the graph-capturable indexed put only for FP8 storage.
    """
    if destination.dtype == getattr(torch, "float8_e4m3fn", None):
        destination.index_put_((index,), source)
    else:
        destination.index_copy_(0, index, source)
    return destination


def quantize_qsa_kv(
    values: torch.Tensor,
    destination: torch.Tensor,
    scales: torch.Tensor | None = None,
) -> torch.Tensor:
    """Store BF16 QSA K/V rows in the configured cache dtype.

    Quantized caches use one scale per token/KV-head.  INT8 preserves much
    finer relative precision than E4M3 at the same resident byte count, while
    FP8 remains useful for A/B measurements.  The function is written in-place
    for CUDA-graph address stability and returns ``destination``.
    """
    if not _qsa_cache_is_quantized(destination.dtype):
        destination.copy_(values)
        return destination
    if scales is None:
        raise ValueError("quantized QSA cache writes require a scale tensor")
    if values.ndim != 3 or destination.shape != values.shape:
        raise ValueError(
            "QSA K/V quantization expects matching [tokens,kv_heads,head_dim] tensors"
        )
    quant_max = (
        127.0
        if destination.dtype == torch.int8
        else float(torch.finfo(destination.dtype).max)
    )
    # Compute scales in FP32, then retain them in FP16.  Clamp away from zero
    # so all-zero rows have a deterministic representation and no NaNs.
    row_max = values.float().abs().amax(dim=-1).clamp_min(1e-8)
    row_scale = (row_max / quant_max).clamp_min(1e-8)
    scales.copy_(row_scale.to(dtype=scales.dtype))
    quantized = (values.float() / row_scale.unsqueeze(-1)).clamp(
        -quant_max, quant_max
    )
    if destination.dtype == torch.int8:
        quantized = quantized.round()
    destination.copy_(quantized.to(destination.dtype))
    return destination


class QSAIndexer(nn.Module):
    """Weight-free-topk indexer: proj, norms, rope, block scoring."""

    def __init__(
        self,
        hidden_size: int = 2560,
        n_heads: int = 4,
        kv_heads: int = 1,
        head_dim: int = 128,
        rotary_dim: int = 64,
        rope_theta: float = 1e7,
        eps: float = 1e-6,
        compress_ratio: int = 4,
        block_topk: int = 512,
        dtype: torch.dtype = torch.bfloat16,
        mrope_section: tuple[int, ...] | None = None,
        mrope_interleaved: bool = False,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.rope_theta = rope_theta
        self.compress_ratio = compress_ratio
        self.block_topk = block_topk
        self.mrope_section = tuple(mrope_section) if mrope_section else None
        self.mrope_interleaved = bool(mrope_interleaved)
        # Building the MRoPE axis map inside every projection used a CUDA
        # boolean-index assignment.  Besides launching an avoidable tiny
        # scatter, that operation is not safe while a CUDA Graph is capturing
        # the target/verify body.  Keep the immutable map as a non-persistent
        # module buffer so it moves with the layer and is reused by both eager
        # and graph paths.
        self.register_buffer(
            "_mrope_axis_map_buffer",
            _mrope_axis_map(
                self.rotary_dim // 2,
                self.mrope_section,
                self.mrope_interleaved,
                torch.device("cpu"),
            ),
            persistent=False,
        )
        self.index_qk_proj = nn.Linear(
            hidden_size, (n_heads + kv_heads) * head_dim, bias=False, dtype=dtype
        )
        self.q_layernorm = nn.Parameter(torch.zeros(head_dim, dtype=dtype))
        self.k_layernorm = nn.Parameter(torch.zeros(head_dim, dtype=dtype))
        self.register_buffer(
            "group_offsets",
            torch.arange(compress_ratio, dtype=torch.long),
            persistent=False,
        )

    @staticmethod
    def _gemma_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        return (xf * torch.rsqrt(var + eps) * (1.0 + weight.float())).to(dtype)

    def _rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Interleaved partial RoPE over the first ``rotary_dim`` dims."""
        return _apply_partial_rope(
            x,
            positions,
            rotary_dim=self.rotary_dim,
            rope_theta=self.rope_theta,
            mrope_section=self.mrope_section,
            mrope_interleaved=self.mrope_interleaved,
            axis_map=self._mrope_axis_map_buffer,
        )

    def project_qk(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qk = self.index_qk_proj(hidden_states)
        q = qk[:, : self.n_heads * self.head_dim].view(-1, self.n_heads, self.head_dim)
        k = qk[:, self.n_heads * self.head_dim :].view(-1, self.head_dim)
        q = self._gemma_norm(q, self.q_layernorm, 1e-6)
        q = self._rope(q, positions)
        # Qwen4-Exp stores unnormalised per-token K in the pending ring.
        # Complete groups are averaged first, then normalised and rotated.
        return q, k

    def pool_keys(
        self,
        k: torch.Tensor,
        *,
        group_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compress complete raw-token K groups exactly like Qwen4-Exp.

        The trailing partial group is appended explicitly by the sparse-index
        expansion and is never included in the compressed cache.
        """
        s = k.shape[0] // self.compress_ratio * self.compress_ratio
        if s == 0:
            return k.new_empty((0, self.head_dim))
        groups = k[:s].view(-1, self.compress_ratio, self.head_dim)
        if group_positions is None:
            return self.pool_key_groups(groups, group_start=0)
        rope = _normalize_rope_positions(group_positions, int(k.shape[0]))
        # A compressed key carries the first member's coordinate.  This is
        # the same choice as sglang's qsa_indexer._rope_from_matrix and keeps
        # the pooled key on the causal boundary of its four-token group.
        first = rope[:, :s].reshape(3, -1, self.compress_ratio)[:, :, 0]
        return self.pool_key_groups_at_positions(groups, first)

    def pool_key_groups(
        self,
        groups: torch.Tensor,
        *,
        group_start: int = 0,
        group_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pool new complete groups for an incremental compressed-key cache."""
        if groups.ndim != 3 or groups.shape[1:] != (
            self.compress_ratio,
            self.head_dim,
        ):
            raise ValueError(
                "QSA key groups must be [groups, compress_ratio, head_dim], "
                f"got {tuple(groups.shape)}"
            )
        if group_positions is None:
            group_positions = torch.arange(
                group_start * self.compress_ratio,
                (group_start + groups.shape[0]) * self.compress_ratio,
                self.compress_ratio,
                dtype=torch.long,
                device=groups.device,
            )
        return self.pool_key_groups_at_positions(groups, group_positions)

    def pool_key_groups_at_positions(
        self,
        groups: torch.Tensor,
        group_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Pool complete groups at explicit positions, including graph replay."""
        if groups.ndim != 3 or groups.shape[1:] != (
            self.compress_ratio,
            self.head_dim,
        ):
            raise ValueError(
                "QSA key groups must be [groups, compress_ratio, head_dim], "
                f"got {tuple(groups.shape)}"
            )
        if group_positions.numel() not in {
            groups.shape[0],
            3 * groups.shape[0],
        }:
            raise ValueError(
                "QSA group positions must match group rows, "
                f"got {tuple(group_positions.shape)}"
            )
        pooled = groups.float().mean(dim=1).to(groups.dtype)
        pooled = self._gemma_norm(pooled, self.k_layernorm, 1e-6)
        positions = _normalize_rope_positions(group_positions, groups.shape[0])
        return self._rope(pooled, positions)

    def update_index_cache_eager(
        self,
        raw_cache: torch.Tensor,
        pooled_cache: torch.Tensor,
        keys: torch.Tensor,
        *,
        start: int,
        rope_cache: torch.Tensor | None = None,
        rope_positions: torch.Tensor | None = None,
    ) -> None:
        """Append contiguous raw index keys to a full cache or bounded ring.

        Only an unfinished compression group needs raw-token retention after
        its pooled key has been written.  Capacity profiles therefore keep a
        small ring instead of ``max_seq`` BF16 rows.  Pooling happens before
        the ring write so a wrap cannot overwrite the prefix of a group that
        the current suffix completes.
        """
        if keys.ndim != 2 or keys.shape[1] != self.head_dim:
            raise ValueError(
                f"QSA raw index keys must be [rows,{self.head_dim}], "
                f"got {tuple(keys.shape)}"
            )
        if start < 0:
            raise ValueError(f"QSA index cache start must be non-negative, got {start}")
        capacity = raw_cache.shape[0]
        if capacity < self.compress_ratio:
            raise ValueError(
                "QSA raw index ring must hold at least one compression group, "
                f"got capacity={capacity}, ratio={self.compress_ratio}"
            )
        if (rope_cache is None) != (rope_positions is None):
            raise ValueError(
                "QSA rope_cache and rope_positions must be provided together"
            )
        rope_rows = None
        if rope_positions is not None:
            rope_rows = _normalize_rope_positions(rope_positions, int(keys.shape[0]))
            if rope_cache is None or rope_cache.ndim != 2 or rope_cache.shape != (capacity, 3):
                raise ValueError(
                    "QSA rope_cache must have shape [raw_capacity, 3], "
                    f"got {None if rope_cache is None else tuple(rope_cache.shape)}"
                )
        end = start + keys.shape[0]
        group_base = start // self.compress_ratio * self.compress_ratio
        complete_end = end // self.compress_ratio * self.compress_ratio
        if complete_end > group_base:
            prefix_count = start - group_base
            pieces = []
            if prefix_count:
                prefix_positions = torch.arange(
                    group_base,
                    start,
                    dtype=torch.long,
                    device=raw_cache.device,
                ).remainder(capacity)
                pieces.append(raw_cache.index_select(0, prefix_positions))
            new_count = complete_end - start
            if new_count > 0:
                pieces.append(keys[:new_count])
            group_rows = torch.cat(pieces).view(
                -1,
                self.compress_ratio,
                self.head_dim,
            )
            group_start = group_base // self.compress_ratio
            group_end = complete_end // self.compress_ratio
            if rope_rows is None:
                pooled = self.pool_key_groups(group_rows, group_start=group_start)
            else:
                rope_pieces = []
                if prefix_count:
                    rope_positions_for_prefix = torch.arange(
                        group_base,
                        start,
                        dtype=torch.long,
                        device=raw_cache.device,
                    ).remainder(capacity)
                    rope_pieces.append(rope_cache.index_select(0, rope_positions_for_prefix))
                if new_count > 0:
                    rope_pieces.append(rope_rows[:, :new_count].transpose(0, 1))
                group_rope = torch.cat(rope_pieces).view(
                    -1, self.compress_ratio, 3
                )[:, 0, :]
                pooled = self.pool_key_groups_at_positions(group_rows, group_rope)
            pooled_cache[group_start:group_end].copy_(pooled)

        # A bounded ring needs only the newest rows.  Limiting the indexed
        # assignment to ``capacity`` also guarantees unique modulo indices.
        keep = min(keys.shape[0], capacity)
        if keep:
            tail_start = keys.shape[0] - keep
            positions = torch.arange(
                end - keep,
                end,
                dtype=torch.long,
                device=raw_cache.device,
            ).remainder(capacity)
            raw_cache.index_copy_(0, positions, keys[tail_start:])
            if rope_rows is not None:
                rope_cache.index_copy_(0, positions, rope_rows[:, tail_start:].transpose(0, 1))

    def update_index_cache_fixed(
        self,
        raw_cache: torch.Tensor,
        pooled_cache: torch.Tensor,
        keys: torch.Tensor,
        positions: torch.Tensor,
        *,
        rope_cache: torch.Tensor | None = None,
        rope_positions: torch.Tensor | None = None,
    ) -> None:
        """Graph-safe fixed-row index update for decode and target verify."""
        if keys.ndim != 2 or keys.shape != (positions.numel(), self.head_dim):
            raise ValueError(
                "QSA fixed index keys must match positions and head_dim, "
                f"got keys={tuple(keys.shape)}, positions={tuple(positions.shape)}"
            )
        capacity = raw_cache.shape[0]
        if capacity < self.compress_ratio + positions.numel():
            raise ValueError(
                "QSA raw index ring is too small for a fixed update: "
                f"capacity={capacity}, ratio={self.compress_ratio}, "
                f"rows={positions.numel()}"
            )
        if (rope_cache is None) != (rope_positions is None):
            raise ValueError(
                "QSA rope_cache and rope_positions must be provided together"
            )
        rope_rows = None
        if rope_positions is not None:
            rope_rows = _normalize_rope_positions(rope_positions, int(keys.shape[0]))
            if rope_cache is None or rope_cache.ndim != 2 or rope_cache.shape != (capacity, 3):
                raise ValueError(
                    "QSA rope_cache must have shape [raw_capacity, 3], "
                    f"got {None if rope_cache is None else tuple(rope_cache.shape)}"
                )
        slots = positions.remainder(capacity)
        raw_cache.index_copy_(0, slots, keys)
        if rope_rows is not None:
            rope_cache.index_copy_(0, slots, rope_rows.transpose(0, 1))
        group_positions = torch.div(
            positions,
            self.compress_ratio,
            rounding_mode="floor",
        ) * self.compress_ratio
        token_positions = (
            group_positions.unsqueeze(1) + self.group_offsets.unsqueeze(0)
        ).remainder(capacity)
        group_rows = raw_cache[token_positions]
        if rope_cache is None:
            pooled = self.pool_key_groups_at_positions(group_rows, group_positions)
        else:
            group_rope = rope_cache[token_positions][:, 0, :]
            pooled = self.pool_key_groups_at_positions(group_rows, group_rope)
        pooled_cache.index_copy_(
            0,
            torch.div(group_positions, self.compress_ratio, rounding_mode="floor"),
            pooled,
        )

    def score_blocks(
        self,
        q: torch.Tensor,
        pooled_k: torch.Tensor,
        row_block_ends: torch.Tensor,
    ) -> torch.Tensor:
        """``q [M, h, d]`` vs ``pooled_k [B, d]`` -> logits ``[M, B]`` masked
        so row m only sees blocks before its causal end."""
        if qsa_mqa_prefill_supported(q):
            return qsa_mqa_prefill(q, pooled_k, row_block_ends)
        scores = torch.einsum("mhd,nd->mnh", q.float(), pooled_k.float())
        logits = torch.relu(scores).sum(dim=-1) / math.sqrt(self.head_dim)
        cols = torch.arange(pooled_k.shape[0], device=q.device).unsqueeze(0)
        valid = cols < row_block_ends.to(q.device).reshape(-1, 1)
        return logits.masked_fill(~valid, -float("inf"))

    def score_blocks_fixed(
        self,
        q: torch.Tensor,
        pooled_k: torch.Tensor,
        row_block_ends: torch.Tensor,
        *,
        out: torch.Tensor,
        column_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Fixed-shape block scoring for CUDA Graph sparse decode.

        ``pooled_k`` is the full fixed pooled-key cache and ``row_block_ends``
        masks away yet-unfilled groups. ``out`` and ``column_ids`` are
        preallocated graph-stable buffers owned by the caller.
        """
        if out.shape != (q.shape[0], pooled_k.shape[0]):
            raise ValueError(
                "QSA fixed score output must be [rows, pooled_rows], "
                f"got {tuple(out.shape)} for rows={q.shape[0]} "
                f"pooled_rows={pooled_k.shape[0]}"
            )
        if column_ids.shape != (1, pooled_k.shape[0]):
            raise ValueError(
                "QSA fixed score column ids must be [1, pooled_rows], "
                f"got {tuple(column_ids.shape)}"
            )
        # ``score_blocks`` already applies the causal mask in the TileLang
        # path.  Re-masking here used to launch a second full-width pass over
        # the fixed pooled cache for every MTP graph row.
        out.copy_(self.score_blocks(q, pooled_k, row_block_ends))
        return out

    def select_blocks(
        self,
        logits: torch.Tensor,
        row_block_ends: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return fixed-width block indices with invalid lanes set to ``-1``."""
        k = min(self.block_topk, logits.shape[1])
        selected = None
        native = None
        # SGLang's QSA fast_topk is a radix selector over the valid prefix;
        # the standalone SM120 adapter has the same semantics without making
        # the runtime import SGLang.  It avoids sorting the unused tail of a
        # 256K fixed pooled-key cache.  Keep the regular torch path for
        # non-production top-k widths and source-only installs.
        if (
            k == 512
            and self.block_topk == 512
            and logits.is_cuda
            and logits.shape[1] >= 512
        ):
            from runtime.kernels.flashnext_qsa_topk import load_native_flashnext_qsa_topk

            native = load_native_flashnext_qsa_topk()
            if native is not None:
                lengths = (
                    torch.isfinite(logits).sum(dim=-1, dtype=torch.int64)
                    if row_block_ends is None
                    else row_block_ends.to(device=logits.device, dtype=torch.int64)
                )
                lengths = lengths.reshape(-1).clamp_(min=0, max=logits.shape[1])
                candidates = native.select(logits.contiguous(), lengths)
                # The radix CTA intentionally emits an unordered set, which
                # is the same contract as SGLang's fast_topk.  Sparse QSA
                # consumes a set of blocks, not a score-ranked sequence, so
                # skipping the extra 512-wide rerank keeps the kernel win on
                # the graph path.  A score-order compatibility switch remains
                # available for numerical A/B checks and debugging.
                if os.environ.get("QSR_FLASHNEXT_QSA_TOPK_RERANK", "0") == "1":
                    safe = candidates.to(torch.long).clamp_(
                        min=0, max=logits.shape[1] - 1
                    )
                    candidate_scores = logits.gather(1, safe)
                    candidate_scores.masked_fill_(candidates < 0, -float("inf"))
                    order = torch.topk(candidate_scores, k, dim=-1).indices
                    candidates = torch.gather(candidates, 1, order)
                # Native output is already torch.long and carries ``-1`` for
                # every lane beyond a short row's valid prefix.  Returning it
                # directly avoids both the conversion kernel and the extra
                # full-width torch.where mask launch on the graph path.
                selected = candidates
        if selected is None:
            selected = torch.topk(logits, k, dim=-1).indices
        if row_block_ends is None:
            row_block_ends = torch.isfinite(logits).sum(dim=-1)
        if selected is not None and native is not None and os.environ.get(
            "QSR_FLASHNEXT_QSA_TOPK_RERANK", "0"
        ) != "1":
            return selected
        ranks = torch.arange(k, device=logits.device).unsqueeze(0)
        valid = ranks < row_block_ends.to(logits.device).reshape(-1, 1).clamp(max=k)
        return torch.where(valid, selected, selected.new_full((), -1))

    def select_blocks_fixed(
        self,
        logits: torch.Tensor,
        row_block_ends: torch.Tensor,
        *,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """Fixed-shape top-k selection with caller-owned output storage."""
        if out.shape != (logits.shape[0], min(self.block_topk, logits.shape[1])):
            raise ValueError(
                "QSA fixed block output has the wrong shape: "
                f"got {tuple(out.shape)} for logits {tuple(logits.shape)}"
            )
        out.copy_(self.select_blocks(logits, row_block_ends))
        return out


    def decode_gather_indices(
        self,
        block_indices: torch.Tensor,
        pos: int,
        device,
        pad_to: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand one decode row, including its uncompressed partial tail."""
        if isinstance(pos, torch.Tensor):
            positions = pos.reshape(1)
        else:
            positions = torch.tensor([pos], dtype=torch.long, device=device)
        tokens, valid = self.batch_decode_gather_indices(
            block_indices.unsqueeze(0), positions, pad_to
        )
        return tokens[0], valid[0]

    def batch_decode_gather_indices(
        self,
        block_indices: torch.Tensor,
        positions: torch.Tensor,
        pad_to: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Vectorized, compact block expansion for a fixed-width verify block.

        ``block_indices`` is ``[M, K]`` and ``positions`` is ``[M]``.  The
        result is ``[M, pad_to]`` so one captured verify graph can service
        every row in ``anchor + drafts`` without a Python attention loop.
        Valid complete-group lanes and the partial causal tail are packed at
        the front of each row.  This keeps the graph shape fixed while giving
        fused QSA an exact loop bound instead of scanning the 2051-lane
        padding tail at short decode positions.
        """
        if block_indices.ndim != 2 or positions.shape != (block_indices.shape[0],):
            raise ValueError(
                "QSA decode blocks/positions must be [rows, topk] and [rows]"
            )
        if pad_to <= 0:
            raise ValueError(f"QSA decode pad_to must be positive, got {pad_to}")
        ratio = self.compress_ratio
        offsets = torch.arange(ratio, device=block_indices.device)
        blocks = block_indices.long()
        expanded = (
            blocks.unsqueeze(-1) * ratio + offsets.view(1, 1, -1)
        ).flatten(1)
        block_valid = blocks.unsqueeze(-1).expand(-1, -1, ratio).flatten(1) >= 0

        token_topk = min(self.block_topk * ratio, pad_to)
        expanded = expanded[:, :token_topk]
        visible_tokens = positions.long() + 1
        block_valid = block_valid[:, :token_topk]
        block_valid &= expanded >= 0
        block_valid &= expanded < visible_tokens.unsqueeze(1)
        tokens = torch.empty(
            block_indices.shape[0], pad_to, dtype=torch.long, device=blocks.device
        )
        valid = torch.empty(
            block_indices.shape[0], pad_to, dtype=torch.bool, device=blocks.device
        )
        self._pack_decode_rows(
            expanded,
            block_valid,
            visible_tokens,
            pad_to,
            out_tokens=tokens,
            out_valid=valid,
        )
        return tokens, valid

    def _pack_decode_rows(
        self,
        expanded: torch.Tensor,
        expanded_valid: torch.Tensor,
        visible_tokens: torch.Tensor,
        pad_to: int,
        *,
        out_tokens: torch.Tensor,
        out_valid: torch.Tensor,
        tail_tokens: torch.Tensor | None = None,
    ) -> None:
        """Pack valid sparse groups and the causal tail into fixed buffers."""
        rows = expanded.shape[0]
        if expanded.ndim != 2 or expanded_valid.shape != expanded.shape:
            raise ValueError("QSA decode expanded/valid rows must have matching shape")
        if visible_tokens.shape != (rows,):
            raise ValueError(
                f"QSA visible token counts must have shape ({rows},), "
                f"got {tuple(visible_tokens.shape)}"
            )
        if out_tokens.shape != (rows, pad_to) or out_valid.shape != out_tokens.shape:
            raise ValueError(
                "QSA packed decode outputs must be [rows, pad_to] with matching masks"
            )
        out_tokens.zero_()
        out_valid.zero_()

        # Scatter each valid expanded lane to its compact rank. Invalid lanes
        # share the last padding slot; a row with a valid final slot has no
        # invalid lanes because its valid count already equals ``pad_to``.
        rank = expanded_valid.to(torch.int64).cumsum(dim=1) - 1
        invalid_destination = torch.full_like(rank, pad_to - 1)
        destinations = torch.where(expanded_valid, rank, invalid_destination)
        destinations = destinations.clamp_(0, pad_to - 1)
        out_tokens.scatter_(
            1,
            destinations,
            torch.where(expanded_valid, expanded, torch.zeros_like(expanded)),
        )
        out_valid.scatter_(1, destinations, expanded_valid)

        ratio = self.compress_ratio
        tail_width = min(ratio - 1, max(pad_to - expanded.shape[1], 0))
        tail_offsets_all = torch.arange(ratio - 1, device=expanded.device)
        tail_start = torch.div(
            visible_tokens,
            ratio,
            rounding_mode="floor",
        ) * ratio
        if tail_tokens is None:
            tail = tail_start.unsqueeze(1) + tail_offsets_all.unsqueeze(0)
        else:
            if tail_tokens.shape != (rows, ratio - 1):
                raise ValueError(
                    "QSA fixed gather tail buffer must be [rows, ratio - 1], "
                    f"got {tuple(tail_tokens.shape)}"
                )
            tail = tail_tokens
            tail.copy_(tail_start.unsqueeze(1) + tail_offsets_all.unsqueeze(0))
        if tail_width <= 0:
            return

        tail = tail[:, :tail_width]
        tail_offsets = torch.arange(tail_width, device=expanded.device)
        tail_valid = tail < visible_tokens.unsqueeze(1)
        block_count = expanded_valid.sum(dim=1, dtype=torch.int64)
        tail_destinations = block_count.unsqueeze(1) + tail_offsets.unsqueeze(0)
        tail_valid &= tail_destinations < pad_to
        safe_tail_destinations = torch.where(
            tail_valid,
            tail_destinations,
            torch.full_like(tail_destinations, pad_to - 1),
        ).clamp_(0, pad_to - 1)
        out_tokens.scatter_(
            1,
            safe_tail_destinations,
            torch.where(tail_valid, tail, torch.zeros_like(tail)),
        )
        out_valid.scatter_(1, safe_tail_destinations, tail_valid)

    def batch_decode_gather_indices_fixed(
        self,
        block_indices: torch.Tensor,
        positions: torch.Tensor,
        pad_to: int,
        *,
        out_tokens: torch.Tensor,
        out_valid: torch.Tensor,
        tail_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Graph-safe fixed-width gather expansion into preallocated buffers."""
        if out_tokens.shape != (block_indices.shape[0], pad_to):
            raise ValueError(
                "QSA fixed gather token buffer must be [rows, pad_to], "
                f"got {tuple(out_tokens.shape)}"
            )
        if out_valid.shape != out_tokens.shape:
            raise ValueError(
                "QSA fixed gather valid buffer must match token buffer, "
                f"got {tuple(out_valid.shape)} vs {tuple(out_tokens.shape)}"
            )
        if positions.shape != (block_indices.shape[0],):
            raise ValueError(
                "QSA fixed decode positions must have one value per row, "
                f"got {tuple(positions.shape)}"
            )
        if pad_to <= 0:
            raise ValueError(f"QSA decode pad_to must be positive, got {pad_to}")
        ratio = self.compress_ratio
        block_offsets = self.group_offsets.to(block_indices.device)
        blocks = block_indices.long()
        expanded = (
            blocks.unsqueeze(-1) * ratio + block_offsets.view(1, 1, -1)
        ).flatten(1)
        token_topk = min(self.block_topk * ratio, pad_to)
        expanded = expanded[:, :token_topk]
        visible_tokens = positions.long() + 1
        expanded_valid = blocks.unsqueeze(-1).expand(-1, -1, ratio).flatten(1)
        expanded_valid = expanded_valid[:, :token_topk] >= 0
        expanded_valid &= expanded >= 0
        expanded_valid &= expanded < visible_tokens.unsqueeze(1)
        self._pack_decode_rows(
            expanded,
            expanded_valid,
            visible_tokens,
            pad_to,
            out_tokens=out_tokens,
            out_valid=out_valid,
            tail_tokens=tail_tokens,
        )
        return out_tokens, out_valid

    def batch_decode_reuse_indices_fixed(
        self,
        shared_indices: torch.Tensor,
        shared_valid: torch.Tensor,
        positions: torch.Tensor,
        captured_len: int | torch.Tensor,
        *,
        out_tokens: torch.Tensor,
        out_valid: torch.Tensor,
        tail_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fixed-shape reuse of a captured sparse row plus appended dense tail."""
        rows = positions.shape[0]
        base_width = shared_indices.shape[1]
        if shared_indices.shape != shared_valid.shape:
            raise ValueError(
                "QSA shared reuse buffers must have matching shapes, "
                f"got {tuple(shared_indices.shape)} vs {tuple(shared_valid.shape)}"
            )
        if out_tokens.shape[0] != rows or out_valid.shape != out_tokens.shape:
            raise ValueError(
                "QSA fixed reuse outputs must match row count and shape, "
                f"got tokens={tuple(out_tokens.shape)} valid={tuple(out_valid.shape)} rows={rows}"
            )
        if tail_tokens.shape != (rows, out_tokens.shape[1] - base_width):
            raise ValueError(
                "QSA fixed reuse tail buffer must be [rows, extra_width], "
                f"got {tuple(tail_tokens.shape)}"
            )
        if out_tokens.shape[1] < base_width:
            raise ValueError(
                f"QSA fixed reuse output width {out_tokens.shape[1]} is smaller than "
                f"shared width {base_width}"
            )
        if not torch.is_tensor(captured_len):
            captured_len = torch.tensor(
                [captured_len],
                dtype=torch.long,
                device=positions.device,
            )
        captured = captured_len.to(device=positions.device, dtype=torch.long).reshape(1, 1)
        out_tokens.zero_()
        out_valid.zero_()
        out_tokens[:, :base_width].copy_(shared_indices.expand(rows, -1))
        out_valid[:, :base_width].copy_(shared_valid.expand(rows, -1))
        extra_width = tail_tokens.shape[1]
        if extra_width > 0:
            tail_offsets = torch.arange(extra_width, device=positions.device).unsqueeze(0)
            tail_tokens.copy_(captured + tail_offsets)
            tail_valid = tail_offsets < (positions.long().unsqueeze(1) - captured + 1)
            out_tokens[:, base_width:].copy_(
                torch.where(tail_valid, tail_tokens, torch.zeros_like(tail_tokens))
            )
            out_valid[:, base_width:].copy_(tail_valid)
        return out_tokens, out_valid

    def batch_prefill_gather_indices(
        self,
        block_indices: torch.Tensor,
        positions: torch.Tensor,
        pad_to: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expand blocks with every valid token packed before padding.

        Decode keeps the partial tail at a fixed address for graph replay.
        Prefill instead needs a compact row so the sparse kernel can stop at
        the actual causal width rather than scanning 2051 lanes for token 0.
        """
        if qsa_prefill_gather_indices_supported(block_indices):
            return qsa_prefill_gather_indices(
                block_indices,
                positions,
                pad_to,
                self.compress_ratio,
            )
        ratio = self.compress_ratio
        rows = block_indices.shape[0]
        blocks = block_indices.long()
        offsets = torch.arange(ratio, device=blocks.device)
        expanded = (blocks.unsqueeze(-1) * ratio + offsets).flatten(1)
        expanded_valid = blocks.unsqueeze(-1).expand(-1, -1, ratio).flatten(1) >= 0
        expanded_capacity = blocks.shape[1] * ratio

        visible = positions.long() + 1
        complete_groups = torch.div(visible, ratio, rounding_mode="floor")
        selected_groups = complete_groups.clamp(max=blocks.shape[1])
        selected_tokens = selected_groups * ratio
        columns = torch.arange(expanded_capacity, device=blocks.device).unsqueeze(0)
        expanded_valid &= columns < selected_tokens.unsqueeze(1)
        expanded_valid &= (expanded >= 0) & (expanded < visible.unsqueeze(1))

        tokens = torch.zeros(rows, pad_to, dtype=torch.long, device=blocks.device)
        valid = torch.zeros(rows, pad_to, dtype=torch.bool, device=blocks.device)
        copied = min(expanded_capacity, pad_to)
        tokens[:, :copied] = torch.where(
            expanded_valid[:, :copied],
            expanded[:, :copied],
            torch.zeros_like(expanded[:, :copied]),
        )
        valid[:, :copied] = expanded_valid[:, :copied]

        tail_offsets = torch.arange(ratio - 1, device=blocks.device)
        tail_start = complete_groups * ratio
        tail = tail_start.unsqueeze(1) + tail_offsets.unsqueeze(0)
        tail_count = visible - tail_start
        tail_valid = tail_offsets.unsqueeze(0) < tail_count.unsqueeze(1)
        destinations = selected_tokens.unsqueeze(1) + tail_offsets.unsqueeze(0)
        destination_valid = destinations < pad_to
        safe_destinations = destinations.clamp(max=pad_to - 1)
        tokens.scatter_(
            1,
            safe_destinations,
            torch.where(tail_valid & destination_valid, tail, torch.zeros_like(tail)),
        )
        valid.scatter_(1, safe_destinations, tail_valid & destination_valid)
        selected_counts = (selected_tokens + tail_count).clamp(max=pad_to).to(torch.int32)
        return tokens, valid, selected_counts

def load_qsa_indexer(
    ckpt: pathlib.Path | str,
    layer_idx: int,
    device: str = "cpu",
    *,
    rope_theta: float = 1e7,
    mrope_section: tuple[int, ...] | None = None,
    mrope_interleaved: bool = False,
) -> QSAIndexer:
    from safetensors import safe_open

    ckpt = pathlib.Path(ckpt)
    with open(ckpt / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]
    prefix = f"model.language_model.layers.{layer_idx}.self_attn.indexer"

    def load(name: str) -> torch.Tensor:
        key = f"{prefix}.{name}"
        with safe_open(str(ckpt / weight_map[key]), framework="pt", device="cpu") as f:
            return f.get_tensor(key)

    module = QSAIndexer(
        rope_theta=rope_theta,
        mrope_section=mrope_section,
        mrope_interleaved=mrope_interleaved,
    )
    with torch.no_grad():
        module.index_qk_proj.weight.copy_(load("index_qk_proj.weight"))
        module.q_layernorm.copy_(load("q_layernorm.weight"))
        module.k_layernorm.copy_(load("k_layernorm.weight"))
    return module.to(device)


class FlashNextQSAAttention(nn.Module):
    """The QSA layer's main attention: GQA (24 Q / 2 KV heads, head_dim 256,
    partial RoPE 64/256) with a per-(head, dim) sigmoid output gate and the
    sparse gather: attend only over the indexer-selected token rows.

    Bring-up path gathers selected KV rows and runs dense attention over the
    compacted set (decode); a paged sparse kernel replaces this in the
    optimization phase.
    """

    def __init__(
        self,
        hidden_size: int = 2560,
        num_heads: int = 24,
        num_kv_heads: int = 2,
        head_dim: int = 256,
        rotary_dim: int = 64,
        rope_theta: float = 1e7,
        dtype: torch.dtype = torch.bfloat16,
        mrope_section: tuple[int, ...] | None = None,
        mrope_interleaved: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.rope_theta = rope_theta
        self.mrope_section = tuple(mrope_section) if mrope_section else None
        self.mrope_interleaved = bool(mrope_interleaved)
        self.register_buffer(
            "_mrope_axis_map_buffer",
            _mrope_axis_map(
                self.rotary_dim // 2,
                self.mrope_section,
                self.mrope_interleaved,
                torch.device("cpu"),
            ),
            persistent=False,
        )
        self.repeat = num_heads // num_kv_heads
        # attn_output_gate: q_proj carries an extra gate half
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim * 2, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False, dtype=dtype)
        self.q_norm = nn.Parameter(torch.zeros(head_dim, dtype=dtype))
        self.k_norm = nn.Parameter(torch.zeros(head_dim, dtype=dtype))

    @staticmethod
    def _norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        return (xf * torch.rsqrt(var + eps) * (1.0 + weight.float())).to(dtype)

    def _rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return _apply_partial_rope(
            x,
            positions,
            rotary_dim=self.rotary_dim,
            rope_theta=self.rope_theta,
            mrope_section=self.mrope_section,
            mrope_interleaved=self.mrope_interleaved,
            axis_map=self._mrope_axis_map_buffer,
        )

    def project(self, hidden: torch.Tensor, positions: torch.Tensor):
        """Returns (q [T, H, D], k [T, KV, D], v [T, KV, D], gate [T, H, D])."""
        qg = self.q_proj(hidden).view(-1, self.num_heads, self.head_dim * 2)
        q, gate = qg.split(self.head_dim, dim=-1)
        k = self.k_proj(hidden).view(-1, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden).view(-1, self.num_kv_heads, self.head_dim)
        q = self._norm(q, self.q_norm)
        k = self._norm(k, self.k_norm)
        q = self._rope(q, positions)
        k = self._rope(k, positions)
        return q, k, v, gate

    def sparse_decode(
        self,
        q: torch.Tensor,
        gate: torch.Tensor,
        k_pool: torch.Tensor,
        v_pool: torch.Tensor,
        selected: torch.Tensor,
    ) -> torch.Tensor:
        """One query row per batch entry; ``selected [B, S]`` token rows to
        gather from the layer's KV pools; dense attention over the compact
        set, sigmoid gate, o_proj."""
        b, sel_len = selected.shape
        k = k_pool[selected.reshape(-1)].view(b, sel_len, self.num_kv_heads, self.head_dim)
        v = v_pool[selected.reshape(-1)].view(b, sel_len, self.num_kv_heads, self.head_dim)
        k = k.repeat_interleave(self.repeat, dim=2)
        v = v.repeat_interleave(self.repeat, dim=2)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.einsum("bhd,bshd->bhs", q.float(), k.float()) * scale
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum("bhs,bshd->bhd", attn, v.float()).to(q.dtype)
        out = out * torch.sigmoid(gate.float()).to(q.dtype)
        return self.o_proj(out.reshape(b, self.num_heads * self.head_dim))


class QsaDecodeAttention(nn.Module):
    """Fixed-width padded gather attention for graph-safe decode."""

    def __init__(self, attn: FlashNextQSAAttention, pad_to: int) -> None:
        super().__init__()
        self.attn = attn
        self.pad_to = pad_to
        self.register_buffer(
            "_dense_indices",
            torch.arange(pad_to, device=attn.q_proj.weight.device),
            persistent=False,
        )

    def forward(
        self,
        q: torch.Tensor,
        gate: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        idx: torch.Tensor,
        valid: torch.Tensor,
        k_scales: torch.Tensor | None = None,
        v_scales: torch.Tensor | None = None,
        selected_counts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend one or more query rows over a shared fixed-address cache.

        Decode passes ``idx/valid [P]`` and verify passes ``[M, P]``.  Both
        paths use the same batched math so the single-row graph remains a
        strict specialization of the K+1 verify graph.
        """
        a = self.attn
        squeeze = idx.ndim == 1
        if squeeze:
            idx = idx.unsqueeze(0)
            valid = valid.unsqueeze(0)
        rows, selected = idx.shape
        if q.shape[0] != rows or gate.shape[0] != rows:
            raise ValueError(
                "QSA query/gate rows must match gather rows: "
                f"q={q.shape[0]}, gate={gate.shape[0]}, idx={rows}"
            )
        if qsa_sparse_attention_supported(q):
            o = qsa_sparse_attention(
                q,
                gate,
                k_cache,
                v_cache,
                idx,
                valid,
                k_scales,
                v_scales,
                selected_counts,
            )
        else:
            ksel = k_cache[idx.reshape(-1)].view(
                rows, selected, a.num_kv_heads, a.head_dim
            )
            vsel = v_cache[idx.reshape(-1)].view_as(ksel)
            if _qsa_cache_is_quantized(k_cache.dtype):
                if k_scales is None or v_scales is None:
                    raise ValueError("FP8 QSA attention requires K/V scale caches")
                row_scales_k = k_scales[idx].to(torch.float32).unsqueeze(-1)
                row_scales_v = v_scales[idx].to(torch.float32).unsqueeze(-1)
                ksel = ksel.to(torch.float32) * row_scales_k
                vsel = vsel.to(torch.float32) * row_scales_v
            ksel = ksel.repeat_interleave(a.repeat, dim=2)  # [M, P, H, D]
            vsel = vsel.repeat_interleave(a.repeat, dim=2)
            scale = 1.0 / math.sqrt(a.head_dim)
            scores = torch.einsum("mhd,mshd->mhs", q.float(), ksel.float()) * scale
            neg_inf = torch.finfo(torch.float32).min
            scores = torch.where(valid.unsqueeze(1), scores, neg_inf)
            att = torch.softmax(scores, dim=-1)
            o = torch.einsum("mhs,mshd->mhd", att, vsel.float()).to(q.dtype)
            o = o * torch.sigmoid(gate.float()).to(q.dtype)
        return a.o_proj(o.reshape(rows, -1))

    def sparse_prefill(
        self,
        indexer: QSAIndexer,
        index_q: torch.Tensor,
        q: torch.Tensor,
        gate: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        index_k_cache: torch.Tensor,
        positions: torch.Tensor,
        *,
        pooled_k_cache: torch.Tensor | None = None,
        k_scales: torch.Tensor | None = None,
        v_scales: torch.Tensor | None = None,
        logits_workspace_bytes: int = 128 * 1024 * 1024,
    ) -> torch.Tensor:
        """Run long-context QSA prefill in bounded row batches.

        The old bring-up fallback re-projected Q/K/V and launched one eager
        attention sequence per query token.  Besides duplicating four large
        projections per QSA layer, that makes long-prefill latency dominated
        by Python and kernel-launch overhead.  Score compressed keys in row
        tiles, then feed each tile to the same grouped sparse-attention kernel
        used by batched verification.

        ``positions`` are absolute sequence positions, so this also supports
        scheduler-style chunked prefill: the K/V and index-key caches contain
        the prefix while only the new query rows are evaluated.
        """
        rows = q.shape[0]
        if rows == 0 or positions.shape != (rows,):
            raise ValueError(
                f"QSA prefill positions must have shape ({rows},), "
                f"got {tuple(positions.shape)}"
            )
        sequence_end = int(positions[-1].item()) + 1
        if sequence_end > k_cache.shape[0]:
            raise ValueError(
                f"QSA prefill end {sequence_end} exceeds K/V cache "
                f"capacity {k_cache.shape[0]}"
            )
        complete_groups = sequence_end // indexer.compress_ratio
        if pooled_k_cache is None:
            pooled = indexer.pool_keys(index_k_cache[:sequence_end])
        else:
            if complete_groups > pooled_k_cache.shape[0]:
                raise ValueError(
                    f"QSA prefill needs {complete_groups} pooled keys but cache "
                    f"capacity is {pooled_k_cache.shape[0]}"
                )
            pooled = pooled_k_cache[:complete_groups]
        compressed_keys = max(int(pooled.shape[0]), 1)
        rows_per_chunk = max(1, logits_workspace_bytes // (compressed_keys * 4))
        # Keep GEMM row tiles regular without turning a small request into a
        # padded allocation.  This mirrors the 128/head-count row granularity
        # used by the reference QSA prefill kernel.
        row_granularity = max(1, 128 // indexer.n_heads)
        if rows_per_chunk >= row_granularity:
            rows_per_chunk = max(
                row_granularity,
                rows_per_chunk // row_granularity * row_granularity,
            )
        rows_per_chunk = min(rows, rows_per_chunk)

        outputs = []
        ratio = indexer.compress_ratio
        for row_start in range(0, rows, rows_per_chunk):
            row_end = min(row_start + rows_per_chunk, rows)
            chunk = slice(row_start, row_end)
            chunk_positions = positions[chunk]
            # Only complete compression groups are eligible.  The unfinished
            # causal tail is appended explicitly by batch_decode_gather_indices.
            complete_groups = torch.div(
                chunk_positions + 1,
                ratio,
                rounding_mode="floor",
            )
            logits = indexer.score_blocks(index_q[chunk], pooled, complete_groups)
            blocks = indexer.select_blocks(logits, complete_groups)
            indices, valid, selected_counts = indexer.batch_prefill_gather_indices(
                blocks,
                chunk_positions,
                self.pad_to,
            )
            if qsa_sparse_attention_supported(q):
                sparse = qsa_sparse_prefill_attention(
                    q[chunk],
                    gate[chunk],
                    k_cache,
                    v_cache,
                    indices,
                    valid,
                    selected_counts,
                    k_scales,
                    v_scales,
                )
                outputs.append(self.attn.o_proj(sparse.reshape(row_end - row_start, -1)))
            else:
                outputs.append(
                    self(
                        q[chunk],
                        gate[chunk],
                        k_cache,
                        v_cache,
                        indices,
                        valid,
                        k_scales,
                        v_scales,
                    )
                )
        return torch.cat(outputs, dim=0)

    def causal_prefix(
        self,
        q: torch.Tensor,
        gate: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        positions: torch.Tensor,
        k_scales: torch.Tensor | None = None,
        v_scales: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend over a short fully-visible prefix without sparse padding.

        Before the QSA token budget is exhausted every causal key is
        selected.  Expanding each query to the fixed 2051-token sparse ABI
        wastes GiBs during MTP prompt sync, and ``repeat_interleave`` then
        duplicates two KV heads twelve times.  Keep the KV-head grouping
        explicit and materialise only the small causal score tensor.
        """
        a = self.attn
        rows = q.shape[0]
        if positions.shape != (rows,):
            raise ValueError(
                f"prefix positions must have shape ({rows},), got {tuple(positions.shape)}"
            )
        end = int(positions[-1].item()) + 1
        if qsa_sparse_attention_supported(q):
            if end > self._dense_indices.shape[0]:
                raise ValueError(
                    f"dense QSA prefix end {end} exceeds capacity "
                    f"{self._dense_indices.shape[0]}"
                )
            indices = self._dense_indices[:end].unsqueeze(0).expand(rows, -1)
            valid = indices <= positions.unsqueeze(1)
            selected_counts = (positions + 1).to(torch.int32)
            out = qsa_sparse_prefill_attention(
                q,
                gate,
                k_cache,
                v_cache,
                indices,
                valid,
                selected_counts,
                k_scales,
                v_scales,
            )
            return a.o_proj(out.reshape(rows, -1))
        k = k_cache[:end]
        v = v_cache[:end]
        if _qsa_cache_is_quantized(k_cache.dtype):
            if k_scales is None or v_scales is None:
                raise ValueError("FP8 QSA attention requires K/V scale caches")
            k = k.float() * k_scales[:end].float().unsqueeze(-1)
            v = v.float() * v_scales[:end].float().unsqueeze(-1)
        query = q.view(rows, a.num_kv_heads, a.repeat, a.head_dim)
        scores = torch.einsum("tgrd,sgd->tgrs", query.float(), k.float())
        scores = scores * (1.0 / math.sqrt(a.head_dim))
        key_positions = torch.arange(end, device=q.device)
        causal = key_positions.unsqueeze(0) <= positions.unsqueeze(1)
        scores.masked_fill_(~causal[:, None, None, :], torch.finfo(torch.float32).min)
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("tgrs,sgd->tgrd", probs, v.float()).to(q.dtype)
        out = out.reshape(rows, a.num_heads, a.head_dim)
        out = out * torch.sigmoid(gate.float()).to(q.dtype)
        return a.o_proj(out.reshape(rows, -1))

    def causal_prefix_fixed(
        self,
        q: torch.Tensor,
        gate: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        positions: torch.Tensor,
        capacity: int,
        k_scales: torch.Tensor | None = None,
        v_scales: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Graph-safe dense-prefix attention over a fixed cache capacity."""
        if capacity <= 0 or capacity > k_cache.shape[0]:
            raise ValueError(
                f"fixed prefix capacity must be in [1,{k_cache.shape[0]}], got {capacity}"
            )
        a = self.attn
        rows = q.shape[0]
        if positions.shape != (rows,):
            raise ValueError(
                f"prefix positions must have shape ({rows},), got {tuple(positions.shape)}"
            )
        if qsa_sparse_attention_supported(q):
            indices = self._dense_indices[:capacity].unsqueeze(0).expand(rows, -1)
            valid = indices <= positions.unsqueeze(1)
            selected_counts = torch.clamp(positions + 1, max=capacity).to(torch.int32)
            out = qsa_sparse_attention(
                q,
                gate,
                k_cache,
                v_cache,
                indices,
                valid,
                k_scales,
                v_scales,
                selected_counts,
            )
            return a.o_proj(out.reshape(rows, -1))
        k = k_cache[:capacity]
        v = v_cache[:capacity]
        if _qsa_cache_is_quantized(k_cache.dtype):
            if k_scales is None or v_scales is None:
                raise ValueError("FP8 QSA attention requires K/V scale caches")
            k = k.float() * k_scales[:capacity].float().unsqueeze(-1)
            v = v.float() * v_scales[:capacity].float().unsqueeze(-1)
        query = q.view(rows, a.num_kv_heads, a.repeat, a.head_dim)
        scores = torch.einsum("tgrd,sgd->tgrs", query.float(), k.float())
        scores = scores * (1.0 / math.sqrt(a.head_dim))
        key_positions = torch.arange(capacity, device=q.device)
        causal = key_positions.unsqueeze(0) <= positions.unsqueeze(1)
        scores.masked_fill_(~causal[:, None, None, :], torch.finfo(torch.float32).min)
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("tgrs,sgd->tgrd", probs, v.float()).to(q.dtype)
        out = out.reshape(rows, a.num_heads, a.head_dim)
        out = out * torch.sigmoid(gate.float()).to(q.dtype)
        return a.o_proj(out.reshape(rows, -1))


def load_qsa_attention(
    ckpt: pathlib.Path | str,
    layer_idx: int,
    device: str = "cpu",
    *,
    rope_theta: float = 1e7,
    mrope_section: tuple[int, ...] | None = None,
    mrope_interleaved: bool = False,
) -> FlashNextQSAAttention:
    from safetensors import safe_open

    ckpt = pathlib.Path(ckpt)
    with open(ckpt / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]
    prefix = f"model.language_model.layers.{layer_idx}.self_attn"

    def load(name: str) -> torch.Tensor:
        key = f"{prefix}.{name}"
        with safe_open(str(ckpt / weight_map[key]), framework="pt", device="cpu") as f:
            return f.get_tensor(key)

    module = FlashNextQSAAttention(
        rope_theta=rope_theta,
        mrope_section=mrope_section,
        mrope_interleaved=mrope_interleaved,
    )
    with torch.no_grad():
        module.q_proj.weight.copy_(load("q_proj.weight"))
        module.k_proj.weight.copy_(load("k_proj.weight"))
        module.v_proj.weight.copy_(load("v_proj.weight"))
        module.o_proj.weight.copy_(load("o_proj.weight"))
        module.q_norm.copy_(load("q_norm.weight"))
        module.k_norm.copy_(load("k_norm.weight"))
    return module.to(device)
