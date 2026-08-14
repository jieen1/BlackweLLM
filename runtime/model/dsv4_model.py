"""DeepSeek-V4-Flash model graph (GGUF IQ2_XS/Q8_0).

Phase 2 complete: full eager graph (containers, RMSNorm, Gate, MoE,
compressor, indexer, attention, Hyper-Connections, Block, Transformer) plus
the zero-exemption GGUF loader. Quantized weights stay packed end-to-end
(plan D2): eager paths dequantize on demand without caching BF16 copies (the
Qwen3.6 dequant-cache memory floor is the standing warning), and Phase 3
replaces the dequant call sites with fused kernels keeping the same numerics
as dsv4_quant.

Weight naming follows the GGUF file (verified 1:1 against llama.cpp's
create_tensor declarations, notes/2026-08-07-dsv4flash-fact-baseline.md §2.1).
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from loader.gguf_header import read_gguf_header
from loader.gguf_quant_tables import IQ2XS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS
from runtime.loading.gguf import GgufTensor, iterate_gguf_checkpoint
from runtime.model.dsv4_config import Dsv4Config, config_from_gguf_kv
from runtime.model.dsv4_quant import dequantize_iq2_xs, dequantize_q8_0

#: Per-device IQ2_XS dequant tables for the fused MoE GEMM (cached once).
_expert_tables_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


class PackedQ8_0Weight(nn.Module):
    """Packed Q8_0 storage plus on-demand fp32 dequantization (no forward).

    Also the container for Q8_0 matrices that are consumed outside a plain
    linear (the HC mixing matrices ``hc_*_fn``, dequantized inside hc_pre).
    """

    def __init__(
        self, out_features: int, in_features: int, *, device: torch.device | str | None = None
    ) -> None:
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.fused_q8 = False
        # SoA planes are the resident storage: code [out, in] int8 + scale
        # [out, in/32] fp16 (total bytes == the interleaved 34B layout, so
        # no extra residency).  The interleaved `packed` compatibility view
        # is rebuilt lazily only for eager oracles and legacy callers.
        nb = in_features // 32
        self.register_buffer(
            "qcode", torch.empty(out_features * in_features, dtype=torch.uint8, device=device)
        )
        self.register_buffer(
            "qscale", torch.empty(out_features * nb, dtype=torch.float16, device=device)
        )

    def soa_planes(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.qcode, self.qscale

    @property
    def packed(self) -> torch.Tensor:
        """Rebuild the interleaved 34B layout from the SoA planes.

        Compatibility view for eager oracles and tests that read the original
        GGUF block layout. Allocates a [out*in/32*34] buffer each call; serving
        GEMM/GEMV paths consume the resident SoA planes directly.
        """
        nb = self.in_features // 32
        q = self.qcode.view(self.out_features, nb, 32)
        d = self.qscale.view(self.out_features, nb, 1)
        out = torch.empty((self.out_features, nb, 34), dtype=torch.uint8, device=self.qcode.device)
        out[:, :, :2].view(torch.float16).copy_(d)
        out[:, :, 2:].copy_(q)
        return out.reshape(-1)

    @packed.setter
    def packed(self, value: torch.Tensor) -> None:
        wv = value.view(self.out_features, self.in_features // 32, 34)
        self.qcode.copy_(wv[:, :, 2:].reshape(-1))
        self.qscale.copy_(wv[:, :, :2].view(torch.float16).squeeze(-1).reshape(-1))

    def load_packed(self, tensor: GgufTensor) -> None:
        if tensor.type_name != "Q8_0":
            raise ValueError(f"{tensor.name}: expected Q8_0, got {tensor.type_name}")
        if tuple(tensor.shape) != (self.out_features, self.in_features):
            raise ValueError(
                f"{tensor.name}: expected {self.out_features}x{self.in_features}, "
                f"got {tensor.shape}"
            )
        # Store directly as SoA planes (setter splits the 34B interleaved
        # layout into code+scale); no 34B packed view is resident.
        self.packed = tensor.data.to(self.qcode.device)

    def dequantized(self) -> torch.Tensor:
        # Directly from the aligned SoA planes (qcode [out, in] int8 +
        # qscale [out, in/32] fp16) -- bit-identical to dequantize_q8_0 of
        # the interleaved packed layout (verified maxdiff 0.0), and avoids
        # rebuilding the 34B view when SoA is the resident storage.
        qf = (
            self.qcode.view(self.out_features, self.in_features // 32, 32)
            .view(torch.int8)
            .to(torch.float32)
        )
        df = self.qscale.view(self.out_features, self.in_features // 32, 1).to(torch.float32)
        return (df * qf).reshape(self.out_features, self.in_features)


class PackedQ8_0Linear(PackedQ8_0Weight):
    """Linear over a packed Q8_0 weight; dequantized per forward (eager).

    ``weight_dtype=None`` (default): fp32 dequant, fp32 compute -- the eager
    accuracy contract. ``weight_dtype=torch.bfloat16``: the dequantized values
    round to bf16 first and the matmul runs bf16 -- the regime the reference
    uses for bf16-declared linears (its ``linear()`` is strict on dtypes),
    and what these layers will run at in production.

    ``fused_q8=True`` routes bf16 projections through the fused Q8_0
    dequant-GEMM tensor-core kernel instead of eager dequant+cuBLAS.  The
    kernel accumulates in fp32, so its output is MORE accurate than cuBLAS
    bf16 (2.6e-5 vs 0.012 vs the exact reference) and never materializes a
    bf16/fp32 weight.  It is an opt-in accelerator (the eager path stays
    the official-reference bit-exact oracle); the serving backend turns it
    on for the decode hot path.
    """

    def __init__(
        self,
        out_features: int,
        in_features: int,
        *,
        weight_dtype: torch.dtype | None = None,
        fused_q8: bool = False,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(out_features, in_features, device=device)
        self.weight_dtype = weight_dtype
        self.fused_q8 = fused_q8
        self.fused_q8_fp32 = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        decode_rows = x.numel() // self.in_features
        if (
            getattr(self, "fused_q8_fp32", False)
            and x.device.type == "cuda"
            and x.dtype is torch.bfloat16
            and decode_rows in (1, 2, 4)
        ):
            if self.weight_dtype is not None:
                raise RuntimeError("fused_q8_fp32 is only valid for FP32-declared Q8_0 linears")
            from runtime.kernels.dsv4_q8_gemm import q8_0_dequant_gemv_fp32

            leading = x.shape[:-1]
            out = q8_0_dequant_gemv_fp32(
                x.reshape(-1, self.in_features),
                self.packed,
                out_features=self.out_features,
                in_features=self.in_features,
            )
            return out.reshape(*leading, self.out_features)
        if self.fused_q8 and x.device.type == "cuda":
            from runtime.kernels.dsv4_q8_gemm import (
                q8_0_dequant_gemv_warp_row_bf16,
                q8_0_soa_dequant_gemm,
            )

            leading = x.shape[:-1]
            x2 = x.reshape(-1, self.in_features)
            if x2.shape[0] == 1 and self.in_features % 32 == 0:
                # M=1 decode: warp-per-row GEMV over the repacked SoA planes
                # with the bf16 weight contract (matches tensor-core tl.dot
                # exactly, verified maxdiff 0) -- the aligned code plane
                # removes the 34B 2-byte alignment penalty (7.1x bandwidth).
                q, d = self.soa_planes()
                out = q8_0_dequant_gemv_warp_row_bf16(
                    x2,
                    q,
                    d,
                    out_features=self.out_features,
                    in_features=self.in_features,
                )
            else:
                q, d = self.soa_planes()
                out = q8_0_soa_dequant_gemm(
                    x2,
                    q,
                    d,
                    out_features=self.out_features,
                    in_features=self.in_features,
                )
            out = out.reshape(*leading, self.out_features)
            if self.weight_dtype is torch.bfloat16:
                return out.to(torch.bfloat16)
            return out
        weight = self.dequantized()
        if self.weight_dtype is not None:
            return F.linear(x, weight.to(self.weight_dtype))
        return F.linear(x.float(), weight)


class DenseLinear(nn.Module):
    """Dense weight stored in the file's own dtype; fp32-matmul forward.

    Used for the router weight (``ffn_gate_inp``, BF16 in the file). The
    reference regime is ``linear(x.float(), weight.float())`` -- an fp32
    matmul over bf16 values (reference Gate.forward).
    """

    def __init__(
        self, out_features: int, in_features: int, *, device: torch.device | str | None = None
    ) -> None:
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.register_buffer(
            "weight",
            torch.empty(out_features, in_features, dtype=torch.bfloat16, device=device),
        )

    def load(self, tensor: GgufTensor) -> None:
        if tensor.type_name != "BF16":
            raise ValueError(f"{tensor.name}: expected BF16, got {tensor.type_name}")
        if tuple(tensor.shape) != (self.out_features, self.in_features):
            raise ValueError(
                f"{tensor.name}: expected {self.out_features}x{self.in_features}, "
                f"got {tensor.shape}"
            )
        self.weight.copy_(tensor.data.to(self.weight.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            x.device.type == "cuda"
            and x.dtype is torch.bfloat16
            and x.shape[-1] == self.in_features
            and x.numel() // self.in_features in (1, 2, 4)
        ):
            # Decode must preserve one arithmetic path per token row.  A
            # regular GEMM selects an M-dependent cuBLAS algorithm, and its
            # few-ULP B=1/B=4 differences can cross a top-k router boundary
            # many layers later.  Strided batched GEMV keeps M=1 inside each
            # batch item, is CUDA-Graph-safe, and matches the established B=1
            # FP32 result bit-for-bit on the production SM120 stack.
            leading = x.shape[:-1]
            flat = x.reshape(-1, self.in_features)
            weight = self.weight.float().t().unsqueeze(0)
            out = torch.bmm(
                flat.float().unsqueeze(1),
                weight.expand(flat.shape[0], -1, -1),
            ).squeeze(1)
            return out.reshape(*leading, self.out_features)
        return F.linear(x.float(), self.weight.float())


class PackedIQ2_XSExperts(nn.Module):
    """Fused routed-expert weights [E, rows, cols] packed IQ2_XS.

    Eager path dequantizes only the experts a batch actually routes to --
    dequantizing all 256 experts of one matrix would transiently need ~4.3 GB
    in fp32, more than the card's headroom.
    """

    def __init__(
        self, num_experts: int, rows: int, cols: int, *, device: torch.device | str | None = None
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.rows = rows
        self.cols = cols
        row_blocks = cols // 256
        self._row_bytes = row_blocks * 74
        total = num_experts * rows * self._row_bytes
        self.register_buffer("packed", torch.empty(total, dtype=torch.uint8, device=device))

    def load_packed(self, tensor: GgufTensor) -> None:
        if tensor.type_name != "IQ2_XS":
            raise ValueError(f"{tensor.name}: expected IQ2_XS, got {tensor.type_name}")
        if tuple(tensor.shape) != (self.num_experts, self.rows, self.cols):
            raise ValueError(
                f"{tensor.name}: expected "
                f"{(self.num_experts, self.rows, self.cols)}, got {tensor.shape}"
            )
        self.packed.copy_(tensor.data.to(self.packed.device))

    def expert_weight(self, expert_id: int) -> torch.Tensor:
        """Dequantize one expert's [rows, cols] matrix (fp32)."""
        start = expert_id * self.rows * self._row_bytes
        end = start + self.rows * self._row_bytes
        return dequantize_iq2_xs(self.packed[start:end]).reshape(self.rows, self.cols)

    def tables(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """IQ2_XS dequant tables on this module's device (cached)."""
        key = str(self.packed.device)
        cache = _expert_tables_cache.get(key)
        if cache is None:
            cache = (
                torch.tensor(IQ2XS_GRID, dtype=torch.int64, device=self.packed.device),
                torch.tensor(KSIGNS_IQ2XS, dtype=torch.int32, device=self.packed.device),
                torch.tensor(KMASK_IQ2XS, dtype=torch.int32, device=self.packed.device),
            )
            _expert_tables_cache[key] = cache
        return cache


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Reference RMSNorm: fp32 internally, weight applied fp32, cast back."""
    if x.device.type == "cuda" and x.dtype is torch.bfloat16:
        from runtime.kernels.fused_rms_norm import rms_norm as fused_rms_norm

        return fused_rms_norm(x, weight, eps)
    dtype = x.dtype
    x = x.float()
    var = x.square().mean(-1, keepdim=True)
    return (weight.float() * (x * torch.rsqrt(var + eps))).to(dtype)


class Dsv4Gate(nn.Module):
    """sqrtsoftplus routing with noaux_tc bias; hash layers skip selection only.

    Bit-exact parity with the reference Gate is proven in
    tests/test_dsv4_reference_parts.py; this is the same computation as a
    graph-owned module.
    """

    def __init__(
        self,
        config: Dsv4Config,
        *,
        hashed: bool,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.top_k = config.n_activated_experts
        self.route_scale = config.route_scale
        self.hashed = hashed
        # ffn_gate_inp is BF16 in the file; reference regime is fp32 matmul
        # over bf16 values (linear(x.float(), weight.float())).
        self.weight = DenseLinear(config.n_routed_experts, config.hidden_size, device=device)
        if hashed:
            self.register_buffer(
                "tid2eid",
                torch.empty(config.vocab_size, self.top_k, dtype=torch.int32, device=device),
            )
        else:
            self.register_buffer(
                "bias",
                torch.empty(config.n_routed_experts, dtype=torch.float32, device=device),
            )

    def forward(
        self, x: torch.Tensor, input_ids: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.weight(x)
        if logits.device.type == "cuda":
            if self.hashed:
                assert input_ids is not None
                from runtime.kernels.dsv4_router import dsv4_route_hashed_scores

                supplied_ids = self.tid2eid[input_ids].contiguous()
                return dsv4_route_hashed_scores(logits, supplied_ids, self.route_scale)
            from runtime.kernels.dsv4_router import dsv4_route_scores

            return dsv4_route_scores(logits, self.bias, self.top_k, self.route_scale)
        scores = F.softplus(logits).sqrt()
        original = scores
        if self.hashed:
            assert input_ids is not None
            indices = self.tid2eid[input_ids].to(torch.int64)
        else:
            selection = scores + self.bias
            indices = selection.topk(self.top_k, dim=-1)[1]
        weights = original.gather(1, indices)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights * self.route_scale, indices


def swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    """Reference Expert clamping: up to [-limit, limit], gate to (-inf, limit]."""
    up = torch.clamp(up, min=-limit, max=limit)
    gate = torch.clamp(gate, max=limit)
    return F.silu(gate) * up


class Dsv4MoE(nn.Module):
    """Top-k routed experts (packed IQ2_XS) + one shared expert (Q8_0).

    Eager routing mirrors the reference loop (per-expert token gather); the
    fused permute/scatter kernel is Phase 3.
    """

    def __init__(
        self,
        config: Dsv4Config,
        *,
        hashed: bool,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.gate = Dsv4Gate(config, hashed=hashed, device=device)
        inter = config.moe_intermediate_size
        hidden = config.hidden_size
        experts = config.n_routed_experts
        self.gate_exps = PackedIQ2_XSExperts(experts, inter, hidden, device=device)
        self.up_exps = PackedIQ2_XSExperts(experts, inter, hidden, device=device)
        self.down_exps = PackedIQ2_XSExperts(experts, hidden, inter, device=device)
        self.shared_w1 = PackedQ8_0Linear(inter, hidden, device=device)
        self.shared_w3 = PackedQ8_0Linear(inter, hidden, device=device)
        self.shared_w2 = PackedQ8_0Linear(hidden, inter, device=device)

    def _shared_forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.shared_w1(x)
        up = self.shared_w3(x)
        decode_rows = x.numel() // self.config.hidden_size
        if x.device.type == "cuda" and x.dtype is torch.bfloat16 and decode_rows in (1, 2, 4):
            from runtime.kernels.dsv4_iq2xs_gemm import swiglu_bf16

            hidden = swiglu_bf16(gate, up, self.config.swiglu_limit)
        else:
            hidden = swiglu(gate, up, self.config.swiglu_limit).to(x.dtype)
        return self.shared_w2(hidden)

    def forward_decode_batch(self, x: torch.Tensor, input_ids: torch.Tensor | None) -> torch.Tensor:
        """Decode-only MoE path for a batch of single-token rows ``[B, 1, H]``.

        CUDA route expansion stays token-major and graph-safe: one fused
        gate/up+SwiGLU launch over ``B*K`` routed rows, one indexed down launch,
        then the fp32 route weights reduce the ``K`` expert contributions back
        to ``[B, H]`` before the shared expert is added once per token.
        """
        if x.ndim != 3:
            raise ValueError(f"forward_decode_batch expects [B, 1, H], got {tuple(x.shape)}")
        if x.shape[1] != 1:
            raise ValueError(f"forward_decode_batch expects seqlen=1, got {x.shape[1]}")
        batch = x.shape[0]
        flat_ids = None
        if input_ids is not None:
            if input_ids.numel() != batch:
                raise ValueError(
                    f"forward_decode_batch input_ids {input_ids.numel()} != batch {batch}"
                )
            flat_ids = input_ids.reshape(batch)
        if x.device.type != "cuda" or batch == 0:
            return self.forward(x, flat_ids)

        flat = x[:, 0]
        weights, indices = self.gate(flat, flat_ids)
        top_k = indices.shape[1]
        hidden = flat.shape[-1]
        limit = self.config.swiglu_limit
        xs_batch = (
            flat.unsqueeze(1)
            .expand(batch, top_k, hidden)
            .reshape(batch * top_k, 1, hidden)
            .contiguous()
        )
        eids = indices.reshape(batch * top_k).contiguous()

        from runtime.kernels.dsv4_iq2xs_gemm import iq2xs_dequant_gemm_batch_indexed_dual_swiglu_b1

        h = iq2xs_dequant_gemm_batch_indexed_dual_swiglu_b1(
            xs_batch,
            self.gate_exps.packed,
            self.up_exps.packed,
            eids,
            rows=self.gate_exps.rows,
            cols=self.gate_exps.cols,
            grid_tables=self.gate_exps.tables(),
            limit=limit,
        )
        routed = self._batch_expert_gemm(self.down_exps, eids, h)
        y = (routed[:, 0].reshape(batch, top_k, routed.shape[-1]) * weights.unsqueeze(-1)).sum(
            dim=1
        )
        y = y.to(torch.float32) + self._shared_forward(flat)
        return y.to(x.dtype).reshape(batch, 1, y.shape[-1])

    def _route_expanded_prefill(
        self,
        flat: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Run every prefill route as one indexed M=1 expert row on CUDA.

        The previous path synchronized expert counts to the host, padded every
        active expert to the largest route count, and launched Python scatter
        loops. DSV4 has a fixed top-6 contract, so the dense token-major route
        expansion is both bounded and smaller than that padded work for the
        production 32-row chunks. Contributions are accumulated in stable
        expert-id order to preserve the reference per-expert reduction order.
        """
        n_tokens, hidden = flat.shape
        top_k = indices.shape[1]
        routes = n_tokens * top_k
        eids = indices.reshape(routes).contiguous()

        # dp4a path: activations are int8-quantized (preq) and the IQ2 codes
        # are decoded in-register with an int8 dp4a inner product; the
        # per-code scale is applied before the code reduction.
        from runtime.kernels.dsv4_iq2xs_gemm import (
            iq2xs_dequant_gemm_indexed_dp4a,
            iq2xs_dequant_gemm_indexed_dual_dp4a,
            preq_activation,
        )

        xr = flat.unsqueeze(1).expand(n_tokens, top_k, hidden).reshape(routes, hidden).contiguous()
        xq, xs = preq_activation(xr)
        gate, up = iq2xs_dequant_gemm_indexed_dual_dp4a(
            xq,
            xs,
            self.gate_exps.packed,
            self.up_exps.packed,
            eids,
            rows=self.gate_exps.rows,
            cols=self.gate_exps.cols,
            grid_tables=self.gate_exps.tables(),
        )
        h = swiglu(gate, up, self.config.swiglu_limit)
        hq, hs = preq_activation(h)
        routed = iq2xs_dequant_gemm_indexed_dp4a(
            hq,
            hs,
            self.down_exps.packed,
            eids,
            rows=self.down_exps.rows,
            cols=self.down_exps.cols,
            grid_tables=self.down_exps.tables(),
        )
        contributions = routed.reshape(n_tokens, top_k, -1) * weights.unsqueeze(-1)
        order = torch.argsort(indices, dim=1, stable=True)
        contributions = contributions.gather(
            1,
            order.unsqueeze(-1).expand_as(contributions),
        )
        y = torch.zeros_like(flat, dtype=torch.float32)
        for top_slot in range(top_k):
            y = y + contributions[:, top_slot]
        return y

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor | None) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1])
        weights, indices = self.gate(flat, input_ids.reshape(-1) if input_ids is not None else None)
        limit = self.config.swiglu_limit
        y = torch.zeros_like(flat, dtype=torch.float32)
        n_tokens = flat.shape[0]
        if flat.device.type == "cuda" and n_tokens > 0:
            if n_tokens == 1:
                # Decode (M=1): the single token is routed to exactly
                # n_activated_experts experts, ALL receiving the same input.
                # Batch them as [E, 1, hidden] -- no padding, no sort, no
                # scatter: one batched dequant-GEMM per (gate/up/down).
                w1 = weights[0]  # [E]
                eids = indices[0]  # [E] int64 on device (capture-safe gather)
                xs_batch = flat.expand(len(eids), 1, -1).contiguous()
                wts = w1
                from runtime.kernels.dsv4_iq2xs_gemm import (
                    iq2xs_dequant_gemm_batch_indexed_dual_swiglu_b1,
                )

                h = iq2xs_dequant_gemm_batch_indexed_dual_swiglu_b1(
                    xs_batch,
                    self.gate_exps.packed,
                    self.up_exps.packed,
                    eids,
                    rows=self.gate_exps.rows,
                    cols=self.gate_exps.cols,
                    grid_tables=self.gate_exps.tables(),
                    limit=limit,
                )
                routed = self._batch_expert_gemm(self.down_exps, eids, h)
                y = (routed * wts[:, None, None]).sum(dim=0).to(torch.float32)
            else:
                from runtime.kernels.iq2_mma16_tc import grouped_moe_prefill_k32

                grid_t, ksigns_t, _ = self.gate_exps.tables()
                y = grouped_moe_prefill_k32(
                    flat,
                    weights,
                    indices,
                    self.gate_exps.packed,
                    self.up_exps.packed,
                    self.down_exps.packed,
                    grid_t,
                    ksigns_t,
                    inter=self.gate_exps.rows,
                    hidden=self.gate_exps.cols,
                    swiglu_limit=limit,
                )
        else:
            # CPU fallback: per-expert eager dequant (small/rare path)
            routed_ids = torch.unique(indices.reshape(-1))
            for expert_id in routed_ids.tolist():
                token_idx, top_slot = torch.where(indices == expert_id)
                xs = flat[token_idx]
                gate = self._expert_gemm(self.gate_exps, expert_id, xs)
                up = self._expert_gemm(self.up_exps, expert_id, xs)
                h = swiglu(gate, up, limit)
                routed_e = self._expert_gemm(self.down_exps, expert_id, h) * weights[
                    token_idx, top_slot
                ].unsqueeze(-1)
                y.index_add_(0, token_idx, routed_e)
        y = y + self._shared_forward(flat)
        return y.to(x.dtype).reshape(*x.shape[:-1], y.shape[-1])

    def forward_dynamic(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor | None,
        *,
        workspace=None,
    ) -> torch.Tensor:
        """Prefill MoE using per-expert compact dynamic GEMMs (llama.cpp style).

        The routed expert call stays a single invocation: its per-expert
        compact dynamic GEMMs are processed in bounded workspace chunks inside
        ``grouped_moe_prefill_k32_dynamic`` (bit-exact, verified) so the routed
        numerics are unchanged.  The shared expert is row-local and chunked
        separately when the batch is large so its full-M gate/up/down
        activations stay bounded (measured shared full-vs-chunk maxdiff
        ~1.75e-5 on real weights -- routed path is the exact oracle).
        """
        flat = x.reshape(-1, x.shape[-1])
        weights, indices = self.gate(flat, input_ids.reshape(-1) if input_ids is not None else None)
        limit = self.config.swiglu_limit
        from runtime.kernels.iq2_mma16_tc import grouped_moe_prefill_k32_dynamic

        grid_t, ksigns_t, _ = self.gate_exps.tables()
        y = grouped_moe_prefill_k32_dynamic(
            flat,
            weights,
            indices,
            self.gate_exps.packed,
            self.up_exps.packed,
            self.down_exps.packed,
            grid_t,
            ksigns_t,
            inter=self.gate_exps.rows,
            hidden=self.gate_exps.cols,
            swiglu_limit=limit,
            workspace=workspace,
        )
        shared = self._shared_chunked(flat)
        y = y + shared
        return y.to(x.dtype).reshape(*x.shape[:-1], y.shape[-1])

    def _shared_chunked(self, flat: torch.Tensor) -> torch.Tensor:
        """Shared-expert forward with bounded full-M activations.

        The shared gate/up/down are ``[M, inter]`` FP32 GEMMs; at a 128K
        prompt they alone reach ~6 GiB of transient activations.  The expert
        is row-local, so splitting the batch into fixed tiles keeps every
        intermediate at tile width while changing only the GEMM's M shape
        (measured maxdiff ~1.75e-5).  Small batches pass through unchanged.
        """
        from runtime.kernels.iq2_mma16_tc import DYNAMIC_MOE_CHUNK

        m = flat.shape[0]
        if m <= DYNAMIC_MOE_CHUNK:
            return self._shared_forward(flat)
        out = torch.empty_like(flat, dtype=torch.float32)
        for start in range(0, m, DYNAMIC_MOE_CHUNK):
            end = min(start + DYNAMIC_MOE_CHUNK, m)
            out[start:end] = self._shared_forward(flat[start:end])
        return out

    def _batch_expert_gemm(
        self,
        exps: PackedIQ2_XSExperts,
        eids: torch.Tensor,
        xs: torch.Tensor,
    ) -> torch.Tensor:
        """Batched ``xs @ W_e^T`` over the routed ``eids`` in one launch.

        ``eids`` is an int64 [E] tensor on the packed's device (decode
        route ids straight from the gate -- no CPU round-trip, so the
        path is CUDA-Graph capture-safe).
        """
        from runtime.kernels.dsv4_iq2xs_gemm import iq2xs_dequant_gemm_batch_indexed

        return iq2xs_dequant_gemm_batch_indexed(
            xs,
            exps.packed,
            eids,
            rows=exps.rows,
            cols=exps.cols,
            grid_tables=exps.tables(),
        )

    def _expert_gemm(
        self, exps: PackedIQ2_XSExperts, expert_id: int, xs: torch.Tensor
    ) -> torch.Tensor:
        """``xs @ W_expert^T`` with the fused IQ2_XS dequant-GEMM.

        The expert is [rows, cols] = [inter, hidden] for gate/up and
        [hidden, inter] for down; ``xs`` is [M, cols] activations (already
        bf16).  Falls back to the eager dequant+fp32-matmul for CPU or
        non-block-aligned shapes (the kernel is a CUDA Triton program).
        """
        if exps.packed.device.type != "cuda" or exps.cols % 256 or xs.device.type != "cuda":
            w = exps.expert_weight(expert_id)
            return xs.float() @ w.t()
        from runtime.kernels.dsv4_iq2xs_gemm import iq2xs_dequant_gemm

        start = expert_id * exps.rows * exps._row_bytes
        end = start + exps.rows * exps._row_bytes
        packed_expert = exps.packed[start:end]
        return iq2xs_dequant_gemm(
            xs,
            packed_expert,
            rows=exps.rows,
            cols=exps.cols,
            grid_tables=exps.tables(),
        )


# ---------------------------------------------------------------------------
# Compressor: gated pooling of every `ratio` tokens into one compressed KV
# entry (reference Compressor, model.py). Eager torch transcription; the
# QAT-simulation quant step uses dsv4_attention.act_quant_simulate.
# ---------------------------------------------------------------------------

from runtime.model.dsv4_attention import (  # noqa: E402
    act_quant_simulate,
    apply_rotary_emb,
    compress_topk_idxs,
    fp4_act_quant_simulate,
    hadamard_transform,
    precompute_freqs_cis,
    sparse_attention_eager,
    window_topk_idxs,
)


class Dsv4Compressor(nn.Module):
    """Compresses the latent KV stream ratio:1 with learned gated pooling.

    ratio-4 layers use overlap (coff=2): the first half of the wide rows is
    the previous window's carry-over, materialized by overlap_transform.
    Decode keeps kv_state/score_state per slot position within the current
    (overlapping) window and emits one compressed entry every `ratio` steps.
    """

    def __init__(
        self,
        config: Dsv4Config,
        layer_id: int,
        *,
        num_slots: int = 1,
        head_dim: int | None = None,
        rotate: bool = False,
        quantize: bool = True,
        bounded_cache: bool = False,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.ratio = config.layer_ratio(layer_id)
        assert self.ratio != 0
        if num_slots < 1:
            raise ValueError(f"num_slots must be >= 1, got {num_slots}")
        self.num_slots = num_slots
        self.overlap = self.ratio == 4
        self.rotate = rotate  # indexer variant: Hadamard + full-dim fp4 simulation
        # False for the kernel-path attention layer: its packed FP8 pages are
        # quantized by the pack kernel (dsv4_kv_pack), so the compressor emits
        # the raw normed/rotated entry and the QAT simulation moves downstream.
        self.quantize = quantize
        # Serving main-attention compressors only need the current emitted
        # entry as a merge source; the eager oracle and indexer retain their
        # historical cache layouts.
        self.bounded_cache = bounded_cache
        self.head_dim = head_dim if head_dim is not None else config.head_dim
        self.rope_head_dim = config.rope_head_dim
        self.eps = config.norm_eps
        self.ue8m0 = True  # production QAT config (fp8-origin weights)
        coff = 2 if self.overlap else 1
        self.coeff = coff
        self.wkv = PackedQ8_0Linear(coff * self.head_dim, config.hidden_size, device=device)
        self.wgate = PackedQ8_0Linear(coff * self.head_dim, config.hidden_size, device=device)
        self.register_buffer(
            "ape", torch.empty(self.ratio, coff * self.head_dim, dtype=torch.float32, device=device)
        )
        self.register_buffer(
            "norm_weight", torch.empty(self.head_dim, dtype=torch.float32, device=device)
        )
        # per-slot decode state; batch dim sized at capture/slot time (eager: 1)
        self.register_buffer(
            "kv_state",
            torch.zeros(
                num_slots,
                coff * self.ratio,
                coff * self.head_dim,
                dtype=torch.float32,
                device=device,
            ),
        )
        self.register_buffer(
            "score_state",
            torch.full(
                (num_slots, coff * self.ratio, coff * self.head_dim),
                float("-inf"),
                dtype=torch.float32,
                device=device,
            ),
        )
        self.register_buffer(
            "_graph_entry_scratch",
            torch.zeros(
                num_slots,
                16,
                self.head_dim,
                dtype=torch.bfloat16,
                device=device,
            ),
            persistent=False,
        )
        # Batch decode (B=1/2/4) needs a contiguous [bsz, 1, head_dim] out
        # slice with zero storage offset for the Triton contract check.
        # _graph_entry_scratch is [num_slots, 16, head_dim] (the seq-prefill
        # kernel's up-to-16 boundary rows), so its [bsz, 1, head_dim] view is
        # non-contiguous for B>1 and must not be reused here.
        self.register_buffer(
            "_decode_batch_out_scratch",
            torch.zeros(
                num_slots,
                1,
                self.head_dim,
                dtype=torch.bfloat16,
                device=device,
            ),
            persistent=False,
        )
        # Shared across slots on purpose: Phase A still captures one
        # slot-bound graph at a time, so the address may be reused safely.
        self.freqs_cis: torch.Tensor | None = None
        self.kv_cache: torch.Tensor | None = None  # assigned by the attention owner

    def _slot_view(self, tensor: torch.Tensor, slot: int) -> torch.Tensor:
        if not 0 <= slot < self.num_slots:
            raise IndexError(f"slot {slot} out of range ({self.num_slots} slots)")
        return tensor.narrow(0, slot, 1)

    def _slot_cache(self, slot: int) -> torch.Tensor:
        assert self.kv_cache is not None
        if self.kv_cache.shape[0] != self.num_slots:
            raise RuntimeError(
                "compressor kv_cache slot arena shape does not match num_slots: "
                f"{self.kv_cache.shape[0]} vs {self.num_slots}"
            )
        if self.bounded_cache:
            return self.kv_cache.as_strided(
                (1, 1, self.head_dim),
                (self.head_dim, 0, 1),
                storage_offset=slot * self.head_dim,
            )
        return self._slot_view(self.kv_cache, slot)

    def _batch_cache(self) -> torch.Tensor:
        """Return the cache view used by the B=1/2/4 decode kernels."""
        assert self.kv_cache is not None
        if self.bounded_cache:
            return self.kv_cache.as_strided(
                (self.num_slots, 1, self.head_dim),
                (self.head_dim, 0, 1),
            )
        return self.kv_cache

    def reset_slot(self, slot: int) -> None:
        self._slot_view(self.kv_state, slot).zero_()
        self._slot_view(self.score_state, slot).fill_(float("-inf"))

    def overlap_transform(self, tensor: torch.Tensor, value: float) -> torch.Tensor:
        """[b, s, r, 2d] -> [b, s, 2r, d]: current window's second half plus
        the previous window's first half shifted down one row."""
        b, s, _, _ = tensor.shape
        ratio, d = self.ratio, self.head_dim
        out = tensor.new_full((b, s, 2 * ratio, d), value)
        out[:, :, ratio:] = tensor[:, :, :, d:]
        out[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return out

    def forward_graph(
        self, x: torch.Tensor, pos: torch.Tensor, *, slot: int = 0
    ) -> torch.Tensor | None:
        """Capture-safe decode step: single token at GPU position ``pos``.

        Mirrors the ``start_pos > 0`` branch of :meth:`forward` for seqlen=1
        but drives the position-dependent steps with the GPU scalar ``pos``
        and mask-outs the conditional write/state-migration so the graph has
        no Python branches and no allocations beyond the fixed buffers.

        Returns the emitted compressed entry ``[1, 1, head_dim]`` or None on
        a non-compress step -- the same contract as the eager path.
        """
        assert self.kv_cache is not None and self.freqs_cis is not None
        ratio, overlap = self.ratio, self.overlap
        dtype = x.dtype
        # Keep the Q8 projection's M=64 execution shape: changing it to one
        # M=1024 GEMM changes Tensor-Core rounding before the overlap softmax.
        full_dim = self.coeff * self.head_dim
        kv = torch.empty((1, x.shape[1], full_dim), dtype=torch.float32, device=x.device)
        score = torch.empty_like(kv)
        for start in range(0, x.shape[1], 64):
            end = start + 64
            xf = x[:, start:end].float()
            kv[:, start:end].copy_(self.wkv(xf))
            score[:, start:end].copy_(self.wgate(xf))
        bsz, seqlen, _ = x.shape
        assert seqlen == 1, "forward_graph is the 1-token decode step"
        kv_state_slot = self._slot_view(self.kv_state, slot)
        score_state_slot = self._slot_view(self.score_state, slot)
        kv_cache_slot = self._slot_cache(slot)
        if x.device.type == "cuda":
            from runtime.kernels.dsv4_compressor import (
                fused_decode_postgemv,
                fused_indexer_decode_postgemv,
                supports_fused_decode_postgemv,
                supports_fused_indexer_decode_postgemv,
            )

            if supports_fused_decode_postgemv(
                ratio=ratio,
                rotate=self.rotate,
                quantize=self.quantize,
                device=x.device,
                batch_size=bsz,
                seq_len=seqlen,
                head_dim=self.head_dim,
                rope_head_dim=self.rope_head_dim,
            ):
                return fused_decode_postgemv(
                    kv_i=kv[:, 0:1],
                    score_i=score[:, 0:1],
                    pos=pos,
                    ratio=ratio,
                    head_dim=self.head_dim,
                    rope_head_dim=self.rope_head_dim,
                    overlap=overlap,
                    ape=self.ape,
                    norm_weight=self.norm_weight,
                    freqs_cis=self.freqs_cis,
                    kv_state=kv_state_slot,
                    score_state=score_state_slot,
                    kv_cache=kv_cache_slot,
                    out=self._graph_entry_scratch.narrow(0, 0, 1).narrow(1, 0, 1),
                    eps=self.eps,
                )
            if supports_fused_indexer_decode_postgemv(
                ratio=ratio,
                rotate=self.rotate,
                quantize=self.quantize,
                device=x.device,
                batch_size=bsz,
                seq_len=seqlen,
                head_dim=self.head_dim,
                rope_head_dim=self.rope_head_dim,
            ):
                return fused_indexer_decode_postgemv(
                    kv_i=kv[:, 0:1],
                    score_i=score[:, 0:1],
                    pos=pos,
                    ape=self.ape,
                    norm_weight=self.norm_weight,
                    freqs_cis=self.freqs_cis,
                    kv_state=kv_state_slot,
                    score_state=score_state_slot,
                    kv_cache=kv_cache_slot,
                    out=self._graph_entry_scratch.narrow(0, 0, 1).narrow(1, 0, 1),
                    eps=self.eps,
                )
        pos0 = pos  # [1] int64
        slot = pos0 % ratio
        should_compress = ((pos0 + 1) % ratio) == 0  # [1] bool
        kv_i = kv[:, 0:1]
        score_i = score[:, 0:1]
        apeslot = self.ape[slot].unsqueeze(0)  # [1,1,coff*head_dim]
        score_i = score_i + apeslot
        if overlap:
            kvs = kv_state_slot  # [1, 2r, 2d]
            scs = score_state_slot
            # GPU advanced index write (capture-safe: fixed buffer, value from pos)
            idx = (ratio + slot).to(torch.long)  # [1]
            kvs[:, idx] = kv_i.squeeze(1)
            scs[:, idx] = score_i.squeeze(1)
            kv_state = torch.cat(
                [kvs[:bsz, :ratio, : self.head_dim], kvs[:bsz, ratio:, self.head_dim :]], dim=1
            )
            score_state = torch.cat(
                [scs[:bsz, :ratio, : self.head_dim], scs[:bsz, ratio:, self.head_dim :]], dim=1
            )
            kv_c = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
            # Masked state migration (only on compress steps).  Selection is
            # required here: score_state deliberately contains -inf sentinels,
            # and a multiply-mask would turn the inactive 0 * -inf branch into
            migrate = should_compress.reshape(1, 1, 1)
            kvs[:, :ratio] = torch.where(migrate, kvs[:, ratio:], kvs[:, :ratio])
            scs[:, :ratio] = torch.where(migrate, scs[:, ratio:], scs[:, :ratio])
        else:
            kvs = kv_state_slot
            scs = score_state_slot
            idx = slot.to(torch.long)
            kvs[:, idx] = kv_i.squeeze(1)
            scs[:, idx] = score_i.squeeze(1)
            kv_c = (kvs[:bsz] * scs[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)

        # finalize (norm + rope + quant), always compute; write masked
        kv_c = self._finalize_graph(kv_c, pos0, dtype)
        kv_cache = kv_cache_slot  # [1, n_entries, head_dim]
        slot_c = (pos0 // ratio).to(torch.long)  # [1] entry index (dim 1)
        sc_mask = should_compress.reshape(1, 1)
        existing = kv_cache[:bsz].index_select(1, slot_c)  # [1, 1, head_dim]
        new_entry = kv_c.squeeze(1)  # [1, head_dim]
        merged = torch.where(
            sc_mask.reshape(1, 1, 1).expand_as(existing), new_entry.unsqueeze(1), existing
        )
        kv_cache[:bsz].index_copy_(1, slot_c, merged)
        # The pack path always runs (graph has no branch); return the entry
        # that EQUALS the current kv_cache slot on every step -- the freshly
        # finalized entry on a compress step, the existing slot otherwise --
        # so packing it into the FP8 pages always mirrors kv_cache.
        return torch.where(
            should_compress.reshape(1, 1).expand_as(new_entry), kv_c.squeeze(1), existing.squeeze(1)
        ).unsqueeze(1)

    def forward_graph_batch(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        slot_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Capture-safe B=1/2/4 decode over distinct slot-arena rows.

        Slot uniqueness and all device-value bounds are validated by the
        backend from host request metadata before graph replay.  This method
        deliberately performs no ``item``/``tolist``/``unique`` device sync.
        """
        assert self.kv_cache is not None and self.freqs_cis is not None
        bsz, seqlen, _ = x.shape
        if bsz not in (1, 2, 4) or seqlen != 1:
            raise ValueError(
                "forward_graph_batch requires x shaped [B, 1, hidden] with "
                f"B in (1, 2, 4), got {tuple(x.shape)}"
            )
        if pos.shape != (bsz,) or pos.dtype != torch.int64:
            raise ValueError(f"pos must be int64 [{bsz}], got {tuple(pos.shape)} {pos.dtype}")
        if slot_ids.shape != (bsz,) or slot_ids.dtype != torch.int64:
            raise ValueError(
                f"slot_ids must be int64 [{bsz}], got {tuple(slot_ids.shape)} {slot_ids.dtype}"
            )
        if x.device.type != "cuda":
            raise RuntimeError("forward_graph_batch requires CUDA")

        xf = x.float()
        kv = self.wkv(xf)
        score = self.wgate(xf)
        out = self._decode_batch_out_scratch.narrow(0, 0, bsz)
        from runtime.kernels.dsv4_compressor import (
            fused_decode_postgemv_batch,
            fused_indexer_decode_postgemv_batch,
            supports_fused_decode_postgemv_batch,
            supports_fused_indexer_decode_postgemv_batch,
        )

        common = {
            "kv_i": kv[:, 0:1],
            "score_i": score[:, 0:1],
            "pos": pos,
            "slot_ids": slot_ids,
            "ape": self.ape,
            "norm_weight": self.norm_weight,
            "freqs_cis": self.freqs_cis,
            "kv_state": self.kv_state,
            "score_state": self.score_state,
            "kv_cache": self._batch_cache(),
            "out": out,
            "eps": self.eps,
        }
        if supports_fused_decode_postgemv_batch(
            ratio=self.ratio,
            rotate=self.rotate,
            quantize=self.quantize,
            device=x.device,
            batch_size=bsz,
            seq_len=seqlen,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
        ):
            return fused_decode_postgemv_batch(
                **common,
                ratio=self.ratio,
                head_dim=self.head_dim,
                rope_head_dim=self.rope_head_dim,
                overlap=self.overlap,
            )
        if supports_fused_indexer_decode_postgemv_batch(
            ratio=self.ratio,
            rotate=self.rotate,
            quantize=self.quantize,
            device=x.device,
            batch_size=bsz,
            seq_len=seqlen,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
        ):
            return fused_indexer_decode_postgemv_batch(**common)
        raise RuntimeError(
            "no native DSV4 batched compressor kernel for "
            f"ratio={self.ratio}, rotate={self.rotate}, quantize={self.quantize}, "
            f"head_dim={self.head_dim}"
        )

    def _finalize(
        self, kv: torch.Tensor, position_count: int, dtype: torch.dtype, pos: int
    ) -> torch.Tensor:
        """Norm + rope + QAT simulation; writes into the owner's kv_cache.

        ``pos`` is the absolute position of the entry's last token: the
        bulk path (pos==0) applies one frame per compressed entry over the
        first ``position_count`` tokens; the decode/chunk path applies the
        single frame of the entry's own window end (``pos+1-ratio``) --
        the exact convention the sequential decode oracle uses, which is
        why a chunk of L tokens equals L single decode steps.
        """
        kv = rms_norm(kv.to(dtype), self.norm_weight, self.eps)
        if pos == 0:
            freqs = self.freqs_cis[: position_count : self.ratio]
        else:
            freqs = self.freqs_cis[pos + 1 - self.ratio].unsqueeze(0)
        apply_rotary_emb(kv[..., -self.rope_head_dim :], freqs)
        if not self.quantize:
            return kv
        if self.rotate:
            # indexer path: Hadamard over the full head dim, then fp4 block-32
            # simulation on everything (rope included, post-rotation).
            kv = hadamard_transform(kv, self.head_dim**-0.5)
            kv = fp4_act_quant_simulate(kv, 32)
        else:
            kv[..., : -self.rope_head_dim] = act_quant_simulate(
                kv[..., : -self.rope_head_dim], 64, ue8m0=self.ue8m0
            )
        return kv

    def _finalize_graph(
        self, kv: torch.Tensor, pos: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        """Capture-safe ``_finalize`` for a single entry at GPU position ``pos``."""
        kv = rms_norm(kv.to(dtype), self.norm_weight, self.eps)
        freqs = self.freqs_cis[pos + 1 - self.ratio]
        apply_rotary_emb(kv[..., -self.rope_head_dim :], freqs)
        if not self.quantize:
            return kv
        if self.rotate:
            kv = hadamard_transform(kv, self.head_dim**-0.5)
            kv = fp4_act_quant_simulate(kv, 32)
        else:
            kv[..., : -self.rope_head_dim] = act_quant_simulate(
                kv[..., : -self.rope_head_dim], 64, ue8m0=self.ue8m0
            )
        return kv

    def forward(self, x: torch.Tensor, start_pos: int, *, slot: int = 0) -> torch.Tensor | None:
        assert self.kv_cache is not None and self.freqs_cis is not None
        bsz, seqlen, _ = x.shape
        ratio, overlap = self.ratio, self.overlap
        dtype = x.dtype
        xf = x.float()
        kv = self.wkv(xf)
        score = self.wgate(xf)
        kv_state_slot = self._slot_view(self.kv_state, slot)
        score_state_slot = self._slot_view(self.score_state, slot)
        kv_cache_slot = self._slot_cache(slot)
        if start_pos == 0 and seqlen < ratio:
            # A cold start of fewer than ``ratio`` tokens cannot form a
            # compression group yet.  Run them through the per-token state
            # machine instead of the bulk branch: the bulk formula assumes
            # complete ratio-blocks and would leave the lower state half
            # empty, producing 0/0 NaN on the next decode step.  This is
            # exactly L sequential decode steps at pos 0..L-1, matching the
            # decode-oracle contract.
            emitted: list[torch.Tensor] = []
            for i in range(seqlen):
                pos = start_pos + i
                kv_i = kv[:, i : i + 1]
                score_i = score[:, i : i + 1]
                should_compress = (pos + 1) % ratio == 0
                score_i = score_i + self.ape[pos % ratio]
                if overlap:
                    kv_state_slot[:bsz, ratio + pos % ratio] = kv_i.squeeze(1)
                    score_state_slot[:bsz, ratio + pos % ratio] = score_i.squeeze(1)
                    if should_compress:
                        kv_state = torch.cat(
                            [
                                kv_state_slot[:bsz, :ratio, : self.head_dim],
                                kv_state_slot[:bsz, ratio:, self.head_dim :],
                            ],
                            dim=1,
                        )
                        score_state = torch.cat(
                            [
                                score_state_slot[:bsz, :ratio, : self.head_dim],
                                score_state_slot[:bsz, ratio:, self.head_dim :],
                            ],
                            dim=1,
                        )
                        kv_c = (kv_state * score_state.softmax(dim=1)).sum(
                            dim=1, keepdim=True
                        )
                        kv_state_slot[:bsz, :ratio] = kv_state_slot[:bsz, ratio:]
                        score_state_slot[:bsz, :ratio] = score_state_slot[:bsz, ratio:]
                else:
                    kv_state_slot[:bsz, pos % ratio] = kv_i.squeeze(1)
                    score_state_slot[:bsz, pos % ratio] = score_i.squeeze(1)
                    if should_compress:
                        kv_c = (
                            kv_state_slot[:bsz] * score_state_slot[:bsz].softmax(dim=1)
                        ).sum(dim=1, keepdim=True)
                if not should_compress:
                    continue
                kv_c = self._finalize(kv_c, 0, dtype, pos)
                if self.bounded_cache:
                    kv_cache_slot[:bsz, :1].copy_(kv_c)
                else:
                    kv_cache_slot[:bsz, pos // ratio] = kv_c.squeeze(1)
                emitted.append(kv_c)
            if not emitted:
                return None
            return torch.cat(emitted, dim=1)
        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            offset = ratio if overlap else 0
            if overlap and cutoff >= ratio:
                kv_state_slot[:bsz, :ratio] = kv[:, cutoff - ratio : cutoff]
                score_state_slot[:bsz, :ratio] = score[:, cutoff - ratio : cutoff] + self.ape
            if remainder > 0:
                kv_head, kv_tail = kv.split([cutoff, remainder], dim=1)
                kv_state_slot[:bsz, offset : offset + remainder] = kv_tail
                score_state_slot[:bsz, offset : offset + remainder] = (
                    score[:, cutoff:] + self.ape[:remainder]
                )
                kv, score = kv_head, score[:, :cutoff]
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio)) + self.ape
            if overlap:
                kv = self.overlap_transform(kv, 0)
                score = self.overlap_transform(score, float("-inf"))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
            if not should_compress:
                return None
            kv = self._finalize(kv, cutoff, dtype, start_pos)
            if self.bounded_cache:
                kv_cache_slot[:bsz, :1].copy_(kv[:, -1:])
            else:
                kv_cache_slot[:bsz, : seqlen // ratio] = kv
            return kv
        # decode / mid-sequence prefill chunk: step the per-token state
        # machine `seqlen` times (a chunk is L sequential decode steps --
        # the eager decode branch is the per-token oracle).
        emitted: list[torch.Tensor] = []
        for i in range(seqlen):
            pos = start_pos + i
            kv_i = kv[:, i : i + 1]
            score_i = score[:, i : i + 1]
            should_compress = (pos + 1) % ratio == 0
            score_i = score_i + self.ape[pos % ratio]
            if overlap:
                kv_state_slot[:bsz, ratio + pos % ratio] = kv_i.squeeze(1)
                score_state_slot[:bsz, ratio + pos % ratio] = score_i.squeeze(1)
                if should_compress:
                    kv_state = torch.cat(
                        [
                            kv_state_slot[:bsz, :ratio, : self.head_dim],
                            kv_state_slot[:bsz, ratio:, self.head_dim :],
                        ],
                        dim=1,
                    )
                    score_state = torch.cat(
                        [
                            score_state_slot[:bsz, :ratio, : self.head_dim],
                            score_state_slot[:bsz, ratio:, self.head_dim :],
                        ],
                        dim=1,
                    )
                    kv_c = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
                    kv_state_slot[:bsz, :ratio] = kv_state_slot[:bsz, ratio:]
                    score_state_slot[:bsz, :ratio] = score_state_slot[:bsz, ratio:]
            else:
                kv_state_slot[:bsz, pos % ratio] = kv_i.squeeze(1)
                score_state_slot[:bsz, pos % ratio] = score_i.squeeze(1)
                if should_compress:
                    kv_c = (kv_state_slot[:bsz] * score_state_slot[:bsz].softmax(dim=1)).sum(
                        dim=1, keepdim=True
                    )
            if not should_compress:
                continue
            kv_c = self._finalize(kv_c, 0, dtype, pos)
            if self.bounded_cache:
                kv_cache_slot[:bsz, :1].copy_(kv_c)
            else:
                kv_cache_slot[:bsz, pos // ratio] = kv_c.squeeze(1)
            emitted.append(kv_c)
        if not emitted:
            return None
        return torch.cat(emitted, dim=1)

    def forward_graph_prefill(
        self,
        x: torch.Tensor,
        pos_tensor: torch.Tensor,
        *,
        slot: int = 0,
        host_start_pos: int | None = None,
    ) -> torch.Tensor:
        """Capture-safe prefill chunk: ``seqlen`` rows at GPU position.

        Keeps wkv/wgate GEMMs batched. Main compressors use one sequential
        tile kernel; ratio-4 indexer compressors use one state/finalize kernel
        plus one Hadamard/FP4 kernel for all boundaries. Returns only the
        compress-boundary NEW entries, exactly like ``forward``'s ``emitted``
        list -- re-packing an existing entry would re-quantise an already-FP8
        page. Tiles are 64-aligned so ``pos_tensor % ratio == 0`` and the
        boundary rows are the fixed ``i % ratio == ratio - 1``.
        """
        bsz, seqlen, _ = x.shape
        assert bsz == 1
        ratio = self.ratio
        xf = x.float()
        kv = self.wkv(xf)
        score = self.wgate(xf)
        kv_state_slot = self._slot_view(self.kv_state, slot)
        score_state_slot = self._slot_view(self.score_state, slot)
        kv_cache_slot = self._slot_cache(slot)
        from runtime.kernels.dsv4_compressor import (
            fused_decode_postgemv_seq,
            supports_fused_decode_postgemv,
        )

        if supports_fused_decode_postgemv(
            ratio=ratio,
            rotate=self.rotate,
            quantize=self.quantize,
            device=x.device,
            batch_size=1,
            seq_len=1,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
        ):
            host_pos = (
                int(pos_tensor.reshape(-1)[0].item())
                if host_start_pos is None
                else host_start_pos
            )
            n_boundaries = (host_pos + seqlen) // ratio - host_pos // ratio
            out = self._graph_entry_scratch.narrow(0, 0, 1).narrow(1, 0, max(1, n_boundaries))
            result = fused_decode_postgemv_seq(
                kv=kv,
                score=score,
                pos0=pos_tensor,
                ratio=ratio,
                head_dim=self.head_dim,
                rope_head_dim=self.rope_head_dim,
                overlap=self.overlap,
                ape=self.ape,
                norm_weight=self.norm_weight,
                freqs_cis=self.freqs_cis,
                kv_state=kv_state_slot,
                score_state=score_state_slot,
                kv_cache=kv_cache_slot,
                out=out,
                eps=self.eps,
            )
            return result.narrow(1, 0, n_boundaries)

        # Indexer compressor (ratio-4 head_dim=128 rotate+quantize): advance
        # the whole tile in two launches (state/finalize + Hadamard/FP4).
        from runtime.kernels.dsv4_compressor import fused_indexer_decode_postgemv_seq

        host_pos = (
            int(pos_tensor.reshape(-1)[0].item()) if host_start_pos is None else host_start_pos
        )
        n_boundaries = (host_pos + seqlen) // ratio - host_pos // ratio
        out = self._graph_entry_scratch.narrow(0, 0, 1).narrow(1, 0, max(1, n_boundaries))
        return fused_indexer_decode_postgemv_seq(
            kv=kv,
            score=score,
            pos0=pos_tensor,
            host_start_pos=host_pos,
            ape=self.ape,
            norm_weight=self.norm_weight,
            freqs_cis=self.freqs_cis,
            kv_state=kv_state_slot,
            score_state=score_state_slot,
            kv_cache=kv_cache_slot,
            out=out,
            eps=self.eps,
        )

    def forward_cold_prefill_parallel(
        self,
        x: torch.Tensor,
        *,
        completed_rows: int,
        slot: int = 0,
    ) -> torch.Tensor:
        """Precompute suffix entries while preserving the 64-row oracle.

        The caller has already advanced ``completed_rows`` through the normal
        cold-prefill path.  Projection is batched over the complete sequence;
        compression boundaries after that prefix run independently on GPU.
        """
        if x.device.type != "cuda" or x.shape[0] != 1:
            raise ValueError("parallel cold prefill requires one CUDA batch")
        if completed_rows != 64 or x.shape[1] % 128 != 0:
            raise ValueError(
                "parallel cold prefill currently requires completed_rows=64 "
                f"and a 128-row-aligned sequence, got {completed_rows}, {x.shape[1]}"
            )
        xf = x.float()
        kv = self.wkv(xf)
        score = self.wgate(xf)
        from runtime.kernels.dsv4_compressor import fused_cold_prefill_postgemv_parallel

        return fused_cold_prefill_postgemv_parallel(
            kv=kv,
            score=score,
            completed_rows=completed_rows,
            ratio=self.ratio,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            overlap=self.overlap,
            ape=self.ape,
            norm_weight=self.norm_weight,
            freqs_cis=self.freqs_cis,
            kv_state=self._slot_view(self.kv_state, slot),
            score_state=self._slot_view(self.score_state, slot),
            kv_cache=self._slot_cache(slot),
            eps=self.eps,
        )


class Dsv4Indexer(nn.Module):
    """Selects top-k compressed-KV positions for CSA layers (ratio 4).

    Reference Indexer transcription: low-rank query from the main q latent,
    rope + Hadamard + fp4 QAT simulation on the query; its own rotating
    compressor builds the compressed K cache; scores = relu(q·k) weighted by
    weights_proj, summed over heads, top-k. Q8_0 file weights round to bf16
    for the query/weight projections (the reference regime for bf16-declared
    linears); the score accumulation after the einsum is fp32.
    """

    def __init__(
        self,
        config: Dsv4Config,
        layer_id: int,
        *,
        num_slots: int = 1,
        max_seq_len: int = 4096,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        assert config.has_indexer(layer_id)
        if num_slots < 1:
            raise ValueError(f"num_slots must be >= 1, got {num_slots}")
        self.num_slots = num_slots
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.rope_head_dim
        self.index_topk = config.index_topk
        self.softmax_scale = self.head_dim**-0.5
        self.wq_b = PackedQ8_0Linear(
            self.n_heads * self.head_dim,
            config.q_lora_rank,
            weight_dtype=torch.bfloat16,
            device=device,
        )
        self.weights_proj = PackedQ8_0Linear(
            self.n_heads, config.hidden_size, weight_dtype=torch.bfloat16, device=device
        )
        self.compressor = Dsv4Compressor(
            config,
            layer_id,
            num_slots=num_slots,
            head_dim=self.head_dim,
            rotate=True,
            device=device,
        )
        # indexer owns its scoring cache (reference layout); its compressor
        # writes into it via the wiring in forward().
        self.register_buffer(
            "kv_cache",
            torch.zeros(
                num_slots,
                max_seq_len // config.layer_ratio(layer_id),
                self.head_dim,
                dtype=torch.bfloat16,
                device=device,
            ),
        )
        self.freqs_cis: torch.Tensor | None = None

    def _slot_cache(self, slot: int) -> torch.Tensor:
        if not 0 <= slot < self.num_slots:
            raise IndexError(f"slot {slot} out of range ({self.num_slots} slots)")
        return self.kv_cache.narrow(0, slot, 1)

    def reset_slot(self, slot: int) -> None:
        """Hard-reset both recurrent state and retained scoring cache."""
        self.reset_state(slot)
        self.clear_cache(slot)

    def reset_state(self, slot: int) -> None:
        """Reset only recursive state, preserving KV-like prefix bytes."""
        self.compressor.reset_slot(slot)

    def clear_cache(self, slot: int) -> None:
        """Discard the slot's retained indexer keys."""
        self._slot_cache(slot).zero_()

    def _score_entries(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        n_entries: int,
        *,
        slot: int = 0,
    ) -> torch.Tensor:
        """Score the live compressed-KV prefix in fp32.

        The eager path must reproduce the reference einsum bit-for-bit
        (``einsum -> relu -> weights -> sum`` all fp32) so that top-k ordering
        is identical; the CUDA kernel path is only used by the fixed-shape
        graph scorer (``forward_graph``) where the reference is the CG's own
        oracle, not the reference module.
        """
        bsz, seqlen = q.shape[:2]
        kv = self._slot_cache(slot)[:bsz, :n_entries]
        score = torch.einsum("bshd,btd->bsht", q, kv)
        return (score.relu() * weights.unsqueeze(-1)).sum(dim=2)

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_pos: int,
        offset: int,
        *,
        slot: int = 0,
        compressor_precomputed: bool = False,
    ) -> torch.Tensor:
        assert self.kv_cache is not None and self.freqs_cis is not None
        bsz, seqlen, _ = x.shape
        ratio = self.compressor.ratio
        end_pos = start_pos + seqlen
        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache
            self.compressor.freqs_cis = self.freqs_cis
        if not compressor_precomputed:
            self.compressor(x, start_pos, slot=slot)

        # Until the compressed history exceeds index_topk, every live entry is
        # attended.  Return the canonical physical order padded to the fixed
        # graph width without scoring: the topk over all-positive scores
        # selects exactly these rows, and the graph scorer must stay fixed
        # shape.  (The set of selected entries is what matters for attention;
        # the reference module's topk may return them in a different order.)
        live_entries = end_pos // ratio
        if seqlen == 1 and start_pos > 0 and live_entries <= self.index_topk:
            indices = torch.arange(self.index_topk, dtype=torch.int64, device=x.device)
            indices = torch.where(indices < live_entries, indices + offset, -1)
            return indices.reshape(1, 1, self.index_topk).expand(bsz, -1, -1)

        freqs = self.freqs_cis[start_pos:end_pos]
        q = self.wq_b(qr)
        q = q.unflatten(-1, (self.n_heads, self.head_dim))
        apply_rotary_emb(q[..., -self.rope_head_dim :], freqs)
        if q.device.type == "cuda":
            from runtime.kernels.dsv4_compressor import hadamard_fp4_query

            q = hadamard_fp4_query(q)
        else:
            q = hadamard_transform(q, self.head_dim**-0.5)
            q = fp4_act_quant_simulate(q, 32)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
        index_score = self._score_entries(q, weights, end_pos // ratio, slot=slot)
        if seqlen > 1:
            # Mid-sequence prefill chunk: each row attends only the
            # compressed entries up to its own position ((pos+1)//ratio,
            # including its just-written entry -- same as single-token
            # decode).  Mask later entries out BEFORE topk so they can
            # never steal a slot, exactly like the start_pos==0 branch.
            bounds = (start_pos + torch.arange(1, seqlen + 1, device=x.device)) // ratio
            causal = torch.arange(end_pos // ratio, device=x.device).unsqueeze(0).repeat(seqlen, 1)
            index_score = index_score + torch.where(
                causal >= bounds.unsqueeze(1),
                float("-inf"),
                0.0,
            )
        elif start_pos == 0:
            causal = torch.arange(seqlen // ratio, device=x.device).repeat(seqlen, 1)
            causal = causal >= torch.arange(1, seqlen + 1, device=x.device).unsqueeze(1) // ratio
            index_score = index_score + torch.where(causal, float("-inf"), 0.0)
        k = min(self.index_topk, end_pos // ratio)
        topk_idxs = index_score.topk(k, dim=-1)[1]
        if seqlen > 1:
            # Absolute bounds per row; masked entries become -1.
            bounds = (start_pos + torch.arange(1, seqlen + 1, device=x.device)) // ratio
            invalid = topk_idxs >= bounds.unsqueeze(1)
            topk_idxs = torch.where(invalid, -1, topk_idxs + offset)
        elif start_pos == 0:
            invalid = topk_idxs >= (
                torch.arange(1, seqlen + 1, device=x.device).unsqueeze(1) // ratio
            )
            topk_idxs = torch.where(invalid, -1, topk_idxs + offset)
        else:
            topk_idxs = topk_idxs + offset
        return topk_idxs

    def forward_graph(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        pos: torch.Tensor,
        *,
        slot: int = 0,
        max_entries: int | None = None,
    ) -> torch.Tensor:
        """Capture-safe decode top-k index selection at GPU position ``pos``.

        Equivalent to :meth:`forward` for ``start_pos>0, seqlen=1`` but
        driven by the GPU scalar ``pos``: the compressed-entry count is
        ``(pos+1)//ratio`` and entries beyond it are masked to -inf before
        a FIXED-k topk (``index_topk``) so the graph has no Python-size
        dependence.  Returns [1, seqlen, index_topk] int32 with -1 padding.
        This method advances the indexer's own compressor just like
        :meth:`forward`; the main attention compressor is a separate state
        machine and cannot stand in for this one.
        """
        assert self.kv_cache is not None and self.freqs_cis is not None
        bsz, seqlen, _ = x.shape
        ratio = self.compressor.ratio
        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache
            self.compressor.freqs_cis = self.freqs_cis
        self.compressor.forward_graph(x, pos, slot=slot)

        n_entries = self.kv_cache.shape[1] if max_entries is None else int(max_entries)
        if not 1 <= n_entries <= self.kv_cache.shape[1]:
            raise ValueError(
                "indexer graph max_entries must be within the kv cache capacity, got "
                f"{n_entries} for capacity {self.kv_cache.shape[1]}"
            )
        if n_entries < self.index_topk:
            raise ValueError(
                "indexer graph max_entries must be >= index_topk, got "
                f"{n_entries} < {self.index_topk}"
            )

        # The smallest graph bucket contains exactly index_topk entries.  Its
        # driver is selected only while the live history fits that bucket, so
        # all live entries are selected and scoring cannot change membership.
        # Emit a stable prefix instead of running two Q8 projections, query
        # transforms, a full score scan and topk merely to permute that set.
        if n_entries == self.index_topk:
            indices = torch.arange(self.index_topk, dtype=torch.int32, device=pos.device)
            limit = ((pos + 1) // ratio).reshape(1, 1)
            return torch.where(indices.reshape(1, 1, -1) < limit, indices, -1)

        freqs = self.freqs_cis[pos]
        q = self.wq_b(qr)
        q = q.unflatten(-1, (self.n_heads, self.head_dim))
        apply_rotary_emb(q[..., -self.rope_head_dim :], freqs)
        q = hadamard_transform(q, self.head_dim**-0.5)
        q = fp4_act_quant_simulate(q, 32)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
        index_score = self._score_entries(q, weights, n_entries, slot=slot)
        # mask entries beyond (pos+1)//ratio (not yet written / not causally
        # available) to -inf so the fixed-k topk can never select them
        valid = torch.arange(n_entries, device=pos.device).unsqueeze(0)
        limit = ((pos + 1) // ratio).reshape(1, 1)  # [1,1]
        index_score = index_score + torch.where(valid >= limit, float("-inf"), 0.0)
        k = self.index_topk
        topk_idxs = index_score.topk(k, dim=-1)[1]
        topk_idxs = torch.where(topk_idxs >= limit, -1, topk_idxs).to(torch.int32)
        return topk_idxs

    def forward_graph_prefill(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        pos_tensor: torch.Tensor,
        *,
        slot: int = 0,
        max_entries: int,
        host_start_pos: int | None = None,
        compressor_precomputed: bool = False,
    ) -> torch.Tensor:
        """Capture-safe prefill indexer: ``seqlen`` rows at GPU position.

        Scores EXACTLY the live ``max_entries`` (= n_entries) so the topk tie
        order matches eager, then pads to a 64-entry bucket (the run width).
        Advances the indexer's own compressor via its batched graph prefill.
        """
        assert self.kv_cache is not None and self.freqs_cis is not None
        bsz, seqlen, _ = x.shape
        ratio = self.compressor.ratio
        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache
            self.compressor.freqs_cis = self.freqs_cis
        if not compressor_precomputed:
            self.compressor.forward_graph_prefill(
                x,
                pos_tensor,
                slot=slot,
                host_start_pos=host_start_pos,
            )

        pos_idx = pos_tensor + torch.arange(seqlen, dtype=torch.int64, device=x.device)
        freqs = self.freqs_cis[pos_idx]
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        apply_rotary_emb(q[..., -self.rope_head_dim :], freqs)
        if q.device.type == "cuda":
            from runtime.kernels.dsv4_compressor import hadamard_fp4_query

            q = hadamard_fp4_query(q)
        else:
            q = hadamard_transform(q, self.head_dim**-0.5)
            q = fp4_act_quant_simulate(q, 32)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
        index_score = self._score_entries(q, weights, max_entries, slot=slot)

        bounds = (pos_tensor + torch.arange(1, seqlen + 1, device=x.device)) // ratio
        causal = torch.arange(max_entries, device=x.device).unsqueeze(0).repeat(seqlen, 1)
        index_score = index_score + torch.where(causal >= bounds.unsqueeze(1), float("-inf"), 0.0)
        k = min(self.index_topk, max_entries)
        topk_idxs = index_score.topk(k, dim=-1)[1]
        invalid = topk_idxs >= bounds.unsqueeze(1)
        topk_idxs = torch.where(invalid, -1, topk_idxs).to(torch.int32)
        bucket = min(self.index_topk, ((max_entries + 63) // 64) * 64)
        if k < bucket:
            topk_idxs = torch.nn.functional.pad(topk_idxs, (0, bucket - k), value=-1)
        return topk_idxs

    def forward_graph_batch(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        pos: torch.Tensor,
        slot_ids: torch.Tensor,
        *,
        max_entries: int | None = None,
    ) -> torch.Tensor:
        """Capture-safe top-k selection for heterogeneous B=1/2/4 rows."""
        assert self.kv_cache is not None and self.freqs_cis is not None
        bsz, seqlen, _ = x.shape
        if bsz not in (1, 2, 4) or seqlen != 1:
            raise ValueError(
                "forward_graph_batch requires x shaped [B, 1, hidden] with "
                f"B in (1, 2, 4), got {tuple(x.shape)}"
            )
        if pos.shape != (bsz,) or slot_ids.shape != (bsz,):
            raise ValueError(
                f"pos and slot_ids must both be [{bsz}], got {tuple(pos.shape)} and "
                f"{tuple(slot_ids.shape)}"
            )
        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache
            self.compressor.freqs_cis = self.freqs_cis
        self.compressor.forward_graph_batch(x, pos, slot_ids)

        n_entries = self.kv_cache.shape[1] if max_entries is None else int(max_entries)
        if not self.index_topk <= n_entries <= self.kv_cache.shape[1]:
            raise ValueError(
                "indexer graph max_entries must be between index_topk and capacity, got "
                f"{n_entries} for [{self.index_topk}, {self.kv_cache.shape[1]}]"
            )
        limits = ((pos + 1) // self.compressor.ratio).reshape(bsz, 1, 1)
        if n_entries == self.index_topk:
            indices = torch.arange(self.index_topk, dtype=torch.int32, device=pos.device)
            return torch.where(indices.reshape(1, 1, -1) < limits, indices, -1)

        freqs = self.freqs_cis[pos]
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        apply_rotary_emb(q[..., -self.rope_head_dim :], freqs)
        from runtime.kernels.dsv4_compressor import hadamard_fp4_query
        from runtime.kernels.dsv4_indexer_score import dsv4_indexer_score_batch

        q = hadamard_fp4_query(q)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
        index_score = dsv4_indexer_score_batch(
            q,
            self.kv_cache,
            weights,
            slot_ids,
            n_entries=n_entries,
        )
        valid = torch.arange(n_entries, device=pos.device).reshape(1, 1, -1)
        index_score = index_score + torch.where(valid >= limits, float("-inf"), 0.0)
        topk_idxs = index_score.topk(self.index_topk, dim=-1)[1]
        return torch.where(topk_idxs >= limits, -1, topk_idxs).to(torch.int32)


class Dsv4Attention(nn.Module):
    """MLA-variant attention with window ring + compressed KV regions.

    Reference Attention transcription for the eager graph: q via low-rank
    projections with per-head renorm, single latent KV (rope on the last 64
    dims, fp8 ue8m0 QAT simulation on the nope part), window-ring cache plus
    the compressor/indexer-owned compressed region, sparse gather attention
    with the learned sink, and the grouped low-rank output (derotate, then
    per-group wo_a einsum, then wo_b).

    Cache layout per slot: [window (128) | compressed (max_seq // ratio)]
    entries of head_dim latents. The compressor writes the compressed region
    directly (its kv_cache is a view of ours); the indexer keeps its own
    scoring cache inside itself.
    """

    def __init__(
        self,
        config: Dsv4Config,
        layer_id: int,
        *,
        max_seq_len: int = 4096,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.ratio = config.layer_ratio(layer_id)
        self.window = config.window_size
        self.n_heads = config.num_heads
        self.head_dim = config.head_dim
        self.rope_head_dim = config.rope_head_dim
        self.n_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.eps = config.norm_eps
        self.softmax_scale = self.head_dim**-0.5

        self.wq_a = PackedQ8_0Linear(
            config.q_lora_rank, config.hidden_size, weight_dtype=torch.bfloat16, device=device
        )
        self.register_buffer(
            "q_norm_weight",
            torch.empty(config.q_lora_rank, dtype=torch.float32, device=device),
        )
        self.wq_b = PackedQ8_0Linear(
            self.n_heads * self.head_dim,
            config.q_lora_rank,
            weight_dtype=torch.bfloat16,
            device=device,
        )
        self.wkv = PackedQ8_0Linear(
            self.head_dim, config.hidden_size, weight_dtype=torch.bfloat16, device=device
        )
        self.register_buffer(
            "kv_norm_weight", torch.empty(self.head_dim, dtype=torch.float32, device=device)
        )
        self.wo_a = PackedQ8_0Linear(
            self.n_groups * self.o_lora_rank,
            self.n_heads * self.head_dim // self.n_groups,
            weight_dtype=torch.bfloat16,
            device=device,
        )
        self.wo_b = PackedQ8_0Linear(
            config.hidden_size,
            self.n_groups * self.o_lora_rank,
            weight_dtype=torch.bfloat16,
            device=device,
        )
        self.register_buffer(
            "attn_sink", torch.empty(self.n_heads, dtype=torch.float32, device=device)
        )

        self.compressor = Dsv4Compressor(config, layer_id, device=device) if self.ratio else None
        self.indexer = (
            Dsv4Indexer(config, layer_id, max_seq_len=max_seq_len, device=device)
            if self.ratio == 4
            else None
        )

        n_compressed = max_seq_len // self.ratio if self.ratio else 0
        self.register_buffer(
            "kv_cache",
            torch.zeros(
                1, self.window + n_compressed, self.head_dim, dtype=torch.bfloat16, device=device
            ),
        )
        if self.ratio:
            freqs = precompute_freqs_cis(
                self.rope_head_dim,
                max_seq_len,
                original_seq_len=config.rope_original_seq_len,
                base=config.compress_rope_theta,
                factor=config.rope_factor,
                beta_fast=config.beta_fast,
                beta_slow=config.beta_slow,
            )
        else:
            # window-only layers: base theta, no YaRN (verified reference behavior)
            freqs = precompute_freqs_cis(
                self.rope_head_dim,
                max_seq_len,
                original_seq_len=0,
                base=config.rope_theta,
                factor=config.rope_factor,
                beta_fast=config.beta_fast,
                beta_slow=config.beta_slow,
            )
        if device is not None:
            freqs = freqs.to(device)
        self.register_buffer("freqs_cis", freqs)

    def _wire_subcaches(self) -> None:
        if self.compressor is not None and self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache[:, self.window :]
            self.compressor.freqs_cis = self.freqs_cis
            if self.indexer is not None:
                self.indexer.freqs_cis = self.freqs_cis

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        self._wire_subcaches()
        bsz, seqlen, _ = x.shape
        win, ratio, rd = self.window, self.ratio, self.rope_head_dim
        freqs = self.freqs_cis[start_pos : start_pos + seqlen]

        # q path
        qr = rms_norm(self.wq_a(x), self.q_norm_weight, self.eps)
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        apply_rotary_emb(q[..., -rd:], freqs)
        # The official attention kernel's q input is BF16 (kernel.py's
        # sparse_attn_kernel signature); the eager path casts to match, so
        # the kernel path (bf16 q by contract) and the eager path compare
        # like for like.
        q = q.to(torch.bfloat16)

        # kv path
        kv = self.kv_norm(self.wkv(x))
        apply_rotary_emb(kv[..., -rd:], freqs)
        kv[..., :-rd] = act_quant_simulate(kv[..., :-rd], 64, ue8m0=True)

        device = x.device
        topk_idxs = window_topk_idxs(win, bsz, seqlen, start_pos, device)
        if ratio:
            offset = kv.size(1) if start_pos == 0 else win
            if self.indexer is not None:
                compress_idxs = self.indexer(x, qr, start_pos, offset).int()
            else:
                compress_idxs = compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset, device)
            topk_idxs = torch.cat([topk_idxs, compress_idxs], dim=-1)

        if start_pos == 0:
            if seqlen <= win:
                self.kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                tail = kv[:, -win:]
                self.kv_cache[:bsz, cutoff:win] = tail[:, : win - cutoff]
                self.kv_cache[:bsz, :cutoff] = tail[:, win - cutoff :]
            attn_kv = kv
            if ratio:
                compressed = self.compressor(x, 0)
                if compressed is not None:
                    attn_kv = torch.cat([kv, compressed], dim=1)
            o = sparse_attention_eager(q, attn_kv, self.attn_sink, topk_idxs, self.softmax_scale)
        else:
            self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
            if ratio:
                self.compressor(x, start_pos)
            o = sparse_attention_eager(
                q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale
            )

        # output path: derotate rope part, grouped low-rank projection
        apply_rotary_emb(o[..., -rd:], freqs, inverse=True)
        o = o.view(bsz, seqlen, self.n_groups, -1)
        wo_a = self.wo_a.dequantized().to(x.dtype).view(self.n_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        return self.wo_b(o.flatten(2))

    def kv_norm(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.kv_norm_weight, self.eps)


# ---------------------------------------------------------------------------
# Hyper-Connections (reference Block, model.py). hc_pre reduces hc_mult
# residual copies to one via Sinkhorn-projected mixing weights; hc_post
# expands back. The Sinkhorn loop ENDS on a column normalization on purpose
# (row sums drift; the reference relies on it) -- verified against the
# tilelang kernel in tests/test_dsv4_reference_parts.py.
# ---------------------------------------------------------------------------


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """mixes layout: [pre(hc) | post(hc) | comb(hc*hc)], per token."""
    hc = hc_mult
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (mixes[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).reshape(
        *mixes.shape[:-1], hc, hc
    )
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


class Dsv4Embedding(nn.Module):
    """Packed Q8_0 token embedding with dequantizing row lookup.

    Rows are exactly ``hidden_size // 32`` Q8_0 blocks (the contiguous
    dimension), so a lookup dequantizes only the requested rows. Output is
    bf16 -- the reference stream dtype.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self._row_bytes = hidden_size // 32 * 34
        self.register_buffer(
            "packed", torch.empty(vocab_size * self._row_bytes, dtype=torch.uint8, device=device)
        )

    def load_packed(self, tensor: GgufTensor) -> None:
        if tensor.type_name != "Q8_0":
            raise ValueError(f"{tensor.name}: expected Q8_0, got {tensor.type_name}")
        if tuple(tensor.shape) != (self.vocab_size, self.hidden_size):
            raise ValueError(
                f"{tensor.name}: expected {(self.vocab_size, self.hidden_size)}, got {tensor.shape}"
            )
        self.packed.copy_(tensor.data.to(self.packed.device))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        flat = input_ids.reshape(-1)
        rows = self.packed.view(-1, self._row_bytes)[flat]
        values = dequantize_q8_0(rows.reshape(-1))
        return values.reshape(*input_ids.shape, self.hidden_size).to(torch.bfloat16)


class Dsv4Block(nn.Module):
    """Transformer block with Hyper-Connections mixing (reference Block).

    hc_pre: hc_mult copies -> 1 via Sinkhorn-weighted sum; hc_post: expand
    1 -> hc_mult copies via post weights + combination matrix. attn/ffn run
    on the single reduced stream, bf16 like the reference.
    """

    def __init__(
        self,
        config: Dsv4Config,
        layer_id: int,
        *,
        max_seq_len: int = 4096,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.eps = config.norm_eps
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.hc_iters = config.hc_sinkhorn_iters
        self.attn = Dsv4Attention(config, layer_id, max_seq_len=max_seq_len, device=device)
        self.moe = Dsv4MoE(config, hashed=layer_id in config.hash_layer_ids, device=device)
        self.register_buffer(
            "attn_norm_weight",
            torch.empty(config.hidden_size, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "ffn_norm_weight",
            torch.empty(config.hidden_size, dtype=torch.float32, device=device),
        )
        self.hc_attn_fn = PackedQ8_0Weight(config.hc_mix_dim, config.hc_dim, device=device)
        self.hc_ffn_fn = PackedQ8_0Weight(config.hc_mix_dim, config.hc_dim, device=device)
        self.register_buffer(
            "hc_attn_base", torch.empty(config.hc_mix_dim, dtype=torch.float32, device=device)
        )
        self.register_buffer(
            "hc_ffn_base", torch.empty(config.hc_mix_dim, dtype=torch.float32, device=device)
        )
        self.register_buffer("hc_attn_scale", torch.empty(3, dtype=torch.float32, device=device))
        self.register_buffer("hc_ffn_scale", torch.empty(3, dtype=torch.float32, device=device))

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: PackedQ8_0Weight,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [b,s,hc,d] -> reduced [b,s,d] plus the post/comb weights.
        shape, dtype = x.shape, x.dtype
        bsz, seq, hc, d = shape
        if x.is_cuda and x.dtype == torch.bfloat16:
            from runtime.kernels.dsv4_mhc import hc_fused_pre

            y, post, comb = hc_fused_pre(
                x.reshape(bsz * seq, hc, d),
                hc_fn.packed,
                hc_scale,
                hc_base,
                hc_mult=self.hc_mult,
                sinkhorn_iters=self.hc_iters,
                eps=self.hc_eps,
            )
            return (
                y.reshape(bsz, seq, d).to(dtype),
                post.reshape(bsz, seq, hc),
                comb.reshape(bsz, seq, hc, hc),
            )
        xf = x.flatten(2)
        rsqrt = torch.rsqrt(xf.float().square().mean(-1, keepdim=True) + self.eps)
        mixes = F.linear(xf.float(), hc_fn.dequantized())
        mixes = mixes * rsqrt
        pre, post, comb = hc_split_sinkhorn(
            mixes, hc_scale, hc_base, self.hc_mult, self.hc_iters, self.hc_eps
        )
        y = (pre.unsqueeze(-1) * x.view(shape)).sum(dim=2)
        return y.to(dtype), post, comb

    @staticmethod
    def hc_post(
        x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor
    ) -> torch.Tensor:
        # x: [b,s,d], residual: [b,s,hc,d] -> [b,s,hc,d]
        if (
            x.is_cuda
            and x.dtype == torch.bfloat16
            and x.shape[-1] == 4096
            and residual.shape[-2] == 4
        ):
            from runtime.kernels.dsv4_mhc import hc_fused_post

            shape = residual.shape
            return hc_fused_post(
                x.reshape(-1, shape[-1]),
                residual.reshape(-1, shape[-2], shape[-1]),
                post.reshape(-1, shape[-2]),
                comb.reshape(-1, shape[-2], shape[-2]),
            ).reshape(shape)
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + (
            comb.unsqueeze(-1) * residual.unsqueeze(-2)
        ).sum(dim=2)
        return y.to(x.dtype)

    def forward(self, x: torch.Tensor, start_pos: int, input_ids: torch.Tensor) -> torch.Tensor:
        residual = x
        x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = rms_norm(x, self.attn_norm_weight, self.eps)
        x = self.attn(x, start_pos)
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = rms_norm(x, self.ffn_norm_weight, self.eps)
        x = self.moe(x, input_ids)
        x = self.hc_post(x, residual, post, comb)
        return x


class Dsv4Transformer(nn.Module):
    """Full eager graph: embed -> HC expand -> 43 blocks -> hc_head -> logits.

    No DSpark/MTP stage by design (the quant-mix GGUF carries no mtp.*
    tensors; DSpark support, if ever attempted, is a later self-quant
    experiment -- plan D9). forward returns fp32 logits [b, s, vocab];
    sampling is the server's job.
    """

    def __init__(
        self,
        config: Dsv4Config,
        *,
        max_seq_len: int = 4096,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.eps = config.norm_eps
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.embed = Dsv4Embedding(config.vocab_size, config.hidden_size, device=device)
        self.blocks = nn.ModuleList(
            Dsv4Block(config, layer_id, max_seq_len=max_seq_len, device=device)
            for layer_id in range(config.num_layers)
        )
        self.register_buffer(
            "norm_weight", torch.empty(config.hidden_size, dtype=torch.float32, device=device)
        )
        self.hc_head_fn = PackedQ8_0Weight(config.hc_mult, config.hc_dim, device=device)
        self.register_buffer(
            "hc_head_base", torch.empty(config.hc_mult, dtype=torch.float32, device=device)
        )
        self.register_buffer("hc_head_scale", torch.empty(1, dtype=torch.float32, device=device))
        self.lm_head = PackedQ8_0Linear(config.vocab_size, config.hidden_size, device=device)

    def hc_head(self, x: torch.Tensor) -> torch.Tensor:
        # x: [b,s,hc,d] -> [b,s,d]; sigmoid weights, no Sinkhorn (reference).
        shape, dtype = x.shape, x.dtype
        xf = x.flatten(2).float()
        rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        mixes = F.linear(xf, self.hc_head_fn.dequantized()) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.hc_eps
        return (pre.unsqueeze(-1) * x.view(shape)).sum(dim=2).to(dtype)

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        h = self.embed(input_ids)
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        for block in self.blocks:
            h = block(h, start_pos, input_ids)
        h = self.hc_head(h)
        return self.lm_head(rms_norm(h, self.norm_weight, self.eps))

    def reset_caches(self) -> None:
        """Zero the recursive compressor state of every layer.

        KV bytes stay (positions are overwritten, stale bytes are never read
        past the sequence length); the recursive kv_state/score_state must be
        cleared because the first step of the next sequence reads them --
        the same rule as the slot pool's ``reset_slot``.
        """
        for block in self.blocks:
            attn = block.attn
            if attn.compressor is not None:
                attn.compressor.kv_state.zero_()
                attn.compressor.score_state.fill_(float("-inf"))
            if attn.indexer is not None:
                attn.indexer.compressor.kv_state.zero_()
                attn.indexer.compressor.score_state.fill_(float("-inf"))
                attn.indexer.kv_cache.zero_()


# ---------------------------------------------------------------------------
# GGUF loading: every one of the 1328 tensors maps by name to exactly one
# home in the graph, with type and shape asserted at the store site. Zero
# exemptions -- an unknown tensor or a missing expectation is a hard error,
# the same refusal posture as the registry.
# ---------------------------------------------------------------------------

_BLK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")


def expected_gguf_tensor_names(config: Dsv4Config) -> set[str]:
    """The full tensor inventory implied by the config (verified 1:1 against
    the real file header; notes/2026-08-07-dsv4flash-fact-baseline.md §2)."""
    names = {
        "token_embd.weight",
        "output.weight",
        "output_norm.weight",
        "output_hc_fn.weight",
        "output_hc_base.weight",
        "output_hc_scale.weight",
    }
    for layer_id in range(config.num_layers):
        prefix = f"blk.{layer_id}."
        names |= {
            prefix + fixed
            for fixed in (
                "attn_norm.weight",
                "ffn_norm.weight",
                "attn_sinks.weight",
                "attn_q_a.weight",
                "attn_q_a_norm.weight",
                "attn_q_b.weight",
                "attn_kv.weight",
                "attn_kv_a_norm.weight",
                "attn_output_a.weight",
                "attn_output_b.weight",
                "hc_attn_fn.weight",
                "hc_attn_base.weight",
                "hc_attn_scale.weight",
                "hc_ffn_fn.weight",
                "hc_ffn_base.weight",
                "hc_ffn_scale.weight",
                "ffn_gate_inp.weight",
                "ffn_gate_exps.weight",
                "ffn_up_exps.weight",
                "ffn_down_exps.weight",
                "ffn_gate_shexp.weight",
                "ffn_up_shexp.weight",
                "ffn_down_shexp.weight",
            )
        }
        if layer_id in config.hash_layer_ids:
            names.add(prefix + "ffn_gate_tid2eid.weight")
        else:
            names.add(prefix + "exp_probs_b.bias")
        if config.has_compressor(layer_id):
            names |= {
                prefix + part
                for part in (
                    "attn_compressor_kv.weight",
                    "attn_compressor_gate.weight",
                    "attn_compressor_ape.weight",
                    "attn_compressor_norm.weight",
                )
            }
        if config.has_indexer(layer_id):
            names |= {
                prefix + part
                for part in (
                    "indexer.attn_q_b.weight",
                    "indexer.proj.weight",
                    "indexer_compressor_kv.weight",
                    "indexer_compressor_gate.weight",
                    "indexer_compressor_ape.weight",
                    "indexer_compressor_norm.weight",
                )
            }
    return names


def _f32_into(tensor: GgufTensor, buffer: torch.Tensor) -> None:
    if tensor.type_name != "F32":
        raise ValueError(f"{tensor.name}: expected F32, got {tensor.type_name}")
    if tuple(tensor.shape) != tuple(buffer.shape):
        raise ValueError(f"{tensor.name}: expected {tuple(buffer.shape)}, got {tensor.shape}")
    buffer.copy_(tensor.data.to(buffer.device))


def _i32_into(tensor: GgufTensor, buffer: torch.Tensor) -> None:
    if tensor.type_name != "I32":
        raise ValueError(f"{tensor.name}: expected I32, got {tensor.type_name}")
    if tuple(tensor.shape) != tuple(buffer.shape):
        raise ValueError(f"{tensor.name}: expected {tuple(buffer.shape)}, got {tensor.shape}")
    buffer.copy_(tensor.data.to(buffer.device))


def _q8_0_dequant_into(tensor: GgufTensor, buffer: torch.Tensor) -> None:
    """APE tensors are Q8_0 in the file but fp32 operands in the graph."""
    if tensor.type_name != "Q8_0":
        raise ValueError(f"{tensor.name}: expected Q8_0, got {tensor.type_name}")
    values = dequantize_q8_0(tensor.data)
    if values.numel() != buffer.numel():
        raise ValueError(f"{tensor.name}: expected {buffer.numel()} values, got {values.numel()}")
    buffer.copy_(values.reshape(buffer.shape).to(buffer.device))


def store_gguf_tensor(model: Dsv4Transformer, tensor: GgufTensor) -> None:
    """Route one streamed tensor into the graph; raises on any mismatch."""
    name = tensor.name
    if name == "token_embd.weight":
        return model.embed.load_packed(tensor)
    if name == "output.weight":
        return model.lm_head.load_packed(tensor)
    if name == "output_norm.weight":
        return _f32_into(tensor, model.norm_weight)
    if name == "output_hc_fn.weight":
        return model.hc_head_fn.load_packed(tensor)
    if name == "output_hc_base.weight":
        return _f32_into(tensor, model.hc_head_base)
    if name == "output_hc_scale.weight":
        return _f32_into(tensor, model.hc_head_scale)

    match = _BLK_RE.match(name)
    if match is None:
        raise ValueError(f"unknown GGUF tensor: {name}")
    layer_id, rest = int(match.group(1)), match.group(2)
    if layer_id >= len(model.blocks):
        raise ValueError(f"{name}: layer {layer_id} out of range ({len(model.blocks)})")
    block = model.blocks[layer_id]
    attn, moe = block.attn, block.moe

    simple_f32 = {
        "attn_norm.weight": block.attn_norm_weight,
        "ffn_norm.weight": block.ffn_norm_weight,
        "hc_attn_base.weight": block.hc_attn_base,
        "hc_attn_scale.weight": block.hc_attn_scale,
        "hc_ffn_base.weight": block.hc_ffn_base,
        "hc_ffn_scale.weight": block.hc_ffn_scale,
        "attn_q_a_norm.weight": attn.q_norm_weight,
        "attn_kv_a_norm.weight": attn.kv_norm_weight,
        "attn_sinks.weight": attn.attn_sink,
    }
    if rest in simple_f32:
        return _f32_into(tensor, simple_f32[rest])

    simple_q8 = {
        "hc_attn_fn.weight": block.hc_attn_fn,
        "hc_ffn_fn.weight": block.hc_ffn_fn,
        "attn_q_a.weight": attn.wq_a,
        "attn_q_b.weight": attn.wq_b,
        "attn_kv.weight": attn.wkv,
        "attn_output_a.weight": attn.wo_a,
        "attn_output_b.weight": attn.wo_b,
        "ffn_gate_shexp.weight": moe.shared_w1,
        "ffn_up_shexp.weight": moe.shared_w3,
        "ffn_down_shexp.weight": moe.shared_w2,
    }
    if rest in simple_q8:
        return simple_q8[rest].load_packed(tensor)

    if rest == "ffn_gate_inp.weight":
        return moe.gate.weight.load(tensor)
    if rest == "ffn_gate_exps.weight":
        return moe.gate_exps.load_packed(tensor)
    if rest == "ffn_up_exps.weight":
        return moe.up_exps.load_packed(tensor)
    if rest == "ffn_down_exps.weight":
        return moe.down_exps.load_packed(tensor)
    if rest == "ffn_gate_tid2eid.weight":
        if not moe.gate.hashed:
            raise ValueError(f"{name}: layer {layer_id} is not a hash layer")
        return _i32_into(tensor, moe.gate.tid2eid)
    if rest == "exp_probs_b.bias":
        if moe.gate.hashed:
            raise ValueError(f"{name}: layer {layer_id} is a hash layer")
        return _f32_into(tensor, moe.gate.bias)

    if attn.compressor is not None:
        compressor = attn.compressor
        if rest == "attn_compressor_kv.weight":
            return compressor.wkv.load_packed(tensor)
        if rest == "attn_compressor_gate.weight":
            return compressor.wgate.load_packed(tensor)
        if rest == "attn_compressor_ape.weight":
            return _q8_0_dequant_into(tensor, compressor.ape)
        if rest == "attn_compressor_norm.weight":
            return _f32_into(tensor, compressor.norm_weight)

    if attn.indexer is not None:
        indexer = attn.indexer
        if rest == "indexer.attn_q_b.weight":
            return indexer.wq_b.load_packed(tensor)
        if rest == "indexer.proj.weight":
            return indexer.weights_proj.load_packed(tensor)
        icomp = indexer.compressor
        if rest == "indexer_compressor_kv.weight":
            return icomp.wkv.load_packed(tensor)
        if rest == "indexer_compressor_gate.weight":
            return icomp.wgate.load_packed(tensor)
        if rest == "indexer_compressor_ape.weight":
            return _q8_0_dequant_into(tensor, icomp.ape)
        if rest == "indexer_compressor_norm.weight":
            return _f32_into(tensor, icomp.norm_weight)

    raise ValueError(f"unknown GGUF tensor: {name}")


def load_weights(model: Dsv4Transformer, path: Path | str, *, device: str = "cuda") -> int:
    """Stream the whole file into the graph; assert zero-exemption coverage."""
    expected = expected_gguf_tensor_names(model.config)
    seen: set[str] = set()
    for tensor in iterate_gguf_checkpoint(Path(path), device=device):
        if tensor.name in seen:
            raise ValueError(f"duplicate tensor in file: {tensor.name}")
        seen.add(tensor.name)
        if tensor.name not in expected:
            raise ValueError(f"GGUF tensor not expected by this config: {tensor.name}")
        store_gguf_tensor(model, tensor)
    missing = expected - seen
    if missing:
        raise ValueError(
            f"{len(missing)} expected tensors missing from the file, e.g. {sorted(missing)[:5]}"
        )
    return len(seen)


def load_dsv4_from_gguf(
    path: Path | str, *, max_seq_len: int = 4096, device: str = "cuda"
) -> tuple[Dsv4Transformer, int]:
    """Build the graph straight onto `device` and stream every tensor in.

    The graph is far larger than host RAM, so construction must never route
    through CPU: every buffer is allocated on the target device from the
    start and the file is streamed tensor-by-tensor into it.
    """
    header = read_gguf_header(Path(path))
    config = config_from_gguf_kv(header.kv)
    model = Dsv4Transformer(config, max_seq_len=max_seq_len, device=device)
    count = load_weights(model, path, device=device)
    return model, count
