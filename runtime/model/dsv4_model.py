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
        numel = out_features * in_features
        n_bytes = numel // 32 * 34
        self.register_buffer("packed", torch.empty(n_bytes, dtype=torch.uint8, device=device))

    def load_packed(self, tensor: GgufTensor) -> None:
        if tensor.type_name != "Q8_0":
            raise ValueError(f"{tensor.name}: expected Q8_0, got {tensor.type_name}")
        if tuple(tensor.shape) != (self.out_features, self.in_features):
            raise ValueError(
                f"{tensor.name}: expected {self.out_features}x{self.in_features}, "
                f"got {tensor.shape}"
            )
        self.packed.copy_(tensor.data.to(self.packed.device))

    def dequantized(self) -> torch.Tensor:
        return dequantize_q8_0(self.packed).reshape(self.out_features, self.in_features)


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused_q8 and x.device.type == "cuda":
            from runtime.kernels.dsv4_q8_gemm import q8_0_dequant_gemm

            leading = x.shape[:-1]
            out = q8_0_dequant_gemm(
                x.reshape(-1, self.in_features),
                self.packed,
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
        scores = F.softplus(self.weight(x)).sqrt()
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
        return self.shared_w2(swiglu(gate, up, self.config.swiglu_limit).to(x.dtype))

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
                eids = indices[0].tolist()
                xs_batch = flat.expand(len(eids), 1, -1).contiguous()
                wts = w1
                gate = self._batch_expert_gemm(self.gate_exps, eids, xs_batch)
                up = self._batch_expert_gemm(self.up_exps, eids, xs_batch)
                h = swiglu(gate, up, limit)
                routed = self._batch_expert_gemm(self.down_exps, eids, h)
                y = (routed * wts[:, None, None]).sum(dim=0).to(torch.float32)
            else:
                # Prefill (M>1): route tokens to their experts in one pass
                # (GPU sort groups tokens by expert), batched GEMM.
                tok_ids = torch.arange(n_tokens, device=flat.device).repeat_interleave(
                    indices.shape[1]
                )
                exp_ids = indices.reshape(-1)
                order = torch.argsort(exp_ids, stable=True)
                exp_sorted = exp_ids[order]
                tok_sorted = tok_ids[order]
                wt_sorted = weights.reshape(-1)[order]
                exp_uniq, exp_counts = torch.unique(exp_sorted, return_counts=True)
                exp_uniq_l, exp_counts_l = exp_uniq.tolist(), exp_counts.tolist()
                E, M_max = len(exp_uniq_l), int(exp_counts.max().item())
                xs_batch = torch.zeros(
                    E, M_max, flat.shape[-1], dtype=torch.bfloat16, device=flat.device
                )
                wt_batch = torch.zeros(E, M_max, dtype=torch.float32, device=flat.device)
                base = 0
                for e in range(E):
                    n = exp_counts_l[e]
                    xs_batch[e, :n] = flat[tok_sorted[base : base + n]]
                    wt_batch[e, :n] = wt_sorted[base : base + n]
                    base += n
                gate = self._batch_expert_gemm(self.gate_exps, exp_uniq_l, xs_batch)
                up = self._batch_expert_gemm(self.up_exps, exp_uniq_l, xs_batch)
                h = swiglu(gate, up, limit)
                routed = self._batch_expert_gemm(self.down_exps, exp_uniq_l, h)
                routed = routed * wt_batch.unsqueeze(-1)
                base = 0
                for e in range(E):
                    n = exp_counts_l[e]
                    y.index_add_(0, tok_sorted[base : base + n], routed[e, :n].to(torch.float32))
                    base += n
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

    def _batch_expert_gemm(
        self,
        exps: PackedIQ2_XSExperts,
        eids: list[int],
        xs: torch.Tensor,
    ) -> torch.Tensor:
        """Batched ``xs @ W_e^T`` over the routed ``eids`` in one launch."""
        from runtime.kernels.dsv4_iq2xs_gemm import iq2xs_dequant_gemm_batch

        rb = exps._row_bytes * exps.rows
        packed = torch.stack([exps.packed[eid * rb : (eid + 1) * rb] for eid in eids])
        return iq2xs_dequant_gemm_batch(
            xs,
            packed,
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
        if (
            exps.packed.device.type != "cuda"
            or exps.cols % 256
            or xs.device.type != "cuda"
        ):
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
        head_dim: int | None = None,
        rotate: bool = False,
        quantize: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.ratio = config.layer_ratio(layer_id)
        assert self.ratio != 0
        self.overlap = self.ratio == 4
        self.rotate = rotate  # indexer variant: Hadamard + full-dim fp4 simulation
        # False for the kernel-path attention layer: its packed FP8 pages are
        # quantized by the pack kernel (dsv4_kv_pack), so the compressor emits
        # the raw normed/rotated entry and the QAT simulation moves downstream.
        self.quantize = quantize
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
                1, coff * self.ratio, coff * self.head_dim, dtype=torch.float32, device=device
            ),
        )
        self.register_buffer(
            "score_state",
            torch.full(
                (1, coff * self.ratio, coff * self.head_dim),
                float("-inf"),
                dtype=torch.float32,
                device=device,
            ),
        )
        self.freqs_cis: torch.Tensor | None = None
        self.kv_cache: torch.Tensor | None = None  # assigned by the attention owner

    def overlap_transform(self, tensor: torch.Tensor, value: float) -> torch.Tensor:
        """[b, s, r, 2d] -> [b, s, 2r, d]: current window's second half plus
        the previous window's first half shifted down one row."""
        b, s, _, _ = tensor.shape
        ratio, d = self.ratio, self.head_dim
        out = tensor.new_full((b, s, 2 * ratio, d), value)
        out[:, :, ratio:] = tensor[:, :, :, d:]
        out[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return out

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
            freqs = self.freqs_cis[:position_count : self.ratio]
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

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor | None:
        assert self.kv_cache is not None and self.freqs_cis is not None
        bsz, seqlen, _ = x.shape
        ratio, overlap = self.ratio, self.overlap
        dtype = x.dtype
        xf = x.float()
        kv = self.wkv(xf)
        score = self.wgate(xf)
        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            offset = ratio if overlap else 0
            if overlap and cutoff >= ratio:
                self.kv_state[:bsz, :ratio] = kv[:, cutoff - ratio : cutoff]
                self.score_state[:bsz, :ratio] = score[:, cutoff - ratio : cutoff] + self.ape
            if remainder > 0:
                kv_head, kv_tail = kv.split([cutoff, remainder], dim=1)
                self.kv_state[:bsz, offset : offset + remainder] = kv_tail
                self.score_state[:bsz, offset : offset + remainder] = (
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
            self.kv_cache[:bsz, : seqlen // ratio] = kv
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
                self.kv_state[:bsz, ratio + pos % ratio] = kv_i.squeeze(1)
                self.score_state[:bsz, ratio + pos % ratio] = score_i.squeeze(1)
                if should_compress:
                    kv_state = torch.cat(
                        [
                            self.kv_state[:bsz, :ratio, : self.head_dim],
                            self.kv_state[:bsz, ratio:, self.head_dim :],
                        ],
                        dim=1,
                    )
                    score_state = torch.cat(
                        [
                            self.score_state[:bsz, :ratio, : self.head_dim],
                            self.score_state[:bsz, ratio:, self.head_dim :],
                        ],
                        dim=1,
                    )
                    kv_c = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
                    self.kv_state[:bsz, :ratio] = self.kv_state[:bsz, ratio:]
                    self.score_state[:bsz, :ratio] = self.score_state[:bsz, ratio:]
            else:
                self.kv_state[:bsz, pos % ratio] = kv_i.squeeze(1)
                self.score_state[:bsz, pos % ratio] = score_i.squeeze(1)
                if should_compress:
                    kv_c = (
                        self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)
                    ).sum(dim=1, keepdim=True)
            if not should_compress:
                continue
            kv_c = self._finalize(kv_c, 0, dtype, pos)
            self.kv_cache[:bsz, pos // ratio] = kv_c.squeeze(1)
            emitted.append(kv_c)
        if not emitted:
            return None
        return torch.cat(emitted, dim=1)


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
        max_seq_len: int = 4096,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        assert config.has_indexer(layer_id)
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
            config, layer_id, head_dim=self.head_dim, rotate=True, device=device
        )
        # indexer owns its scoring cache (reference layout); its compressor
        # writes into it via the wiring in forward().
        self.register_buffer(
            "kv_cache",
            torch.zeros(
                1,
                max_seq_len // config.layer_ratio(layer_id),
                self.head_dim,
                dtype=torch.bfloat16,
                device=device,
            ),
        )
        self.freqs_cis: torch.Tensor | None = None

    def forward(
        self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, offset: int
    ) -> torch.Tensor:
        assert self.kv_cache is not None and self.freqs_cis is not None
        bsz, seqlen, _ = x.shape
        ratio = self.compressor.ratio
        end_pos = start_pos + seqlen
        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache
            self.compressor.freqs_cis = self.freqs_cis
        freqs = self.freqs_cis[start_pos:end_pos]
        q = self.wq_b(qr)
        q = q.unflatten(-1, (self.n_heads, self.head_dim))
        apply_rotary_emb(q[..., -self.rope_head_dim :], freqs)
        q = hadamard_transform(q, self.head_dim**-0.5)
        q = fp4_act_quant_simulate(q, 32)
        self.compressor(x, start_pos)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
        index_score = torch.einsum("bshd,btd->bsht", q, self.kv_cache[:bsz, : end_pos // ratio])
        index_score = (index_score.relu() * weights.unsqueeze(-1)).sum(dim=2)
        if seqlen > 1:
            # Mid-sequence prefill chunk: each row attends only the
            # compressed entries up to its own position ((pos+1)//ratio,
            # including its just-written entry -- same as single-token
            # decode).  Mask later entries out BEFORE topk so they can
            # never steal a slot, exactly like the start_pos==0 branch.
            bounds = ((start_pos + torch.arange(1, seqlen + 1, device=x.device)) // ratio)
            causal = (
                torch.arange(end_pos // ratio, device=x.device)
                .unsqueeze(0)
                .repeat(seqlen, 1)
            )
            index_score = index_score + torch.where(
                causal >= bounds.unsqueeze(1),
                torch.tensor(float("-inf"), device=x.device),
                0.0,
            )
        elif start_pos == 0:
            causal = torch.arange(seqlen // ratio, device=x.device).repeat(seqlen, 1)
            causal = causal >= torch.arange(1, seqlen + 1, device=x.device).unsqueeze(1) // ratio
            index_score = index_score + torch.where(
                causal, torch.tensor(float("-inf"), device=x.device), 0.0
            )
        k = min(self.index_topk, end_pos // ratio)
        topk_idxs = index_score.topk(k, dim=-1)[1]
        if seqlen > 1:
            # Absolute bounds per row; masked entries become -1.
            bounds = ((start_pos + torch.arange(1, seqlen + 1, device=x.device)) // ratio)
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
