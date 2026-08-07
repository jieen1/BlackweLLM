"""DeepSeek-V4-Flash model graph (GGUF IQ2_XS/Q8_0).

Phase 2 status: FFN half (weight containers, RMSNorm, Gate, MoE) with eager
dequant-on-demand numerics; attention half and assembly land in the next
increments. Quantized weights stay packed end-to-end (plan D2): eager paths
dequantize on demand without caching BF16 copies (the Qwen3.6 dequant-cache
memory floor is the standing warning), and Phase 3 replaces the dequant call
sites with fused kernels keeping the same numerics as dsv4_quant.

Weight naming follows the GGUF file (verified 1:1 against llama.cpp's
create_tensor declarations, notes/2026-08-07-dsv4flash-fact-baseline.md §2.1).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from runtime.loading.gguf import GgufTensor
from runtime.model.dsv4_config import Dsv4Config
from runtime.model.dsv4_quant import dequantize_iq2_xs, dequantize_q8_0


class PackedQ8_0Linear(nn.Module):
    """Linear over a packed Q8_0 weight; dequantized per forward (eager)."""

    def __init__(self, out_features: int, in_features: int) -> None:
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        numel = out_features * in_features
        n_bytes = numel // 32 * 34
        self.register_buffer("packed", torch.empty(n_bytes, dtype=torch.uint8))

    def load_packed(self, tensor: GgufTensor) -> None:
        if tuple(tensor.shape) != (self.out_features, self.in_features):
            raise ValueError(
                f"{tensor.name}: expected {self.out_features}x{self.in_features}, "
                f"got {tensor.shape}"
            )
        self.packed.copy_(tensor.data.to(self.packed.device))

    def dequantized(self) -> torch.Tensor:
        return dequantize_q8_0(self.packed).reshape(self.out_features, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # fp32 compute is the eager contract (accuracy scaffold; kernels later
        # keep fp32 accumulation).
        weight = self.dequantized()
        return F.linear(x.float(), weight)


class PackedIQ2_XSExperts(nn.Module):
    """Fused routed-expert weights [E, rows, cols] packed IQ2_XS.

    Eager path dequantizes only the experts a batch actually routes to --
    dequantizing all 256 experts of one matrix would transiently need ~4.3 GB
    in fp32, more than the card's headroom.
    """

    def __init__(self, num_experts: int, rows: int, cols: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.rows = rows
        self.cols = cols
        row_blocks = cols // 256
        self._row_bytes = row_blocks * 74
        total = num_experts * rows * self._row_bytes
        self.register_buffer("packed", torch.empty(total, dtype=torch.uint8))

    def load_packed(self, tensor: GgufTensor) -> None:
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

    def __init__(self, config: Dsv4Config, *, hashed: bool) -> None:
        super().__init__()
        self.top_k = config.n_activated_experts
        self.route_scale = config.route_scale
        self.hashed = hashed
        self.weight = PackedQ8_0Linear(config.n_routed_experts, config.hidden_size)
        if hashed:
            self.register_buffer(
                "tid2eid", torch.empty(config.vocab_size, self.top_k, dtype=torch.int32)
            )
        else:
            self.register_buffer("bias", torch.empty(config.n_routed_experts, dtype=torch.float32))

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

    def __init__(self, config: Dsv4Config, *, hashed: bool) -> None:
        super().__init__()
        self.config = config
        self.gate = Dsv4Gate(config, hashed=hashed)
        inter = config.moe_intermediate_size
        hidden = config.hidden_size
        experts = config.n_routed_experts
        self.gate_exps = PackedIQ2_XSExperts(experts, inter, hidden)
        self.up_exps = PackedIQ2_XSExperts(experts, inter, hidden)
        self.down_exps = PackedIQ2_XSExperts(experts, hidden, inter)
        self.shared_w1 = PackedQ8_0Linear(inter, hidden)
        self.shared_w3 = PackedQ8_0Linear(inter, hidden)
        self.shared_w2 = PackedQ8_0Linear(hidden, inter)

    def _shared_forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.shared_w1(x)
        up = self.shared_w3(x)
        return self.shared_w2(swiglu(gate, up, self.config.swiglu_limit).to(x.dtype))

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor | None) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1])
        weights, indices = self.gate(flat, input_ids.reshape(-1) if input_ids is not None else None)
        y = torch.zeros_like(flat, dtype=torch.float32)
        limit = self.config.swiglu_limit
        counts = torch.bincount(indices.reshape(-1), minlength=self.config.n_routed_experts)
        for expert_id in range(self.config.n_routed_experts):
            if int(counts[expert_id]) == 0:
                continue
            token_idx, top_slot = torch.where(indices == expert_id)
            w1 = self.gate_exps.expert_weight(expert_id)
            w3 = self.up_exps.expert_weight(expert_id)
            w2 = self.down_exps.expert_weight(expert_id)
            xs = flat[token_idx].float()
            gate = xs @ w1.t()
            up = xs @ w3.t()
            h = swiglu(gate, up, limit)
            routed = (h @ w2.t()) * weights[token_idx, top_slot].unsqueeze(-1)
            y.index_add_(0, token_idx, routed)
        y = y + self._shared_forward(flat)
        return y.to(x.dtype).reshape(*x.shape[:-1], y.shape[-1])


# ---------------------------------------------------------------------------
# Compressor: gated pooling of every `ratio` tokens into one compressed KV
# entry (reference Compressor, model.py). Eager torch transcription; the
# QAT-simulation quant step uses dsv4_attention.act_quant_simulate.
# ---------------------------------------------------------------------------

from runtime.model.dsv4_attention import (  # noqa: E402
    act_quant_simulate,
    apply_rotary_emb,
)


class Dsv4Compressor(nn.Module):
    """Compresses the latent KV stream ratio:1 with learned gated pooling.

    ratio-4 layers use overlap (coff=2): the first half of the wide rows is
    the previous window's carry-over, materialized by overlap_transform.
    Decode keeps kv_state/score_state per slot position within the current
    (overlapping) window and emits one compressed entry every `ratio` steps.
    """

    def __init__(self, config: Dsv4Config, layer_id: int) -> None:
        super().__init__()
        self.ratio = config.layer_ratio(layer_id)
        assert self.ratio != 0
        self.overlap = self.ratio == 4
        self.head_dim = config.head_dim
        self.rope_head_dim = config.rope_head_dim
        self.eps = config.norm_eps
        self.ue8m0 = True  # production QAT config (fp8-origin weights)
        coff = 2 if self.overlap else 1
        self.coeff = coff
        self.wkv = PackedQ8_0Linear(coff * self.head_dim, config.hidden_size)
        self.wgate = PackedQ8_0Linear(coff * self.head_dim, config.hidden_size)
        self.register_buffer(
            "ape", torch.empty(self.ratio, coff * self.head_dim, dtype=torch.float32)
        )
        self.register_buffer("norm_weight", torch.empty(self.head_dim, dtype=torch.float32))
        # per-slot decode state; batch dim sized at capture/slot time (eager: 1)
        self.register_buffer(
            "kv_state", torch.zeros(1, coff * self.ratio, coff * self.head_dim, dtype=torch.float32)
        )
        self.register_buffer(
            "score_state",
            torch.full(
                (1, coff * self.ratio, coff * self.head_dim), float("-inf"), dtype=torch.float32
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
        self, kv: torch.Tensor, position_count: int, dtype: torch.dtype, start_pos: int
    ) -> torch.Tensor:
        """Norm + rope + QAT simulation; writes into the owner's kv_cache."""
        kv = rms_norm(kv.to(dtype), self.norm_weight, self.eps)
        if start_pos == 0:
            freqs = self.freqs_cis[: position_count : self.ratio]
        else:
            freqs = self.freqs_cis[start_pos + 1 - self.ratio].unsqueeze(0)
        apply_rotary_emb(kv[..., -self.rope_head_dim :], freqs)
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
        # decode step
        should_compress = (start_pos + 1) % ratio == 0
        score = score + self.ape[start_pos % ratio]
        if overlap:
            self.kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1)
            self.score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)
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
                kv = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
                self.kv_state[:bsz, :ratio] = self.kv_state[:bsz, ratio:]
                self.score_state[:bsz, :ratio] = self.score_state[:bsz, ratio:]
        else:
            self.kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
            self.score_state[:bsz, start_pos % ratio] = score.squeeze(1)
            if should_compress:
                kv = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(
                    dim=1, keepdim=True
                )
        if not should_compress:
            return None
        kv = self._finalize(kv, 0, dtype, start_pos)
        self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
        return kv
