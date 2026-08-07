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
