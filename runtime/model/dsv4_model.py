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
    """Linear over a packed Q8_0 weight; dequantized per forward (eager).

    ``weight_dtype=None`` (default): fp32 dequant, fp32 compute -- the eager
    accuracy contract. ``weight_dtype=torch.bfloat16``: the dequantized values
    round to bf16 first and the matmul runs bf16 -- the regime the reference
    uses for bf16-declared linears (its ``linear()`` is strict on dtypes),
    and what these layers will run at in production.
    """

    def __init__(
        self, out_features: int, in_features: int, *, weight_dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.weight_dtype = weight_dtype
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
        weight = self.dequantized()
        if self.weight_dtype is not None:
            return F.linear(x, weight.to(self.weight_dtype))
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
    ) -> None:
        super().__init__()
        self.ratio = config.layer_ratio(layer_id)
        assert self.ratio != 0
        self.overlap = self.ratio == 4
        self.rotate = rotate  # indexer variant: Hadamard + full-dim fp4 simulation
        self.head_dim = head_dim if head_dim is not None else config.head_dim
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


class Dsv4Indexer(nn.Module):
    """Selects top-k compressed-KV positions for CSA layers (ratio 4).

    Reference Indexer transcription: low-rank query from the main q latent,
    rope + Hadamard + fp4 QAT simulation on the query; its own rotating
    compressor builds the compressed K cache; scores = relu(q·k) weighted by
    weights_proj, summed over heads, top-k. Q8_0 file weights round to bf16
    for the query/weight projections (the reference regime for bf16-declared
    linears); the score accumulation after the einsum is fp32.
    """

    def __init__(self, config: Dsv4Config, layer_id: int, *, max_seq_len: int = 4096) -> None:
        super().__init__()
        assert config.has_indexer(layer_id)
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.rope_head_dim
        self.index_topk = config.index_topk
        self.softmax_scale = self.head_dim**-0.5
        self.wq_b = PackedQ8_0Linear(
            self.n_heads * self.head_dim, config.q_lora_rank, weight_dtype=torch.bfloat16
        )
        self.weights_proj = PackedQ8_0Linear(
            self.n_heads, config.hidden_size, weight_dtype=torch.bfloat16
        )
        self.compressor = Dsv4Compressor(config, layer_id, head_dim=self.head_dim, rotate=True)
        # indexer owns its scoring cache (reference layout); its compressor
        # writes into it via the wiring in forward().
        self.register_buffer(
            "kv_cache",
            torch.zeros(
                1,
                max_seq_len // config.layer_ratio(layer_id),
                self.head_dim,
                dtype=torch.bfloat16,
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
        if start_pos == 0:
            causal = torch.arange(seqlen // ratio, device=x.device).repeat(seqlen, 1)
            causal = causal >= torch.arange(1, seqlen + 1, device=x.device).unsqueeze(1) // ratio
            index_score = index_score + torch.where(
                causal, torch.tensor(float("-inf"), device=x.device), 0.0
            )
        k = min(self.index_topk, end_pos // ratio)
        topk_idxs = index_score.topk(k, dim=-1)[1]
        if start_pos == 0:
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

    def __init__(self, config: Dsv4Config, layer_id: int, *, max_seq_len: int = 4096) -> None:
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
            config.q_lora_rank, config.hidden_size, weight_dtype=torch.bfloat16
        )
        self.register_buffer("q_norm_weight", torch.empty(config.q_lora_rank, dtype=torch.float32))
        self.wq_b = PackedQ8_0Linear(
            self.n_heads * self.head_dim, config.q_lora_rank, weight_dtype=torch.bfloat16
        )
        self.wkv = PackedQ8_0Linear(self.head_dim, config.hidden_size, weight_dtype=torch.bfloat16)
        self.register_buffer("kv_norm_weight", torch.empty(self.head_dim, dtype=torch.float32))
        self.wo_a = PackedQ8_0Linear(
            self.n_groups * self.o_lora_rank,
            self.n_heads * self.head_dim // self.n_groups,
            weight_dtype=torch.bfloat16,
        )
        self.wo_b = PackedQ8_0Linear(
            config.hidden_size, self.n_groups * self.o_lora_rank, weight_dtype=torch.bfloat16
        )
        self.register_buffer("attn_sink", torch.empty(self.n_heads, dtype=torch.float32))

        self.compressor = Dsv4Compressor(config, layer_id) if self.ratio else None
        self.indexer = (
            Dsv4Indexer(config, layer_id, max_seq_len=max_seq_len) if self.ratio == 4 else None
        )

        n_compressed = max_seq_len // self.ratio if self.ratio else 0
        self.register_buffer(
            "kv_cache",
            torch.zeros(1, self.window + n_compressed, self.head_dim, dtype=torch.bfloat16),
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
