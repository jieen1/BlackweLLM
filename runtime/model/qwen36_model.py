"""Self-built Qwen3.6-27B (``Qwen3_5ForConditionalGeneration``, text-only)
model graph -- Track B / B1 (``docs/implementation-plan.md`` §7.1,
``docs/qwen36-rebuild-spec.md``).

**Scope, stated once here rather than per class** (matches the B1 mandate
exactly: "eager, batch=1, no CUDA graph, no speculative decoding, no
prefix cache -- only ship it correct"):

- Every forward path below assumes ``batch_size == 1`` and asserts it where
  it matters. No slot manager, no continuous batching, no ``BlockPool``/
  ``RecurrentStatePool`` integration -- :class:`Qwen36GenerationState` is a
  plain per-sequence container a caller owns directly, not a resource this
  module allocates from a pool. Wiring this into ``ModelBackend``
  (``runtime/backends/protocol.py``) is B2 scope, not attempted here.
- Quantized Linears (``runtime/model/modelopt_linear.py``) dequantize to
  BF16 once and run BF16xBF16 matmul -- not this checkpoint's intended
  FP8xFP8 / block-scaled-FP4xFP4 GEMM path. See that module's docstring.
- Full-attention layers use sparkinfer's paged-attention kernel
  (B0-3-verified correct for this exact shape:
  ``head_dim=256``/``gqa_group=6``/SM120) with a **BF16** KV cache, not
  FP8 -- this checkpoint declares ``kv_cache_quant_algo: FP8`` but ships
  zero ``k_scale``/``v_scale`` tensors (B0-2), and resolving "what scale
  to use" is explicitly deferred to B3 (``docs/qwen36-rebuild-spec.md``
  §7). BF16 KV sidesteps that open question entirely for B1's
  correctness gate rather than guessing a default.
- GDN layers use ``fla.ops.gated_delta_rule`` directly (B0-4's decision
  ①: verified correct against HF's own torch fallback for these exact
  shapes, cosine >= 0.99998) -- not the torch fallback, and not a
  from-scratch reimplementation.
- No MTP head. B1 does not do speculative decoding; ``mtp.*`` checkpoint
  tensors are recognized and explicitly skipped by the loader (see
  ``load_weights`` below), not silently dropped through a name-mismatch.

Transcribed from ``transformers==5.8.0``'s
``transformers/models/qwen3_5/modeling_qwen3_5.py`` (read-only reference,
never imported into this module -- ``docs/qwen36-rebuild-spec.md`` §1.0
already established that file, not ``oracle/qwen36_vllm/``, is where the
actual model math lives upstream). Two conventions are deliberately
preserved exactly because getting them subtly wrong is the class of bug
B1's layer-by-layer cosine gate exists to catch:

- ``Qwen36RMSNorm`` uses the *zero-centered* ``(1.0 + weight)`` convention
  (``Qwen3_5RMSNorm``, "unlike Llama... Qwen3_5 is (x * w).to(dtype)" --
  its own comment). ``Qwen36RMSNormGated`` (GDN's output norm) uses the
  *plain* ``weight`` convention with no ``+1`` (``Qwen3_5RMSNormGated``) --
  these are NOT the same formula despite both being "RMSNorm" in this
  model, and ``runtime/kernels/fused_rms_norm.py``'s existing
  ``TritonRMSNorm`` (Laguna's convention, plain ``x * w``) must not be
  reused for either one without checking which formula it implements.
- ``attn_output_gate``: ``q_proj`` outputs ``num_heads * head_dim * 2``
  (verified against real safetensors headers, B0-2/B0 fact baseline §4)
  -- the extra half is a per-(head, dim) sigmoid gate applied to the
  attention output, split out of the *same* Linear as the query
  projection, not a separate weight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from sparkinfer.attention import paged
from torch import nn

from runtime.kernels.rope import apply_rotary_embedding_inplace, compute_cos_sin_cache_default
from runtime.loading.modelopt import (
    QUANT_ALGO_FP8,
    QUANT_ALGO_NVFP4,
    QUANT_ALGO_UNQUANTIZED,
    classify_module,
    quantized_layers_map,
)
from runtime.model._weight_loading import default_weight_loader
from runtime.model.modelopt_linear import ModelOptFP8Linear, ModelOptNVFP4Linear
from runtime.model.plain_linear import PlainLinear

#: Checkpoint tensor suffixes this loader deliberately never consumes into
#: a Parameter -- see runtime/model/modelopt_linear.py's module docstring
#: for why ``.input_scale`` specifically is dead weight for a
#: dequantize-to-BF16 B1 implementation.
_IGNORED_WEIGHT_SUFFIXES: tuple[str, ...] = (".input_scale",)


def _make_linear(
    quantized: dict[str, str],
    dotted_name: str,
    in_features: int,
    out_features: int,
) -> nn.Module:
    """Pick the Linear class for ``dotted_name`` from the checkpoint's own
    ``quantized_layers`` declaration -- never hardcoded per-projection, so
    a checkpoint that quantizes something differently fails loud at
    construction time instead of silently loading raw bytes as BF16.
    """
    algo = classify_module(dotted_name, quantized)
    if algo == QUANT_ALGO_FP8:
        return ModelOptFP8Linear(in_features, out_features, bias=False)
    if algo == QUANT_ALGO_NVFP4:
        return ModelOptNVFP4Linear(in_features, out_features, bias=False)
    assert algo == QUANT_ALGO_UNQUANTIZED
    return PlainLinear(in_features, out_features, bias=False)


# ---------------------------------------------------------------------------
# Norms -- see module docstring for why these are two different formulas.
# ---------------------------------------------------------------------------


class Qwen36RMSNorm(nn.Module):
    """Zero-centered RMSNorm: ``out = norm(x) * (1.0 + weight)``.

    Transcribed from ``Qwen3_5RMSNorm``. Checkpoint stores ``weight``
    already offset by ``-1.0`` from a plain-RMSNorm weight (i.e. an
    all-zeros checkpoint tensor is a no-op norm, matching Gemma's
    convention) -- do not "simplify" this to plain ``x * weight``.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))
        self.weight.weight_loader = default_weight_loader

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        out = x_f32 * torch.rsqrt(variance + self.eps)
        out = out * (1.0 + self.weight.float())
        return out.to(input_dtype)


class Qwen36RMSNormGated(nn.Module):
    """GDN's output norm+gate: ``out = (norm(x) * weight) * silu(gate)``.

    Transcribed from ``Qwen3_5RMSNormGated`` (the torch fallback HF itself
    falls back to when ``fla.modules.FusedRMSNormGated`` is unavailable --
    mathematically the same class, not a different one, per that module's
    own conditional construction). Plain ``weight`` multiply, NOT
    ``(1.0 + weight)`` -- do not conflate with :class:`Qwen36RMSNorm`
    above. Order matters: the weight multiply happens in ``input_dtype``
    (cast back down before multiplying), the gate multiply happens against
    ``silu`` computed in FP32 -- both preserved exactly as HF has them,
    not simplified to "cast once at the end".
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.weight.weight_loader = default_weight_loader

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = self.weight * x.to(input_dtype)
        x = x * F.silu(gate.to(torch.float32))
        return x.to(input_dtype)


# ---------------------------------------------------------------------------
# Per-sequence generation state -- plain containers, no pooling/eviction.
# ---------------------------------------------------------------------------


@dataclass
class GdnLayerState:
    """One GDN layer's recurrent state for one sequence.

    ``conv_state``: ``[1, conv_dim, kernel_size - 1]``. ``recurrent_state``:
    ``[1, num_v_heads, head_k_dim, head_v_dim]``. Both persisted in BF16
    between steps (matching ``transformers/cache_utils.py``'s
    ``LinearAttentionLayer``: FLA's kernels compute each step in FP32
    internally regardless of this buffer's dtype -- B0-4/B0-7 -- and this
    class reproduces the "round to BF16 between steps" half of that
    mechanism explicitly, in :meth:`Qwen36GatedDeltaNet.forward`, not
    silently by relying on dtype coercion).
    """

    conv_state: torch.Tensor
    recurrent_state: torch.Tensor
    has_previous_state: bool = False


class Qwen36PagedAttentionCache:
    """Single-page (batch=1), BF16 KV cache for one full-attention layer,
    read/written through sparkinfer's paged-attention kernel.

    One page sized to hold the entire sequence -- deliberately avoids
    page-table bookkeeping complexity that has no payoff at batch=1 (that
    complexity belongs to B2's real slot manager, not this correctness
    check). ``max_seq_len`` must be decided upfront (caller's
    responsibility); this is a hard cap, not a growable buffer.
    """

    def __init__(
        self,
        *,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        kv_shape = (1, max_seq_len, num_kv_heads, head_dim)
        self.k_cache = torch.zeros(kv_shape, dtype=dtype, device=device)
        self.v_cache = torch.zeros(kv_shape, dtype=dtype, device=device)
        self.page_table = torch.zeros(1, 1, dtype=torch.int32, device=device)
        self.seq_len = 0  # tokens currently resident

    def append(self, new_k: torch.Tensor, new_v: torch.Tensor) -> tuple[int, int]:
        """Write ``new_k``/``new_v`` (``[seq_len, num_kv_heads, head_dim]``)
        at the tail. Returns ``(past_len, new_total_len)``."""
        seq_len = new_k.shape[0]
        past_len = self.seq_len
        new_total = past_len + seq_len
        if new_total > self.max_seq_len:
            raise RuntimeError(
                f"Qwen36PagedAttentionCache overflow: max_seq_len={self.max_seq_len}, "
                f"attempted to reach {new_total}"
            )
        self.k_cache[0, past_len:new_total] = new_k
        self.v_cache[0, past_len:new_total] = new_v
        self.seq_len = new_total
        return past_len, new_total


@dataclass
class Qwen36GenerationState:
    """Per-sequence state a caller owns and threads through forward calls.

    Indexed by *global* layer index (0..num_hidden_layers-1); entries for
    a layer of the "wrong" kind for that index are simply never
    constructed (``None``), matching ``ArchitectureSpec.layers[i].cache``
    (paged_kv vs recurrent) -- see ``runtime/architecture.py``.
    """

    gdn_states: list[GdnLayerState | None]
    attn_caches: list[Qwen36PagedAttentionCache | None]
    num_tokens_seen: int = 0


# ---------------------------------------------------------------------------
# Gated DeltaNet (linear attention) layer.
# ---------------------------------------------------------------------------


class Qwen36GatedDeltaNet(nn.Module):
    """Transcribed from ``Qwen3_5GatedDeltaNet``. Uses ``fla.ops.
    gated_delta_rule`` directly (B0-4 decision ①), not HF's torch
    fallback and not a from-scratch reimplementation. ``causal_conv1d``
    (the compiled package) is NOT installed in this runtime's environment
    (verified, B0/B1) -- the plain-torch conv path below is exercised
    unconditionally, which is also what HF's own reference falls back to
    on this same machine, so there is no cross-implementation divergence
    from that choice.
    """

    def __init__(self, config: dict[str, Any], layer_idx: int, quantized: dict[str, str]) -> None:
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_v_heads = config["linear_num_value_heads"]
        self.num_k_heads = config["linear_num_key_heads"]
        self.head_k_dim = config["linear_key_head_dim"]
        self.head_v_dim = config["linear_value_head_dim"]
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.repeat = self.num_v_heads // self.num_k_heads
        assert self.repeat * self.num_k_heads == self.num_v_heads

        self.conv_kernel_size = config["linear_conv_kernel_dim"]
        self.layer_idx = layer_idx
        self.eps = config["rms_norm_eps"]
        assert config["hidden_act"] == "silu"

        self.conv_dim = self.key_dim * 2 + self.value_dim
        # Depthwise causal conv1d: same parameterization as
        # Qwen3_5GatedDeltaNet.conv1d, but we only ever read .weight (bias
        # is False in the real checkpoint -- there is no conv1d.bias
        # tensor; verified against the real safetensors index). A plain
        # nn.Module() container (not a bare Parameter attribute) so the
        # checkpoint's own dotted name (``linear_attn.conv1d.weight``)
        # matches this module's parameter path exactly -- load_weights
        # below does no per-tensor name remapping, only a fixed top-level
        # prefix strip, so this has to line up 1:1.
        self.conv1d = nn.Module()
        self.conv1d.weight = nn.Parameter(
            torch.empty(self.conv_dim, 1, self.conv_kernel_size), requires_grad=False
        )
        self.conv1d.weight.weight_loader = default_weight_loader

        self.dt_bias = nn.Parameter(torch.empty(self.num_v_heads), requires_grad=False)
        self.dt_bias.weight_loader = default_weight_loader
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads), requires_grad=False)
        self.A_log.weight_loader = default_weight_loader

        self.norm = Qwen36RMSNormGated(self.head_v_dim, eps=self.eps)

        prefix = f"model.language_model.layers.{layer_idx}.linear_attn"
        self.in_proj_qkv = _make_linear(
            quantized, f"{prefix}.in_proj_qkv", self.hidden_size, self.key_dim * 2 + self.value_dim
        )
        self.in_proj_z = _make_linear(
            quantized, f"{prefix}.in_proj_z", self.hidden_size, self.value_dim
        )
        self.out_proj = _make_linear(
            quantized, f"{prefix}.out_proj", self.value_dim, self.hidden_size
        )
        # in_proj_a/in_proj_b are never in quantized_layers (verified, B0-2)
        # -- plain BF16 Linears, small (hidden_size -> num_v_heads).
        self.in_proj_b = PlainLinear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = PlainLinear(self.hidden_size, self.num_v_heads, bias=False)

    def new_state(self, *, batch: int, device: torch.device, dtype: torch.dtype) -> GdnLayerState:
        assert batch == 1, "Qwen36GatedDeltaNet (B1) only supports batch=1"
        return GdnLayerState(
            # NOTE: length is the FULL kernel size (4), not kernel_size-1.
            # Derived (not guessed) from transformers/cache_utils.py's
            # LinearAttentionLayer.lazy_initialization, which sets its own
            # conv_kernel_size = conv_states.shape[-1] from whatever the
            # model passes on the first update_conv_state call --
            # Qwen3_5GatedDeltaNet.forward always passes
            # F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0)),
            # whose last-dim length is exactly self.conv_kernel_size (a
            # negative left-pad here means "trim to the last K elements",
            # not "pad to K-1") -- see this file's git history / the B1
            # handoff notes for the full derivation and the GPU check that
            # confirmed it against a real Cache object.
            conv_state=torch.zeros(
                batch, self.conv_dim, self.conv_kernel_size, device=device, dtype=dtype
            ),
            recurrent_state=torch.zeros(
                batch, self.num_v_heads, self.head_k_dim, self.head_v_dim,
                device=device, dtype=dtype,
            ),
            has_previous_state=False,
        )

    def _conv1d_causal(self, x: torch.Tensor) -> torch.Tensor:
        """Exactly reproduces ``F.silu(self.conv1d(x)[:, :, :x.shape[-1]])``
        where HF's ``self.conv1d`` is ``nn.Conv1d(kernel_size=K,
        groups=conv_dim, padding=K-1)`` -- i.e. padding=K-1 on BOTH sides,
        then keep only the first ``x.shape[-1]`` output columns. This is
        NOT the same as "left-pad by K-1, then padding=0" (that would drop
        the wrong end) -- the right-side padding's outputs are real, just
        discarded by the truncation, exactly as HF's real module produces
        and discards them. See :meth:`new_state`'s comment for why calling
        this on ``state.conv_state``-prefixed input (continuation case) is
        still correct despite re-adding a spurious left zero-pad: the
        corrupted positions land entirely inside the discarded range.
        """
        input_len = x.shape[-1]
        out = F.conv1d(
            x, self.conv1d.weight, bias=None,
            padding=self.conv_kernel_size - 1, groups=self.conv_dim,
        )
        return F.silu(out[:, :, :input_len])

    def forward(self, hidden_states: torch.Tensor, state: GdnLayerState) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        assert batch_size == 1

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)  # [b, conv_dim, seq]
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        if state.has_previous_state and seq_len == 1:
            # Single-token decode: torch_causal_conv1d_update equivalent.
            state_len = state.conv_state.shape[-1]
            catted = torch.cat([state.conv_state, mixed_qkv], dim=-1).to(self.conv1d.weight.dtype)
            state.conv_state.copy_(catted[:, :, -state_len:])
            out = F.conv1d(catted, self.conv1d.weight, bias=None, padding=0, groups=self.conv_dim)
            mixed_qkv = F.silu(out[:, :, -seq_len:])
        else:
            if state.has_previous_state:
                mixed_qkv = torch.cat([state.conv_state, mixed_qkv], dim=-1)
            new_conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
            state.conv_state.copy_(new_conv_state)
            mixed_qkv = self._conv1d_causal(mixed_qkv)
            if state.has_previous_state:
                mixed_qkv = mixed_qkv[:, :, -seq_len:]

        mixed_qkv = mixed_qkv.transpose(1, 2)  # [b, seq, conv_dim]
        split_sizes = [self.key_dim, self.key_dim, self.value_dim]
        query, key, value = torch.split(mixed_qkv, split_sizes, dim=-1)
        query = query.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if self.repeat > 1:
            query = query.repeat_interleave(self.repeat, dim=2)
            key = key.repeat_interleave(self.repeat, dim=2)

        initial_state = state.recurrent_state if state.has_previous_state else None
        if state.has_previous_state and seq_len == 1:
            core_attn_out, last_state = fused_recurrent_gated_delta_rule(
                query, key, value, g=g, beta=beta,
                initial_state=initial_state, output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, last_state = chunk_gated_delta_rule(
                query, key, value, g=g, beta=beta,
                initial_state=initial_state, output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )

        # Cross-step BF16 rounding (B0-4/B0-7): FLA computes this step in
        # FP32 internally regardless of state.recurrent_state's dtype; we
        # explicitly round the persisted copy back down here, matching
        # transformers/cache_utils.py's LinearAttentionLayer exactly
        # rather than relying on an implicit dtype coercion somewhere.
        state.recurrent_state = last_state.to(state.recurrent_state.dtype)
        state.has_previous_state = True

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z_flat = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z_flat)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        return self.out_proj(core_attn_out)


# ---------------------------------------------------------------------------
# Full attention layer (sparkinfer paged attention).
# ---------------------------------------------------------------------------


class Qwen36Attention(nn.Module):
    """Transcribed from ``Qwen3_5Attention``. Uses sparkinfer's paged
    attention kernel (B0-3-verified for this exact shape) with a BF16 KV
    cache -- see module docstring for why BF16, not FP8.
    """

    def __init__(
        self,
        config: dict[str, Any],
        layer_idx: int,
        quantized: dict[str, str],
        *,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.num_kv_heads = config["num_key_value_heads"]
        self.head_dim = config["head_dim"]
        self.scaling = self.head_dim**-0.5
        self.eps = config["rms_norm_eps"]
        self.max_seq_len = max_seq_len
        assert not config.get("attention_bias", False)
        assert config.get("attn_output_gate", False), (
            "Qwen36Attention always applies the sigmoid output gate baked into "
            "q_proj's doubled output width; a checkpoint with attn_output_gate=False "
            "would need a different q_proj shape and forward path, not handled here"
        )

        prefix = f"model.language_model.layers.{layer_idx}.self_attn"
        qkv_in = self.hidden_size
        kv_out = self.num_kv_heads * self.head_dim
        q_out = self.num_heads * self.head_dim * 2
        self.q_proj = _make_linear(quantized, f"{prefix}.q_proj", qkv_in, q_out)
        self.k_proj = _make_linear(quantized, f"{prefix}.k_proj", qkv_in, kv_out)
        self.v_proj = _make_linear(quantized, f"{prefix}.v_proj", qkv_in, kv_out)
        self.o_proj = _make_linear(
            quantized, f"{prefix}.o_proj", self.num_heads * self.head_dim, self.hidden_size
        )
        self.q_norm = Qwen36RMSNorm(self.head_dim, eps=self.eps)
        self.k_norm = Qwen36RMSNorm(self.head_dim, eps=self.eps)

    def new_cache(self, *, device: torch.device, dtype: torch.dtype) -> Qwen36PagedAttentionCache:
        return Qwen36PagedAttentionCache(
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            dtype=dtype,
            device=device,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        cache: Qwen36PagedAttentionCache,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        assert batch_size == 1

        q_and_gate = self.q_proj(hidden_states)
        q_and_gate = q_and_gate.view(batch_size, seq_len, self.num_heads, self.head_dim * 2)
        query, gate = torch.chunk(q_and_gate, 2, dim=-1)
        gate = gate.reshape(batch_size, seq_len, -1)

        kv_shape = (batch_size, seq_len, self.num_kv_heads, self.head_dim)
        query = self.q_norm(query)
        key = self.k_norm(self.k_proj(hidden_states).view(*kv_shape))
        value = self.v_proj(hidden_states).view(*kv_shape)

        # Flatten batch=1 away for RoPE + sparkinfer, both of which are
        # token-major ([total_q, num_heads, head_dim]) by convention.
        query = query.reshape(seq_len, self.num_heads, self.head_dim)
        key = key.reshape(seq_len, self.num_kv_heads, self.head_dim)
        value = value.reshape(seq_len, self.num_kv_heads, self.head_dim)

        query_flat = query.reshape(seq_len, self.num_heads * self.head_dim).contiguous()
        key_flat = key.reshape(seq_len, self.num_kv_heads * self.head_dim).contiguous()
        apply_rotary_embedding_inplace(positions, query_flat, self.head_dim, cos_sin_cache)
        apply_rotary_embedding_inplace(positions, key_flat, self.head_dim, cos_sin_cache)
        query = query_flat.view(seq_len, self.num_heads, self.head_dim)
        key = key_flat.view(seq_len, self.num_kv_heads, self.head_dim)

        past_len, total_len = cache.append(key.to(cache.dtype), value.to(cache.dtype))
        mode = "decode" if seq_len == 1 else "extend"

        plan = paged.plan(
            paged.Caps(
                device=query.device,
                mode=mode,
                dtype=query.dtype,
                kv_dtype=cache.k_cache.dtype,
                num_q_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.head_dim,
                head_dim_vo=self.head_dim,
                page_size=cache.max_seq_len,
                max_total_q=seq_len,
                max_batch=1,
                max_page_table_width=1,
                max_work_items=4096,
                max_partial_rows=65536,
                num_cache_pages=1,
                use_cuda_graph=False,
            )
        )
        scratch_spec = plan.scratch_specs()[0]
        scratch = torch.empty(scratch_spec.shape, dtype=scratch_spec.dtype, device=query.device)
        output = torch.empty(
            seq_len, self.num_heads, self.head_dim, dtype=query.dtype, device=query.device
        )
        cache_seqlens = torch.tensor([total_len], dtype=torch.int32, device=query.device)
        cu_seqlens_q = torch.tensor([0, seq_len], dtype=torch.int32, device=query.device)
        binding = paged.bind(
            plan,
            scratch=scratch,
            q=query.to(cache.k_cache.dtype) if query.dtype != cache.k_cache.dtype else query,
            k_cache=cache.k_cache,
            v_cache=cache.v_cache,
            output=output,
            page_table=cache.page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            active_total_q=seq_len,
            k_descale=None,
            v_descale=None,
        )
        attn_out, _lse = paged.run(binding=binding)
        del past_len

        attn_out = attn_out.reshape(batch_size, seq_len, -1).contiguous()
        attn_out = attn_out * torch.sigmoid(gate)
        return self.o_proj(attn_out)


# ---------------------------------------------------------------------------
# Dense SwiGLU MLP.
# ---------------------------------------------------------------------------


class Qwen36MLP(nn.Module):
    def __init__(self, config: dict[str, Any], layer_idx: int, quantized: dict[str, str]) -> None:
        super().__init__()
        hidden_size = config["hidden_size"]
        intermediate_size = config["intermediate_size"]
        prefix = f"model.language_model.layers.{layer_idx}.mlp"
        self.gate_proj = _make_linear(
            quantized, f"{prefix}.gate_proj", hidden_size, intermediate_size
        )
        self.up_proj = _make_linear(quantized, f"{prefix}.up_proj", hidden_size, intermediate_size)
        self.down_proj = _make_linear(
            quantized, f"{prefix}.down_proj", intermediate_size, hidden_size
        )
        assert config["hidden_act"] == "silu"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Decoder layer / text model / causal LM.
# ---------------------------------------------------------------------------


class Qwen36DecoderLayer(nn.Module):
    def __init__(
        self,
        config: dict[str, Any],
        layer_idx: int,
        quantized: dict[str, str],
        *,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config["layer_types"][layer_idx]
        assert self.layer_type in ("linear_attention", "full_attention")
        eps = config["rms_norm_eps"]

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen36GatedDeltaNet(config, layer_idx, quantized)
            self.self_attn = None
        else:
            self.self_attn = Qwen36Attention(config, layer_idx, quantized, max_seq_len=max_seq_len)
            self.linear_attn = None

        self.mlp = Qwen36MLP(config, layer_idx, quantized)
        self.input_layernorm = Qwen36RMSNorm(config["hidden_size"], eps=eps)
        self.post_attention_layernorm = Qwen36RMSNorm(config["hidden_size"], eps=eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        gdn_state: GdnLayerState | None,
        attn_cache: Qwen36PagedAttentionCache | None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            assert gdn_state is not None
            hidden_states = self.linear_attn(hidden_states, gdn_state)
        else:
            assert attn_cache is not None
            hidden_states = self.self_attn(hidden_states, positions, cos_sin_cache, attn_cache)

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen36TextModelSelfBuilt(nn.Module):
    """Embedding -> N decoder layers -> final norm. Batch=1, no vision."""

    def __init__(
        self, config: dict[str, Any], quantized: dict[str, str], *, max_seq_len: int
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config["hidden_size"]
        self.num_hidden_layers = config["num_hidden_layers"]
        self.max_seq_len = max_seq_len

        self.embed_tokens = nn.Embedding(config["vocab_size"], self.hidden_size)
        self.embed_tokens.weight.weight_loader = default_weight_loader

        self.layers = nn.ModuleList(
            [
                Qwen36DecoderLayer(config, i, quantized, max_seq_len=max_seq_len)
                for i in range(self.num_hidden_layers)
            ]
        )
        self.norm = Qwen36RMSNorm(self.hidden_size, eps=config["rms_norm_eps"])

        rope_params = config["rope_parameters"]
        assert rope_params.get("rope_type", "default") == "default", (
            "B0-6: mrope only degenerates to standard 1D RoPE for rope_type "
            "'default'; a checkpoint using a different rope_type needs new "
            "verification before this class can be trusted for it"
        )
        rotary_dim = int(self.head_dim_for_rope() * rope_params.get("partial_rotary_factor", 1.0))
        self.rotary_dim = rotary_dim
        # Built on the ambient (construction-time) device -- not
        # hardcoded to CPU -- so runtime.model.forward doesn't pay an
        # ~33 MiB H2D copy (262144 positions x rotary_dim x bf16) on
        # every single call, including every decode step.
        cos_sin_cache = compute_cos_sin_cache_default(
            rotary_dim,
            config["max_position_embeddings"],
            float(rope_params["rope_theta"]),
            torch.bfloat16,
            device=self.embed_tokens.weight.device,
        )
        self.register_buffer("cos_sin_cache", cos_sin_cache, persistent=False)

    def head_dim_for_rope(self) -> int:
        return self.config["head_dim"]

    def new_generation_state(
        self, *, device: torch.device, dtype: torch.dtype
    ) -> Qwen36GenerationState:
        gdn_states: list[GdnLayerState | None] = []
        attn_caches: list[Qwen36PagedAttentionCache | None] = []
        for layer in self.layers:
            if layer.layer_type == "linear_attention":
                gdn_states.append(layer.linear_attn.new_state(batch=1, device=device, dtype=dtype))
                attn_caches.append(None)
            else:
                gdn_states.append(None)
                attn_caches.append(layer.self_attn.new_cache(device=device, dtype=dtype))
        return Qwen36GenerationState(gdn_states=gdn_states, attn_caches=attn_caches)

    def forward(
        self,
        input_ids: torch.Tensor,
        state: Qwen36GenerationState,
        *,
        capture_hidden_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        assert input_ids.dim() == 2 and input_ids.shape[0] == 1, "B1 only supports batch=1"
        seq_len = input_ids.shape[1]
        past_len = state.num_tokens_seen
        positions = torch.arange(
            past_len, past_len + seq_len, device=input_ids.device, dtype=torch.long
        )

        hidden_states = self.embed_tokens(input_ids)
        cos_sin_cache = self.cos_sin_cache.to(hidden_states.device)

        per_layer_hidden: list[torch.Tensor] = []
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                positions,
                cos_sin_cache,
                state.gdn_states[layer.layer_idx],
                state.attn_caches[layer.layer_idx],
            )
            if capture_hidden_states:
                per_layer_hidden.append(hidden_states)

        hidden_states = self.norm(hidden_states)
        state.num_tokens_seen = past_len + seq_len

        if capture_hidden_states:
            return hidden_states, per_layer_hidden
        return hidden_states


class Qwen36ForCausalLMSelfBuilt(nn.Module):
    """Top-level model: :class:`Qwen36TextModelSelfBuilt` + ``lm_head``
    (NVFP4-quantized, per B0-2). Not tied to ``embed_tokens`` (checkpoint
    declares ``tie_word_embeddings: false``, verified)."""

    def __init__(self, config: dict[str, Any], *, max_seq_len: int) -> None:
        super().__init__()
        self.config = config
        # `config` is the merged dict runtime.model_loading.load_qwen36_model
        # builds (text_config's fields + top-level quantization_config
        # injected under the same key) -- see that function's docstring.
        quantized = quantized_layers_map(config)
        self.quantized = quantized
        self.model = Qwen36TextModelSelfBuilt(config, quantized, max_seq_len=max_seq_len)
        assert not config.get("tie_word_embeddings", False)
        self.lm_head = _make_linear(
            quantized, "lm_head", config["hidden_size"], config["vocab_size"]
        )

    def new_generation_state(
        self, *, device: torch.device, dtype: torch.dtype
    ) -> Qwen36GenerationState:
        return self.model.new_generation_state(device=device, dtype=dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        state: Qwen36GenerationState,
        *,
        capture_hidden_states: bool = False,
    ):
        return self.model(input_ids, state, capture_hidden_states=capture_hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    # -- Weight loading ------------------------------------------------

    def load_weights(self, weights) -> set[str]:
        """Route every checkpoint tensor by its own dotted name.

        No stacked_params_mapping needed (Track A/Laguna's own
        docstrings flag this as needed for *fused* checkpoint tensors --
        this checkpoint has none: every projection this graph constructs
        (q/k/v/o_proj, in_proj_qkv/z/a/b, out_proj, gate/up/down_proj) is
        already a 1:1 checkpoint tensor, verified against the real
        safetensors index, B0-2/B1). Top-level prefixes recognized:
        ``model.language_model.`` (backbone), ``lm_head.`` (head),
        ``mtp.`` (skipped -- B1 has no MTP head, see module docstring),
        ``model.visual.`` (should already be filtered out by the caller's
        ``language_model_only`` loader stage; if any slip through here,
        that is exactly the B0-1a/b guarantee failing, so this raises
        rather than silently accepting them).
        """
        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()
        self.skipped_mtp_count = 0
        unrecognized: list[str] = []

        for name, tensor in weights:
            if name.endswith(_IGNORED_WEIGHT_SUFFIXES):
                continue
            if name.startswith("mtp."):
                self.skipped_mtp_count += 1
                continue
            if name.startswith("model.visual."):
                raise RuntimeError(
                    f"vision tensor {name!r} reached Qwen36ForCausalLMSelfBuilt.load_weights "
                    "-- the caller must filter these BEFORE calling load_weights "
                    "(runtime.loading.language_model_only.filter_language_model_only "
                    "with language_model_only=True); this is not this method's job."
                )
            if name.startswith("model.language_model."):
                mapped = "model." + name[len("model.language_model.") :]
            elif name.startswith("lm_head."):
                mapped = name
            else:
                unrecognized.append(name)
                continue

            if mapped not in params_dict:
                continue
            param = params_dict[mapped]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, tensor)
            loaded.add(mapped)

        if unrecognized:
            raise ValueError(
                f"Qwen36ForCausalLMSelfBuilt.load_weights: {len(unrecognized)} tensor(s) with "
                f"an unrecognized top-level prefix, e.g. {unrecognized[:3]!r}. Expected every "
                "checkpoint key to start with 'model.language_model.', 'lm_head.', 'mtp.', "
                "or 'model.visual.'."
            )
        return loaded
