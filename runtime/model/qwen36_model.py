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
  FP8. The original reason (B0-2) was that "this checkpoint declares
  ``kv_cache_quant_algo: FP8`` but ships zero ``k_scale``/``v_scale``
  tensors", so there was no scale to use and BF16 KV sidestepped the
  question rather than guessing a default.

  🔴 **That premise is false for the standard model** (measured
  2026-08-03; the B0-2 observation was made against ``nvidia/``, which
  every script pointed at then):

  ====================  ========  ========
  checkpoint            k_scale   v_scale
  ====================  ========  ========
  ``nvidia/`` (B0-2)           0         0
  ``unsloth/`` standard      16        16
  ====================  ========  ========

  The standard checkpoint ships a complete static per-tensor symmetric
  FP8 KV scheme (``kv_cache_scheme``: ``num_bits=8``, ``strategy=tensor``,
  ``symmetric=True``, ``observer=static_minmax``) plus one
  ``k_scale``/``v_scale`` per full-attention layer. So the deferred
  question has a checkpoint-provided answer here, and nothing consumes
  it -- ``load_qwen36_model`` does not call
  ``apply_kv_cache_scale_post_load`` the way the Laguna path does, which
  is how ``warn_on_unconsumed_tensor_families`` surfaced these 32 tensors
  on its first real run.

  BF16 KV is still what ships and is still correct; this is an unclaimed
  opportunity, not a bug. It is a large one: KV is 8192 MiB/slot, the
  single biggest line in the 72.39 GiB resident audit
  (``notes/2026-08-03-production-memory-audit.md``), and FP8 KV would
  halve it. Any attempt must clear B1-R first. Uses a per-layer
  fixed-capacity workspace (:class:`Qwen36AttentionWorkspace`), not the
  higher-level ``sparkinfer.attention.paged.{plan,bind,run}`` convenience
  API directly -- the convenience API JIT-recompiles per distinct
  ``(seq_len, cache_seqlens)`` shape (confirmed on real GPU: an 8-token
  prompt paid a fresh ~24s extend compile a 5-token prompt moments
  earlier had already separately paid for), a real usability blocker for
  any traffic that does not repeat exact shapes. Ports the same fix
  Laguna already has (``SparkinferPrefillWorkspace``,
  ``runtime/backends/laguna_sparkinfer_attn.py``), plus a second piece
  Laguna does not need at its own geometry: the plan's ``cta_tile_q`` is
  part of sparkinfer's compile key and the eager planner derives it from
  the live query length, so ``PagedPlanBudget`` is passed to derive it
  from declared capacity instead. With both pieces there is exactly ONE
  extend compile and ONE decode compile per process, at any prompt
  length -- see :class:`Qwen36AttentionWorkspace`'s docstring for the
  measured evidence and for why the earlier "extend recompiles per novel
  shape" reading was one bucket boundary misread as an unbounded axis.
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

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from b12x.attention.gated_delta_rule import (
    fused_recurrent_gated_delta_rule_multistep,
    fused_recurrent_gated_delta_rule_multistep_indexed,
    fused_recurrent_gated_delta_rule_multistep_indexed_gated,
)
from b12x.attention.paged._forward import paged_attention_forward
from b12x.attention.paged._scratch import build_paged_attention_binding
from b12x.attention.paged.planner import (
    PagedPlanBudget,
    create_paged_plan,
    plan_decode_graph_capacity,
)
from b12x.attention.paged.workspace import PagedAttentionWorkspace
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from torch import nn

from runtime.kernels.rope import apply_rotary_embedding_inplace, compute_cos_sin_cache_default
from runtime.loading.compressed_tensors import (
    QUANT_ALGO_MP_FP8_CHANNEL,
    QUANT_ALGO_MP_NVFP4,
    mixed_precision_quant_map,
)
from runtime.loading.modelopt import (
    QUANT_ALGO_FP8,
    QUANT_ALGO_NVFP4,
    QUANT_ALGO_UNQUANTIZED,
    quantized_layers_map,
)
from runtime.model._weight_loading import default_weight_loader
from runtime.model.compressed_tensors_linear import (
    CompressedTensorsFP8ChannelLinear,
    CompressedTensorsNVFP4Linear,
    FusedFP8ChannelGateUp,
    FusedFP8ChannelQKV,
    _native_w8a8_fp8_channel_enabled,
    _native_w8a8_fp8_gate_up_enabled,
    _native_w8a8_library_for_cuda,
    fp8_channel_raw_execution_uses_all_layers,
)
from runtime.model.flashinfer_gdn import FlashInferGDNPrefill, load_chunk_gated_delta_rule
from runtime.model.flashinfer_prefill import FlashInferPagedPrefill
from runtime.model.modelopt_linear import ModelOptFP8Linear, ModelOptNVFP4Linear
from runtime.model.plain_linear import PlainLinear

#: MTP-draft-head FP8 KV gate (2026-08-16): the backbone FP8 KV is gated by
#: ``QSR_QWEN36_FP8_KV``; this separate switch lets an A/B keep the backbone
#: FP8 and toggle only the draft head's cache dtype.  DEFAULT OFF: measured
#: on real text (temperature 0, fixed prompt, 3x120 tokens) that FP8 MTP KV
#: drops draft acceptance 46.4% -> 36.0%, which costs more tokens per round
#: (2.38 -> 2.08) than the ~4% faster round it buys -- net negative.  The
#: digit-filler bench (100% acceptance) does not expose this, so the switch
#: stays available for any future scale strategy (e.g. per-token dynamic).
QSR_QWEN36_MTP_FP8_KV_ENV = "QSR_QWEN36_MTP_FP8_KV"

# vLLM/SGLang's optimized GDN extend path prepares q/k and the headwise gates
# outside the chunk kernel: q/k are normalized in FP32 and beta is sigmoid'ed
# in FP32 before being stored.  The diagnostic switch remains explicit so a
# regression can be bisected, but the reference-aligned path is the default;
# the decode/recurrent path is intentionally unaffected.
QSR_QWEN36_GDN_PREFILL_VLLM_PREP_ENV = "QSR_QWEN36_GDN_PREFILL_VLLM_PREP"

# SGLang's Qwen hybrid full-attention prefill uses FlashInfer FA2.  ``auto``
# selects it on CUDA when the optional package is available and retains the
# existing SparkInfer path otherwise; ``sparkinfer`` is the explicit rollback
# switch used for bisects and environments without the reference package.
QSR_QWEN36_PREFILL_ATTN_BACKEND_ENV = "QSR_QWEN36_PREFILL_ATTN_BACKEND"

# SGLang/vLLM's Qwen GDN extend path uses the FlashInfer SM120 CuTeDSL chunk
# kernel.  ``auto`` selects it when the optional package is available and
# retains the tested FLA path otherwise; ``fla`` is the explicit bisect switch.
QSR_QWEN36_GDN_PREFILL_BACKEND_ENV = "QSR_QWEN36_GDN_PREFILL_BACKEND"

# Temporary, opt-in CUDA synchronization for identifying which decoder layer
# dominates a long prefill.  It is off by default and must not affect serving.
QSR_QWEN36_PREFILL_LAYER_PROFILE_ENV = "QSR_QWEN36_PREFILL_LAYER_PROFILE"
# Temporary, opt-in sub-layer timing.  The value is either ``1`` (all
# layers) or a comma-separated list of decoder layer ids, e.g. ``3,4``.
# Keeping this separate from the layer timer lets us isolate attention/GDN
# from the dense MLP without synchronizing the production path.
QSR_QWEN36_PREFILL_OP_PROFILE_ENV = "QSR_QWEN36_PREFILL_OP_PROFILE"

# SGLang's CUDA Qwen3.5 path fuses Q/K Gemma RMSNorm, partial NeoX RoPE,
# and q/gate deinterleaving, then applies the attention output gate in one
# Triton launch.  Keep an explicit A/B switch because the old sequence of
# kernels remains the correctness fallback for unusual dtypes/layouts.
QSR_QWEN36_FUSED_ATTN_PREP_ENV = "QSR_QWEN36_FUSED_ATTN_PREP"

logger = logging.getLogger("qwen_sm120_runtime.qwen36_model")


def _prefill_op_profile_enabled(layer_idx: int, device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    value = os.environ.get(QSR_QWEN36_PREFILL_OP_PROFILE_ENV, "").strip()
    if not value:
        return False
    if value == "1":
        return True
    try:
        return layer_idx in {int(item.strip()) for item in value.split(",") if item.strip()}
    except ValueError:
        logger.warning("ignoring invalid %s=%r", QSR_QWEN36_PREFILL_OP_PROFILE_ENV, value)
        return False


def _gdn_prefill_l2norm(x: torch.Tensor) -> torch.Tensor:
    """Use the FLA Triton L2-normalizer with vLLM/SGLang's output contract.

    The editable FLA checkout used by this runtime returns ``(normalized,
    rstd)`` while SGLang's copy returns just ``normalized``.  Accept both
    shapes so the call site documents the algorithmic contract rather than a
    fork-specific helper return type.
    """

    from fla.modules.l2norm import l2norm_fwd

    normalized = l2norm_fwd(x.contiguous())
    if isinstance(normalized, tuple):
        normalized = normalized[0]
    return normalized

#: Checkpoint tensor suffixes this loader deliberately never consumes into
#: a Parameter. Empty since the 2026-08-03 FP8 follow-up: ``ModelOptFP8Linear``
#: now has a real ``input_scale`` Parameter (see its docstring), so that
#: suffix is routed like any other tensor -- and every quantization format's
#: activation-side scale that this model graph has no Parameter for
#: (NVFP4's weight-only ``.input_scale``/modelopt, unsloth's mixed-precision
#: NVFP4 ``.input_global_scale`` -- see ``runtime/model/
#: compressed_tensors_linear.py``'s module docstring for why that one is
#: never read either) still falls through harmlessly via the ``mapped not in
#: params_dict: continue`` check below, same as it always has for any
#: checkpoint tensor with no matching Parameter.
_IGNORED_WEIGHT_SUFFIXES: tuple[str, ...] = ()


#: Which ``nn.Module`` factory each classifier's algo string maps to.
#: Deliberately the single place that knows about every quantization format
#: this model graph can build Linears for -- two checkpoint *formats*
#: (modelopt, compressed-tensors mixed-precision) can both declare "FP8" or
#: "NVFP4" in spirit, but never the same algo *string* (see
#: ``runtime/loading/compressed_tensors.py``'s module docstring for why
#: they are genuinely different physical layouts, not just different
#: names for the same bytes), so a checkpoint's own classifier output
#: dispatches here unambiguously regardless of which format produced it.
_LINEAR_FACTORY_FOR_ALGO: dict[str, type[nn.Module]] = {
    QUANT_ALGO_FP8: ModelOptFP8Linear,
    QUANT_ALGO_NVFP4: ModelOptNVFP4Linear,
    QUANT_ALGO_MP_FP8_CHANNEL: CompressedTensorsFP8ChannelLinear,
    QUANT_ALGO_MP_NVFP4: CompressedTensorsNVFP4Linear,
}


def _make_linear(
    quantized: dict[str, str],
    dotted_name: str,
    in_features: int,
    out_features: int,
) -> nn.Module:
    """Pick the Linear class for ``dotted_name`` from the checkpoint's own
    per-module quantization classification (``quantized``, produced by
    :func:`~runtime.loading.modelopt.quantized_layers_map` for a modelopt
    checkpoint or :func:`~runtime.loading.compressed_tensors.
    mixed_precision_quant_map` for a compressed-tensors mixed-precision one
    -- see :meth:`Qwen36ForCausalLMSelfBuilt.__init__`, the one place that
    decides which) -- never hardcoded per-projection, so a checkpoint that
    quantizes something differently fails loud at construction time instead
    of silently loading raw bytes as BF16.
    """
    return _make_linear_for_algorithm(
        quantized.get(dotted_name, QUANT_ALGO_UNQUANTIZED),
        dotted_name,
        in_features,
        out_features,
    )


def _make_linear_for_algorithm(
    algo: str,
    dotted_name: str,
    in_features: int,
    out_features: int,
) -> nn.Module:
    """Build a Linear from a known checkpoint quantization algorithm.

    Most projections call :func:`_make_linear`, because their module name is
    also their checkpoint name.  Qwen GDN's historical execution contract is
    the important exception: the checkpoint stores qkv/z separately, while
    the execution matrix is one concatenated qkvz projection.  Keeping the
    algorithm dispatch here makes that fixed-layout exception use the exact
    same raw-FP8 factory as every ordinary projection.
    """
    if algo == QUANT_ALGO_UNQUANTIZED:
        return PlainLinear(in_features, out_features, bias=False)
    factory = _LINEAR_FACTORY_FOR_ALGO.get(algo)
    if factory is None:
        raise ValueError(
            f"module {dotted_name!r} declares quant_algo {algo!r}, which this loader does "
            f"not know how to dequantize; known algos are "
            f"{sorted(_LINEAR_FACTORY_FOR_ALGO)} (plus {QUANT_ALGO_UNQUANTIZED!r}). Failing "
            "loudly here beats silently loading this module's raw checkpoint bytes as if "
            "they were plain BF16."
        )
    return factory(in_features, out_features, bias=False)


def _bmm_project(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Apply a Linear-like ``module`` (:class:`~runtime.model.plain_linear.
    PlainLinear`, :class:`~runtime.model.modelopt_linear.ModelOptFP8Linear`,
    or ``...ModelOptNVFP4Linear``) to ``x`` (``[1, seq_len, in_features]``)
    via ``torch.bmm`` over an explicit batch dimension -- one independent
    ``[1, in] @ [in, out]`` matmul per position, weight broadcast in with
    ``expand`` (a stride-0 view, no copy) -- used by
    :meth:`Qwen36GatedDeltaNet.spec_forward` instead of calling ``module``
    directly.

    **Why this exists, not just ``module(x)``**: measured directly
    (2026-08-02, ``notes/2026-08-02-gdn-spec-forward-batching.md``) that
    calling a Linear once over all ``seq_len`` positions at once produces a
    DIFFERENT BF16 rounding (~1-2 ULP, up to ``0.00195`` absolute on real
    checkpoint weights) than calling it once per position in a loop --
    ordinary batched GEMM (``F.linear``/``torch.matmul`` broadcasting/
    ``torch.einsum`` all measured equally non-bit-exact) picks a reduction
    order that depends on the row count, so naively "batching the matmul"
    silently breaks bit-exactness. ``torch.bmm`` over an explicit batch
    dimension does not: measured bit-exact against ``seq_len`` sequential
    single-row calls, for both unquantized (:class:`PlainLinear`) and
    dequantize-to-BF16 quantized (FP8/NVFP4) weights alike -- it evidently
    dispatches to a genuinely per-batch-independent kernel rather than
    fusing rows into one wider reduction, so it reproduces each position's
    exact sequential-call result while still costing ONE kernel launch
    instead of ``seq_len``.
    """
    # The historical W8A8 verify path projected its complete ``B * (K + 1)``
    # query matrix through the channel-scaled FP8 GEMM.  Do that before the
    # old BF16 ``bmm`` compatibility fallback: invoking ``_ensure_ready``
    # here would materialize an otherwise avoidable full BF16 copy of every
    # GDN ``in_proj_a``/``in_proj_b`` weight.  This is intentionally limited
    # to CUDA's all-layer raw-FP8 contract.  CPU tests and the explicit
    # legacy fallback retain the independently-rounded BF16 ``bmm`` path
    # documented above.
    if (
        x.device.type == "cuda"
        and isinstance(module, CompressedTensorsFP8ChannelLinear)
        and fp8_channel_raw_execution_uses_all_layers()
    ):
        return module(x)

    if hasattr(module, "_ensure_ready"):
        module._ensure_ready()  # ModelOptFP8Linear / ModelOptNVFP4Linear
        weight = module._weight_bf16  # [out, in], BF16 (dequantized once, cached)
    else:
        weight = module.weight  # PlainLinear -- already BF16
    bias = getattr(module, "bias", None)
    assert bias is None, "_bmm_project: every GDN projection is bias=False; bias path untested"

    batch_size, seq_len, _ = x.shape
    rows = batch_size * seq_len
    x2d = x.reshape(rows, -1)  # [B * seq_len, in]
    weight_batched = weight.t().unsqueeze(0).expand(rows, -1, -1)  # [B * seq_len, in, out]
    out = torch.bmm(x2d.unsqueeze(1), weight_batched).squeeze(1)  # [B * seq_len, out]
    return out.reshape(batch_size, seq_len, -1)


def _gdn_fused_checkpoint_slice(
    mapped_name: str,
    tensor: torch.Tensor,
    params: dict[str, nn.Parameter],
) -> tuple[str, str] | None:
    """Copy one legacy Qwen GDN checkpoint tensor into a fused parameter.

    Qwen3.6 saves four independent tensors (``qkv``, ``z``, ``b``, ``a``),
    but historical production deliberately executed two physical matrices in
    ``[qkv, z]`` and ``[b, a]`` output-row order.  Both FP8 values and their
    per-channel scale tensors have their output channel on dimension zero,
    so direct row copies preserve the checkpoint bytes exactly.

    The second returned value is the legacy source name used by
    :meth:`Qwen36ForCausalLMSelfBuilt.load_weights` to prove both slices of
    each fused Parameter arrived before marking it loaded.
    """
    source_to_fused = (
        (".linear_attn.in_proj_qkv.", ".linear_attn.in_proj_qkvz.", True),
        (".linear_attn.in_proj_z.", ".linear_attn.in_proj_qkvz.", False),
        (".linear_attn.in_proj_b.", ".linear_attn.in_proj_ba.", True),
        (".linear_attn.in_proj_a.", ".linear_attn.in_proj_ba.", False),
    )
    for source, fused, first_shard in source_to_fused:
        if source not in mapped_name:
            continue
        fused_name = mapped_name.replace(source, fused)
        param = params.get(fused_name)
        if param is None:
            return None
        if tensor.ndim != param.ndim or tensor.shape[1:] != param.shape[1:]:
            raise RuntimeError(
                f"GDN fused load shape mismatch for {mapped_name!r}: checkpoint "
                f"{tuple(tensor.shape)} cannot fill {fused_name!r} with shape "
                f"{tuple(param.shape)}"
            )
        offset = 0 if first_shard else param.shape[0] - tensor.shape[0]
        destination = param.data.narrow(0, offset, tensor.shape[0])
        if destination.shape != tensor.shape:
            raise RuntimeError(
                f"GDN fused load row mismatch for {mapped_name!r}: destination "
                f"{tuple(destination.shape)} vs checkpoint {tuple(tensor.shape)}"
            )
        destination.copy_(tensor)
        return fused_name, mapped_name
    return None


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
        self._w_plus_one: torch.Tensor | None = None

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run Qwen's zero-centred RMSNorm, optionally carrying the residual.

        Keeping the residual as a separate return value lets all decoder
        entry points share one control-flow shape.  The actual add remains
        the original BF16 operation before normalisation: the historical
        Triton fused-add variant was correct, but its controlled W1-S A/B was
        slower in this self-built graph and is therefore deliberately not the
        production default.
        """
        combined = x if residual is None else x + residual
        input_dtype = combined.dtype
        combined_f32 = combined.to(torch.float32)
        variance = combined_f32.pow(2).mean(-1, keepdim=True)
        if combined.is_cuda:
            # Bit-exact tail fusion (tests/test_norm_tail_bit_parity.py):
            # variance keeps torch's reduction order; only the two final fp32
            # multiplies + bf16 round move into one kernel -- deterministic
            # RN ops, so the acceptance anchor cannot move.
            if self._w_plus_one is None:
                self._w_plus_one = (1.0 + self.weight.float()).contiguous()
            from runtime.kernels.fused_rms_norm import rms_norm_tail

            rstd = torch.rsqrt(variance + self.eps)
            orig_shape = combined_f32.shape
            out = rms_norm_tail(
                combined_f32.reshape(-1, orig_shape[-1]),
                rstd.reshape(-1),
                self._w_plus_one,
            ).view(orig_shape)
        else:
            out = combined_f32 * torch.rsqrt(variance + self.eps)
            out = out * (1.0 + self.weight.float())
            out = out.to(input_dtype)
        if residual is None:
            return out
        return out, combined


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


#: sparkinfer's paged-attention planner rejects any other page_size
#: (confirmed directly, 2026-08-02: ``page_size=512`` raised
#: "primary paged backend expects page_size=64 or page_size=128, got
#: 512" from ``sparkinfer/attention/paged/planner.py``) -- matches B0-3's
#: own probe, which only ever exercised these two values. 128 chosen
#: arbitrarily between the two (B0-3 measured both as correct and of
#: comparable throughput for this shape; B1 does not need to pick a
#: winner).
_PAGED_ATTENTION_PAGE_SIZE = 128


class Qwen36PagedAttentionCache:
    """Batch=1 BF16 KV cache for one full-attention layer, paged (fixed
    ``page_size=128``, see :data:`_PAGED_ATTENTION_PAGE_SIZE`) and
    read/written through sparkinfer's paged-attention kernel.

    Real multi-page addressing (not one giant page) is required, not a
    simplification choice -- sparkinfer's planner hard-rejects any
    ``page_size`` other than 64/128 (confirmed directly; see the constant
    above), so a single page sized to the whole sequence only works by
    accident when ``max_seq_len`` happens to equal 64 or 128. The page
    *table* itself may map logical pages onto a shared physical pool.  The
    B1-owned constructor keeps the trivial identity mapping; the serving
    slot pool supplies its fixed logical-to-physical row explicitly.  This
    makes batch=1, batched decode, and eventual prefix-cache sharing use the
    exact same addressing contract.
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
        self.page_size = _PAGED_ATTENTION_PAGE_SIZE
        self.num_pages = (max_seq_len + self.page_size - 1) // self.page_size
        self.physical_num_pages = self.num_pages
        self.max_seq_len = self.num_pages * self.page_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        kv_shape = (self.num_pages, self.page_size, num_kv_heads, head_dim)
        self.k_cache = torch.zeros(kv_shape, dtype=dtype, device=device)
        self.v_cache = torch.zeros(kv_shape, dtype=dtype, device=device)
        # Batch=1, one sequence: physical pages are just 0..num_pages-1,
        # in order -- no sharing/eviction to address (see class docstring).
        self.page_table = torch.arange(self.num_pages, dtype=torch.int32, device=device)
        self.page_table = self.page_table.unsqueeze(0)  # [1, num_pages]
        self.seq_len = 0  # tokens currently resident

    @classmethod
    def wrap(
        cls,
        *,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_size: int,
        page_table: torch.Tensor | None = None,
    ) -> Qwen36PagedAttentionCache:
        """Build a cache **over storage someone else owns** (B2's slot pool).

        When ``page_table`` is omitted, the storage is interpreted as this
        cache's complete local page range and the identity mapping is used.
        The slot pool instead passes its whole per-layer physical allocation
        plus one ``[1, logical_pages]`` table row.  That keeps a slot's B=1
        prefill/decode on the same page ids that batched decode and CUDA
        graphs consume, rather than relying on a hidden contiguous slice.

        Deliberately a second constructor rather than an ``__init__``
        keyword: ``__init__`` allocating is the invariant B1's callers
        rely on, and a ``storage=None`` parameter would make "who owns
        this memory" a runtime question at every call site instead of a
        choice visible in the constructor's name.
        """
        obj = cls.__new__(cls)
        obj.page_size = page_size
        if k_cache.shape != v_cache.shape:
            raise ValueError("wrapped Qwen KV caches must have the same shape")
        if page_table is not None and (
            page_table.ndim != 2 or page_table.shape[0] != 1 or page_table.shape[1] < 1
        ):
            raise ValueError("wrapped Qwen page_table must have shape [1, logical_pages]")
        obj.num_pages = (
            int(page_table.shape[1]) if page_table is not None else int(k_cache.shape[0])
        )
        obj.physical_num_pages = int(k_cache.shape[0])
        obj.max_seq_len = obj.num_pages * page_size
        obj.num_kv_heads = int(k_cache.shape[2])
        obj.head_dim = int(k_cache.shape[3])
        obj.dtype = k_cache.dtype
        obj.device = k_cache.device
        obj.k_cache = k_cache
        obj.v_cache = v_cache
        obj.page_table = (
            page_table
            if page_table is not None
            else torch.arange(obj.num_pages, dtype=torch.int32, device=k_cache.device).unsqueeze(0)
        )
        obj.seq_len = 0
        return obj

    def append(self, new_k: torch.Tensor, new_v: torch.Tensor) -> tuple[int, int]:
        """Write ``new_k``/``new_v`` (``[seq_len, num_kv_heads, head_dim]``)
        at the tail, scattering across page boundaries as needed. Returns
        ``(past_len, new_total_len)``."""
        seq_len = new_k.shape[0]
        past_len = self.seq_len
        new_total = past_len + seq_len
        if new_total > self.max_seq_len:
            raise RuntimeError(
                f"Qwen36PagedAttentionCache overflow: max_seq_len={self.max_seq_len}, "
                f"attempted to reach {new_total}"
            )
        positions = torch.arange(past_len, new_total, device=self.device)
        page_ids = positions // self.page_size
        offsets = positions % self.page_size
        physical_page_ids = self.page_table[0, page_ids]
        self.k_cache[physical_page_ids, offsets] = new_k.to(self.dtype)
        self.v_cache[physical_page_ids, offsets] = new_v.to(self.dtype)
        self.seq_len = new_total
        return past_len, new_total


@dataclass
class Qwen36DecodeBatch:
    """Everything one batched decode step needs, addressed globally (B2).

    Built once per step by :class:`runtime.model.qwen36_slots.Qwen36SlotPool`
    and threaded down to the layers, so no layer has to know what a "slot"
    is: by the time this object exists, slot identity has already been
    reduced to (a) ``slot_index``, the one gather/scatter key the recurrent
    layers use, and (b) ``page_table``/``write_index``, the global KV
    addresses the attention layers use.

    Lists are indexed by **global layer index** and carry ``None`` for
    layers of the other kind, exactly like
    :class:`Qwen36GenerationState` -- same convention, so the two paths
    stay readable against each other.

    Every tensor here is a persistent buffer owned by the pool, refilled
    in place each step. That is what makes the whole step CUDA-Graph
    replayable: a replay re-reads these same addresses, so "run the graph"
    and "run eager" differ only in whether the pool's contents were
    updated before the launch.
    """

    input_ids: torch.Tensor  # [B, 1] int64
    positions: torch.Tensor  # [B] int64
    write_index: torch.Tensor  # [B] int64, flat row into k_pool.view(-1, H, D)
    slot_index: torch.Tensor  # [B] int64, which pool row each batch entry is
    #: Shared across every full-attention layer; owns ``page_table`` (global
    #: page ids) and ``cache_seqlens`` (lengths INCLUDING this step's token).
    #: Which concrete driver this is -- eager or graph-replay -- is the whole
    #: of the difference between an eager step and a captured one.
    attn: Any
    k_pools: list[torch.Tensor | None]
    v_pools: list[torch.Tensor | None]
    conv_pools: list[torch.Tensor | None]
    recurrent_pools: list[torch.Tensor | None]
    attn_outputs: list[torch.Tensor | None]
    #: NVFP4 KV pools (None when QSR_QWEN36_NVFP4_KV is off).
    nvfp4_k_codes: list[torch.Tensor | None] | None = None
    nvfp4_k_scales: list[torch.Tensor | None] | None = None
    nvfp4_v_codes: list[torch.Tensor | None] | None = None
    nvfp4_v_scales: list[torch.Tensor | None] | None = None

    @property
    def page_table(self) -> torch.Tensor:
        return self.attn.page_table

    @property
    def cache_seqlens(self) -> torch.Tensor:
        return self.attn.cache_seqlens


@dataclass
class Qwen36PrefillBatch:
    """One uniform ``B x Q`` extend forward over slot-owned pools.

    This deliberately describes only the homogeneous prefill shape: every
    request contributes the same number of new tokens and enters with the
    same recurrent-state regime.  That is the high-throughput admission
    shape used by W1-S.  Ragged/prefix-mismatched admissions retain the
    established B=1 path rather than padding real GDN recurrence with fake
    tokens (which would be silently wrong).
    """

    input_ids: torch.Tensor  # [B, Q]
    positions: torch.Tensor  # [B * Q], request-major
    write_index: torch.Tensor  # [B * Q], flat rows into each KV pool
    slot_index: torch.Tensor  # [B], pooled GDN row per request
    attn: Any
    k_pools: list[torch.Tensor | None]
    v_pools: list[torch.Tensor | None]
    conv_pools: list[torch.Tensor | None]
    recurrent_pools: list[torch.Tensor | None]
    attn_outputs: list[torch.Tensor | None]
    has_previous_state: bool
    #: NVFP4 KV pools (None when QSR_QWEN36_NVFP4_KV is off).
    nvfp4_k_codes: list[torch.Tensor | None] | None = None
    nvfp4_k_scales: list[torch.Tensor | None] | None = None
    nvfp4_v_codes: list[torch.Tensor | None] | None = None
    nvfp4_v_scales: list[torch.Tensor | None] | None = None


@dataclass
class Qwen36VerifyBatch:
    """Persistent inputs for one captured K-token target verify.

    Unlike :meth:`Qwen36TextModelSelfBuilt.verify_forward`, this descriptor
    owns no Python cache lengths and never calls ``Qwen36PagedAttentionCache
    .append``.  The graph driver updates page metadata and candidate write
    rows in place before replay, which is the distinction that makes a graph
    replay at a different round position correct rather than a replay of the
    capture round's addresses.
    """

    input_ids: torch.Tensor
    positions: torch.Tensor
    write_index: torch.Tensor
    k_pools: list[torch.Tensor | None]
    v_pools: list[torch.Tensor | None]
    attn_drivers: list[Any | None]
    attn_outputs: list[torch.Tensor | None]
    gdn_source_index: torch.Tensor
    #: Request-major ``[B, qo_len]`` global row ids.  This is the graph
    #: safe form of ``gdn_state_rows``: replay updates ids, while the captured
    #: body keeps reading and writing the same pooled storage addresses.
    gdn_destination_index: torch.Tensor | None
    gdn_conv_pools: list[torch.Tensor | None]
    gdn_recurrent_pools: list[torch.Tensor | None]
    #: Eager compatibility path.  CUDA graphs use ``gdn_destination_index``
    #: instead, so slot identity remains data rather than captured addresses.
    gdn_state_rows: dict[int, list[list[GdnLayerState]]] | None
    #: NVFP4 KV pools (None when QSR_QWEN36_NVFP4_KV is off).
    nvfp4_k_codes: list[torch.Tensor | None] | None = None
    nvfp4_k_scales: list[torch.Tensor | None] | None = None
    nvfp4_v_codes: list[torch.Tensor | None] | None = None
    nvfp4_v_scales: list[torch.Tensor | None] | None = None
    #: Ragged DSpark verify metadata.  ``input_ids``/``positions``/writes are
    #: compact request-major rows with capacity ``batch * max_verify_tokens``;
    #: the GDN path temporarily views those rows through the padded maps below
    #: while full attention consumes ``cu_seqlens_q`` directly.
    ragged: bool = False
    max_verify_tokens: int | None = None
    cu_seqlens_q: torch.Tensor | None = None
    gdn_padded_to_compact: torch.Tensor | None = None
    gdn_compact_to_padded: torch.Tensor | None = None


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
        # The checkpoint stores qkv and z as two tensors, but the historical
        # production path joins their *output rows* into one qkvz matrix and
        # executes exactly one input projection.  This is a storage-layout
        # fusion only: load_weights below copies the original E4M3 rows and
        # their original per-channel scales directly into the two slices --
        # no BF16 materialization and no re-quantization.
        qkv_name = f"{prefix}.in_proj_qkv"
        z_name = f"{prefix}.in_proj_z"
        qkv_algo = quantized.get(qkv_name, QUANT_ALGO_UNQUANTIZED)
        z_algo = quantized.get(z_name, QUANT_ALGO_UNQUANTIZED)
        if qkv_algo != z_algo:
            raise ValueError(
                f"GDN layer {layer_idx}: in_proj_qkv uses {qkv_algo!r}, but "
                f"in_proj_z uses {z_algo!r}; historical qkvz fusion requires "
                "one physical projection format"
            )
        self.in_proj_qkvz = _make_linear_for_algorithm(
            qkv_algo,
            f"{prefix}.in_proj_qkvz",
            self.hidden_size,
            self.conv_dim + self.value_dim,
        )
        self.out_proj = _make_linear(
            quantized, f"{prefix}.out_proj", self.value_dim, self.hidden_size
        )
        # b/a are plain BF16 in this checkpoint.  Like qkv/z, historical
        # execution concatenates their output rows in [b, a] order; the
        # shard-aware loader records the original two checkpoint tensors.
        for projection_name in (f"{prefix}.in_proj_b", f"{prefix}.in_proj_a"):
            algo = quantized.get(projection_name, QUANT_ALGO_UNQUANTIZED)
            if algo != QUANT_ALGO_UNQUANTIZED:
                raise ValueError(
                    f"GDN layer {layer_idx}: {projection_name} uses unexpected "
                    f"quantization {algo!r}; the verified [b, a] fusion is BF16-only"
                )
        self.in_proj_ba = PlainLinear(
            self.hidden_size,
            self.num_v_heads * 2,
            shard_sizes=[self.num_v_heads, self.num_v_heads],
            bias=False,
        )
        self._flashinfer_gdn: FlashInferGDNPrefill | None = None
        self._flashinfer_gdn_checked = False

    def _flashinfer_gdn_prefill_enabled(self, hidden_states: torch.Tensor) -> bool:
        backend = os.environ.get(QSR_QWEN36_GDN_PREFILL_BACKEND_ENV, "auto").lower()
        if backend not in {"auto", "flashinfer", "fla"}:
            raise ValueError(
                f"{QSR_QWEN36_GDN_PREFILL_BACKEND_ENV} must be auto, flashinfer, or fla; "
                f"got {backend!r}"
            )
        if backend == "fla" or not hidden_states.is_cuda:
            return False
        if not self._flashinfer_gdn_checked:
            self._flashinfer_gdn_checked = True
            if load_chunk_gated_delta_rule() is not None:
                self._flashinfer_gdn = FlashInferGDNPrefill()
            elif backend == "flashinfer":
                raise RuntimeError("FlashInfer GDN prefill was requested but is unavailable")
        if self._flashinfer_gdn is None and backend == "flashinfer":
            raise RuntimeError("FlashInfer GDN prefill was requested but is unavailable")
        return self._flashinfer_gdn is not None

    def _run_prefill_core(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        z: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        *,
        state: GdnLayerState,
    ) -> torch.Tensor:
        """Run the SGLang-compatible chunk path or the legacy FLA fallback."""
        use_flashinfer = self._flashinfer_gdn_prefill_enabled(query)
        use_vllm_prefill_prep = (
            os.environ.get(QSR_QWEN36_GDN_PREFILL_VLLM_PREP_ENV, "1") == "1"
            and query.is_cuda
        )
        if use_vllm_prefill_prep:
            query = _gdn_prefill_l2norm(query)
            key = _gdn_prefill_l2norm(key)
            beta = b.float().sigmoid()
        else:
            beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        initial_state = state.recurrent_state if state.has_previous_state else None
        if use_flashinfer:
            assert self._flashinfer_gdn is not None
            batch_size, seq_len = query.shape[:2]
            if initial_state is None:
                # The pool is already zeroed on admission, but passing an
                # explicit zero state keeps the FlashInfer state bridge and
                # all subsequent chunks on one fixed shape/ownership path.
                initial_state = state.recurrent_state
            cu_seqlens = torch.arange(
                0,
                (batch_size + 1) * seq_len,
                seq_len,
                device=query.device,
                dtype=torch.int64,
            )
            core_attn_out = self._flashinfer_gdn.run(
                query=query,
                key=key,
                value=value,
                log_decay=g,
                beta=beta,
                recurrent_state=initial_state,
                cu_seqlens=cu_seqlens,
                num_value_heads=self.num_v_heads,
                head_k_dim=self.head_k_dim,
                head_v_dim=self.head_v_dim,
            )
            state.has_previous_state = True
        else:
            if self.repeat > 1:
                query = query.repeat_interleave(self.repeat, dim=2)
                key = key.repeat_interleave(self.repeat, dim=2)
            core_attn_out, last_state = chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=not use_vllm_prefill_prep,
            )
            state.recurrent_state.copy_(last_state)
            state.has_previous_state = True

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z_flat = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z_flat)
        core_attn_out = core_attn_out.reshape(query.shape[0], query.shape[1], -1)
        return self.out_proj(core_attn_out)

    def new_state(self, *, batch: int, device: torch.device, dtype: torch.dtype) -> GdnLayerState:
        if batch < 1:
            raise ValueError(f"GDN state batch must be positive, got {batch}")
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
                batch,
                self.num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
                device=device,
                dtype=dtype,
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
            x,
            self.conv1d.weight,
            bias=None,
            padding=self.conv_kernel_size - 1,
            groups=self.conv_dim,
        )
        return F.silu(out[:, :, :input_len])

    def forward(self, hidden_states: torch.Tensor, state: GdnLayerState) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        mixed_qkvz = self.in_proj_qkvz(hidden_states)
        mixed_qkv = mixed_qkvz[..., : self.conv_dim].transpose(1, 2)  # [b, conv_dim, seq]
        z = mixed_qkvz[..., self.conv_dim :].reshape(batch_size, seq_len, -1, self.head_v_dim)
        b, a = self.in_proj_ba(hidden_states).split(self.num_v_heads, dim=-1)

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

        # vLLM's Qwen GDN prefill and SGLang's FlashInfer/CuTeDSL extend
        # paths normalize q/k before the chunk kernel and keep beta in FP32.
        # That is numerically distinct from the historical eager path here:
        # ``b.sigmoid()`` rounds beta to BF16 before FLA sees it.  Keep the
        # The reference-aligned form is the default for CUDA prefill/extend.
        # Set the variable to 0 only when bisecting a numerical or acceptance
        # regression; single-token recurrent decode remains on the old path.
        use_vllm_prefill_prep = (
            os.environ.get(QSR_QWEN36_GDN_PREFILL_VLLM_PREP_ENV, "1") == "1"
            and query.is_cuda
            and not (state.has_previous_state and seq_len == 1)
        )
        if use_vllm_prefill_prep:
            query = _gdn_prefill_l2norm(query)
            key = _gdn_prefill_l2norm(key)
            beta = b.float().sigmoid()
        else:
            beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        initial_state = state.recurrent_state if state.has_previous_state else None
        if state.has_previous_state and seq_len == 1:
            core_attn_out, last_state = fused_recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            if self.repeat > 1:
                query = query.repeat_interleave(self.repeat, dim=2)
                key = key.repeat_interleave(self.repeat, dim=2)
            core_attn_out, last_state = chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=not use_vllm_prefill_prep,
            )

        # Cross-step BF16 rounding (B0-4/B0-7): FLA computes this step in
        # FP32 internally regardless of state.recurrent_state's dtype; we
        # explicitly round the persisted copy back down here, matching
        # transformers/cache_utils.py's LinearAttentionLayer exactly
        # rather than relying on an implicit dtype coercion somewhere.
        #
        # ``.copy_()``, never rebinding (B2): this used to be
        # ``state.recurrent_state = last_state.to(dtype)``, which allocates
        # a fresh tensor and rebinds the Python attribute. That is the one
        # thing B0-5 identified as fatal for CUDA Graph capture
        # (notes/2026-08-02-trackB-b0-gpu-facts.md §B0-5: "状态 buffer 只分配
        # 一次、mark_static_address 标记、永远 .copy_() 写回、永不重新绑定
        # 引用"), and it also breaks pooling: B2 hands this dataclass a
        # *view* into a per-slot pool buffer (runtime/model/qwen36_slots.py),
        # and a rebind would silently detach the sequence from its own slot
        # -- the state would keep being computed correctly and keep being
        # thrown away. Numerically identical: both round fp32 -> bf16 with
        # torch's single round-to-nearest-even path.
        state.recurrent_state.copy_(last_state)
        state.has_previous_state = True

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z_flat = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z_flat)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        return self.out_proj(core_attn_out)

    def decode_batch(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> torch.Tensor:
        """One decode step for ``B`` independent sequences at once (B2).

        ``hidden_states`` is ``[B, 1, hidden_size]``; ``conv_state`` is
        ``[B, conv_dim, conv_kernel_size]`` and ``recurrent_state`` is
        ``[B, num_v_heads, head_k_dim, head_v_dim]``, both **written in
        place** (``.copy_()``, never rebound -- see :meth:`forward`).

        This is :meth:`forward`'s ``state.has_previous_state and
        seq_len == 1`` branch with the batch axis left free instead of
        asserted to 1. Every op in it is already batch-elementwise:
        ``torch.cat``/``F.conv1d`` with ``groups=conv_dim`` treat dim 0 as
        independent, and FLA's ``fused_recurrent_gated_delta_rule``
        parallelizes over ``(batch, head)`` -- so B==1 through here must
        reproduce :meth:`forward` bit-for-bit, which is a claim B2's GPU
        gate checks rather than assumes (a per-batch-element reduction
        order change would be invisible to any shape assertion).

        Callers hand batched (gathered) state rather than a per-slot view
        on purpose: the gather/scatter is the only place slot identity
        enters, so nothing below this line needs to know which physical
        slot a row came from -- which is also what makes the whole call
        CUDA-Graph replayable against a fixed index buffer.
        """
        batch_size, seq_len, _ = hidden_states.shape
        assert seq_len == 1, "decode_batch is the single-token continuation path"

        mixed_qkvz = self.in_proj_qkvz(hidden_states)
        mixed_qkv = mixed_qkvz[..., : self.conv_dim].transpose(1, 2)  # [b, conv_dim, 1]
        z = mixed_qkvz[..., self.conv_dim :].reshape(batch_size, seq_len, -1, self.head_v_dim)
        b, a = self.in_proj_ba(hidden_states).split(self.num_v_heads, dim=-1)

        state_len = conv_state.shape[-1]
        catted = torch.cat([conv_state, mixed_qkv], dim=-1).to(self.conv1d.weight.dtype)
        conv_state.copy_(catted[:, :, -state_len:])
        out = F.conv1d(catted, self.conv1d.weight, bias=None, padding=0, groups=self.conv_dim)
        mixed_qkv = F.silu(out[:, :, -seq_len:])

        mixed_qkv = mixed_qkv.transpose(1, 2)  # [b, 1, conv_dim]
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

        core_attn_out, last_state = fused_recurrent_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        recurrent_state.copy_(last_state)

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z_flat = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z_flat)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        return self.out_proj(core_attn_out)

    def prefill_batch(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        *,
        has_previous_state: bool,
    ) -> torch.Tensor:
        """Uniform ``B x Q`` GDN extend over pooled states.

        ``forward`` already implements independent batch elements; its state
        container only has a scalar ``has_previous_state`` because B1 owned
        one sequence.  The prefill scheduler calls this only when every row
        has the same regime, so one scalar remains exact.  The explicit
        wrapper keeps that admission invariant at the batch boundary instead
        of silently applying one row's recurrence mode to another.
        """
        batch_size, seq_len, _ = hidden_states.shape
        mixed_qkvz = self.in_proj_qkvz(hidden_states)
        mixed_qkv = mixed_qkvz[..., : self.conv_dim].transpose(1, 2)
        z = mixed_qkvz[..., self.conv_dim :].reshape(
            batch_size, seq_len, -1, self.head_v_dim
        )
        b, a = self.in_proj_ba(hidden_states).split(self.num_v_heads, dim=-1)

        if has_previous_state:
            # ``conv_state`` and ``mixed_qkv`` are both [B, C, T].  Keep the
            # continuation window in that layout so this path does not make
            # two transient [B, T, C] views and a second transpose before the
            # causal convolution.
            mixed_qkv = torch.cat([conv_state, mixed_qkv], dim=-1)
        new_conv_state = F.pad(
            mixed_qkv,
            (self.conv_kernel_size - mixed_qkv.shape[-1], 0),
        )
        conv_state.copy_(new_conv_state)
        mixed_qkv = self._conv1d_causal(mixed_qkv)
        if has_previous_state:
            mixed_qkv = mixed_qkv[:, :, -seq_len:]
        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        state = GdnLayerState(
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            has_previous_state=has_previous_state,
        )
        return self._run_prefill_core(query, key, value, z, b, a, state=state)

    def spec_forward(
        self,
        hidden_states: torch.Tensor,
        state: GdnLayerState,
        *,
        spec_state_rows: list[GdnLayerState] | list[list[GdnLayerState]] | None = None,
        spec_source_index: torch.Tensor | None = None,
        spec_destination_index: torch.Tensor | None = None,
        spec_conv_pool: torch.Tensor | None = None,
        spec_recurrent_pool: torch.Tensor | None = None,
        batch_large_projections: bool = False,
    ) -> tuple[torch.Tensor, list[GdnLayerState] | None]:
        """B3: MTP verify's GDN forward -- K candidate positions, K+1
        materialized state snapshots, no chunk algorithm involved.

        **Why this exists** (``docs/implementation-plan.md`` §7.1 B3, the
        "主模型侧 GDN 递归状态回滚" item -- see also
        ``runtime/recurrent_state_pool.py``'s ``spec_row``/
        ``runtime/block_pool.py``'s ``_ssm_spec_row``, whose docstrings
        describe the addressing scheme this method's output is meant to
        feed): verify runs ``K`` draft tokens through the model at once.
        If accept/reject later keeps only the first ``m < K`` of them, the
        recurrent state must resume from "as if only those ``m`` tokens
        happened" -- but :func:`chunk_gated_delta_rule` (the ``forward``
        method's ``seq_len > 1`` branch) only ever returns the state after
        ALL ``K`` positions. Re-deriving an intermediate state from that one
        call is not possible without re-deriving the kernel's internal
        chunking (Qwen3.6's chunk size is 64 > any realistic ``K``, so a
        verify-length span is always exactly one chunk -- no chunk-boundary
        state is exposed either).

        **The mechanism** (the "ReplaySSM ring-buffer" idea from
        ``investigation-queue.md`` D-3, adapted to what this runtime already
        has rather than re-derived from vLLM's custom kernel -- see the B3
        report for the full derivation): call the exact same single-token
        ``fused_recurrent_gated_delta_rule`` path :meth:`forward` already
        uses for ordinary decode, once per candidate position, in a Python
        loop, and keep every intermediate ``(conv_state, recurrent_state)``
        pair instead of discarding all but the last. Because this is
        *literally* the same kernel call ordinary single-token decode makes,
        with the same inputs, snapshot ``j`` is not merely close to what ``j``
        ordinary decode steps would have produced from the same anchor -- it
        is the same floating-point computation, so it is bit-identical
        (verified on GPU, see ``scripts/b3_probe_gdn_spec_rollback.py``).
        This is strictly stronger than the B3 correctness bar in
        ``docs/b1-correctness-criterion.md`` §7 ("接受/拒绝后GDN状态与非投机
        路径的状态张量对比"), which only asks the two to agree, not to agree
        by being the same code path.

        **The cost this trades for that guarantee**: ``K`` sequential
        kernel launches per GDN layer instead of one chunked call. This is
        real (measured in the B3 report), not free -- it is the price of
        never needing a recompute-forward repair on partial rejection,
        which is what every other rollback strategy this runtime considered
        (checkpoint + recompute the accepted prefix through the FULL 64-layer
        model) would have cost instead, and that price scales with the
        number of REJECTED tokens times the full model's per-token cost, not
        just this one layer's.

        **Batching everything except the recurrence itself** (this
        session, 2026-08-02): of the ~12.6ms this method cost for one
        layer at K=16 (measured, B3 report), only ~6.8ms is the K
        sequential :func:`fused_recurrent_gated_delta_rule` calls
        themselves -- the rest (~5.7ms) was ``in_proj_qkv``/``in_proj_z``/
        ``in_proj_b``/``in_proj_a``, the causal conv1d state update,
        ``beta``/``g``, ``norm``, ``out_proj``, and per-step ``clone()``
        calls being *re-run once per candidate position inside the Python
        loop*, even though none of them has a cross-step dependency: each
        is a pure per-position map from ``hidden_states[:, t, :]`` (plus,
        for conv1d only, a window of raw pre-conv values that are already
        fully known before the recurrent loop starts -- they never depend
        on the recurrence's output). This method now runs each of those
        exactly once, over all ``seq_len`` positions at once, using the
        same batched causal-conv construction :meth:`forward`'s own
        ``seq_len > 1`` branch already uses for ordinary prefill (a single
        conv1d call over ``[state.conv_state, mixed_qkv]``, ``padding=0``)
        -- and recovers every intermediate conv_state snapshot by slicing
        a fixed ``kernel_size``-wide window out of that one concatenated
        tensor, since a causal conv's window contents at position ``t``
        don't depend on how many further positions are computed alongside
        it. Only the recurrent kernel call itself stays in a ``for t in
        range(seq_len)`` loop, because it is the one part with a genuine
        sequential dependency: each step's ``initial_state`` is the
        previous step's output. Proven bit-exact against sequential decode
        by the same probe as before
        (``scripts/b3_probe_gdn_spec_rollback.py``) -- batching changes
        WHICH kernel call computes a value, never the value itself, but
        this turned out to be a narrower claim than it first looked:

        - conv1d's causal window and ``Qwen36RMSNormGated``'s per-row
          reduction ARE both per-position maps whose result at position
          ``t`` is bit-identical whether ``t`` rides along with other
          positions in the same kernel call or not (measured directly) --
          both are batched here, unconditionally.
        - ``in_proj_a``/``in_proj_b`` (output dim = ``num_v_heads``, e.g.
          48 on the real checkpoint) are ALSO safe to batch, but only
          through :func:`_bmm_project` (``torch.bmm`` over an explicit
          batch dimension) -- a plain batched ``F.linear`` call measurably
          does NOT reproduce ``seq_len`` sequential single-row calls
          bit-for-bit (BF16 GEMM's reduction order depends on row count on
          this stack).
        - ``in_proj_qkv``/``in_proj_z``/``out_proj`` (output dims in the
          thousands -- ``conv_dim``/``value_dim``/``hidden_size``) do not
          preserve bit-exactness when batched. ``batch_large_projections``
          exposes that faster experimental form for the B3 quality gate;
          it remains off until the gate also measures its NLL and full-logit
          cosine legs. The exact sequential form is the production default.

        The legacy snapshot mode never touches ``state`` and operates on
        clones throughout, preserving crash-safety. The row-addressed mode
        writes fixed candidate rows as it advances; the caller selects the
        accepted row after the decision, so no snapshot copy is needed.

        Returns ``(output, snapshots)``: ``output`` is
        ``[1, seq_len, hidden_size]``, this layer's contribution for every
        candidate position (needed to feed the rest of the model, same as
        :meth:`forward`'s return). ``snapshots`` has ``seq_len + 1`` entries;
        ``snapshots[0]`` is the (cloned) anchor, unmodified;
        ``snapshots[j]`` for ``j >= 1`` is the state after processing the
        first ``j`` candidate positions. Pass ``snapshots[m]`` (``m`` =
        accepted count) to :func:`commit_spec_snapshot` to resume decoding.
        When ``spec_state_rows`` is supplied, the second value is ``None``
        because the rows themselves are the committed-state candidates.
        """
        batch_size, seq_len, _ = hidden_states.shape
        if spec_state_rows is None and spec_destination_index is None and batch_size != 1:
            raise ValueError(
                "snapshot verify only supports one sequence; use fixed GDN rows for batches"
            )
        if not state.has_previous_state:
            raise ValueError(
                "spec_forward continues from a committed anchor (the token "
                "immediately before the draft), which by construction "
                "always has a previous state -- even the first verify round "
                "of a sequence follows a real prefill's chunked forward. A "
                "fresh, never-prefilled slot reaching this call is a caller "
                "bug, not a case this method degrades gracefully for."
            )

        # ---- The exact sequential form is the production default. The
        # batched branch exists solely for the full-model B3 quality gate.
        # A single-token qkvz call is deliberately retained here: it keeps
        # the old per-position arithmetic while matching historical storage
        # and eliminating the former qkv-plus-z duplicate FP8 launch. ------
        if batch_large_projections:
            mixed_qkvz = self.in_proj_qkvz(hidden_states)
            mixed_qkv = mixed_qkvz[..., : self.conv_dim].transpose(1, 2)
            z = mixed_qkvz[..., self.conv_dim :].reshape(batch_size, seq_len, -1, self.head_v_dim)
        else:
            mixed_qkv_steps: list[torch.Tensor] = []
            z_steps: list[torch.Tensor] = []
            for t in range(seq_len):
                h_t = hidden_states[:, t : t + 1, :]
                qkvz_t = self.in_proj_qkvz(h_t)
                mixed_qkv_steps.append(qkvz_t[..., : self.conv_dim].transpose(1, 2))
                z_steps.append(qkvz_t[..., self.conv_dim :])
            mixed_qkv = torch.cat(mixed_qkv_steps, dim=-1)  # [1, conv_dim, seq_len]
            z = torch.cat(z_steps, dim=1).reshape(batch_size, seq_len, -1, self.head_v_dim)

        # ---- The historical [b, a] matrix stays on bmm's independent-row
        # execution path, then splits without a copy. ----------------------
        b, a = _bmm_project(self.in_proj_ba, hidden_states).split(self.num_v_heads, dim=-1)

        # ---- one causal conv1d call covers every position's state
        # update, fed the bit-exact per-position `mixed_qkv` collected
        # above. Exactly :meth:`forward`'s ``seq_len == 1`` branch's
        # window construction (`cat([state.conv_state, mixed_qkv]);
        # conv1d(padding=0)`), just with `seq_len` new positions appended
        # instead of 1 in a single call -- ``state.conv_state`` itself is
        # only ever read here, never mutated (this method's "never
        # touches state" contract). ----------------------------------
        state_len = state.conv_state.shape[-1]
        catted = torch.cat([state.conv_state, mixed_qkv], dim=-1).to(self.conv1d.weight.dtype)
        conv_out = F.conv1d(catted, self.conv1d.weight, bias=None, padding=0, groups=self.conv_dim)
        mixed_qkv = F.silu(conv_out[:, :, -seq_len:]).transpose(1, 2)  # [1, seq_len, conv_dim]

        split_sizes = [self.key_dim, self.key_dim, self.value_dim]
        query, key, value = torch.split(mixed_qkv, split_sizes, dim=-1)
        query = query.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        fuse_indexed_gates = (
            os.environ.get("QSR_QWEN36_GDN_FUSED_GATES", "0") == "1"
            and batch_large_projections
            and spec_destination_index is not None
            and query.is_cuda
            and query.dtype == torch.bfloat16
        )
        if not fuse_indexed_gates:
            beta = b.sigmoid()
            g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if self.repeat > 1:
            query = query.repeat_interleave(self.repeat, dim=2)
            key = key.repeat_interleave(self.repeat, dim=2)

        # The multistep kernel advances the whole speculative sequence in one
        # launch and returns every recurrent state, including the initial one.
        # This removes the per-token recurrent-state copies from the graph path
        # while preserving the exact K+1 row contract used for rollback-free
        # verification.
        if spec_state_rows is not None and spec_destination_index is not None:
            raise ValueError("spec verify accepts row views or row indices, not both")
        if spec_destination_index is not None:
            if (
                spec_destination_index.shape != (batch_size, seq_len)
                or spec_conv_pool is None
                or spec_recurrent_pool is None
            ):
                raise ValueError(
                    "indexed spec verify requires [B, qo_len] row ids and both recurrent pools"
                )
            spec_rows_by_request = None
        elif spec_state_rows is not None:
            # Preserve the original single-request public shape while letting
            # the captured batch descriptor carry one row list per request.
            # Normalising once here keeps every write below request-major.
            if spec_state_rows and isinstance(spec_state_rows[0], GdnLayerState):
                spec_rows_by_request = [spec_state_rows]
            else:
                spec_rows_by_request = spec_state_rows
            if len(spec_rows_by_request) != batch_size or any(
                len(rows) != seq_len for rows in spec_rows_by_request
            ):
                raise ValueError(
                    "spec_state_rows must provide one row per verify position "
                    f"(batch={batch_size}, seq_len={seq_len})"
                )
        else:
            spec_rows_by_request = None
        # The multistep kernel is CUDA + BF16 only. Production is exactly
        # that, so it is the fast path -- but `spec_forward`'s own CPU
        # equivalence gate builds a float32 layer on purpose (the layer's
        # parameters are uninitialized allocations, so a CPU fixture is the
        # only way to compare the row path against the snapshot oracle
        # deterministically). Calling the kernel unconditionally makes that
        # gate raise `q must be bfloat16`, i.e. removes the test that guards
        # this exact code. Fall back to K sequential calls instead: same
        # contract, same K+1 states, just without the fusion.
        use_multistep = query.is_cuda and query.dtype == torch.bfloat16
        use_indexed_multistep = (
            use_multistep
            and spec_destination_index is not None
            and spec_source_index is not None
            and spec_source_index.shape == (batch_size,)
        )
        if use_multistep:
            # The multistep kernel requires an innermost-contiguous layout;
            # FLA's sequential entry point does not, which is why this never
            # surfaced before. `mixed_qkv` arrives transposed and is then
            # `torch.split`, so these are strided views. Cheap to fix here:
            # q/k/v are [1, K, H, 128] with K=4, orders of magnitude smaller
            # than the K+1 recurrent-state clones this path exists to delete.
            if use_indexed_multistep:
                assert spec_recurrent_pool is not None and spec_source_index is not None
                if fuse_indexed_gates:
                    core_attn_out = fused_recurrent_gated_delta_rule_multistep_indexed_gated(
                        query.contiguous(),
                        key.contiguous(),
                        value.contiguous(),
                        a=a.contiguous(),
                        b=b.contiguous(),
                        A_log=self.A_log.float(),
                        dt_bias=self.dt_bias,
                        state_pool=spec_recurrent_pool,
                        source_index=spec_source_index,
                        destination_index=spec_destination_index,
                    )
                else:
                    core_attn_out = fused_recurrent_gated_delta_rule_multistep_indexed(
                        query.contiguous(),
                        key.contiguous(),
                        value.contiguous(),
                        g=g.contiguous(),
                        beta=beta.contiguous(),
                        state_pool=spec_recurrent_pool,
                        source_index=spec_source_index,
                        destination_index=spec_destination_index,
                    )
                recurrent_states = None
            else:
                core_attn_out, recurrent_states = fused_recurrent_gated_delta_rule_multistep(
                    query.contiguous(),
                    key.contiguous(),
                    value.contiguous(),
                    g=g.contiguous(),
                    beta=beta.contiguous(),
                    initial_state=state.recurrent_state.contiguous(),
                    output_all_states=True,
                )
        else:
            outs = []
            states = [state.recurrent_state]
            running = state.recurrent_state
            for t in range(seq_len):
                out_t, running = fused_recurrent_gated_delta_rule(
                    query[:, t : t + 1],
                    key[:, t : t + 1],
                    value[:, t : t + 1],
                    g=g[:, t : t + 1],
                    beta=beta[:, t : t + 1],
                    initial_state=running,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                )
                outs.append(out_t)
                states.append(running)
            core_attn_out = torch.cat(outs, dim=1)
            recurrent_states = torch.stack(states, dim=1)
        if spec_rows_by_request is None and spec_destination_index is None:
            assert recurrent_states is not None
            recurrent_snapshots = [recurrent_states[:, j].clone() for j in range(seq_len + 1)]
        else:
            # The source was gathered before this loop, so row zero can be
            # overwritten with the anchor result.  Every verify position has
            # exactly one candidate row -- no separate incoming-state row.
            for j in range(seq_len):
                if spec_destination_index is not None:
                    # The indexed Triton recurrence has already performed
                    # this exact row-addressed write.  Retaining a separate
                    # gather/snapshot/index_copy here would recreate the
                    # historical bottleneck this path removes.
                    if recurrent_states is not None:
                        spec_recurrent_pool.index_copy_(
                            0, spec_destination_index[:, j], recurrent_states[:, j + 1]
                        )
                else:
                    assert spec_rows_by_request is not None
                    assert recurrent_states is not None
                    for batch_idx, rows in enumerate(spec_rows_by_request):
                        rows[j].recurrent_state.copy_(
                            recurrent_states[batch_idx : batch_idx + 1, j + 1]
                        )

        # ---- batched norm, once over all `seq_len` positions' recurrent
        # outputs (was: once per position, inside the loop). RMSNormGated
        # measured bit-exact when batched -- see docstring above. ---------
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z_flat = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z_flat)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        # ---- out_proj stays per-position, sequential: output dim is
        # hidden_size, past the `_bmm_project` bit-exactness threshold
        # (same reasoning as in_proj_qkv/in_proj_z above). ------------------
        if batch_large_projections:
            output = self.out_proj(core_attn_out)
        else:
            output = torch.cat(
                [self.out_proj(core_attn_out[:, t : t + 1, :]) for t in range(seq_len)], dim=1
            )

        # ---- assemble the `seq_len + 1` snapshots: conv_state via pure
        # slicing of `catted` (position j's window is
        # ``catted[:, :, j : j + state_len]`` -- j=0 is the untouched
        # anchor, j=seq_len is the state after all candidates), recurrent
        # state from the multistep kernel above. -------------------------
        if spec_rows_by_request is None and spec_destination_index is None:
            assert recurrent_snapshots is not None
            snapshots: list[GdnLayerState] = [
                GdnLayerState(
                    conv_state=catted[:, :, j : j + state_len].clone(),
                    recurrent_state=recurrent_snapshots[j],
                    has_previous_state=True,
                )
                for j in range(seq_len + 1)
            ]
            return output, snapshots

        for j in range(seq_len):
            if spec_destination_index is not None:
                spec_conv_pool.index_copy_(
                    0, spec_destination_index[:, j], catted[:, :, j + 1 : j + 1 + state_len]
                )
            else:
                assert spec_rows_by_request is not None
                for batch_idx, rows in enumerate(spec_rows_by_request):
                    rows[j].conv_state.copy_(
                        catted[batch_idx : batch_idx + 1, :, j + 1 : j + 1 + state_len]
                    )
                    rows[j].has_previous_state = True
        return output, None


def commit_spec_snapshot(
    state: GdnLayerState, snapshots: list[GdnLayerState], accepted_count: int
) -> None:
    """Resume ``state`` from ``snapshots[accepted_count]`` -- the O(1) half
    of B3's rollback (the expensive half is :meth:`Qwen36GatedDeltaNet.
    spec_forward`, already paid before this is ever called).

    ``.copy_()`` into the caller's existing buffers, never a rebind -- same
    B0-5/B2 discipline as :meth:`Qwen36GatedDeltaNet.forward` and
    :meth:`Qwen36SlotPool.restore_recurrent_state`, for the same reason: if
    ``state`` is a slot pool's persistent, ``mark_static_address``-marked
    view, rebinding would silently detach the sequence from its own slot.

    ``accepted_count=0`` (every draft token rejected) is not a special case
    here -- ``snapshots[0]`` is the untouched anchor clone
    :meth:`spec_forward` made before running anything, so "roll all the way
    back" and "roll back partway" are the same operation at a different
    index. This is also what makes :meth:`spec_forward`'s "never touches
    ``state``" property load-bearing for crash-safety: a caller that never
    reaches this call (an exception between the two) has left ``state``
    exactly where it was -- equivalent to having called this with
    ``accepted_count=0``, not to having silently applied every candidate.
    """
    if not (0 <= accepted_count < len(snapshots)):
        raise ValueError(
            f"accepted_count={accepted_count} out of range for "
            f"{len(snapshots)} snapshots (0..{len(snapshots) - 1})"
        )
    chosen = snapshots[accepted_count]
    state.conv_state.copy_(chosen.conv_state)
    state.recurrent_state.copy_(chosen.recurrent_state)
    state.has_previous_state = True


# ---------------------------------------------------------------------------
# Full attention layer (sparkinfer paged attention).
# ---------------------------------------------------------------------------


class Qwen36AttentionWorkspace:
    """Fixed-capacity sparkinfer paged-attention workspace for one
    ``Qwen36Attention`` layer, covering ``mode="extend"`` (prefill),
    ``mode="decode"`` (single-token continuation) and ``mode="verify"``
    (MTP's K-token speculative verify, added 2026-08-03 -- B1 itself had
    no speculative decoding, and routing verify onto the extend workspace
    is what broke MTP under FP8 KV; see
    notes/2026-08-03-mtp-verify-mode.md). No ``window_left``/SWA (Qwen3.6's
    ``layer_types`` has no ``"sliding_attention"`` entry, only
    ``"full_attention"``/``"linear_attention"`` -- verified against the
    real checkpoint's ``config.json``, B0-6/B1).

    **Why this exists, not a nice-to-have**: the higher-level
    ``sparkinfer.attention.paged.{plan,bind,run}`` convenience API this
    module used before builds a *fresh* plan from the CALLER's actual
    ``(seq_len, cache_seqlens)`` on every single call. sparkinfer's CuTe
    launch wrapper JIT-compiles keyed on a snapshot of several tensors'
    shapes (confirmed directly, 2026-08-02: an 8-token prompt during B1's
    own smoke test re-triggered a fresh ~24s extend compile that a
    5-token prompt moments earlier had already separately paid for) -- so
    real serving traffic, which essentially never repeats a shape, would
    pay a ~25-60s compile on every distinct prompt length forever. This is
    not a hypothetical: it is the same root cause Laguna already found
    and fixed for its own attention path
    (``notes/2026-08-01-prefill-shape-buckets-root-cause.md``,
    ``SparkinferPrefillWorkspace`` in
    ``runtime/backends/laguna_sparkinfer_attn.py``) -- this class is that
    same fix, ported to Qwen3.6's shape (BF16 KV rather than FP8, no SWA,
    no verify), not a new design. Building ONE persistent
    ``PagedAttentionWorkspace.for_fixed_capacity`` instead means every
    call within the declared capacity reuses the SAME compiled kernel and
    the SAME scratch buffers -- only their *contents* change per call.

    **What is scoped out of this port, and why that's a real gap, not
    laziness**: Laguna shares ONE ``SparkinferPrefillWorkspace`` across an
    entire *group* of layers with matching shapes (all its full-attention
    layers at once). This class was originally constructed **per layer**
    (``Qwen36Attention._workspace_for`` built its own) -- correct (each
    instance's fixed capacity is still honored, and sparkinfer's own
    compile cache is keyed by shape parameters below the level of this
    Python object, so cross-layer compiles very likely still dedupe), but
    it allocated ``Qwen36Attention.num_layers``-times the scratch memory a
    single shared instance would. **2026-08-15 (plan §4.5 P0-M2 step 1):
    closed.** Given all 16 full-attention layers in this checkpoint share
    identical ``num_q_heads``/``num_kv_heads``/``head_dim``/``page_size``,
    ``_workspace_for`` now resolves through the module-level
    ``_SHARED_ATTN_WORKSPACES`` registry keyed by full geometry, so one
    arena per mode serves the whole group (recovering ~795 MiB of
    duplicate scratch); the per-layer K/V descale is handed in at
    ``forward`` time because it is the one per-layer input.

    **The second, independent axis: ``cta_tile_q`` (2026-08-02)**. The
    fixed-capacity workspace above pins every *buffer* shape, and that is
    genuinely sufficient for ``mode="decode"``. It was NOT sufficient for
    ``mode="extend"``, and the earlier version of this docstring recorded
    that as an unexplained contradiction. It is explained now, and the
    explanation is a second capacity-vs-live-shape leak in a place the
    workspace object cannot reach on its own:

    sparkinfer's compile cache key includes ``_traits_compile_key(traits)``
    (``attention/paged/_forward.py``), and ``traits`` comes from
    ``select_paged_forward_traits_from_plan(plan)`` -- so ``plan.cta_tile_q``
    is *part of the compile key*. For an eager plan the planner derives it
    from the LIVE query length
    (``planner.py``, ``create_paged_plan``'s ``enable_cuda_graph=False``
    branch)::

        avg_packed_qo_len = sum(packed_qo_len_arr) // max(batch, 1)
        if mode in ("extend", "verify") and plan_budget is not None \\
                and plan_budget.max_total_q is not None:
            avg_packed_qo_len = max(
                avg_packed_qo_len,
                int(plan_budget.max_total_q) * gqa_group_size // max(batch, 1),
            )
        cta_tile_q = _paged_determine_cta_tile_q(packed_qo_len=avg_packed_qo_len, ...)

    With ``gqa_group_size=6`` and ``head_dim=256``,
    ``_paged_determine_cta_tile_q`` returns 16 while
    ``packed_qo_len = 6 * seq_len <= 32`` and 64 above it -- i.e. it flips at
    ``seq_len == 6``. That is exactly one boundary, and B1's smoke test
    (prompts of 5, 5, 8 tokens) walked straight across it, which is why the
    third prompt looked like "every novel length recompiles". Measured
    directly (``scripts/b1_probe_extend_jit_buckets.py``, 14 mutually
    distinct novel lengths from 5 to 2584): pre-fix, exactly TWO lengths pay
    a compile (5 -> ``cta_tile_q=16``, 8 -> ``cta_tile_q=64``) and the other
    twelve cost ~4 ms. So the earlier "recompiles forever" reading was too
    pessimistic -- but the bug is real and is a usability blocker anyway,
    because *which* bucket a request lands in is decided by its prompt
    length, so any warmup that exercises one bucket leaves the other cold
    and a short prompt still stalls ~26 s in production.

    Fix: pass sparkinfer's own :class:`PagedPlanBudget` into
    ``create_paged_plan``. That ``max(...)`` branch above exists precisely
    so a fixed-capacity caller can have ``cta_tile_q`` derived from its
    declared capacity instead of the live shape; with
    ``max_total_q = max_seq_len`` every extend call lands in the same tile
    bucket, so there is ONE extend compile per (geometry, capacity) for the
    life of the process regardless of prompt length. The budget's other
    fields (``max_batch``, ``max_page_table_width``) additionally make
    ``create_paged_plan`` fail loudly and early if a caller ever exceeds the
    capacity this workspace was built for, instead of failing deeper inside
    ``_ensure_capacity``.

    Two things deliberately NOT changed, having been read and ruled out:

    - ``max_partial_rows=0`` stays correct. ``create_paged_plan`` hard-sets
      ``split_kv = False; disable_split_kv = True`` for ``mode == "extend"``
      unconditionally (``planner.py``), so an extend plan can never request
      the split-KV merge buffer. Measured across all 14 probe lengths:
      ``split_kv`` is ``False`` everywhere. This also means the split-KV
      chunk-count divergence documented in
      ``notes/2026-08-02-eager-verify-cg-verify-divergence.md`` is
      structurally out of reach on this path.
    - ``mode="decode"`` is unaffected by the budget:
      ``_paged_determine_cta_tile_q`` returns a hard-coded 16 for decode
      before it ever looks at ``packed_qo_len``, and the ``max(...)``
      branch is gated on ``mode in ("extend", "verify")``. Passing the
      budget for decode buys only the capacity assertions.
    """

    def __init__(
        self,
        *,
        mode: str,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        max_total_q: int,
        max_page_table_width: int,
        num_cache_pages: int,
        dtype: torch.dtype,
        kv_dtype: torch.dtype,
        device: torch.device,
        max_batch: int = 1,
        k_descale: torch.Tensor | None = None,
        v_descale: torch.Tensor | None = None,
        # Required for mode="verify" only: the largest query length any
        # verify call will present (the speculator's K), plus the real KV
        # caches, so the worst-case plan below is built against the same
        # tensors the live calls will use rather than a guess at their
        # layout. Unused by extend/decode.
        max_verify_query_len: int = 0,
        k_cache: torch.Tensor | None = None,
        v_cache: torch.Tensor | None = None,
    ) -> None:
        if mode not in ("extend", "decode", "verify"):
            raise ValueError(f"Qwen36AttentionWorkspace: unsupported mode {mode!r}")
        self.mode = mode
        self.max_batch = max_batch
        # k_descale/v_descale (2026-08-03, FP8 KV): the real per-layer
        # k_scale/v_scale Parameters (Qwen36Attention.k_scale/.v_scale) when
        # this layer's KV cache is FP8, or the harmless 1.0 no-op descale
        # (unchanged from before FP8 KV existed) when it is BF16. K and V
        # are NEVER equal on the real checkpoint (e.g. layer 3 measures
        # k_scale=0.0262, v_scale=0.0344) -- one shared descale for both
        # would silently apply K's scale to V or vice versa, so these are
        # kept as two independent tensors rather than one `_descale`.
        self._k_descale = (
            k_descale
            if k_descale is not None
            else torch.ones(1, dtype=torch.float32, device=device)
        )
        self._v_descale = (
            v_descale
            if v_descale is not None
            else torch.ones(1, dtype=torch.float32, device=device)
        )
        # eager_extend_work_items_capacity is sparkinfer's own estimator
        # for exactly this pair of modes (its name and design track
        # max_total_q, which is what extend/decode's real work-item count
        # scales with). It does NOT generalize to mode="verify" -- see
        # SparkinferPrefillWorkspace's docstring, which recorded this the
        # first time (notes/2026-08-01-c1-c2-gpu-investigation.md §C-1):
        # the same estimator applied to verify under-provisioned and
        # sparkinfer's _ensure_capacity hard-failed with "fixed-capacity
        # paged workspace exceeded" before any attention math ran.
        # mode="verify" therefore takes the recipe that investigation
        # landed on: run the REAL eager planner once, up front, at this
        # workspace's own declared worst case, and trust its numbers.
        if mode == "verify":
            max_work_items, max_partial_rows = self._verify_capacity(
                max_verify_query_len=max_verify_query_len,
                k_cache=k_cache,
                v_cache=v_cache,
                num_q_heads=num_q_heads,
                dtype=dtype,
                device=device,
                head_dim=head_dim,
            )
        else:
            max_work_items = PagedAttentionWorkspace.eager_extend_work_items_capacity(
                max_total_q=max_total_q, num_q_heads=num_q_heads, num_kv_heads=num_kv_heads
            )
            # matches PagedExtendGraphCapacity: no split-KV merge buffer
            max_partial_rows = 0
        self._workspace = PagedAttentionWorkspace.for_fixed_capacity(
            mode=mode,
            device=device,
            dtype=dtype,
            kv_dtype=kv_dtype,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            head_dim_vo=head_dim,
            page_size=page_size,
            max_total_q=max_total_q,
            max_batch=max_batch,
            max_page_table_width=max_page_table_width,
            max_work_items=max_work_items,
            max_partial_rows=max_partial_rows,
            num_cache_pages=num_cache_pages,
            use_cuda_graph=False,
        )
        # Declares this workspace's capacity to the planner so the plan's
        # own policy knobs -- above all ``cta_tile_q``, which is part of
        # sparkinfer's compile cache key -- are derived from capacity, not
        # from the live request's query length. See the class docstring.
        #
        # mode="verify" is deliberately EXCLUDED, and this is load-bearing
        # rather than conservatism -- SparkinferPrefillWorkspace's docstring
        # records why: _paged_determine_cta_tile_q selects the M64 verifier
        # by an exact match on packed_qo_len, and several downstream
        # kernel-policy flags (use_laguna_verify_kernel,
        # laguna_verify_two_wave_b1, the FP8 PV MMA path) are gated on
        # plan.cta_tile_q == 64. A capacity-derived packed_qo_len misses
        # that match and silently drops verify onto cta_tile_q=16. Verify
        # does not have extend's multi-bucket problem anyway: its query
        # length is a fixed K-token window, so it is single-bucket already.
        self._plan_budget = (
            None
            if mode == "verify"
            else PagedPlanBudget(
                max_total_q=max_total_q,
                max_batch=max_batch,
                max_page_table_width=max_page_table_width,
            )
        )
        self._prepared_metadata: object | None = None

    @staticmethod
    def _verify_capacity(
        *,
        max_verify_query_len: int,
        k_cache: torch.Tensor | None,
        v_cache: torch.Tensor | None,
        num_q_heads: int,
        dtype: torch.dtype,
        device: torch.device,
        head_dim: int,
    ) -> tuple[int, int]:
        """``(max_work_items, max_partial_rows)`` for a fixed-capacity
        ``mode="verify"`` workspace.

        Built by running sparkinfer's REAL eager planner
        (``create_paged_plan(enable_cuda_graph=False, mode="verify", ...)``
        -- the exact function every live verify call below will use) once
        against a synthetic worst case: every cache page occupied, query at
        the declared K. This is the recipe
        ``SparkinferPrefillWorkspace._capacity_for`` already validated on
        real GPU; its docstring also records the dead end not to repeat
        (``plan_verify_graph_capacity`` predicted 47/112 where the real
        eager plan needed 96/256 -- the graph and eager paths compute
        different schedules and are not interchangeable capacity sources).

        Sizing at the max KV bound is a genuine upper bound, not another
        guess: work items were confirmed monotonically increasing with
        kv_len on real GPU for this same full-attention geometry.
        """
        if max_verify_query_len <= 0:
            raise ValueError(
                "Qwen36AttentionWorkspace: mode='verify' requires "
                "max_verify_query_len > 0 (the speculator's K). Guessing one "
                "here would repeat the under-provisioning bug recorded in "
                "notes/2026-08-01-c1-c2-gpu-investigation.md, which surfaces "
                "as sparkinfer's 'fixed-capacity paged workspace exceeded'."
            )
        if k_cache is None or v_cache is None:
            raise ValueError(
                "Qwen36AttentionWorkspace: mode='verify' requires the real "
                "k_cache/v_cache to build its worst-case plan."
            )
        num_cache_pages = int(k_cache.shape[0])
        page_size = int(k_cache.shape[1])
        max_kv = max(num_cache_pages * page_size - 1, 1)
        worst_plan = create_paged_plan(
            torch.empty(max_verify_query_len, num_q_heads, head_dim, dtype=dtype, device=device),
            k_cache,
            v_cache,
            torch.arange(num_cache_pages, dtype=torch.int32, device=device).unsqueeze(0),
            torch.tensor([max_kv], dtype=torch.int32, device=device),
            torch.tensor([0, max_verify_query_len], dtype=torch.int32, device=device),
            mode="verify",
            enable_cuda_graph=False,
            window_left=-1,
        )
        max_partial_rows = int(worst_plan.total_num_partial_rows) if worst_plan.split_kv else 0
        return int(worst_plan.new_batch_size), max_partial_rows

    def forward(
        self,
        *,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        output: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        k_descale: torch.Tensor | None = None,
        v_descale: torch.Tensor | None = None,
    ) -> None:
        """Run attention, writing into ``output`` in place. First call
        against this instance pays sparkinfer's one-time CuTe compile
        (or hits its on-disk cache from a prior process); every later
        call at any shape within this instance's declared capacity reuses
        it -- including query lengths never seen before, which is what
        ``plan_budget`` buys (class docstring).

        ``k_descale``/``v_descale`` (2026-08-15, plan §4.5 P0-M2): the
        workspace is shared across layers with identical geometry, so the
        per-layer FP8 K/V scales are passed in per call instead of being
        captured at construction. ``None`` falls back to the descale the
        workspace was built with (the harmless 1.0 no-op for BF16 KV)."""
        plan = create_paged_plan(
            q,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            mode=self.mode,
            enable_cuda_graph=False,
            window_left=-1,
            plan_budget=self._plan_budget,
        )
        ws = self._workspace
        ws._ensure_capacity(plan)
        ws._copy_runtime_metadata(page_table, cache_seqlens, cu_seqlens_q)
        ws._copy_plan_metadata(plan)
        ws._plan = plan
        binding = build_paged_attention_binding(
            scratch=ws,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            output=output,
            k_descale=k_descale if k_descale is not None else self._k_descale,
            v_descale=v_descale if v_descale is not None else self._v_descale,
        )
        paged_attention_forward(binding=binding)


#: Shared fixed-capacity attention workspaces (plan §4.5 P0-M2 step 1).
#: All 16 full-attention layers of this checkpoint share one geometry
#: (``num_q_heads``/``num_kv_heads``/``head_dim``/``page_size``/capacity),
#: so one workspace per (mode, geometry, device, verify-capacity) suffices
#: instead of a per-layer arena; the workspace itself is layer-agnostic --
#: k/v descale and the real k/v caches are handed in per call. Keyed by
#: every input that fixes the arena's capacity, so a layer that diverges
#: (different max_seq_len, different verify capacity, FP8 vs BF16 KV)
#: simply gets its own entry.
_SHARED_ATTN_WORKSPACES: dict[tuple[object, ...], Qwen36AttentionWorkspace] = {}


def _shared_attention_workspace_key(
    *,
    mode: str,
    layer: Qwen36Attention,
    cache: Qwen36PagedAttentionCache,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[object, ...]:
    return (
        mode,
        layer.num_heads,
        layer.num_kv_heads,
        layer.head_dim,
        cache.page_size,
        layer.max_seq_len,
        cache.num_pages,
        cache.physical_num_pages,
        dtype,
        cache.k_cache.dtype,
        device,
        layer._max_verify_query_len,  # noqa: SLF001
    )


def release_shared_decode_workspaces() -> int:
    """Drop every shared ``mode="decode"`` workspace (plan §4.5 P0-M2
    step 3).

    Decode is CUDA-graph-replayed in production once capture succeeds; the
    eager per-layer decode path (serial control group, capture-failure
    fallback) is the only consumer left, and a later call rebuilds a
    workspace on demand. Returns the number of workspaces dropped.
    """
    decode_keys = [key for key in _SHARED_ATTN_WORKSPACES if key[0] == "decode"]
    for key in decode_keys:
        del _SHARED_ATTN_WORKSPACES[key]
    return len(decode_keys)


class Qwen36BatchedDecodeAttention:
    """One shared paged-attention driver for a whole batched decode step (B2).

    Owns the three metadata tensors (``page_table``, ``cache_seqlens``,
    ``cu_seqlens_q``) rather than taking them per call, which is what makes
    it interchangeable with :class:`Qwen36DecodeGraphAttention`: in graph
    mode those tensors have to be sparkinfer's own persistent buffers, not
    the caller's, and a driver object is the only place that difference can
    live without leaking into every layer's forward.

    **Shared across every full-attention layer**, unlike B1's per-layer
    :class:`Qwen36AttentionWorkspace`. That class's own docstring already
    flagged the per-layer construction as "a real, concrete follow-up --
    not attempted here because it was not needed to fix the actual bug
    within this pass's time budget"; batched decode forces the issue, since
    graph mode needs one set of metadata buffers for the whole step rather
    than sixteen that must all be written identically. All 16 full-attention
    layers in this checkpoint share
    ``num_q_heads``/``num_kv_heads``/``head_dim``/``page_size``, so one
    instance is correct as well as cheaper.
    """

    def __init__(
        self,
        *,
        batch: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        pages_per_slot: int,
        num_cache_pages: int,
        dtype: torch.dtype,
        kv_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.batch = batch
        self.device = device
        self.page_size = page_size
        # Fallback descale used when a caller's forward() doesn't pass its
        # own -- BF16 KV's harmless no-op (1.0). This driver is SHARED
        # across every full-attention layer for a given batch size (class
        # docstring), but real per-layer k_scale/v_scale differ across
        # layers (see Qwen36AttentionWorkspace's own comment on the same
        # point), so the real per-layer descale is a forward()-time
        # argument here, not something this driver can hold fixed at
        # construction like the old single `self._descale` used to.
        self._default_descale = torch.ones(1, dtype=torch.float32, device=device)
        self.page_table = torch.zeros(batch, pages_per_slot, dtype=torch.int32, device=device)
        self.cache_seqlens = torch.ones(batch, dtype=torch.int32, device=device)
        self.cu_seqlens_q = torch.arange(batch + 1, dtype=torch.int32, device=device)
        # ``eager_extend_work_items_capacity`` is the WRONG estimator here and
        # says so in its name: it scales with ``max_total_q * gqa / 16``, which
        # for a decode step is one work item per request and ignores the KV
        # axis entirely. A batch-2 decode over a 512-token context blew
        # straight through it ("fixed-capacity paged workspace exceeded",
        # measured 2026-08-02). ``plan_decode_graph_capacity`` is sparkinfer's
        # own worst-case policy for exactly this shape -- a decode bucket of a
        # given batch over a given page count -- and returns ``max_work_items``
        # and ``max_partial_rows`` as a consistent pair, which matters because
        # a split-KV plan needs partial rows to merge into and a hand-picked
        # ``max_partial_rows=0`` silently forbids the schedule the planner is
        # about to ask for.
        if device.type == "cuda":
            capacity = plan_decode_graph_capacity(
                device=device,
                q_dtype=dtype,
                kv_dtype=kv_dtype,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                head_dim_vo=head_dim,
                page_size=page_size,
                batch=batch,
                max_cache_page_count=num_cache_pages,
                window_left=-1,
            )
            max_work_items = capacity.max_work_items
            max_partial_rows = capacity.max_partial_rows
        else:
            # CPU: this driver is never actually run (the kernels are
            # CUDA-only); it exists so the pool's bookkeeping is testable.
            # plan_decode_graph_capacity refuses a non-CUDA device outright,
            # so fall back to a value that only has to be self-consistent.
            max_work_items = max(batch, 1)
            max_partial_rows = 0
        self._workspace = PagedAttentionWorkspace.for_fixed_capacity(
            mode="decode",
            device=device,
            dtype=dtype,
            kv_dtype=kv_dtype,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            head_dim_vo=head_dim,
            page_size=page_size,
            max_total_q=batch,
            max_batch=batch,
            max_page_table_width=pages_per_slot,
            max_work_items=max_work_items,
            max_partial_rows=max_partial_rows,
            num_cache_pages=num_cache_pages,
            use_cuda_graph=False,
        )
        self._plan_budget = PagedPlanBudget(
            max_total_q=batch,
            max_batch=batch,
            max_page_table_width=pages_per_slot,
        )

    def forward(
        self,
        *,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        output: torch.Tensor,
        k_descale: torch.Tensor | None = None,
        v_descale: torch.Tensor | None = None,
    ) -> None:
        """``k_descale``/``v_descale`` default to the harmless 1.0 no-op
        (BF16 KV); a caller with an FP8 KV pool passes its own layer's
        real ``k_scale``/``v_scale`` Parameters here instead (see class
        docstring -- this driver is shared across layers, so the real
        descale cannot live on ``self``)."""
        plan = create_paged_plan(
            q,
            k_cache,
            v_cache,
            self.page_table,
            self.cache_seqlens,
            self.cu_seqlens_q,
            mode="decode",
            enable_cuda_graph=False,
            window_left=-1,
            plan_budget=self._plan_budget,
        )
        ws = self._workspace
        ws._ensure_capacity(plan)
        ws._copy_runtime_metadata(self.page_table, self.cache_seqlens, self.cu_seqlens_q)
        ws._copy_plan_metadata(plan)
        ws._plan = plan
        binding = build_paged_attention_binding(
            scratch=ws,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            output=output,
            k_descale=k_descale if k_descale is not None else self._default_descale,
            v_descale=v_descale if v_descale is not None else self._default_descale,
        )
        paged_attention_forward(binding=binding)


class Qwen36BatchedExtendAttention:
    """Shared paged-attention driver for a uniform batched prefill.

    Unlike decode, prefill is not CUDA-graph replayed: the scheduler may
    submit a different prompt length on every admission.  It is still worth
    sharing the driver across all full-attention layers in a single forward:
    the 16 layers have identical attention geometry, while the page table,
    sequence lengths, and query offsets are also identical.  This is the
    no-vLLM counterpart of the historical ``_forward_batch(...,
    is_decode=False)`` metadata construction.
    """

    def __init__(
        self,
        *,
        batch: int,
        tokens_per_slot: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        pages_per_slot: int,
        num_cache_pages: int,
        dtype: torch.dtype,
        kv_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if batch < 1 or tokens_per_slot < 1:
            raise ValueError("batched extend requires positive batch and tokens_per_slot")
        self.batch = batch
        self.tokens_per_slot = tokens_per_slot
        self.device = device
        self._default_descale = torch.ones(1, dtype=torch.float32, device=device)
        self.page_table = torch.zeros(batch, pages_per_slot, dtype=torch.int32, device=device)
        self.cache_seqlens = torch.zeros(batch, dtype=torch.int32, device=device)
        self.cu_seqlens_q = torch.arange(batch + 1, dtype=torch.int32, device=device).mul_(
            tokens_per_slot
        )
        total_q = batch * tokens_per_slot
        # Decoder layers execute sequentially, so one output scratch is
        # sufficient for every full-attention layer in this prefill. Keeping
        # one per layer would retain 16 copies of a BxQxH tensor and turn a
        # throughput optimisation into a long-context OOM.
        self.output = torch.empty(total_q, num_q_heads, head_dim, dtype=dtype, device=device)
        self._flashinfer: FlashInferPagedPrefill | None = None
        self._prefill_kv_lengths: tuple[int, ...] | None = None
        self._prefill_metadata_generation = 0
        requested_backend = os.environ.get(QSR_QWEN36_PREFILL_ATTN_BACKEND_ENV, "auto").lower()
        if requested_backend not in {"auto", "flashinfer", "fa2", "sparkinfer", "b12x"}:
            raise ValueError(
                f"{QSR_QWEN36_PREFILL_ATTN_BACKEND_ENV} must be one of "
                f"auto/flashinfer/fa2/sparkinfer/b12x, got {requested_backend!r}"
            )
        if device.type == "cuda" and requested_backend in {"auto", "flashinfer", "fa2"}:
            try:
                self._flashinfer = FlashInferPagedPrefill(
                    batch=batch,
                    tokens_per_slot=tokens_per_slot,
                    num_q_heads=num_q_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    page_size=page_size,
                    pages_per_slot=pages_per_slot,
                    num_cache_pages=num_cache_pages,
                    dtype=dtype,
                    kv_dtype=kv_dtype,
                    device=device,
                )
            except (RuntimeError, ValueError, ImportError) as exc:
                if requested_backend in {"flashinfer", "fa2"}:
                    raise
                logger.warning(
                    "Qwen3.6 FlashInfer FA2 prefill unavailable; using SparkInfer: %s", exc
                )
        self.prefill_backend = "flashinfer" if self._flashinfer is not None else "sparkinfer"
        self._workspace: PagedAttentionWorkspace | None = None
        self._plan_budget: PagedPlanBudget | None = None
        if self._flashinfer is None:
            max_work_items = PagedAttentionWorkspace.eager_extend_work_items_capacity(
                max_total_q=total_q, num_q_heads=num_q_heads, num_kv_heads=num_kv_heads
            )
            self._workspace = PagedAttentionWorkspace.for_fixed_capacity(
                mode="extend",
                device=device,
                dtype=dtype,
                kv_dtype=kv_dtype,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                head_dim_vo=head_dim,
                page_size=page_size,
                max_total_q=total_q,
                max_batch=batch,
                max_page_table_width=pages_per_slot,
                max_work_items=max_work_items,
                max_partial_rows=0,
                num_cache_pages=num_cache_pages,
                use_cuda_graph=False,
            )
            self._plan_budget = PagedPlanBudget(
                max_total_q=total_q,
                max_batch=batch,
                max_page_table_width=pages_per_slot,
            )

    def set_kv_lengths(self, lengths: list[int] | tuple[int, ...]) -> None:
        """Record host-known KV lengths for FlashInfer's compact page list."""
        normalized = tuple(int(length) for length in lengths)
        if len(normalized) != self.batch:
            raise ValueError(
                f"batched extend expected {self.batch} KV lengths, got {len(normalized)}"
            )
        self._prefill_kv_lengths = normalized
        self._prefill_metadata_generation += 1

    def warmup_flashinfer(
        self,
        *,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_table: torch.Tensor,
        kv_length: int,
    ) -> None:
        """Compile and run the configured FA2 shape before serving traffic."""
        if self._flashinfer is None:
            return
        self.page_table.copy_(page_table[: self.batch])
        self.cache_seqlens.fill_(kv_length)
        self.set_kv_lengths([kv_length] * self.batch)
        self._flashinfer.run(
            q=torch.zeros_like(self.output),
            k_cache=k_cache,
            v_cache=v_cache,
            output=self.output,
            page_table=self.page_table,
            kv_lengths=self._prefill_kv_lengths,
            qo_indptr=self.cu_seqlens_q,
            generation=self._prefill_metadata_generation,
            k_scale=None,
            v_scale=None,
        )

    def forward(
        self,
        *,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        output: torch.Tensor,
        k_descale: torch.Tensor | None = None,
        v_descale: torch.Tensor | None = None,
    ) -> None:
        total_q = self.batch * self.tokens_per_slot
        if q.shape[0] != total_q:
            raise ValueError(f"batched extend expected {total_q} query rows, got {q.shape[0]}")
        if self._flashinfer is not None:
            if self._prefill_kv_lengths is None:
                raise RuntimeError(
                    "FlashInfer prefill metadata was not initialized; "
                    "call set_kv_lengths before the first forward"
                )
            self._flashinfer.run(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                output=output,
                page_table=self.page_table,
                kv_lengths=self._prefill_kv_lengths,
                qo_indptr=self.cu_seqlens_q,
                generation=self._prefill_metadata_generation,
                k_scale=k_descale,
                v_scale=v_descale,
            )
            return
        plan = create_paged_plan(
            q,
            k_cache,
            v_cache,
            self.page_table,
            self.cache_seqlens,
            self.cu_seqlens_q,
            mode="extend",
            enable_cuda_graph=False,
            window_left=-1,
            plan_budget=self._plan_budget,
        )
        ws = self._workspace
        assert ws is not None
        ws._ensure_capacity(plan)
        ws._copy_runtime_metadata(self.page_table, self.cache_seqlens, self.cu_seqlens_q)
        ws._copy_plan_metadata(plan)
        ws._plan = plan
        binding = build_paged_attention_binding(
            scratch=ws,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            output=output,
            k_descale=k_descale if k_descale is not None else self._default_descale,
            v_descale=v_descale if v_descale is not None else self._default_descale,
        )
        paged_attention_forward(binding=binding)


class Qwen36DecodeGraphAttention:
    """The same driver, in sparkinfer's CUDA-Graph replay mode (B2).

    Why a second class instead of a flag on the first: the eager driver
    calls ``create_paged_plan`` on every forward, and that planner reads
    tensor *contents* on the host. Inside a capture that is not merely
    slow, it raises -- which is the good outcome; the bad one would be a
    planner that silently baked one step's schedule into a graph replayed
    forever. sparkinfer already provides the alternative
    (``prepare_decode_graph_replay_state`` + metadata updated by a device
    kernel from ``cache_seqlens``), and ``SparkinferDecodeWorkspace`` in
    ``runtime/backends/laguna_sparkinfer_attn.py`` is the same shape for
    Laguna's geometry -- this is that pattern at head_dim=256 / BF16 KV /
    batch>1, which B0-3 explicitly left untested and handed to B2
    ("未实测 ``use_cuda_graph=True``，留给 B2").

    Per-step cost is what it is for Laguna: int32 writes into
    :attr:`cache_seqlens` / :attr:`page_table`, then ``graph.replay()``.
    """

    def __init__(
        self,
        *,
        batch: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        pages_per_slot: int,
        num_cache_pages: int,
        max_seq_len: int,
        dtype: torch.dtype,
        kv_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.batch = batch
        self.device = device
        # See Qwen36BatchedDecodeAttention's matching comment: this driver
        # is also shared across every full-attention layer, so the real
        # per-layer descale is a forward()-time argument, not fixed here.
        self._default_descale = torch.ones(1, dtype=torch.float32, device=device)
        self._workspace = PagedAttentionWorkspace.for_contract(
            mode="decode",
            device=device,
            dtype=dtype,
            kv_dtype=kv_dtype,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            head_dim_vo=head_dim,
            page_size=page_size,
            max_total_q=batch,
            num_cache_pages=num_cache_pages,
            use_cuda_graph=True,
        )
        # ``max_cache_page_count`` is per REQUEST, not the physical pool size:
        # it bounds how many pages one sequence's page-table row may address,
        # and sparkinfer plans the replay schedule against
        # ``max_cache_page_count * page_size`` tokens. Passing the pool total
        # here raised "page_table width is smaller than required by
        # cache_seqlens" (measured 2026-08-02) -- the planner asked for a
        # 12-page row out of a 4-page-wide table. ``num_cache_pages`` above is
        # the other number, the physical page count the kernel may index into,
        # and that one IS the pool total.
        self._workspace.prepare_decode_graph_replay_state(
            batch=batch,
            max_page_table_width=pages_per_slot,
            total_q_capacity=batch,
            max_cache_page_count=pages_per_slot,
            window_left=-1,
        )
        # Bind the WORST case before capture, not a representative one: the
        # graph's schedule is fixed at capture time, so a shorter context
        # captured here would silently under-serve every longer one later.
        capture_page_table = (
            torch.arange(pages_per_slot, dtype=torch.int32, device=device)
            .unsqueeze(0)
            .repeat(batch, 1)
        )
        capture_cache_seqlens = torch.full((batch,), max_seq_len, dtype=torch.int32, device=device)
        self.cu_seqlens_q = torch.arange(batch + 1, dtype=torch.int32, device=device)
        self._workspace._copy_runtime_metadata(
            capture_page_table, capture_cache_seqlens, self.cu_seqlens_q
        )
        # From here on these ARE the buffers the caller writes each step.
        self.page_table = self._workspace.page_table
        self.cache_seqlens = self._workspace.cache_seqlens
        self.page_size = page_size

    def update_replay_metadata(self) -> None:
        """Re-derive the split-KV chunking from the LIVE cache lengths.

        The graph's grid and work-item buffers are capture-static, but the
        chunk size itself is a device scalar the replay updater rewrites
        from ``cache_seqlens`` through the decode chunk-pages LUT. Without
        this call every replay keeps the worst-case (full-context) chunking
        captured at load -- measured 2026-08-06 as ~1.0-1.1 ms/call at
        128K q=1 where the live-length chunking runs ~0.4 ms
        (scripts/probe_draft_graph_attn.py).
        """
        self._workspace.update_decode_graph_replay_metadata_from_runtime_cache_seqlens()

    def forward(
        self,
        *,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        output: torch.Tensor,
        k_descale: torch.Tensor | None = None,
        v_descale: torch.Tensor | None = None,
    ) -> None:
        """See :meth:`Qwen36BatchedDecodeAttention.forward`'s matching
        docstring -- same default-to-1.0, caller-supplies-the-real-scale
        contract. Safe under CUDA Graph capture/replay: whichever tensor
        object is passed becomes part of ``binding`` and the kernel reads
        its address directly (``sparkinfer.attention.paged._scratch.
        build_paged_attention_binding`` stores it, no intermediate copy
        into shared scratch) -- since a real caller always passes the SAME
        Parameter object (never reallocated after load), capture bakes in
        the correct per-layer address and replay keeps reading it."""
        binding = build_paged_attention_binding(
            scratch=self._workspace,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            output=output,
            k_descale=k_descale if k_descale is not None else self._default_descale,
            v_descale=v_descale if v_descale is not None else self._default_descale,
        )
        paged_attention_forward(binding=binding)


class Qwen36VerifyGraphAttention:
    """Fixed-capacity ``mode="verify"`` paged-attention replay driver.

    ``Qwen36AttentionWorkspace`` deliberately remains the eager extend path;
    this small driver is the graph-only counterpart used by MTP. Sparkinfer's
    prefill replay metadata updater is called before every replay, so the
    captured plan is capacity-sized while page ids, cache length, and query
    offsets remain round-specific.
    """

    def __init__(
        self,
        *,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        pages_per_slot: int,
        num_cache_pages: int,
        max_seq_len: int,
        verify_tokens: int,
        batch: int = 1,
        dtype: torch.dtype,
        kv_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.verify_tokens = verify_tokens
        self.batch = batch
        self.device = device
        self.page_size = page_size
        self._default_descale = torch.ones(1, dtype=torch.float32, device=device)
        self._workspace = PagedAttentionWorkspace.for_contract(
            mode="verify",
            device=device,
            dtype=dtype,
            kv_dtype=kv_dtype,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            head_dim_vo=head_dim,
            page_size=page_size,
            max_total_q=batch * verify_tokens,
            num_cache_pages=num_cache_pages,
            use_cuda_graph=True,
        )
        # arange(B+1) * qo_len -- every slot in one spec-decode verify submits
        # the SAME k+1 tokens, so the query offsets are uniform and this is not
        # a ragged batch. That is the historical contract verbatim
        # (`spec_query_start_loc` in oracle/.../metadata_builders.py:524's
        # build_gdn_metadata_spec_batch, which states qo_len "is always
        # num_spec + 1" and is "NOT generalized to a ragged per-request list").
        self._cu_seqlens_q = (
            torch.arange(batch + 1, dtype=torch.int32, device=device) * verify_tokens
        )
        self._workspace.prepare_prefill_graph_replay_state(
            batch=batch,
            total_q_capacity=batch * verify_tokens,
            max_page_table_width=pages_per_slot,
            max_cache_seqlen=max_seq_len,
            cu_seqlens_q=self._cu_seqlens_q,
            window_left=-1,
        )
        self.page_table = self._workspace.page_table
        self.cache_seqlens = self._workspace.cache_seqlens

    def update_metadata(
        self,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        host_cache_seqlens: object | None = None,
    ) -> None:
        replay_page_key = None
        if host_cache_seqlens is not None:
            # Host-known lengths (the verify fill has already written them
            # into the pinned staging array): the split-KV worklist is a pure
            # function of the per-request page count, so the workspace can
            # skip its three Triton rebuilds while the count is unchanged.
            replay_page_key = (
                len(host_cache_seqlens),
                tuple(
                    (int(seq_len) + self.page_size - 1) // self.page_size
                    for seq_len in host_cache_seqlens
                ),
            )
        self._workspace.update_prefill_graph_replay_metadata(
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            replay_page_key=replay_page_key,
        )

    def forward(
        self,
        *,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        output: torch.Tensor,
        k_descale: torch.Tensor | None = None,
        v_descale: torch.Tensor | None = None,
    ) -> None:
        binding = build_paged_attention_binding(
            scratch=self._workspace,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            output=output,
            k_descale=k_descale if k_descale is not None else self._default_descale,
            v_descale=v_descale if v_descale is not None else self._default_descale,
        )
        paged_attention_forward(binding=binding)


#: sparkinfer's one FP8 KV-cache storage dtype (its own module docstring:
#: "BF16/FP16 queries, BF16/FP16/FP8-e4m3 KV cache"). Named here so every
#: FP8-KV-aware branch below spells the same torch dtype object, not a
#: string literal repeated at each call site.
_FP8_KV_DTYPE = torch.float8_e4m3fn


def _kv_to_cache_dtype(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    cache_dtype: torch.dtype,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cast ``(key, value)`` to ``cache_dtype`` for KV-cache storage.

    BF16 cache (``cache_dtype != _FP8_KV_DTYPE``): a plain dtype cast --
    exactly what every call site did before FP8 KV existed, so the default
    (``enable_fp8_kv=False``) path is byte-for-byte unchanged.

    FP8 cache: scale-DIVIDE before casting -- ``fp8_stored = real_value /
    scale`` -- the same convention ``runtime/kernels/fused_kv_scatter.py``
    (``k_fp8 = (k_val / k_scale).to(tl.float8e4nv)``) and Laguna's own FP8
    KV write path use (``runtime/backends/laguna_cuda_graph.py``:
    ``k_cache[...] = (key / layer._k_scale).to(torch.float8_e4m3fn)``).
    This is the exact inverse of what sparkinfer's kernel does on READ via
    ``k_descale``/``v_descale`` (a multiply: ``real ~= fp8_stored *
    scale`` -- confirmed by Laguna's matching read-side convention,
    ``ws._k_descale = layer._k_scale.detach()``, no reciprocal), so
    ``k_descale``/``v_descale`` must always be handed these same
    ``k_scale``/``v_scale`` tensors DIRECTLY, never their reciprocal.
    Getting this inverted is exactly the class of bug this codebase has
    hit before with a different scale pair (``CompressedTensorsNVFP4Linear``'s
    ``weight_global_scale`` vs. modelopt's ``weight_scale_2`` -- see that
    class's docstring for what a missed reciprocal there looked like on a
    real GPU run: degenerate ``"!!!!!!!!!!!!"`` output) -- this codebase's
    scale conventions are genuinely inconsistent across formats, so this
    function's docstring states its own convention explicitly rather than
    assuming a reader infers it from a sibling.
    """
    if cache_dtype == _FP8_KV_DTYPE:
        assert k_scale is not None and v_scale is not None, (
            "FP8 KV cache requires real k_scale/v_scale tensors -- a cache built "
            "with cache_dtype=float8_e4m3fn but k_scale/v_scale=None is a "
            "construction bug (Qwen36Attention.__init__ always creates both "
            "together), not a runtime condition to silently degrade for"
        )
        return (key / k_scale).to(cache_dtype), (value / v_scale).to(cache_dtype)
    return key.to(cache_dtype), value.to(cache_dtype)


class Qwen36Attention(nn.Module):
    """Transcribed from ``Qwen3_5Attention``. Uses sparkinfer's paged
    attention kernel (B0-3-verified for this exact shape). KV cache dtype
    is BF16 by default; ``enable_fp8_kv=True`` (module docstring's
    2026-08-03 update -- gated by ``QSR_QWEN36_FP8_KV``, resolved once by
    ``runtime.model_loading.load_qwen36_model`` and threaded down
    explicitly rather than re-read from the environment per layer) makes
    it FP8-e4m3, consuming the standard checkpoint's real per-layer
    ``k_scale``/``v_scale`` tensors.
    """

    def __init__(
        self,
        config: dict[str, Any],
        layer_idx: int,
        quantized: dict[str, str],
        *,
        max_seq_len: int,
        weight_prefix: str | None = None,
        enable_fp8_kv: bool = False,
        kv_scale_buffer: bool = False,
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

        # FP8 KV (2026-08-03 follow-up, default OFF -- see module
        # docstring): `kv_cache_dtype` is this layer's own KV storage
        # dtype, consulted by `new_cache`/`Qwen36SlotPool` instead of the
        # ambient compute dtype the two used to share unconditionally.
        # `k_scale`/`v_scale` are real Parameters ONLY when enabled --
        # named exactly like the checkpoint's own
        # `self_attn.{k,v}_scale` tensors (no remapping needed:
        # `load_weights`'s `model.language_model.` -> `model.` prefix
        # strip already lines this module's dotted name up with the
        # checkpoint's), so a checkpoint that declares
        # QSR_QWEN36_FP8_KV=1 but ships neither tensor (e.g. the modelopt
        # checkpoint -- its quantization_config.kv_cache_scheme exists but
        # it ships ZERO k_scale/v_scale tensors, verified directly, see
        # module docstring) fails loudly at `assert_all_params_loaded`
        # rather than silently running FP8 KV with an unset 1.0 scale.
        self.enable_fp8_kv = enable_fp8_kv
        self.kv_cache_dtype = _FP8_KV_DTYPE if enable_fp8_kv else torch.bfloat16
        if enable_fp8_kv:
            if kv_scale_buffer:
                # Buffers instead of Parameters for callers whose checkpoint
                # ships no k_scale/v_scale tensors (the MTP draft head --
                # verified against its tensor list): buffers still move with
                # .cuda()/.to() and are consumed identically by the write and
                # descale paths, but they are invisible to
                # assert_all_params_loaded's named_parameters() sweep, so a
                # checkpoint that legitimately has no scales does not trip
                # the loader gate.
                self.register_buffer("k_scale", torch.ones(1, dtype=torch.float32))
                self.register_buffer("v_scale", torch.ones(1, dtype=torch.float32))
            else:
                self.k_scale = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=False)
                self.v_scale = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=False)
        else:
            self.k_scale = None
            self.v_scale = None

        # ``weight_prefix`` (B3): overrides the derived
        # ``model.language_model.layers.{layer_idx}`` checkpoint prefix used
        # for quantization classification (``_make_linear``'s ``dotted_name``
        # arg -- see ``classify_module``). Every real call site before B3
        # left this ``None`` (backbone full-attention layers, whose
        # checkpoint tensors really do live at that prefix); the MTP draft
        # head (``Qwen36MTPHead``) is the first caller to pass a real
        # override, because its checkpoint tensors live under ``mtp.layers.0``
        # instead -- reusing this class unmodified for the MTP head's
        # self-attention sublayer beats hand-duplicating its forward() (RoPE
        # + sparkinfer paged attention + sigmoid output gate), which has
        # already been through B1/B2 GPU verification for every OTHER
        # full-attention layer in this model.
        prefix = (
            weight_prefix
            if weight_prefix is not None
            else (f"model.language_model.layers.{layer_idx}.self_attn")
        )
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

        # Fused one-launch QKV W8A8 GEMM (native route only), built lazily on
        # first forward from the three projections' raw FP8 weights.  Kept as
        # a plain attribute so named_parameters()/state_dict and the loader
        # stay untouched.
        self._fused_qkv: FusedFP8ChannelQKV | None = None

        # Built lazily on first forward() call, from the actual observed
        # cache/dtype -- safe because every Qwen36GenerationState this
        # layer instance will ever see comes from the SAME model instance
        # (same max_seq_len, same cache.num_pages every time; a second
        # model instance with a different max_seq_len gets its own
        # Qwen36Attention layers, hence its own workspaces). See
        # Qwen36AttentionWorkspace's docstring for why this exists.
        #
        # 2026-08-15 (plan §4.5 P0-M2 step 1): the three per-layer
        # attributes below are retired in favor of the shared module-level
        # registry ``_SHARED_ATTN_WORKSPACES`` -- all 16 full-attention
        # layers share one workspace per mode, recovering ~795 MiB of
        # duplicate scratch. The attributes are kept as the *cache* of the
        # shared instance for this layer's own fast path; ``_workspace_for``
        # resolves through the registry keyed by full geometry.
        self._extend_workspace: Qwen36AttentionWorkspace | None = None
        self._decode_workspace: Qwen36AttentionWorkspace | None = None
        # Speculative verify (K>1 tokens against an existing KV span) is a
        # THIRD mode, not a short extend. sparkinfer distinguishes the two
        # throughout its planner, and routing verify onto the extend
        # workspace is what broke MTP under FP8 KV -- see
        # notes/2026-08-03-mtp-verify-mode.md.
        self._verify_workspace: Qwen36AttentionWorkspace | None = None
        # Declared by declare_verify_capacity() before any verify traffic.
        self._max_verify_query_len = 0

    def declare_verify_capacity(self, max_query_len: int) -> None:
        """Declare the largest ``mode="verify"`` query length this layer
        will ever be asked for (the speculator's K).

        Must be called before the first verify forward: the verify
        workspace is fixed-capacity and sparkinfer hard-fails rather than
        growing it, by design. Raising K after the workspace exists
        discards it so the next call rebuilds at the larger bound --
        silently keeping the old one would under-provision exactly the way
        notes/2026-08-01-c1-c2-gpu-investigation.md describes.
        """
        max_query_len = int(max_query_len)
        if max_query_len <= 0:
            raise ValueError("declare_verify_capacity: max_query_len must be positive")
        if max_query_len > self._max_verify_query_len:
            self._max_verify_query_len = max_query_len
            self._verify_workspace = None

    def _workspace_for(
        self,
        mode: str,
        cache: Qwen36PagedAttentionCache,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Qwen36AttentionWorkspace:
        attr = {
            "extend": "_extend_workspace",
            "decode": "_decode_workspace",
            "verify": "_verify_workspace",
        }[mode]
        existing = getattr(self, attr)
        if existing is not None:
            return existing
        # Shared registry (plan §4.5 P0-M2 step 1): every layer with the
        # same geometry and capacity reuses the same fixed-capacity arena;
        # per-layer K/V descale is passed at forward time.
        key = _shared_attention_workspace_key(
            mode=mode,
            layer=self,
            cache=cache,
            dtype=dtype,
            device=device,
        )
        workspace = _SHARED_ATTN_WORKSPACES.get(key)
        if workspace is None:
            workspace = Qwen36AttentionWorkspace(
                mode=mode,
                num_q_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                page_size=cache.page_size,
                max_total_q=self.max_seq_len,
                max_page_table_width=cache.num_pages,
                num_cache_pages=cache.physical_num_pages,
                dtype=dtype,
                kv_dtype=cache.k_cache.dtype,
                device=device,
                # None when FP8 KV is disabled -- Qwen36AttentionWorkspace
                # treats None as "use the harmless 1.0 no-op descale", same
                # as before this parameter existed.
                k_descale=self.k_scale,
                v_descale=self.v_scale,
                # Only consulted for mode="verify"; see _verify_capacity.
                max_verify_query_len=self._max_verify_query_len,
                k_cache=cache.k_cache,
                v_cache=cache.v_cache,
            )
            _SHARED_ATTN_WORKSPACES[key] = workspace
        setattr(self, attr, workspace)
        return workspace

    def new_cache(self, *, device: torch.device, dtype: torch.dtype) -> Qwen36PagedAttentionCache:
        """KV storage dtype is this layer's own ``kv_cache_dtype`` (BF16
        unless FP8 KV was enabled for this layer -- see ``__init__``), not
        the compute ``dtype`` this method used to be handed by every call
        site (they still pass one, always the model's BF16 compute dtype,
        for backward call-site compatibility, but it is unused now that
        the two dtypes can diverge)."""
        del dtype
        return Qwen36PagedAttentionCache(
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            dtype=self.kv_cache_dtype,
            device=device,
        )

    def _kv_nvfp4_roundtrip(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        nvfp4_k_codes: torch.Tensor | None,
        nvfp4_k_scales: torch.Tensor | None,
        nvfp4_v_codes: torch.Tensor | None,
        nvfp4_v_scales: torch.Tensor | None,
        write_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """NVFP4 KV write path (notes/2026-08-16-nvfp4-kv-plan.md S3):
        pack ``key``/``value`` into the packed pools at ``write_index`` and
        return the dequantized values so the caller can keep the fp8 shadow
        pool (b12x prefill reads) coherent.  Returns None (no-op) unless
        the pools were allocated (QSR_QWEN36_NVFP4_KV=1).
        """
        if nvfp4_k_codes is None or write_index is None:
            return None
        from runtime.kernels.nvfp4_kv import pack_nvfp4_kv_into_pool, unpack_nvfp4_kv

        if os.environ.get("QSR_NVFP4_ATTN_DEBUG", "0") == "1":
            import time as _t

            _t0 = _t.perf_counter()
        pack_nvfp4_kv_into_pool(
            key,
            value,
            nvfp4_k_codes,
            nvfp4_k_scales,
            nvfp4_v_codes,
            nvfp4_v_scales,
            write_index,
            head_dim=self.head_dim,
        )
        rows = write_index.shape[0]
        kv_heads = self.num_kv_heads
        k_codes_rows = nvfp4_k_codes.view(-1, kv_heads, self.head_dim // 2)[
            write_index
        ].reshape(-1, self.head_dim // 2)
        k_scales_rows = nvfp4_k_scales.view(-1, kv_heads, self.head_dim // 16)[
            write_index
        ].reshape(-1, self.head_dim // 16)
        v_codes_rows = nvfp4_v_codes.view(-1, kv_heads, self.head_dim // 2)[
            write_index
        ].reshape(-1, self.head_dim // 2)
        v_scales_rows = nvfp4_v_scales.view(-1, kv_heads, self.head_dim // 16)[
            write_index
        ].reshape(-1, self.head_dim // 16)
        k_un = unpack_nvfp4_kv(k_codes_rows, k_scales_rows).to(torch.bfloat16).view(
            rows, kv_heads, self.head_dim
        )
        v_un = unpack_nvfp4_kv(v_codes_rows, v_scales_rows).to(torch.bfloat16).view(
            rows, kv_heads, self.head_dim
        )
        if os.environ.get("QSR_NVFP4_ATTN_DEBUG", "0") == "1":
            print(f"[nvfp4-write] rows={rows} {( _t.perf_counter()-_t0)*1e3:.2f}ms", flush=True)
        return k_un, v_un

    def _qkv_proj(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """q/gate, k, v projections -- one fused native W8A8 launch when possible.

        Mirrors the three projections' own ``forward_native_w8a8`` routing
        condition (CUDA + native W8A8 enabled + all-layers raw execution);
        anything else (fallback env, missing raw FP8 weights, CPU) falls
        back to the three separate modules so behavior is unchanged.  The
        fused path is bit-exact with the three-GEMM path (same per-column
        dots and scales, deterministic shared activation quantizer).
        """
        if (
            hidden_states.device.type == "cuda"
            and _native_w8a8_fp8_channel_enabled()
            and fp8_channel_raw_execution_uses_all_layers()
        ):
            if self._fused_qkv is None:
                try:
                    if not all(
                        isinstance(p, CompressedTensorsFP8ChannelLinear)
                        for p in (self.q_proj, self.k_proj, self.v_proj)
                    ):
                        raise TypeError("fused QKV requires three FP8-channel projections")
                    self._fused_qkv = FusedFP8ChannelQKV(self.q_proj, self.k_proj, self.v_proj)
                    self._fused_qkv._ensure()
                except (RuntimeError, ValueError, TypeError, AttributeError):
                    self._fused_qkv = None
            if self._fused_qkv is not None:
                library = _native_w8a8_library_for_cuda()
                if library is not None:
                    return self._fused_qkv.forward_native(hidden_states, library)
        return (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        cache: Qwen36PagedAttentionCache,
        paged_mode: str | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        assert batch_size == 1

        q_and_gate, key_raw, value_raw = self._qkv_proj(hidden_states)
        q_and_gate = q_and_gate.view(batch_size, seq_len, self.num_heads, self.head_dim * 2)
        query, gate = torch.chunk(q_and_gate, 2, dim=-1)
        gate = gate.reshape(batch_size, seq_len, -1)

        kv_shape = (batch_size, seq_len, self.num_kv_heads, self.head_dim)
        query = self.q_norm(query)
        key = self.k_norm(key_raw.view(*kv_shape))
        value = value_raw.view(*kv_shape)

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

        _rt = self._kv_nvfp4_roundtrip(
            key, value, None, None, None, None, write_index=None
        )
        if _rt is not None:
            key, value = _rt
        k_to_store, v_to_store = _kv_to_cache_dtype(
            key, value, cache_dtype=cache.dtype, k_scale=self.k_scale, v_scale=self.v_scale
        )
        _past_len, total_len = cache.append(k_to_store, v_to_store)
        # paged_mode is passed explicitly by speculative verify, which is a
        # K>1 query against an existing KV span -- shape-indistinguishable
        # from a K-token prefill here, but a different sparkinfer mode with
        # a different plan and different kernel-policy flags. Inferring it
        # from seq_len alone silently routed verify onto extend; see
        # notes/2026-08-03-mtp-verify-mode.md.
        mode = paged_mode or ("decode" if seq_len == 1 else "extend")

        workspace = self._workspace_for(mode, cache, query.dtype, query.device)
        output = torch.empty(
            seq_len, self.num_heads, self.head_dim, dtype=query.dtype, device=query.device
        )
        cache_seqlens = torch.tensor([total_len], dtype=torch.int32, device=query.device)
        cu_seqlens_q = torch.tensor([0, seq_len], dtype=torch.int32, device=query.device)
        # Query stays in its own compute dtype (BF16) regardless of the KV
        # cache's dtype -- sparkinfer's paged-attention kernel wants
        # "BF16/FP16 queries, BF16/FP16/FP8-e4m3 KV cache (FP8 KV needs
        # BF16 queries + k/v descales)" (sparkinfer/attention/paged/
        # __init__.py's own module docstring), never a query cast to match
        # the cache. Casting query to cache.dtype here (the old
        # `needs_cast`/`q_for_kernel`) was latently harmless only because
        # cache.dtype always equaled query.dtype before FP8 KV existed --
        # it would have been actively wrong with FP8 KV (casting a BF16
        # query down to FP8), and the workspace's own scratch independently
        # enforces `q.dtype == self.dtype`, constructed as the BF16 compute
        # dtype, never kv_dtype (`Qwen36AttentionWorkspace.forward`'s
        # `_workspace._validate_static_shapes`).
        workspace.forward(
            q=query,
            k_cache=cache.k_cache,
            v_cache=cache.v_cache,
            output=output,
            page_table=cache.page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            k_descale=self.k_scale,
            v_descale=self.v_scale,
        )

        attn_out = output.reshape(batch_size, seq_len, -1).contiguous()
        attn_out = attn_out * torch.sigmoid(gate)
        return self.o_proj(attn_out)

    def decode_batch(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        *,
        k_pool: torch.Tensor,
        v_pool: torch.Tensor,
        write_index: torch.Tensor,
        attn: Qwen36BatchedDecodeAttention | Qwen36DecodeGraphAttention,
        output: torch.Tensor,
        nvfp4_k_codes: torch.Tensor | None = None,
        nvfp4_k_scales: torch.Tensor | None = None,
        nvfp4_v_codes: torch.Tensor | None = None,
        nvfp4_v_scales: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One decode step for ``B`` sequences sharing this layer's KV pool.

        Unlike :meth:`forward`, which owns a single sequence's cache object
        and appends through it, this takes the **whole layer pool** plus
        pre-computed global addresses:

        * ``k_pool``/``v_pool``: ``[num_pages_total, page_size,
          num_kv_heads, head_dim]`` -- every slot's pages in one
          allocation.
        * ``write_index``: ``[B]`` int64, flat row index into
          ``k_pool.view(-1, num_kv_heads, head_dim)`` for this step's new
          token -- i.e. ``global_page * page_size + offset``. Computed by
          the caller because it depends only on each slot's ``kv_len``,
          which the slot bookkeeping already owns; doing it here would
          mean handing this layer slot identities it otherwise never sees.
        * ``attn``: the step's shared attention driver, which owns
          ``page_table`` (``[B, pages_per_slot]`` int32, **global** page
          ids) and ``cache_seqlens`` (``[B]`` int32, each sequence's length
          **including** the token written this step).

        ``output`` and ``attn`` are caller-owned so a CUDA Graph capture
        can pin them; every tensor this method touches is either an
        argument or derived by a shape-stable op, so nothing here
        allocates a buffer whose address a replay could invalidate.
        """
        batch_size, seq_len, _ = hidden_states.shape
        assert seq_len == 1, "decode_batch is the single-token continuation path"

        q_and_gate, key_raw, value_raw = self._qkv_proj(hidden_states)
        q_and_gate = q_and_gate.view(batch_size, self.num_heads, self.head_dim * 2)
        query, gate = torch.chunk(q_and_gate, 2, dim=-1)
        gate = gate.reshape(batch_size, 1, -1)

        kv_shape = (batch_size, self.num_kv_heads, self.head_dim)
        query = self.q_norm(query)
        key = self.k_norm(key_raw.view(*kv_shape))
        value = value_raw.view(*kv_shape)

        query_flat = query.reshape(batch_size, self.num_heads * self.head_dim).contiguous()
        key_flat = key.reshape(batch_size, self.num_kv_heads * self.head_dim).contiguous()
        apply_rotary_embedding_inplace(positions, query_flat, self.head_dim, cos_sin_cache)
        apply_rotary_embedding_inplace(positions, key_flat, self.head_dim, cos_sin_cache)
        query = query_flat.view(batch_size, self.num_heads, self.head_dim)
        key = key_flat.view(batch_size, self.num_kv_heads, self.head_dim)

        k_flat = k_pool.view(-1, self.num_kv_heads, self.head_dim)
        v_flat = v_pool.view(-1, self.num_kv_heads, self.head_dim)
        _rt = self._kv_nvfp4_roundtrip(
            key, value, nvfp4_k_codes, nvfp4_k_scales, nvfp4_v_codes, nvfp4_v_scales, write_index
        )
        if _rt is not None:
            key, value = _rt
        k_to_store, v_to_store = _kv_to_cache_dtype(
            key, value, cache_dtype=k_pool.dtype, k_scale=self.k_scale, v_scale=self.v_scale
        )
        # Plain advanced-indexing assignment, NOT index_copy_: measured
        # directly (2026-08-03, a real FP8 KV server load) that
        # `index_copy_cuda` raises `NotImplementedError` for
        # `Float8_e4m3fn` on this torch build, while `tensor[idx] = value`
        # (dispatches to `aten::index_put_`) does not -- this is exactly
        # the op :class:`Qwen36PagedAttentionCache`'s own `append` already
        # uses successfully for FP8 (`self.k_cache[page_ids, offsets] =
        # new_k.to(self.dtype)`, proven correct by the B1-R gap-error gate
        # on GPU). Semantically identical here: `write_index` has one
        # distinct row per batch entry, never a repeat, so index_copy_'s
        # only behavioral edge over index_put_ (last-write-wins on
        # duplicate indices) never applies.
        k_flat[write_index] = k_to_store
        v_flat[write_index] = v_to_store

        # Query stays BF16 regardless of k_pool's dtype -- see forward()'s
        # matching comment for why casting it to the pool's dtype (the old
        # `needs_cast`/`q_for_kernel`) would be wrong for FP8 KV.
        if nvfp4_k_codes is not None:
            from runtime.kernels.nvfp4_decode_attn import nvfp4_decode_attention

            q2r = torch.arange(batch_size, device=query.device)
            qpos = attn.cache_seqlens - 1
            o = nvfp4_decode_attention(
                query,
                nvfp4_k_codes,
                nvfp4_k_scales,
                nvfp4_v_codes,
                nvfp4_v_scales,
                attn.page_table[:, 0] * attn.page_size,
                attn.cache_seqlens,
                q2r,
                qpos,
                num_q_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                gqa=self.num_heads // self.num_kv_heads,
            )
            output.copy_(o)
        else:
            attn.forward(
                q=query,
                k_cache=k_pool,
                v_cache=v_pool,
                output=output,
                k_descale=self.k_scale,
                v_descale=self.v_scale,
            )

        attn_out = output.reshape(batch_size, 1, -1)
        attn_out = attn_out * torch.sigmoid(gate)
        return self.o_proj(attn_out)

    def prefill_batch(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        *,
        k_pool: torch.Tensor,
        v_pool: torch.Tensor,
        write_index: torch.Tensor,
        attn: Qwen36BatchedExtendAttention,
        output: torch.Tensor,
        nvfp4_k_codes: torch.Tensor | None = None,
        nvfp4_k_scales: torch.Tensor | None = None,
        nvfp4_v_codes: torch.Tensor | None = None,
        nvfp4_v_scales: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Batched extend attention over globally addressed slot pages."""
        batch_size, seq_len, _ = hidden_states.shape
        if batch_size != attn.batch or seq_len != attn.tokens_per_slot:
            raise ValueError(
                "prefill batch shape does not match its attention driver: "
                f"got [{batch_size}, {seq_len}], expected [{attn.batch}, {attn.tokens_per_slot}]"
            )
        total_q = batch_size * seq_len
        q_and_gate, key_raw, value_raw = self._qkv_proj(hidden_states)
        use_fused_prepare = (
            hidden_states.is_cuda
            and hidden_states.dtype in (torch.bfloat16, torch.float16)
            and os.environ.get(QSR_QWEN36_FUSED_ATTN_PREP_ENV, "1") != "0"
        )
        if use_fused_prepare:
            try:
                from runtime.kernels.qwen36_fused_attn import (
                    fused_qk_rmsnorm_rope_gate,
                )

                q_gate = q_and_gate.reshape(total_q, self.num_heads, 2 * self.head_dim)
                key_raw_3d = key_raw.reshape(total_q, self.num_kv_heads, self.head_dim)
                query_flat, key_flat, gate_flat = fused_qk_rmsnorm_rope_gate(
                    q_gate,
                    key_raw_3d,
                    self.q_norm.weight,
                    self.k_norm.weight,
                    cos_sin_cache,
                    positions,
                    self.eps,
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    cos_sin_cache.shape[-1],
                )
                query = query_flat.view(total_q, self.num_heads, self.head_dim)
                key = key_flat.view(total_q, self.num_kv_heads, self.head_dim)
                value = value_raw.reshape(total_q, self.num_kv_heads, self.head_dim)
            except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
                logger.warning(
                    "Qwen3.6 fused attention prefill prepare unavailable at layer %s; "
                    "using fallback: %s",
                    self.layer_idx,
                    exc,
                )
                use_fused_prepare = False
        if not use_fused_prepare:
            q_and_gate = q_and_gate.view(
                batch_size, seq_len, self.num_heads, self.head_dim * 2
            )
            query, gate = torch.chunk(q_and_gate, 2, dim=-1)
            gate = gate.reshape(batch_size, seq_len, -1)
            kv_shape = (batch_size, seq_len, self.num_kv_heads, self.head_dim)
            query = self.q_norm(query).reshape(total_q, self.num_heads, self.head_dim)
            key = self.k_norm(key_raw.view(*kv_shape)).reshape(
                total_q, self.num_kv_heads, self.head_dim
            )
            value = value_raw.view(*kv_shape).reshape(
                total_q, self.num_kv_heads, self.head_dim
            )
            query_flat = query.reshape(total_q, self.num_heads * self.head_dim).contiguous()
            key_flat = key.reshape(total_q, self.num_kv_heads * self.head_dim).contiguous()
            apply_rotary_embedding_inplace(positions, query_flat, self.head_dim, cos_sin_cache)
            apply_rotary_embedding_inplace(positions, key_flat, self.head_dim, cos_sin_cache)
            query = query_flat.view(total_q, self.num_heads, self.head_dim)
            key = key_flat.view(total_q, self.num_kv_heads, self.head_dim)
        _rt = self._kv_nvfp4_roundtrip(
            key, value, nvfp4_k_codes, nvfp4_k_scales, nvfp4_v_codes, nvfp4_v_scales, write_index
        )
        if _rt is not None:
            key, value = _rt
        k_to_store, v_to_store = _kv_to_cache_dtype(
            key, value, cache_dtype=k_pool.dtype, k_scale=self.k_scale, v_scale=self.v_scale
        )
        k_pool.view(-1, self.num_kv_heads, self.head_dim)[write_index] = k_to_store
        v_pool.view(-1, self.num_kv_heads, self.head_dim)[write_index] = v_to_store
        attn.forward(
            q=query,
            k_cache=k_pool,
            v_cache=v_pool,
            output=output,
            k_descale=self.k_scale,
            v_descale=self.v_scale,
        )
        if use_fused_prepare:
            from runtime.kernels.qwen36_fused_attn import fused_sigmoid_mul

            fused_sigmoid_mul(output.reshape(total_q, -1), gate_flat)
            attn_out = output.reshape(batch_size, seq_len, -1)
        else:
            attn_out = output.reshape(batch_size, seq_len, -1) * torch.sigmoid(gate)
        return self.o_proj(attn_out)

    def verify_batch(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        *,
        k_pool: torch.Tensor,
        v_pool: torch.Tensor,
        write_index: torch.Tensor,
        attn: Qwen36VerifyGraphAttention,
        output: torch.Tensor,
        nvfp4_k_codes: torch.Tensor | None = None,
        nvfp4_k_scales: torch.Tensor | None = None,
        nvfp4_v_codes: torch.Tensor | None = None,
        nvfp4_v_scales: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """K-token graph-safe verify attention over a pooled KV cache.

        This mirrors :meth:`decode_batch`, but uses sparkinfer's graph-mode
        ``verify`` workspace and caller-owned K-row metadata. It deliberately
        does not touch a Python ``Qwen36PagedAttentionCache`` or its
        ``seq_len``; the MTP engine resolves the accepted prefix after all
        layers finish.
        """
        batch_size, seq_len, _ = hidden_states.shape
        assert batch_size == attn.batch and seq_len == attn.verify_tokens

        q_and_gate, key_raw, value_raw = self._qkv_proj(hidden_states)
        q_and_gate = q_and_gate.view(
            batch_size, seq_len, self.num_heads, self.head_dim * 2
        )
        query, gate = torch.chunk(q_and_gate, 2, dim=-1)
        gate = gate.reshape(batch_size, seq_len, -1)
        kv_shape = (batch_size, seq_len, self.num_kv_heads, self.head_dim)
        query = self.q_norm(query)
        key = self.k_norm(key_raw.view(*kv_shape))
        value = value_raw.view(*kv_shape)
        total_q = batch_size * seq_len
        query = query.reshape(total_q, self.num_heads, self.head_dim)
        key = key.reshape(total_q, self.num_kv_heads, self.head_dim)
        value = value.reshape(total_q, self.num_kv_heads, self.head_dim)

        # ``verify_batch`` is request-major ``[B, qo_len]``.  RoPE and the
        # paged-attention binding are token-major, so flatten *all* requests
        # rather than retaining the old B=1 ``seq_len`` shape here.
        query_flat = query.reshape(total_q, self.num_heads * self.head_dim).contiguous()
        key_flat = key.reshape(total_q, self.num_kv_heads * self.head_dim).contiguous()
        apply_rotary_embedding_inplace(positions, query_flat, self.head_dim, cos_sin_cache)
        apply_rotary_embedding_inplace(positions, key_flat, self.head_dim, cos_sin_cache)
        query = query_flat.view(total_q, self.num_heads, self.head_dim)
        key = key_flat.view(total_q, self.num_kv_heads, self.head_dim)

        _rt = self._kv_nvfp4_roundtrip(
            key, value, nvfp4_k_codes, nvfp4_k_scales, nvfp4_v_codes, nvfp4_v_scales, write_index
        )
        if _rt is not None:
            key, value = _rt
        k_to_store, v_to_store = _kv_to_cache_dtype(
            key, value, cache_dtype=k_pool.dtype, k_scale=self.k_scale, v_scale=self.v_scale
        )
        k_pool.view(-1, self.num_kv_heads, self.head_dim)[write_index] = k_to_store
        v_pool.view(-1, self.num_kv_heads, self.head_dim)[write_index] = v_to_store
        if nvfp4_k_codes is not None:
            from runtime.kernels.nvfp4_decode_attn import nvfp4_decode_attention

            qo_len = total_q // attn.batch
            q2r = torch.repeat_interleave(
                torch.arange(attn.batch, device=query.device), qo_len
            )
            qpos = torch.repeat_interleave(attn.cache_seqlens, qo_len) + torch.arange(
                qo_len, device=query.device
            ).repeat(attn.batch)
            o = nvfp4_decode_attention(
                query,
                nvfp4_k_codes,
                nvfp4_k_scales,
                nvfp4_v_codes,
                nvfp4_v_scales,
                attn.page_table[:, 0] * attn.page_size,
                attn.cache_seqlens,
                q2r,
                qpos,
                num_q_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                gqa=self.num_heads // self.num_kv_heads,
            )
            output.copy_(o)
        else:
            attn.forward(
                q=query,
                k_cache=k_pool,
                v_cache=v_pool,
                output=output,
                k_descale=self.k_scale,
                v_descale=self.v_scale,
            )

        attn_out = output.reshape(batch_size, seq_len, -1)
        attn_out = attn_out * torch.sigmoid(gate)
        return self.o_proj(attn_out)

    def verify_ragged_batch(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        *,
        k_pool: torch.Tensor,
        v_pool: torch.Tensor,
        write_index: torch.Tensor,
        attn: Qwen36VerifyGraphAttention,
        output: torch.Tensor,
        nvfp4_k_codes: torch.Tensor | None = None,
        nvfp4_k_scales: torch.Tensor | None = None,
        nvfp4_v_codes: torch.Tensor | None = None,
        nvfp4_v_scales: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one compact request-major ragged verify attention.

        The leading dimension is the graph capacity, not ``B * Q`` with a
        request-local padded width.  ``Qwen36VerifyGraphAttention`` owns the
        dynamic query indptr, so its paged kernel consumes only the valid
        prefix of each request while the graph keeps a stable allocation for
        the largest possible batch.  Rows after the indptr end are harmless
        scratch rows used only to keep the captured tensor shapes static.
        """
        if hidden_states.ndim != 2:
            raise ValueError(
                "Qwen36 ragged verify hidden states must be [capacity, H], "
                f"got {tuple(hidden_states.shape)}"
            )
        capacity = int(hidden_states.shape[0])
        if positions.shape != (capacity,) or write_index.shape != (capacity,):
            raise ValueError("Qwen36 ragged verify positions/write_index must match capacity")
        if output.shape[0] != capacity:
            raise ValueError("Qwen36 ragged attention output must match hidden capacity")

        q_and_gate, key_raw, value_raw = self._qkv_proj(hidden_states)
        q_and_gate = q_and_gate.view(capacity, self.num_heads, self.head_dim * 2)
        query, gate = torch.chunk(q_and_gate, 2, dim=-1)
        query = self.q_norm(query)
        key = self.k_norm(key_raw.view(capacity, self.num_kv_heads, self.head_dim))
        value = value_raw.view(capacity, self.num_kv_heads, self.head_dim)

        query_flat = query.reshape(capacity, self.num_heads * self.head_dim).contiguous()
        key_flat = key.reshape(capacity, self.num_kv_heads * self.head_dim).contiguous()
        apply_rotary_embedding_inplace(positions, query_flat, self.head_dim, cos_sin_cache)
        apply_rotary_embedding_inplace(positions, key_flat, self.head_dim, cos_sin_cache)
        query = query_flat.view(capacity, self.num_heads, self.head_dim)
        key = key_flat.view(capacity, self.num_kv_heads, self.head_dim)

        _rt = self._kv_nvfp4_roundtrip(
            key, value, nvfp4_k_codes, nvfp4_k_scales, nvfp4_v_codes, nvfp4_v_scales, write_index
        )
        if _rt is not None:
            key, value = _rt
        k_to_store, v_to_store = _kv_to_cache_dtype(
            key, value, cache_dtype=k_pool.dtype, k_scale=self.k_scale, v_scale=self.v_scale
        )
        k_pool.view(-1, self.num_kv_heads, self.head_dim)[write_index] = k_to_store
        v_pool.view(-1, self.num_kv_heads, self.head_dim)[write_index] = v_to_store
        if nvfp4_k_codes is not None:
            raise RuntimeError("NVFP4 KV attention is not wired for ragged DSpark verify")
        attn.forward(
            q=query,
            k_cache=k_pool,
            v_cache=v_pool,
            output=output,
            k_descale=self.k_scale,
            v_descale=self.v_scale,
        )
        return self.o_proj(output * torch.sigmoid(gate))


# ---------------------------------------------------------------------------
# Dense SwiGLU MLP.
# ---------------------------------------------------------------------------


def _mlp_w4a4_prefill_enabled() -> bool:
    """Routes large-M (prefill) MLP forwards through the true W4A4
    ``blockscaled.mm`` path. Default ON as part of the combined
    historical-kernel mode -- measured 2026-08-04 W1-S c=4 twice: wall
    33.3/33.2 s vs 58.7-60.2 s baseline, acceptance 72.3% (historical
    anchor 70.29%) (see ``notes/2026-08-04-w4a4-sparkinfer-headtohead.md``).
    ``QSR_QWEN36_MLP_W4A4=0`` stays the diagnostic fallback."""
    return (
        os.environ.get("QSR_QWEN36_MLP_W4A4", "1") != "0"
        or os.environ.get("QSR_QWEN36_HIST_KERNELS") == "1"
    )


#: Smallest row count routed to the W4A4 prefill path. Decode (M<=4 per
#: slot batch) and MTP verify (M=16 at c=4) stay on the W4A16 fused kernel
#: -- they are CUDA-graph-captured and bandwidth-bound, where W4A16 holds
#: its own (measured 2026-08-04: decode GEMM bandwidth parity, and the
#: fused FC1->activation->FC2 launch structure beats three separate dense
#: launches there).
_W4A4_PREFILL_MIN_ROWS = 64


def _w4a4_all_rows_enabled() -> bool:
    """Routes EVERY MLP forward (decode and verify included) through the
    W4A4 blockscaled path -- the historical single-kernel-family /
    single-weight-residency layout. Default ON inside the combined
    historical-kernel mode (see ``_mlp_w4a4_prefill_enabled``): the e2e
    win includes the raw packed-weight residency, and the measured
    small-M bandwidth deficit (~330-440 vs ~830 GB/s) does not survive
    the fused-round accounting. ``QSR_QWEN36_MLP_W4A4_ALL=0`` stays the
    diagnostic fallback.

    Since 2026-08-15 (plan §4.4 P0-M1) this flag also switches the
    prepare lifecycle: all-W4A4 prepares the W4A4 operands directly and
    never builds the independent W4A16 repack (~7.84 GiB) or its per-M
    graph runtime buffers; setting it to 0 restores the historical
    W4A16-first layout for the fresh-process small-M A/B."""
    return (
        os.environ.get("QSR_QWEN36_MLP_W4A4_ALL", "1") != "0"
        or os.environ.get("QSR_QWEN36_HIST_KERNELS") == "1"
    )


class Qwen36MLP(nn.Module):
    """Dense SwiGLU MLP: ``down_proj(silu(gate_proj(x)) * up_proj(x))``.

    **NVFP4 fused fast path (``work/nvfp4-gemm-20260802`` follow-up)**: when
    the checkpoint declares ``gate_proj``/``up_proj``/``down_proj`` all
    ``W4A16_NVFP4`` (true for every real MLP layer per B0-2 -- the
    checkpoint quantizes ``mlp.{gate,up,down}_proj`` uniformly, never a
    mix), ``forward()`` does NOT call the three NVFP4 Linear submodules
    individually. Instead it fuses them into one call to
    ``sparkinfer.moe._shared.kernels.w4a16.kernel.run_w4a16_moe`` -- the
    weight-only kernel that dequantizes NVFP4 *inside* the kernel against
    the real BF16 activation (``packed_dequant_e2m1x4_to_bfloat2x2`` +
    ``bf16_mma_m16n8k16_f32``), not a kernel that requires both operands
    pre-quantized like ``sparkinfer.gemm.blockscaled.mm`` (the previous
    attempt on this branch -- see git history: that one turned a genuine
    W4A16 checkpoint into an unintended W4A4 approximation because it needs
    a quantized activation operand, and failed B1-R's calibrated gap-error
    bars).

    **Two checkpoint formats, one fused path (2026-08-03 follow-up)**: this
    fast path was originally gated on ``isinstance(..., ModelOptNVFP4Linear)``
    only, so unsloth's ``unsloth/Qwen3.6-27B-NVFP4`` checkpoint (compressed-
    tensors mixed-precision format, :class:`~runtime.model.
    compressed_tensors_linear.CompressedTensorsNVFP4Linear`) silently fell
    through to the plain per-Linear BF16-dequant-and-cache path below and
    never got the memory/throughput win nvidia's modelopt checkpoint did
    (measured: 80.65 GiB resident vs nvidia's 55-57 GiB). The gate now
    accepts either NVFP4 Linear class, and every place this method used to
    read a submodule's raw Parameters by name (``.weight``/``.weight_scale``/
    ``.weight_scale_2``) instead calls that submodule's own
    ``nvfp4_components_for_fuse()``/``free_nvfp4_raw_params()`` (defined on
    both classes -- see ``runtime/model/modelopt_linear.py``,
    ``runtime/model/compressed_tensors_linear.py``), so this class no longer
    needs to know which checkpoint format it holds beyond the isinstance
    check itself. The two formats' Parameter names genuinely differ
    (``weight``/``weight_scale_2`` vs ``weight_packed``/``weight_global_scale``)
    -- more importantly, **their global scales are RECIPROCALS of each
    other**, not the same value under a different name (unsloth's real
    ``layers.0.mlp.gate_proj``: ``weight_global_scale=6624.0``; nvidia's:
    ``weight_scale_2=0.0002``; ``1/6624 ≈ 0.000151``, same order of
    magnitude -- see ``CompressedTensorsNVFP4Linear``'s own docstring for
    the GPU-measured evidence this exact mismatch produced degenerate
    ``"!!!!!!!!!!!!"`` output the first time it was missed). Each format's
    ``nvfp4_components_for_fuse()`` normalizes to modelopt's convention (a
    direct multiplier, matching ``sparkinfer.moe._shared.kernels.w4a16.
    prepare.prepare_w4a16_modelopt_nvfp4_weights``'s own documented
    expectation of "raw ModelOpt weight global scales") before returning,
    so everything in ``_ensure_w4a16_fused_ready`` below is format-agnostic.

    Also correctly inert for unsloth's own FP8-channel MLP layers: that
    checkpoint's ``config_groups`` quantizes ``layers.{56..63}.mlp.
    {gate,up,down}_proj`` as per-channel FP8, not NVFP4 (verified against
    the real safetensors index/config, 2026-08-03) -- those three
    submodules are :class:`~runtime.model.compressed_tensors_linear.
    CompressedTensorsFP8ChannelLinear`, so ``_nvfp4_fused`` is False for
    them and they fall through to the plain per-Linear forward below,
    unaffected by any of this.

    Why fusion is required, not optional: ``run_w4a16_moe`` is shaped like
    one full MoE expert's gated MLP block (FC1 = ``w13`` fused gate+up ->
    activation -> FC2 = ``w2`` down-proj) in a single launch -- there is no
    way to call it for "just gate_proj alone" or "just down_proj alone" the
    way the individual NVFP4 Linear's own ``forward()`` used to. This class
    degenerates the call into a 1-expert/top-1 MoE (``topk_ids`` is always
    ``[[0]]``, ``topk_weights`` is always ``[[1.0]]``) -- exactly the
    untested-on-GPU path B0 already flagged
    (``notes/2026-08-02-trackB-b0-facts.md`` references
    ``prepare_w4a16_modelopt_native_weights`` + ``num_experts=1``).

    ``gate_proj``/``up_proj``/``down_proj`` stay real NVFP4 Linear
    submodules (unchanged Parameter shapes, unchanged ``weight_loader``
    wiring) purely so the existing checkpoint-loading machinery keeps
    working unmodified -- ``_ensure_w4a16_fused_ready`` below reads their
    scale/weight tensors (via ``nvfp4_components_for_fuse()``) to build the
    fused ``w13``/``w2`` representation once, lazily, and never calls their
    own ``.forward()``. Their ``_ensure_ready()``/``_weight_bf16`` legacy
    dequant path is left completely alone (still there for whatever
    standalone diagnostics construct a bare NVFP4 Linear, e.g.
    ``scripts/verify_nvfp4_gemm_single_layer.py``) -- this class just never
    triggers it.

    ``gate_proj``'s and ``up_proj``'s fused-global-scale must be exactly
    equal for the fused ``w13`` global scale to be valid (the kernel's
    packed W4A16 format has ONE global scale per w13 tensor, shared by both
    halves). Empirically true for every real NVFP4 MLP layer in BOTH
    checkpoints this runtime loads (verified directly off safetensors
    headers, not assumed -- nvidia's modelopt checkpoint: all 64 layers;
    unsloth's compressed-tensors checkpoint: all 56 NVFP4 layers, i.e.
    every layer except the 8 FP8-channel ones above) --
    ``_ensure_w4a16_fused_ready`` asserts it rather than silently
    averaging/picking one if a future checkpoint variant differs.

    **Raw-Parameter freeing (2026-08-03 follow-up)**: once
    ``_w4a16_prepared`` is built, ``gate_proj``/``up_proj``/``down_proj``'s
    raw weight/scale Parameters (~9.15 GiB across 64 real layers on
    nvidia's checkpoint) are no longer read by this class -- by default
    ``_ensure_w4a16_fused_ready`` frees them (see
    ``_free_raw_nvfp4_weights``, which now calls each submodule's own
    ``free_nvfp4_raw_params()`` rather than looping over one format's fixed
    attribute names), so the two copies (raw + repacked) don't stay
    resident together the way ``notes/2026-08-03-nvfp4-gemm-memory-
    audit.md`` measured. Set ``self._keep_raw_nvfp4_weights = True`` before
    the first fused forward to opt out -- see ``__init__``'s docstring on
    that attribute for exactly which diagnostic/verification scripts need
    this and why.
    """

    # The production graph matrix never has more than four slots and its
    # longest backbone verify is anchor + K (K defaults to 3). Keep some
    # headroom for MTP's ragged-sync buckets, while deliberately not caching
    # arbitrary prefill chunk sizes as resident scratch.
    _W4A16_GRAPH_BUFFER_MAX_M = 32

    def __init__(
        self,
        config: dict[str, Any],
        layer_idx: int,
        quantized: dict[str, str],
        *,
        weight_prefix: str | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        hidden_size = config["hidden_size"]
        intermediate_size = config["intermediate_size"]
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        # See Qwen36Attention.__init__'s ``weight_prefix`` docstring -- same
        # override, same reason (Qwen36MTPHead's checkpoint prefix is
        # ``mtp.layers.0.mlp``, not ``model.language_model.layers.N.mlp``).
        prefix = (
            weight_prefix
            if weight_prefix is not None
            else (f"model.language_model.layers.{layer_idx}.mlp")
        )
        self.gate_proj = _make_linear(
            quantized, f"{prefix}.gate_proj", hidden_size, intermediate_size
        )
        self.up_proj = _make_linear(quantized, f"{prefix}.up_proj", hidden_size, intermediate_size)
        self.down_proj = _make_linear(
            quantized, f"{prefix}.down_proj", intermediate_size, hidden_size
        )
        assert config["hidden_act"] == "silu"

        # All-or-nothing: only fuse when every one of the three is a real
        # NVFP4 Linear -- either checkpoint format (modelopt's
        # ModelOptNVFP4Linear or compressed-tensors mixed-precision's
        # CompressedTensorsNVFP4Linear; 2026-08-03 follow-up, see both
        # classes' `nvfp4_components_for_fuse`/`free_nvfp4_raw_params`
        # methods -- this class no longer touches either one's raw
        # Parameters by name, so it does not need to know which format it
        # holds beyond this isinstance check). Never true for the
        # mixed/unquantized configs some unit tests build, e.g.
        # tests/test_qwen36_mtp_head.py's TestWeightPrefixOverride, which
        # only quantizes gate_proj -- those fall through to the plain
        # per-Linear forward below, unaffected. Also correctly False for
        # unsloth's own FP8-channel MLP layers (that checkpoint's
        # config_groups quantizes layers 56-63's mlp.{gate,up,down}_proj as
        # FP8, not NVFP4 -- verified against the real safetensors index,
        # 2026-08-03): those three submodules are CompressedTensorsFP8ChannelLinear
        # instances, so this check is False for them and they fall through
        # to the same plain per-Linear forward, same as ever.
        self._nvfp4_fused = (
            isinstance(self.gate_proj, (ModelOptNVFP4Linear, CompressedTensorsNVFP4Linear))
            and isinstance(self.up_proj, (ModelOptNVFP4Linear, CompressedTensorsNVFP4Linear))
            and isinstance(self.down_proj, (ModelOptNVFP4Linear, CompressedTensorsNVFP4Linear))
        )
        # vLLM's Qwen3NextMLP uses one merged gate/up projection.  The
        # corresponding runtime fusion is built lazily only for the native
        # FP8-channel route; NVFP4 MLPs use their existing W4A4/W4A16 paths.
        self._fused_fp8_gate_up: FusedFP8ChannelGateUp | None = None
        self._w4a16_prepared = None  # built lazily, once, on first fused forward
        self._w4a4_prepared = None  # W4A4 blockscaled operands (opt-in, prefill)
        self._w4a4_unavailable = False  # checkpoint lacks activation global scales
        self._w4a4_share_gate_up_quant = False
        # SGLang's Qwen3.5/Qwen3.8 MLP uses one merged gate+up projection
        # (MergedColumnParallelLinear).  Keep the W4A4 equivalent explicit so
        # a checkpoint with mismatched calibration scales can still use the
        # old two-GEMM fallback without silently changing its numerics.
        self._w4a4_gate_up_fused = False
        # ``run_w4a16_moe`` takes caller-owned route, output, and GEMM scratch.
        # Decode/verify used to recreate all of them for every NVFP4 layer on
        # every replay. Cache only CUDA-graph-sized M values so each captured
        # shape has stable addresses without retaining prefill-sized scratch.
        self._w4a16_graph_runtime: dict[int, tuple[torch.Tensor, torch.Tensor, Any]] = {}

        # Memory-audit follow-up (2026-08-03, notes/2026-08-03-nvfp4-gemm-
        # memory-audit.md "What was not verified"): once `_w4a16_prepared`
        # exists, gate/up/down_proj's raw NVFP4 Parameters
        # (`.weight`/`.weight_scale`/`.weight_scale_2`, ~9.15 GiB across 64
        # real layers) are never read again by this class's own forward
        # path -- `_ensure_w4a16_fused_ready` below frees them by default.
        # Set this True BEFORE the first fused forward to opt out (keeps
        # them resident) -- required by any caller that alternates between
        # this fused path and a submodule's own legacy
        # `ModelOptNVFP4Linear.forward()` / `_ensure_ready()` on the SAME
        # instance after that point, since that path independently
        # dequantizes straight from the raw Parameters:
        # `scripts/verify_nvfp4_gemm_full_model_gap.py` (oracle/candidate
        # monkeypatch of `Qwen36MLP.forward`, re-run per workload on one
        # loaded model), `scripts/verify_nvfp4_gemm_single_layer.py`
        # (legacy-vs-fused comparison repeated across M values on one MLP
        # instance), `scripts/b3_probe_batching_bar.py` (calls
        # `gate_proj`/`down_proj` directly, for a DIFFERENT diagnostic
        # purpose, right after the fused `mlp(x)` call on the same
        # instance). `scripts/b1_verify_greedy_alignment.py` does NOT need
        # this: it dequantizes every submodule once, up front, strictly
        # before this class's fused forward ever runs for the first time.
        self._keep_raw_nvfp4_weights = False

    def _ensure_w4a16_fused_ready(self) -> None:
        """Build the fused ``w13``/``w2`` W4A16 packed representation from
        the three NVFP4 submodules' checkpoint tensors. Lazy + cached: runs
        once per module instance, not once per forward call. Also frees the
        three submodules' raw NVFP4 Parameters afterward, unless
        ``self._keep_raw_nvfp4_weights`` is set -- see that attribute's
        docstring in ``__init__``."""
        if self._w4a16_prepared is not None:
            return
        from runtime.backends._sparkinfer_import import ensure_sparkinfer_path

        ensure_sparkinfer_path()
        from b12x._lib.intrinsics import swizzle_block_scale
        from b12x.moe._shared.kernels.w4a16.prepare import (
            prepare_w4a16_modelopt_nvfp4_weights,
        )

        gate, up, down = self.gate_proj, self.up_proj, self.down_proj
        # `nvfp4_components_for_fuse()` normalizes both checkpoint formats
        # to the SAME (modelopt-style, direct-multiplier) global-scale
        # convention -- see that method's docstring on each class
        # (runtime/model/modelopt_linear.py,
        # runtime/model/compressed_tensors_linear.py) for why unsloth's
        # compressed-tensors format needs a reciprocal here and modelopt's
        # doesn't. Everything below this line is format-agnostic.
        gate_w, gate_scale, gate_gs = gate.nvfp4_components_for_fuse()
        up_w, up_scale, up_gs = up.nvfp4_components_for_fuse()
        down_w, down_scale, down_gs = down.nvfp4_components_for_fuse()
        if gate_gs.item() != up_gs.item():
            raise ValueError(
                "Qwen36MLP: gate_proj's fused-global-scale "
                f"({gate_gs.item()!r}) != up_proj's ({up_gs.item()!r}) -- "
                "the fused w13 kernel needs one shared global scale for "
                "both halves; this checkpoint layer breaks the assumption "
                "every real layer was verified to satisfy."
            )

        # w13 physical row order: [gate rows; up rows] -- pass
        # w13_layout="gate_up" so the kernel knows this is already its own
        # native (no-row-rotation) order. See class docstring.
        w13_fp4 = torch.cat([gate_w, up_w], dim=0).unsqueeze(0).contiguous()
        w13_blockscale = swizzle_block_scale(
            torch.cat([gate_scale, up_scale], dim=0).unsqueeze(0).contiguous()
        )
        w13_global_scale = gate_gs.reshape(1).contiguous()

        w2_fp4 = down_w.unsqueeze(0).contiguous()
        w2_blockscale = swizzle_block_scale(down_scale.unsqueeze(0).contiguous())
        w2_global_scale = down_gs.reshape(1).contiguous()

        self._w4a16_prepared = prepare_w4a16_modelopt_nvfp4_weights(
            w13_fp4,
            w13_blockscale,
            w13_global_scale,
            w2_fp4,
            w2_blockscale,
            w2_global_scale,
            activation="silu",
            params_dtype=torch.bfloat16,
            w13_layout="gate_up",
        )
        # Drop this function's own references before freeing the raw
        # Parameters below: `w2_fp4`/`w2_blockscale` are (for w2 -- w13's
        # `torch.cat` above already decoupled it) VIEWS that can alias
        # `down.weight`/`down.weight_scale`'s storage (`.unsqueeze(0)
        # .contiguous()` is a no-op on an already-contiguous tensor), and a
        # still-live local reference would keep that storage resident even
        # after the Parameter itself is reassigned below. `_repack_weight`/
        # `swizzle_block_scale` (verified directly, not assumed) always
        # write `self._w4a16_prepared`'s tensors into freshly allocated
        # storage -- never a view of their inputs -- so nothing above holds
        # onto these once this function returns.
        del w13_fp4, w13_blockscale, w2_fp4, w2_blockscale
        # `gate_w`/`up_w`/`down_w`/`gate_scale`/`up_scale`/`down_scale`
        # (unpacked from `nvfp4_components_for_fuse()` above) are themselves
        # direct references to the raw Parameter `.data` tensors -- no
        # copy. Unlike the pre-refactor code, which accessed
        # `<submodule>.weight.data` inline inside `torch.cat(...)` and so
        # never bound a persistent local name to it, these six ARE
        # persistent local names in this frame and must be dropped for the
        # same reason `w2_fp4`/`w2_blockscale` are above: a live local
        # reference to the OLD tensor object keeps its storage resident
        # even after `_free_raw_nvfp4_weights` reassigns the owning
        # Parameter's `.data` to a fresh empty tensor.
        del gate_w, up_w, down_w, gate_scale, up_scale, down_scale
        if _mlp_w4a4_prefill_enabled():
            # Build the W4A4 operand set while the raw Parameters are still
            # live; the operands ALIAS the raw packed weights (no copy, one
            # weight residency -- the historical layout), so the free below
            # must be skipped whenever the aliasing path is active.
            self._ensure_w4a4_ready()
        if not self._keep_raw_nvfp4_weights and self._w4a4_prepared is None:
            self._free_raw_nvfp4_weights()

    def _free_raw_nvfp4_weights(self) -> None:
        """Release gate/up/down_proj's raw NVFP4 Parameter storage
        (``.weight``/``.weight_scale``/``.weight_scale_2``) now that
        ``self._w4a16_prepared`` holds an independent repacked copy --
        ``_forward_w4a16_fused`` never reads these three submodules'
        Parameters again. Reassigns each Parameter's ``.data`` to a
        0-element tensor rather than deleting the ``nn.Parameter`` itself or
        setting it to ``None``, so anything that walks ``named_parameters()``
        or reads ``module.weight`` directly (this runtime never re-saves a
        checkpoint, so nothing depends on the shape matching the real
        checkpoint header afterward) still finds a real tensor of the
        correct dtype/device at the expected attribute -- just shape
        ``(0,)`` -- instead of hitting a missing attribute or ``None``.

        Also returns the freed storage to the driver via
        ``torch.cuda.empty_cache()`` -- measured directly (2026-08-03),
        not assumed: without it, ``torch.cuda.memory_allocated()`` drops
        as expected but PyTorch's caching allocator keeps the underlying
        blocks reserved for its own reuse, so external ``nvidia-smi``
        polling (what every memory-audit script in this repo actually
        measures) barely moved -- 67.10 -> 64.58 GiB despite ~9.15 GiB of
        real Parameter storage being dropped, because only a fraction of
        it happened to get reused by later allocations before that run's
        peak. Runs once per real MLP layer, ever (guarded the same way
        ``_ensure_w4a16_fused_ready`` itself is -- this whole method only
        runs once per instance), so the cost is a bounded ~64 calls over
        the model's lifetime, not a per-token or per-forward recurring
        one."""
        # `weight_scale` is the one Parameter name both NVFP4 Linear
        # formats share verbatim (modelopt's `weight`/`weight_scale_2` vs
        # compressed-tensors' `weight_packed`/`weight_global_scale` differ,
        # `weight_scale` doesn't -- see both classes' `__init__`), so it is
        # read here for the device check before `free_nvfp4_raw_params()`
        # (below) reassigns it to a 0-element tensor on the SAME device.
        device_type = self.gate_proj.weight_scale.data.device.type
        for lin in (self.gate_proj, self.up_proj, self.down_proj):
            lin.free_nvfp4_raw_params()
        if device_type == "cuda":
            torch.cuda.empty_cache()

    def _ensure_w4a4_ready(self) -> None:
        """Build per-projection W4A4 ``blockscaled.mm`` operands (packed
        values + swizzled e4m3 scale view + alpha) from the three NVFP4
        submodules' checkpoint tensors. Opt-in prefill path
        (``QSR_QWEN36_MLP_W4A4=1``); silently marks the path unavailable
        when the checkpoint format does not ship activation global scales
        (nvidia modelopt). Measured 2026-08-04 vs the W4A16 fused path on
        real layer-5 weights: single-layer cosine 0.9955 (the genuine
        scheme-level W4A4 activation-quantization delta -- numerically
        validated against the historical NVFP4 recipe, see
        ``notes/2026-08-04-w4a4-sparkinfer-headtohead.md``), GEMM speed
        within 20% of flashinfer b12x at M=16 and ~4x faster at prefill M.
        The operand tensors are independent copies (the raw Parameters are
        freed right after this returns), +one packed-weight copy resident
        while the path is enabled."""
        if self._w4a4_prepared is not None or self._w4a4_unavailable:
            return
        from runtime.backends._sparkinfer_import import ensure_sparkinfer_path

        ensure_sparkinfer_path()
        from b12x._lib.intrinsics import (
            as_grouped_scale_view,
            swizzle_block_scale,
        )

        prepared: dict[str, tuple] = {}
        raw_components: dict[str, tuple[torch.Tensor, ...]] = {}
        for name in ("gate", "up", "down"):
            lin = getattr(self, f"{name}_proj")
            comps = getattr(lin, "nvfp4_w4a4_components_for_fuse", None)
            if comps is None:
                self._w4a4_unavailable = True
                return
            w_packed, w_scale, w_gs, igs = comps()
            if igs.numel() == 0 or not torch.isfinite(igs.float()).all():
                self._w4a4_unavailable = True
                return
            raw_components[name] = (w_packed, w_scale, w_gs, igs)
            out_dim, in_dim = w_packed.shape[0], w_packed.shape[1] * 2
            # NO weight copy: the historical implementation kept exactly one
            # NVFP4 weight residency. ``unsqueeze(-1)`` on a contiguous
            # ``[N, K/2]`` tensor is a no-op view, so alias the raw packed
            # weights and keep them resident (this method's caller then skips
            # ``_free_raw_nvfp4_weights`` while the W4A4 path is enabled).
            # Cost: the ~8.65 GiB raw weights stay up; a cloned operand set
            # was measured to push this box into allocator pressure and slow
            # BOTH phases of the full-model W1-S run.
            b_p = w_packed.unsqueeze(-1)
            b_sf = as_grouped_scale_view(
                swizzle_block_scale(w_scale.unsqueeze(0).contiguous()).view(torch.uint8),
                out_dim,
                in_dim,
            )
            alpha = (1.0 / (igs.to(torch.float32) * w_gs.to(torch.float32))).reshape(1)
            prepared[name] = (b_p, b_sf, alpha, igs.to(torch.float32).reshape(1))

        # Match SGLang's merged FC1.  The blockscaled kernel accepts one
        # weight operand with [gate rows; up rows], so one launch replaces the
        # two otherwise identical input-quantized GEMMs.  Both halves must
        # share input and weight calibration; if they do not, keep the exact
        # per-projection path below rather than approximating one scale.
        gate_w, gate_scale, gate_gs, gate_igs = raw_components["gate"]
        up_w, up_scale, up_gs, up_igs = raw_components["up"]
        can_fuse_gate_up = (
            gate_w.shape == up_w.shape
            and gate_scale.shape == up_scale.shape
            and torch.equal(gate_gs.reshape(()), up_gs.reshape(()))
            and torch.equal(gate_igs.reshape(()), up_igs.reshape(()))
            and torch.equal(prepared["gate"][2], prepared["up"][2])
        )
        if can_fuse_gate_up:
            gate_up_w = torch.cat((gate_w, up_w), dim=0).contiguous().unsqueeze(-1)
            gate_up_scale = as_grouped_scale_view(
                swizzle_block_scale(
                    torch.cat((gate_scale, up_scale), dim=0)
                    .unsqueeze(0)
                    .contiguous()
                ).view(torch.uint8),
                gate_w.shape[0] * 2,
                gate_w.shape[1] * 2,
            )
            prepared["gate_up"] = (
                gate_up_w,
                gate_up_scale,
                prepared["gate"][2],
                prepared["gate"][3],
            )
            # Drop the views into the two raw tensors before optionally
            # releasing their Parameter storage.  The merged packed weights
            # and merged block scales above are independent allocations.
            del prepared["gate"], prepared["up"]
            self._w4a4_gate_up_fused = True
            if not self._keep_raw_nvfp4_weights:
                self.gate_proj.free_nvfp4_raw_params()
                self.up_proj.free_nvfp4_raw_params()
                if gate_w.device.type == "cuda":
                    torch.cuda.empty_cache()
        del raw_components
        self._w4a4_share_gate_up_quant = (
            False
            if self._w4a4_gate_up_fused
            else prepared["gate"][3].item() == prepared["up"][3].item()
        )
        self._w4a4_prepared = prepared
        if os.environ.get("QSR_QWEN36_MLP_PROFILE", "0") == "1":
            logger.info(
                "qwen36_mlp_w4a4 layer=%s gate_up_fused=%s shared_input_scale=%s",
                self.layer_idx,
                self._w4a4_gate_up_fused,
                self._w4a4_share_gate_up_quant,
            )

    def _forward_w4a4_blockscaled(self, x: torch.Tensor) -> torch.Tensor:
        """True W4A4 NVFP4 MLP (both operands quantized, FP4 tensor-core
        GEMM) via ``sparkinfer.gemm.blockscaled.mm`` -- the prefill fast
        path (measured ~4x faster than the W4A16 fused kernel at M=16384).
        Decode/verify shapes stay on ``_forward_w4a16_fused``: see
        ``forward``'s dispatch. Activation quantization is the bit-exact
        Triton kernel (``runtime/kernels/nvfp4_quant.py``, gated by
        ``tests/test_nvfp4_quant_triton.py``), NOT the pure-Torch oracle."""
        from b12x._lib.intrinsics import (
            as_grouped_scale_view,
            swizzle_block_scale,
        )
        from b12x.gemm import blockscaled

        from runtime.kernels.nvfp4_quant import quantize_nvfp4_activation

        orig_shape = x.shape
        x2d = x.reshape(-1, self.hidden_size)
        if not x2d.is_contiguous():
            x2d = x2d.contiguous()
        m = x2d.shape[0]
        prepared = self._w4a4_prepared
        assert prepared is not None, "_forward_w4a4_blockscaled before _ensure_w4a4_ready"

        def quant_operand(t2d: torch.Tensor, gs: torch.Tensor):
            a_packed, sf_linear = quantize_nvfp4_activation(t2d, gs)
            sw = swizzle_block_scale(sf_linear.view(torch.float8_e4m3fn).unsqueeze(0))
            a_sf = as_grouped_scale_view(sw.view(torch.uint8), m, t2d.shape[1])
            # dense_gemm takes [M, K, L] group-major operands; num_groups=1
            return a_packed.unsqueeze(-1), a_sf

        def gemm(name: str, a_packed: torch.Tensor, a_sf: torch.Tensor) -> torch.Tensor:
            b_p, b_sf, alpha, _igs = prepared[name]
            return blockscaled.mm(
                (a_packed, a_sf),
                (b_p, b_sf),
                alpha=alpha,
                ab_dtype="float4_e2m1fn",
                sf_dtype="float8_e4m3fn",
                c_dtype="bfloat16",
                sf_vec_size=16,
                expected_m=m,
            )[:, :, 0]

        if self._w4a4_gate_up_fused:
            a_gate_up_p, a_gate_up_sf = quant_operand(x2d, prepared["gate_up"][3])
            gate_up = gemm("gate_up", a_gate_up_p, a_gate_up_sf)
            gate, up = gate_up.chunk(2, dim=-1)
        else:
            a_gate_p, a_gate_sf = quant_operand(x2d, prepared["gate"][3])
            gate = gemm("gate", a_gate_p, a_gate_sf)
            if self._w4a4_share_gate_up_quant:
                up = gemm("up", a_gate_p, a_gate_sf)
            else:
                a_up_p, a_up_sf = quant_operand(x2d, prepared["up"][3])
                up = gemm("up", a_up_p, a_up_sf)
        inter = F.silu(gate) * up
        a_down_p, a_down_sf = quant_operand(inter.contiguous(), prepared["down"][3])
        out = gemm("down", a_down_p, a_down_sf)
        return out.reshape(*orig_shape[:-1], self.hidden_size)

    def _forward_w4a16_fused(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_w4a16_fused_ready()
        from b12x.moe._shared.kernels.w4a16.kernel import run_w4a16_moe
        from b12x.moe._shared.kernels.w4a16.prepare import make_w4a16_packed_buffers

        if x.dtype != torch.bfloat16:
            raise TypeError(
                f"Qwen36MLP._forward_w4a16_fused expects bf16 activations, got {x.dtype}"
            )
        orig_shape = x.shape
        x2d = x.reshape(-1, self.hidden_size).contiguous()
        m = x2d.shape[0]
        device = x2d.device

        # NVFP4-standard-model follow-up (2026-08-03, notes/2026-08-03-std-
        # model-serving-acceptance.md section 3): decode CUDA Graph capture
        # of this fused path used to fail here with RuntimeError("W4A16 GEMM
        # scratch is not initialized for CUDA graph capture ...") because
        # sparkinfer's make_w4a16_packed_buffers()/plan_w4a16_buffers()
        # under-sized fc1_c_tmp/fc2_c_tmp for run_w4a16_moe's small-M direct
        # top-k routes / TC-decode fast path (which this class's degenerate
        # 1-expert/top-1 MoE always takes): it sized scratch via
        # max_packed_route_slots (the *packed/grouped* route-kernel's bound)
        # while the kernel's own fast path needs
        # route_slots_for_scratch=m*topk*block_size_m instead -- a genuine
        # sparkinfer bug (9 vs 16 route slots for this deployment's decode
        # batch=2), invisible in eager mode (silently absorbed by a fresh
        # torch.empty() fallback) but fatal under torch.cuda.graph capture
        # (which correctly refuses that fallback). Worked around here with a
        # separate, persistent, deliberately-oversized scratch buffer
        # (``_w4a16_c_tmp_scratch``, since removed).
        #
        # Root-caused and fixed upstream instead (sparkinfer worktree
        # work/w4a16-scratch-20260803, plan_w4a16_buffers now unions the
        # packed-mode bound with the direct-topk-routes bound when sizing
        # fc1_c_tmp_elements/fc2_c_tmp_elements) -- verified against a real
        # checkpoint layer at this exact decode shape
        # (scripts/verify_w4a16_cuda_graph_scratch_rootcause.py: CUDA Graph
        # capture succeeds and replays bit-exact vs eager using
        # make_w4a16_packed_buffers's own fc1_c_tmp/fc2_c_tmp directly, no
        # workaround). The workaround is gone; buffers.fc1_c_tmp/fc2_c_tmp
        # are passed straight through again, same as every other buffer
        # here.
        cached = self._w4a16_graph_runtime.get(m)
        if cached is None:
            # Degenerate 1-expert/top-1 MoE: every row routes to expert 0
            # with router weight 1.0. Those values are immutable for dense
            # Qwen MLPs, so they are valid graph replay inputs.
            topk_ids = torch.zeros((m, 1), dtype=torch.int32, device=device)
            topk_weights = torch.ones((m, 1), dtype=torch.float32, device=device)
            buffers = make_w4a16_packed_buffers(
                self._w4a16_prepared,
                m=m,
                topk=1,
                dtype=torch.bfloat16,
                device=device,
            )
            if m <= self._W4A16_GRAPH_BUFFER_MAX_M:
                self._w4a16_graph_runtime[m] = (topk_ids, topk_weights, buffers)
        else:
            topk_ids, topk_weights, buffers = cached
        out = run_w4a16_moe(
            x2d,
            self._w4a16_prepared,
            topk_weights,
            topk_ids,
            activation="silu",
            intermediate_cache13=buffers.intermediate_cache13,
            intermediate_cache2=buffers.intermediate_cache2,
            output=buffers.output,
            fc1_c_tmp=buffers.fc1_c_tmp,
            fc2_c_tmp=buffers.fc2_c_tmp,
            packed_route_indices=buffers.packed_route_indices,
            block_expert_ids=buffers.block_expert_ids,
            packed_route_count=buffers.packed_route_count,
            expert_offsets=buffers.expert_offsets,
            # Without this the route-packing path falls back to a fresh
            # torch.empty per call -- invisible in eager mode, fatal under
            # CUDA Graph capture (measured 2026-08-16: MTP verify capture
            # failed, decode degraded to eager, 108 -> 15.8 tok/s).
            expert_counts=buffers.expert_counts,
            fast_math=True,
        )
        return out.reshape(*orig_shape[:-1], self.hidden_size)

    def _forward_native_fp8_gate_up(self, x: torch.Tensor) -> torch.Tensor | None:
        """Run vLLM-style merged gate/up W8A8 when the raw route is active."""
        if (
            x.device.type != "cuda"
            or self._nvfp4_fused
            or not _native_w8a8_fp8_channel_enabled()
            or not _native_w8a8_fp8_gate_up_enabled()
            or (
                self.gate_proj.output_size != 17408
                and not fp8_channel_raw_execution_uses_all_layers()
            )
            or not isinstance(self.gate_proj, CompressedTensorsFP8ChannelLinear)
            or not isinstance(self.up_proj, CompressedTensorsFP8ChannelLinear)
        ):
            return None
        if self._fused_fp8_gate_up is None:
            try:
                fused = FusedFP8ChannelGateUp(self.gate_proj, self.up_proj)
                fused._ensure()
            except (RuntimeError, ValueError, TypeError, AttributeError):
                self._fused_fp8_gate_up = None
            else:
                self._fused_fp8_gate_up = fused
        if self._fused_fp8_gate_up is None:
            return None
        library = _native_w8a8_library_for_cuda()
        if library is None:
            return None
        gate_up = self._fused_fp8_gate_up.forward_native(x, library)
        gate, up = torch.chunk(gate_up, 2, dim=-1)
        return self.down_proj(F.silu(gate) * up)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._nvfp4_fused:
            if _w4a4_all_rows_enabled() and _mlp_w4a4_prefill_enabled():
                # W4A4-only residency (plan §4.4 P0-M1): every row -- decode,
                # verify, and prefill -- goes through the W4A4 blockscaled
                # path, so the independent W4A16 repack (~7.84 GiB) and its
                # per-M graph runtime buffers are never built. The raw NVFP4
                # Parameters stay resident because the W4A4 operands alias
                # them (one weight residency). The historical behavior -- W4A16
                # first, then W4A4 on top -- stays reachable with
                # ``QSR_QWEN36_MLP_W4A4_ALL=0`` for the fresh-process
                # W4A16 small-M performance/quality comparison.
                self._ensure_w4a4_ready()
                if self._w4a4_prepared is not None:
                    return self._forward_w4a4_blockscaled(x)
                # Checkpoint lacks activation global scales: W4A16 fallback.
                return self._forward_w4a16_fused(x)
            if _mlp_w4a4_prefill_enabled() and self._w4a4_prepared is not None:
                rows = x.reshape(-1, x.shape[-1]).shape[0]
                if rows >= _W4A4_PREFILL_MIN_ROWS:
                    return self._forward_w4a4_blockscaled(x)
            return self._forward_w4a16_fused(x)
        fused_gate_up = self._forward_native_fp8_gate_up(x)
        if fused_gate_up is not None:
            return fused_gate_up
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
        enable_fp8_kv: bool = False,
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
            self.self_attn = Qwen36Attention(
                config,
                layer_idx,
                quantized,
                max_seq_len=max_seq_len,
                enable_fp8_kv=enable_fp8_kv,
            )
            self.linear_attn = None

        self.mlp = Qwen36MLP(config, layer_idx, quantized)
        self.input_layernorm = Qwen36RMSNorm(config["hidden_size"], eps=eps)
        self.post_attention_layernorm = Qwen36RMSNorm(config["hidden_size"], eps=eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        gdn_state: GdnLayerState | None,
        attn_cache: Qwen36PagedAttentionCache | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if self.layer_type == "linear_attention":
            assert gdn_state is not None
            hidden_states = self.linear_attn(hidden_states, gdn_state)
        else:
            assert attn_cache is not None
            hidden_states = self.self_attn(hidden_states, positions, cos_sin_cache, attn_cache)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def decode_batch(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        cos_sin_cache: torch.Tensor,
        batch: Qwen36DecodeBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """:meth:`forward`'s structure with the batched, globally addressed
        decode kernels substituted for the single-sequence ones.

        The gather/scatter of the recurrent state lives here rather than
        inside :meth:`Qwen36GatedDeltaNet.decode_batch` so that method
        stays a pure function of its arguments (see its docstring): this
        is the only place in the decode path where "batch row i is slot
        ``slot_index[i]``" is known.
        """
        i = self.layer_idx
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if self.layer_type == "linear_attention":
            slot_index = batch.slot_index
            conv = batch.conv_pools[i].index_select(0, slot_index)
            recurrent = batch.recurrent_pools[i].index_select(0, slot_index)
            hidden_states = self.linear_attn.decode_batch(hidden_states, conv, recurrent)
            batch.conv_pools[i].index_copy_(0, slot_index, conv)
            batch.recurrent_pools[i].index_copy_(0, slot_index, recurrent)
        else:
            hidden_states = self.self_attn.decode_batch(
                hidden_states,
                batch.positions,
                cos_sin_cache,
                k_pool=batch.k_pools[i],
                v_pool=batch.v_pools[i],
                write_index=batch.write_index,
                attn=batch.attn,
                output=batch.attn_outputs[i],
                nvfp4_k_codes=(
                    None if batch.nvfp4_k_codes is None else batch.nvfp4_k_codes[i]
                ),
                nvfp4_k_scales=(
                    None if batch.nvfp4_k_scales is None else batch.nvfp4_k_scales[i]
                ),
                nvfp4_v_codes=(
                    None if batch.nvfp4_v_codes is None else batch.nvfp4_v_codes[i]
                ),
                nvfp4_v_scales=(
                    None if batch.nvfp4_v_scales is None else batch.nvfp4_v_scales[i]
                ),
            )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def prefill_batch(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        cos_sin_cache: torch.Tensor,
        batch: Qwen36PrefillBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One homogeneous multi-slot extend layer over pooled state."""
        i = self.layer_idx
        profile_ops = _prefill_op_profile_enabled(i, hidden_states.device)
        op_times: dict[str, float] = {}

        def mark(name: str, started: float) -> None:
            if profile_ops:
                torch.cuda.synchronize(hidden_states.device)
                op_times[name] = (time.perf_counter() - started) * 1000.0

        if profile_ops:
            torch.cuda.synchronize(hidden_states.device)
            op_started = time.perf_counter()
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        mark("input_norm", op_started if profile_ops else 0.0)

        if profile_ops:
            op_started = time.perf_counter()
        if self.layer_type == "linear_attention":
            slot_index = batch.slot_index
            conv = batch.conv_pools[i].index_select(0, slot_index)
            recurrent = batch.recurrent_pools[i].index_select(0, slot_index)
            hidden_states = self.linear_attn.prefill_batch(
                hidden_states,
                conv,
                recurrent,
                has_previous_state=batch.has_previous_state,
            )
            batch.conv_pools[i].index_copy_(0, slot_index, conv)
            batch.recurrent_pools[i].index_copy_(0, slot_index, recurrent)
        else:
            hidden_states = self.self_attn.prefill_batch(
                hidden_states,
                batch.positions,
                cos_sin_cache,
                k_pool=batch.k_pools[i],
                v_pool=batch.v_pools[i],
                write_index=batch.write_index,
                attn=batch.attn,
                output=batch.attn_outputs[i],
                nvfp4_k_codes=(
                    None if batch.nvfp4_k_codes is None else batch.nvfp4_k_codes[i]
                ),
                nvfp4_k_scales=(
                    None if batch.nvfp4_k_scales is None else batch.nvfp4_k_scales[i]
                ),
                nvfp4_v_codes=(
                    None if batch.nvfp4_v_codes is None else batch.nvfp4_v_codes[i]
                ),
                nvfp4_v_scales=(
                    None if batch.nvfp4_v_scales is None else batch.nvfp4_v_scales[i]
                ),
            )
        mark("attention", op_started if profile_ops else 0.0)

        if profile_ops:
            op_started = time.perf_counter()
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        mark("post_norm", op_started if profile_ops else 0.0)

        if profile_ops:
            op_started = time.perf_counter()
        hidden_states = self.mlp(hidden_states)
        mark("mlp", op_started if profile_ops else 0.0)
        if profile_ops:
            logger.info(
                "qwen36_prefill_ops layer=%d kind=%s tokens=%d input_norm_ms=%.3f "
                "attention_ms=%.3f post_norm_ms=%.3f mlp_ms=%.3f",
                i,
                self.layer_type,
                hidden_states.shape[1],
                op_times["input_norm"],
                op_times["attention"],
                op_times["post_norm"],
                op_times["mlp"],
            )
        return hidden_states, residual


# ---------------------------------------------------------------------------
# B3: MTP (multi-token-prediction) draft head.
# ---------------------------------------------------------------------------
#
# Checkpoint fact (verified directly against two real checkpoints, plus
# re-confirmed against ``nvidia/Qwen3.6-27B-NVFP4``'s own
# ``model.safetensors.index.json`` while building this): exactly 15
# ``mtp.*`` tensors, none of them ``linear_attn.*`` -- ``mtp.fc.weight``,
# ``mtp.pre_fc_norm_embedding.weight``, ``mtp.pre_fc_norm_hidden.weight``,
# ``mtp.norm.weight``, and ``mtp.layers.0.{input_layernorm,
# post_attention_layernorm, self_attn.{q,k,v,o}_proj, self_attn.{q,k}_norm,
# mlp.{gate,up,down}_proj}.weight``. Structurally this is ONE ordinary
# full-attention decoder layer (``config["layer_types"][0]`` says
# "linear_attention" -- irrelevant here, the MTP head is unconditionally
# full-attention regardless of what layer 0 of the *backbone* is) plus a
# small fusion block in front of it. No GDN anywhere in the draft head, so
# none of B3's GDN-state-rollback machinery
# (``Qwen36GatedDeltaNet.spec_forward``/``commit_spec_snapshot``) applies to
# this class -- that machinery exists entirely for the TARGET model's
# recursive layers, which the draft head never touches.
#
# All 15 tensors are plain BF16 in the real checkpoint (no ``*_scale``
# siblings; confirmed via direct safetensors dtype inspection) --
# ``_make_linear`` naturally resolves them to :class:`PlainLinear` because
# ``mtp.*`` never appears in ``quantization_config.quantized_layers``. This
# matches a known NVFP4-checkpoint quirk vLLM works around explicitly
# (``/home/bot/vllm/vllm/model_executor/models/qwen3_5_mtp.py``, read
# ONLY as an external reference while designing this class, never
# imported: "mtp.fc is stored as BF16 in NVFP4 checkpoints but is missing
# from hf_quant_config.json exclude_modules. Force unquantized.").
#
# Forward structure and the embedding-then-hidden concat order are
# transcribed from that same file's ``Qwen3_5MultiTokenPredictor.forward``
# (read-only reference -- this repository has no vLLM runtime dependency,
# see module docstring). The target hidden state MTP consumes is the
# TARGET model's post-final-norm hidden state -- confirmed by tracing
# vLLM's ``gpu_model_runner.py`` (``target_hidden_states = hidden_states``,
# where ``hidden_states`` is exactly what ``Qwen3NextModel.forward``
# returns after its own ``self.norm(...)`` call, for the non-EAGLE3
# ``use_aux_hidden_state_outputs=False`` path this checkpoint uses) --
# i.e. the SAME hidden state ``compute_logits`` reads for the target's own
# prediction, not a pre-norm intermediate.
class Qwen36MTPLayer(nn.Module):
    """One MTP decoder layer with the historical carried residual stream.

    Its numerical sequence is byte-for-byte the same as
    :meth:`Qwen36DecoderLayer.forward`'s full-attention branch
    (deliberately not shared code with that method -- see this class's
    module-level docstring for why instantiating ``Qwen36DecoderLayer``
    itself is wrong here: it picks GDN vs attention from
    ``config["layer_types"][layer_idx]``, which for ``layer_idx=0`` says
    "linear_attention" in the real backbone and would build a GDN layer
    instead of the full-attention one this checkpoint's ``mtp.layers.0.*``
    tensors actually are).
    """

    def __init__(
        self,
        config: dict[str, Any],
        quantized: dict[str, str],
        *,
        max_seq_len: int,
        weight_prefix: str,
        enable_fp8_kv: bool = False,
    ) -> None:
        super().__init__()
        eps = config["rms_norm_eps"]
        hidden_size = config["hidden_size"]
        self.self_attn = Qwen36Attention(
            config,
            0,
            quantized,
            max_seq_len=max_seq_len,
            weight_prefix=f"{weight_prefix}.self_attn",
            enable_fp8_kv=enable_fp8_kv,
            kv_scale_buffer=enable_fp8_kv,
        )
        self.mlp = Qwen36MLP(config, 0, quantized, weight_prefix=f"{weight_prefix}.mlp")
        self.input_layernorm = Qwen36RMSNorm(hidden_size, eps=eps)
        self.post_attention_layernorm = Qwen36RMSNorm(hidden_size, eps=eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        attn_cache: Qwen36PagedAttentionCache,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(hidden_states, positions, cos_sin_cache, attn_cache)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen36MTPHead(nn.Module):
    """B3 draft head: ``fc(cat([norm_embed(next_token), norm_hidden(prev)]))
    -> Qwen36MTPLayer -> norm``.

    Shares ``embed_tokens``/``lm_head`` with the target model (checkpoint
    declares ``mtp_use_dedicated_embeddings: false``, and there is no
    ``mtp.embed_tokens``/``mtp.lm_head`` tensor in the checkpoint to load
    even if it didn't) -- callers pass in an already-computed token
    embedding and read the returned hidden state back out through the
    target model's own ``lm_head``, rather than this class owning either.

    Has its own :class:`Qwen36PagedAttentionCache` (via its own
    ``Qwen36Attention`` instance's ``new_cache()``), independent of the
    target model's per-layer caches, because the draft head's self-attention
    needs causal context over the sequence of positions IT has processed
    -- which, during chained multi-token drafting, includes its own
    previously-drafted positions, not just positions the target model has
    committed. Rollback after a partial accept is a plain ``cache.seq_len``
    rewind (same trick ``Qwen36SlotPool.rewind_slot`` uses for prefix-cache
    truncation) -- unlike the target's GDN layers, nothing here needs
    snapshot/restore, because a causal KV cache's "state at position m" is
    just "the first m rows", always recoverable by truncation alone.
    """

    def __init__(
        self,
        config: dict[str, Any],
        quantized: dict[str, str],
        *,
        max_seq_len: int,
        enable_fp8_kv: bool = False,
    ) -> None:
        super().__init__()
        hidden_size = config["hidden_size"]
        eps = config["rms_norm_eps"]
        num_mtp_layers = config.get("mtp_num_hidden_layers", 1)
        assert num_mtp_layers == 1, (
            f"Qwen36MTPHead only verified against mtp_num_hidden_layers=1 (the real "
            f"checkpoint's value); got {num_mtp_layers}. A checkpoint with more MTP "
            "layers needs new verification (which of the N layers a given spec_step "
            "picks, matching vLLM's `spec_step_idx % self.num_mtp_layers`) before "
            "this class can be trusted for it."
        )
        self.pre_fc_norm_embedding = Qwen36RMSNorm(hidden_size, eps=eps)
        self.pre_fc_norm_hidden = Qwen36RMSNorm(hidden_size, eps=eps)
        self.fc = _make_linear(quantized, "mtp.fc", hidden_size * 2, hidden_size)
        self.layers = nn.ModuleList(
            [
                Qwen36MTPLayer(
                    config,
                    quantized,
                    max_seq_len=max_seq_len,
                    weight_prefix=f"mtp.layers.{i}",
                    enable_fp8_kv=enable_fp8_kv,
                )
                for i in range(num_mtp_layers)
            ]
        )
        if enable_fp8_kv:
            # The checkpoint has no ``mtp.layers.*.self_attn.k_scale`` /
            # ``v_scale`` tensors (verified against its tensor list), so the
            # FP8 KV write/read scales fall back to the fixed 1/448
            # convention FP8Linear uses for activations.  fp8 e4m3 relative
            # precision is scale-independent; the fixed scale only needs to
            # keep typical draft-head KV magnitudes inside the representable
            # range, which O(1) post-norm values satisfy.
            for layer in self.layers:
                attn = layer.self_attn
                attn.k_scale.data.fill_(1.0 / float(torch.finfo(torch.float8_e4m3fn).max))
                attn.v_scale.data.fill_(1.0 / float(torch.finfo(torch.float8_e4m3fn).max))
        self.norm = Qwen36RMSNorm(hidden_size, eps=eps)

    def new_cache(self, *, device: torch.device, dtype: torch.dtype) -> Qwen36PagedAttentionCache:
        return self.layers[0].self_attn.new_cache(device=device, dtype=dtype)

    def forward(
        self,
        next_token_embeds: torch.Tensor,
        prev_hidden: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        attn_cache: Qwen36PagedAttentionCache,
        *,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """One MTP step. ``next_token_embeds``/``prev_hidden``:
        ``[1, seq_len, hidden_size]`` each -- the (already embedded) real or
        self-drafted token(s) this step conditions on, and the hidden state
        (target's own, for the very first draft step of a round; this
        head's own previous output, for every later chained step) from the
        position immediately before. Returns the post-``norm`` hidden state,
        ready for the shared ``lm_head`` to turn into draft logits.
        """
        fused = torch.cat(
            [self.pre_fc_norm_embedding(next_token_embeds), self.pre_fc_norm_hidden(prev_hidden)],
            dim=-1,
        )
        hidden_states = self.fc(fused)
        hidden_states, residual = self.layers[layer_idx](
            hidden_states, None, positions, cos_sin_cache, attn_cache
        )
        normed, _ = self.norm(hidden_states, residual)
        return normed


class Qwen36TextModelSelfBuilt(nn.Module):
    """Embedding -> N decoder layers -> final norm. Batch=1, no vision."""

    def __init__(
        self,
        config: dict[str, Any],
        quantized: dict[str, str],
        *,
        max_seq_len: int,
        enable_fp8_kv: bool = False,
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
                Qwen36DecoderLayer(
                    config,
                    i,
                    quantized,
                    max_seq_len=max_seq_len,
                    enable_fp8_kv=enable_fp8_kv,
                )
                for i in range(self.num_hidden_layers)
            ]
        )
        self.norm = Qwen36RMSNorm(self.hidden_size, eps=config["rms_norm_eps"])

        # DSpark uses post-layer residual states at the target layer ids from
        # the external draft config (the same 1-based convention as the
        # existing Laguna DFlash aux-state path).  Keep this opt-in: ordinary
        # Qwen/MTP forwards must not retain five extra [tokens, hidden]
        # tensors.
        self.aux_hidden_state_layers: tuple[int, ...] = ()
        # Bounded, opt-in cross-engine diagnostics.  This is intentionally
        # separate from ``aux_hidden_state_layers``: DSpark needs exactly five
        # taps for its projector, while a numerical comparison may need every
        # post-layer residual state without changing the model contract.
        self._debug_last_all_layer_hidden: list[torch.Tensor] | None = None

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

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        """Select 1-based post-layer outputs for an external draft model."""

        if any(layer_id <= 0 or layer_id > self.num_hidden_layers for layer_id in layers):
            raise ValueError(
                f"aux hidden layer ids must be in 1..{self.num_hidden_layers}, got {layers}"
            )
        self.aux_hidden_state_layers = tuple(layers)

    def _maybe_add_aux_hidden_state(
        self,
        aux_hidden_states: list[torch.Tensor],
        layer_id: int,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> None:
        if layer_id in self.aux_hidden_state_layers:
            aux_hidden_states.append(
                hidden_states + residual if residual is not None else hidden_states
            )

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
        capture_aux_hidden_states: bool = False,
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
        aux_hidden_states: list[torch.Tensor] = []
        debug_all_layers = (
            capture_aux_hidden_states
            and os.environ.get("QSR_DSPARK_DUMP_ALL_LAYERS")
            not in (None, "", "0")
        )
        debug_row_cap = max(1, int(os.environ.get("QSR_DSPARK_DUMP_ALL_LAYERS_ROWS", "8")))
        debug_all: list[torch.Tensor] = []
        residual: torch.Tensor | None = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states,
                residual,
                positions,
                cos_sin_cache,
                state.gdn_states[layer.layer_idx],
                state.attn_caches[layer.layer_idx],
            )
            if capture_hidden_states:
                per_layer_hidden.append(hidden_states)
            if capture_aux_hidden_states:
                self._maybe_add_aux_hidden_state(
                    aux_hidden_states,
                    layer.layer_idx + 1,
                    hidden_states,
                    residual,
                )
            if debug_all_layers:
                assert residual is not None
                debug_all.append(
                    (hidden_states + residual)[..., :debug_row_cap, :].detach().clone()
                )

        assert residual is not None
        self._debug_last_all_layer_hidden = debug_all if debug_all_layers else None
        hidden_states, _ = self.norm(hidden_states, residual)
        state.num_tokens_seen = past_len + seq_len

        if capture_aux_hidden_states:
            return hidden_states, aux_hidden_states
        if capture_hidden_states:
            return hidden_states, per_layer_hidden
        return hidden_states

    def prefill_batch(
        self,
        batch: Qwen36PrefillBatch,
        *,
        capture_aux_hidden_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the pool-backed uniform batched prefill body.

        State-length bookkeeping remains in :class:`Qwen36SlotPool`: this
        graph only consumes explicit device metadata and mutates the pooled
        KV/GDN tensors, mirroring the existing batched decode split.
        """
        hidden_states = self.embed_tokens(batch.input_ids)
        cos_sin_cache = self.cos_sin_cache
        residual: torch.Tensor | None = None
        aux_hidden_states: list[torch.Tensor] = []
        for layer in self.layers:
            profile_layer = (
                os.environ.get(QSR_QWEN36_PREFILL_LAYER_PROFILE_ENV, "0") == "1"
                and hidden_states.is_cuda
            )
            if profile_layer:
                torch.cuda.synchronize(hidden_states.device)
                layer_t0 = time.perf_counter()
            hidden_states, residual = layer.prefill_batch(
                hidden_states, residual, cos_sin_cache, batch
            )
            if profile_layer:
                torch.cuda.synchronize(hidden_states.device)
                logger.info(
                    "qwen36_prefill_layer layer=%d kind=%s tokens=%d ms=%.3f",
                    layer.layer_idx,
                    layer.layer_type,
                    hidden_states.shape[1],
                    (time.perf_counter() - layer_t0) * 1000.0,
                )
            if capture_aux_hidden_states:
                self._maybe_add_aux_hidden_state(
                    aux_hidden_states,
                    layer.layer_idx + 1,
                    hidden_states,
                    residual,
                )
        assert residual is not None
        hidden_states, _ = self.norm(hidden_states, residual)
        if capture_aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def verify_forward(
        self,
        draft_token_ids: torch.Tensor,
        state: Qwen36GenerationState,
        *,
        spec_state_rows: dict[int, list[GdnLayerState] | list[list[GdnLayerState]]] | None = None,
        capture_aux_hidden_states: bool = False,
    ) -> (
        tuple[torch.Tensor, dict[int, list[GdnLayerState]] | None]
        | tuple[torch.Tensor, dict[int, list[GdnLayerState]] | None, list[torch.Tensor]]
    ):
        """B3: run ``K`` draft tokens through every layer in ONE pass,
        WITHOUT committing ``state`` -- the target-model half of MTP verify.

        ``draft_token_ids``: ``[1, K]``, the K speculative token ids to
        verify (NOT including the anchor -- ``state`` already reflects
        having processed the anchor, exactly like
        :meth:`Qwen36GatedDeltaNet.spec_forward`'s own precondition on each
        GDN layer's state -- see that method's docstring).

        For each layer, in order:
          * ``linear_attention`` (GDN) layers use
            :meth:`Qwen36GatedDeltaNet.spec_forward` instead of the
            ordinary ``forward()`` -- the only layer type that cannot
            expose intermediate states from a single multi-token call (its
            docstring explains why: the chunk algorithm only returns the
            state after ALL K positions). Its ``K+1`` state snapshots are
            collected here, keyed by layer index, for :meth:`commit_verify`
            to resolve later once accept/reject is decided.
          * ``full_attention`` layers use the SAME ``self_attn`` call an
            ordinary multi-token prefill already uses (B1-proven,
            unmodified) -- a causal attention kernel over a K-token query
            block against an already-materialized KV cache computes every
            position's correct output in one shot, unlike GDN's fused
            recurrence, so no per-position snapshot mechanism is needed
            here. This eagerly appends all K positions' KV to the cache;
            :meth:`commit_verify` rewinds ``cache.seq_len`` back down on a
            partial accept (the appended-but-rejected KV bytes are simply
            never read again -- same trick
            ``Qwen36SlotPool.rewind_slot`` uses for prefix-cache
            truncation).

        Returns ``(post_norm_hidden, gdn_snapshots)``: ``post_norm_hidden``
        is ``[1, K, hidden_size]`` -- feed to ``lm_head`` for verify logits,
        and (row-wise) to :class:`Qwen36MTPHead` as the next round's
        ``prev_hidden`` once accept/reject picks which row. ``gdn_snapshots
        [layer_idx]`` has ``K+1`` entries, same contract as
        :meth:`Qwen36GatedDeltaNet.spec_forward`'s own return.

        Does not touch ``state.num_tokens_seen``; :meth:`commit_verify` is
        the only place ``state`` is actually resolved.
        """
        assert draft_token_ids.dim() == 2 and draft_token_ids.shape[0] == 1
        seq_len = draft_token_ids.shape[1]
        past_len = state.num_tokens_seen
        positions = torch.arange(
            past_len, past_len + seq_len, device=draft_token_ids.device, dtype=torch.long
        )
        hidden_states = self.embed_tokens(draft_token_ids)
        cos_sin_cache = self.cos_sin_cache.to(hidden_states.device)

        gdn_snapshots: dict[int, list[GdnLayerState]] | None = (
            {} if spec_state_rows is None else None
        )
        aux_hidden_states: list[torch.Tensor] = []
        residual: torch.Tensor | None = None
        for layer in self.layers:
            if residual is None:
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)
            else:
                hidden_states, residual = layer.input_layernorm(hidden_states, residual)

            if layer.layer_type == "linear_attention":
                gdn_state = state.gdn_states[layer.layer_idx]
                assert gdn_state is not None
                out, snapshots = layer.linear_attn.spec_forward(
                    hidden_states,
                    gdn_state,
                    spec_state_rows=(
                        None if spec_state_rows is None else spec_state_rows[layer.layer_idx]
                    ),
                    batch_large_projections=fp8_channel_raw_execution_uses_all_layers(),
                )
                if gdn_snapshots is not None:
                    assert snapshots is not None
                    gdn_snapshots[layer.layer_idx] = snapshots
                hidden_states = out
            else:
                attn_cache = state.attn_caches[layer.layer_idx]
                assert attn_cache is not None
                layer.self_attn.declare_verify_capacity(seq_len)
                hidden_states = layer.self_attn(
                    hidden_states, positions, cos_sin_cache, attn_cache, paged_mode="verify"
                )

            hidden_states, residual = layer.post_attention_layernorm(hidden_states, residual)
            hidden_states = layer.mlp(hidden_states)
            if capture_aux_hidden_states:
                self._maybe_add_aux_hidden_state(
                    aux_hidden_states,
                    layer.layer_idx + 1,
                    hidden_states,
                    residual,
                )

        assert residual is not None
        hidden_states, _ = self.norm(hidden_states, residual)
        if capture_aux_hidden_states:
            return hidden_states, gdn_snapshots, aux_hidden_states
        return hidden_states, gdn_snapshots

    def verify_batch(
        self,
        batch: Qwen36VerifyBatch,
        *,
        capture_hidden_states: bool = False,
        capture_aux_hidden_states: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, list[torch.Tensor]]
        | tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]
    ):
        """Run the graph-safe K-token target verify forward.

        GDN rows and full-attention metadata are supplied by the persistent
        graph descriptor. No Python cache length is read or advanced here;
        this makes the captured body independent of which speculative row was
        accepted by the preceding round.
        """
        hidden_states = self.embed_tokens(batch.input_ids)
        cos_sin_cache = self.cos_sin_cache.to(hidden_states.device)
        residual: torch.Tensor | None = None
        # Both optional capture seams are kept on the graph-safe body.  MTP
        # uses the per-layer list for diagnostics; DSpark uses the selected
        # post-layer taps inside its captured verify graph.  Neither path
        # substitutes ``verify_forward``: that method has different
        # cache/GDN addressing and is the eager fallback only.
        per_layer_hidden: list[torch.Tensor] = []
        aux_hidden_states: list[torch.Tensor] = []
        for layer in self.layers:
            if residual is None:
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)
            else:
                hidden_states, residual = layer.input_layernorm(hidden_states, residual)
            if layer.layer_type == "linear_attention":
                conv_pool = batch.gdn_conv_pools[layer.layer_idx]
                recurrent_pool = batch.gdn_recurrent_pools[layer.layer_idx]
                assert conv_pool is not None and recurrent_pool is not None
                source = GdnLayerState(
                    conv_state=conv_pool.index_select(0, batch.gdn_source_index),
                    # The indexed recurrence reads its actual incoming row
                    # inside the Triton kernel.  This stable view is only a
                    # compatibility placeholder for the CPU/legacy branch;
                    # do not gather the large recurrent state before every
                    # graph replay.
                    recurrent_state=recurrent_pool[: hidden_states.shape[0]],
                    has_previous_state=True,
                )
                rows = (
                    None if batch.gdn_state_rows is None else batch.gdn_state_rows[layer.layer_idx]
                )
                hidden_states, snapshots = layer.linear_attn.spec_forward(
                    hidden_states,
                    source,
                    spec_state_rows=rows,
                    spec_source_index=batch.gdn_source_index,
                    spec_destination_index=batch.gdn_destination_index,
                    spec_conv_pool=conv_pool,
                    spec_recurrent_pool=recurrent_pool,
                    # Historical production used one qkvz/ba projection on
                    # the complete request-major B*(K+1) verify matrix. The
                    # raw-FP8 all-layer contract is the only format we have
                    # validated for that physical layout; legacy BF16 paths
                    # retain their exact single-position compatibility form.
                    batch_large_projections=fp8_channel_raw_execution_uses_all_layers(),
                )
                assert snapshots is None
            else:
                attn_cache = batch.k_pools[layer.layer_idx]
                v_cache = batch.v_pools[layer.layer_idx]
                driver = batch.attn_drivers[layer.layer_idx]
                output = batch.attn_outputs[layer.layer_idx]
                assert attn_cache is not None
                assert v_cache is not None and driver is not None and output is not None
                hidden_states = layer.self_attn.verify_batch(
                    hidden_states,
                    batch.positions,
                    cos_sin_cache,
                    k_pool=attn_cache,
                    v_pool=v_cache,
                    write_index=batch.write_index,
                    attn=driver,
                    output=output,
                    nvfp4_k_codes=(
                        None
                        if batch.nvfp4_k_codes is None
                        else batch.nvfp4_k_codes[layer.layer_idx]
                    ),
                    nvfp4_k_scales=(
                        None
                        if batch.nvfp4_k_scales is None
                        else batch.nvfp4_k_scales[layer.layer_idx]
                    ),
                    nvfp4_v_codes=(
                        None
                        if batch.nvfp4_v_codes is None
                        else batch.nvfp4_v_codes[layer.layer_idx]
                    ),
                    nvfp4_v_scales=(
                        None
                        if batch.nvfp4_v_scales is None
                        else batch.nvfp4_v_scales[layer.layer_idx]
                    ),
                )
            hidden_states, residual = layer.post_attention_layernorm(hidden_states, residual)
            hidden_states = layer.mlp(hidden_states)
            if capture_hidden_states:
                per_layer_hidden.append(hidden_states)
            if capture_aux_hidden_states:
                self._maybe_add_aux_hidden_state(
                    aux_hidden_states,
                    layer.layer_idx + 1,
                    hidden_states,
                    residual,
                )
        assert residual is not None
        hidden_states, _ = self.norm(hidden_states, residual)
        if capture_hidden_states and capture_aux_hidden_states:
            return hidden_states, per_layer_hidden, aux_hidden_states
        if capture_aux_hidden_states:
            return hidden_states, aux_hidden_states
        if capture_hidden_states:
            return hidden_states, per_layer_hidden
        return hidden_states

    def verify_ragged_batch(
        self,
        batch: Qwen36VerifyBatch,
        *,
        capture_hidden_states: bool = False,
        capture_aux_hidden_states: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, list[torch.Tensor]]
        | tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]
    ):
        """Run one compact request-major DSpark verify body.

        Full-attention rows use ``batch.cu_seqlens_q`` through the graph
        driver, so only the request-local ``verify_lens`` prefix participates
        in attention.  GDN is still evaluated through its existing batched
        ``spec_forward`` contract: the compact rows are viewed through a
        padded ``[B, max_verify_tokens, H]`` map, and invalid padding writes
        only to the scratch candidate rows.  This keeps the recurrent state
        update graph-safe without putting a second target verify graph in the
        scheduler.
        """
        if not batch.ragged:
            raise ValueError("verify_ragged_batch requires a ragged batch descriptor")
        if batch.max_verify_tokens is None:
            raise ValueError("ragged verify descriptor is missing max_verify_tokens")
        if batch.gdn_padded_to_compact is None or batch.gdn_compact_to_padded is None:
            raise ValueError("ragged verify descriptor is missing GDN compact maps")
        if batch.cu_seqlens_q is None:
            raise ValueError("ragged verify descriptor is missing cu_seqlens_q")

        capacity = int(batch.input_ids.shape[0])
        batch_size = int(batch.gdn_source_index.shape[0])
        max_verify_tokens = int(batch.max_verify_tokens)
        if capacity != batch_size * max_verify_tokens:
            raise ValueError(
                "ragged verify capacity must equal B*max_verify_tokens: "
                f"capacity={capacity}, B={batch_size}, max={max_verify_tokens}"
            )

        hidden_states = self.embed_tokens(batch.input_ids)
        cos_sin_cache = self.cos_sin_cache.to(hidden_states.device)
        residual: torch.Tensor | None = None
        per_layer_hidden: list[torch.Tensor] = []
        aux_hidden_states: list[torch.Tensor] = []
        for layer in self.layers:
            if residual is None:
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)
            else:
                hidden_states, residual = layer.input_layernorm(hidden_states, residual)

            if layer.layer_type == "linear_attention":
                conv_pool = batch.gdn_conv_pools[layer.layer_idx]
                recurrent_pool = batch.gdn_recurrent_pools[layer.layer_idx]
                assert conv_pool is not None and recurrent_pool is not None
                padded_hidden = hidden_states.index_select(
                    0, batch.gdn_padded_to_compact
                ).view(batch_size, max_verify_tokens, -1)
                source = GdnLayerState(
                    conv_state=conv_pool.index_select(0, batch.gdn_source_index),
                    recurrent_state=recurrent_pool[:batch_size],
                    has_previous_state=True,
                )
                padded_out, snapshots = layer.linear_attn.spec_forward(
                    padded_hidden,
                    source,
                    spec_source_index=batch.gdn_source_index,
                    spec_destination_index=batch.gdn_destination_index,
                    spec_conv_pool=conv_pool,
                    spec_recurrent_pool=recurrent_pool,
                    batch_large_projections=fp8_channel_raw_execution_uses_all_layers(),
                )
                assert snapshots is None
                hidden_states = padded_out.reshape(capacity, -1).index_select(
                    0, batch.gdn_compact_to_padded
                )
            else:
                attn_cache = batch.k_pools[layer.layer_idx]
                v_cache = batch.v_pools[layer.layer_idx]
                driver = batch.attn_drivers[layer.layer_idx]
                output = batch.attn_outputs[layer.layer_idx]
                assert attn_cache is not None
                assert v_cache is not None and driver is not None and output is not None
                hidden_states = layer.self_attn.verify_ragged_batch(
                    hidden_states,
                    batch.positions,
                    cos_sin_cache,
                    k_pool=attn_cache,
                    v_pool=v_cache,
                    write_index=batch.write_index,
                    attn=driver,
                    output=output,
                    nvfp4_k_codes=(
                        None
                        if batch.nvfp4_k_codes is None
                        else batch.nvfp4_k_codes[layer.layer_idx]
                    ),
                    nvfp4_k_scales=(
                        None
                        if batch.nvfp4_k_scales is None
                        else batch.nvfp4_k_scales[layer.layer_idx]
                    ),
                    nvfp4_v_codes=(
                        None
                        if batch.nvfp4_v_codes is None
                        else batch.nvfp4_v_codes[layer.layer_idx]
                    ),
                    nvfp4_v_scales=(
                        None
                        if batch.nvfp4_v_scales is None
                        else batch.nvfp4_v_scales[layer.layer_idx]
                    ),
                )

            hidden_states, residual = layer.post_attention_layernorm(hidden_states, residual)
            hidden_states = layer.mlp(hidden_states)
            if capture_hidden_states:
                per_layer_hidden.append(hidden_states)
            if capture_aux_hidden_states:
                self._maybe_add_aux_hidden_state(
                    aux_hidden_states,
                    layer.layer_idx + 1,
                    hidden_states,
                    residual,
                )

        assert residual is not None
        hidden_states, _ = self.norm(hidden_states, residual)
        if capture_hidden_states and capture_aux_hidden_states:
            return hidden_states, per_layer_hidden, aux_hidden_states
        if capture_aux_hidden_states:
            return hidden_states, aux_hidden_states
        if capture_hidden_states:
            return hidden_states, per_layer_hidden
        return hidden_states

    def commit_verify(
        self,
        state: Qwen36GenerationState,
        gdn_snapshots: dict[int, list[GdnLayerState]] | None,
        *,
        past_len: int,
        accepted_count: int,
    ) -> None:
        """Resolve a :meth:`verify_forward` call: roll every GDN layer's
        state back to ``accepted_count`` (the O(1) half of B3's rollback,
        :func:`commit_spec_snapshot`) and rewind every attention layer's
        KV cache to the same effective length (plain integer truncation --
        see :meth:`verify_forward`'s docstring for why that alone is
        sufficient for attention, unlike GDN).

        ``past_len`` must be the ``state.num_tokens_seen`` value
        :meth:`verify_forward` read before running -- passed explicitly,
        not re-read from ``state`` here, so a caller that has already
        mutated ``state.num_tokens_seen`` for some other reason between
        the two calls cannot silently corrupt this one.
        """
        for layer in self.layers:
            if layer.layer_type == "linear_attention":
                # `gdn_snapshots is None` means the caller used the K+1
                # `spec_row` path instead of clone-and-restore: every
                # candidate's state already lives at its own permanent row,
                # and the accepted one is selected by the caller's
                # `Qwen36MTPGDNRows.activate(slot, m)` immediately after this
                # returns -- a pointer swap, not a copy. So there is nothing
                # to commit here for a GDN layer in that mode.
                #
                # This branch used to read `layer_type == "linear_attention"
                # and gdn_snapshots is not None`, which sent GDN layers into
                # the KV branch below as soon as the row path returned None --
                # and GDN layers have no `attn_cache`, so it tripped
                # `assert attn_cache is not None` on the first graph-enabled
                # round. Branching on layer type is the invariant; whether
                # snapshots exist is a mode question that belongs inside it.
                if gdn_snapshots is not None:
                    commit_spec_snapshot(
                        state.gdn_states[layer.layer_idx],
                        gdn_snapshots[layer.layer_idx],
                        accepted_count,
                    )
            else:
                attn_cache = state.attn_caches[layer.layer_idx]
                assert attn_cache is not None
                attn_cache.seq_len = past_len + accepted_count
        state.num_tokens_seen = past_len + accepted_count

    def decode_batch(self, batch: Qwen36DecodeBatch) -> torch.Tensor:
        """Batched single-token continuation for ``B`` slots -> ``[B, 1, H]``.

        Deliberately does **not** touch any Python-side bookkeeping (no
        ``state.num_tokens_seen +=``): every quantity this step depends on
        arrives pre-computed inside ``batch``'s device tensors, which is
        the property that lets the whole call be captured into a CUDA
        Graph and replayed. Advancing lengths is the slot pool's job, on
        the host, outside the graph.
        """
        hidden_states = self.embed_tokens(batch.input_ids)
        cos_sin_cache = self.cos_sin_cache
        residual: torch.Tensor | None = None
        for layer in self.layers:
            hidden_states, residual = layer.decode_batch(
                hidden_states, residual, cos_sin_cache, batch
            )
        assert residual is not None
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


def _quantized_layers_map_for_checkpoint(config: dict[str, Any]) -> dict[str, str]:
    """Pick which checkpoint format's classifier :func:`_make_linear` should
    use, from ``config["quantization_config"]["quant_method"]``.

    Both real formats this backend can load (``runtime.model_registry``'s
    ``SUPPORTED_QUANT_FORMATS`` for the ``qwen36`` backend) reach here
    already validated -- this function does not re-implement that gate, it
    only decides *which* per-module classifier to build, and fails loudly
    for anything ``model_registry.resolve_checkpoint`` should have refused
    before construction ever got this far (a caller that constructs this
    class directly, bypassing the registry, is the one real way to reach
    that branch).
    """
    quant_config = config.get("quantization_config")
    if quant_config is None:
        return {}
    method = quant_config.get("quant_method") if isinstance(quant_config, dict) else None
    if method == "modelopt":
        return quantized_layers_map(config)
    if method == "compressed-tensors":
        return mixed_precision_quant_map(config)
    raise ValueError(
        f"Qwen36ForCausalLMSelfBuilt: quantization_config.quant_method {method!r} has no "
        "loader adapter wired into this model graph (known: 'modelopt', "
        "'compressed-tensors'); runtime.model_registry.resolve_checkpoint should have "
        "refused this checkpoint before construction reached here."
    )


class Qwen36ForCausalLMSelfBuilt(nn.Module):
    """Top-level model: :class:`Qwen36TextModelSelfBuilt` + ``lm_head``
    (NVFP4-quantized, per B0-2). Not tied to ``embed_tokens`` (checkpoint
    declares ``tie_word_embeddings: false``, verified)."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        max_seq_len: int,
        enable_mtp: bool = False,
        enable_fp8_kv: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        # `config` is the merged dict runtime.model_loading.load_qwen36_model
        # builds (text_config's fields + top-level quantization_config
        # injected under the same key) -- see that function's docstring.
        quantized = _quantized_layers_map_for_checkpoint(config)
        self.quantized = quantized
        # enable_fp8_kv (module docstring's 2026-08-03 update, default
        # OFF): threaded to the backbone's own full-attention layers only
        # -- NOT to Qwen36MTPHead below, whose checkpoint tensors
        # (`mtp.layers.0.self_attn.*`) never include a k_scale/v_scale
        # (verified directly against the real checkpoint's tensor list,
        # see the block comment above Qwen36MTPHead), so its attention
        # stays hardcoded BF16 KV regardless of this flag.
        self.model = Qwen36TextModelSelfBuilt(
            config, quantized, max_seq_len=max_seq_len, enable_fp8_kv=enable_fp8_kv
        )
        assert not config.get("tie_word_embeddings", False)
        self.lm_head = _make_linear(
            quantized, "lm_head", config["hidden_size"], config["vocab_size"]
        )
        # B3: MTP draft head, off by default (B1/B2 never load or construct
        # it -- see module docstring above Qwen36MTPHead). Constructing it
        # adds ~1 GiB of BF16 params on top of the backbone's much larger
        # dequant-cache floor (notes/2026-08-02-qwen36-dequant-cache-memory-
        # floor.md); negligible relative to that, but still opt-in so every
        # existing B1/B2 caller's memory footprint is unchanged byte-for-byte.
        self.mtp: Qwen36MTPHead | None = (
            Qwen36MTPHead(
                config,
                quantized,
                max_seq_len=max_seq_len,
                enable_fp8_kv=(
                    enable_fp8_kv
                    and os.environ.get(QSR_QWEN36_MTP_FP8_KV_ENV, "0") != "0"
                ),
            )
            if enable_mtp
            else None
        )

    def new_generation_state(
        self, *, device: torch.device, dtype: torch.dtype
    ) -> Qwen36GenerationState:
        return self.model.new_generation_state(device=device, dtype=dtype)

    def free_fp8_raw_weights(self, *, keep_all_raw: bool = False) -> int:
        """Release raw FP8 weights only for the explicit BF16 fallback.

        The FP8 counterpart to :meth:`Qwen36MLP._free_raw_nvfp4_weights`,
        which freed the NVFP4 MLP layers' raw parameters and took the resident
        set from 76.34 to 53.08 GiB. That fix covered the 56 NVFP4 MLP layers
        and nothing else; the other 237 FP8 tensors -- attention q/k/v/o, the
        GDN projections, ``lm_head``, and layers 56-63's MLP -- went on
        holding both the FP8 original and its BF16 dequantization, which
        ``forward`` never reads. Measured at 9.99 GiB of originals against
        19.99 GiB of cache (``notes/2026-08-03-production-memory-audit.md``).

        Each call materializes the BF16 cache if it does not exist yet, so
        this can run before any forward -- it pulls the lazy dequantization
        forward to load time rather than first token, which is where it was
        going to happen regardless. Peak stays bounded: one extra tensor at a
        time, not a second full copy of the model.

        Normal CUDA serving is a raw W8A8 executor and therefore leaves every
        checkpoint-native FP8 tensor untouched, without materializing a BF16
        cache. ``keep_all_raw`` exists for callers that need the same
        guarantee independently of the selected backend.

        Returns the number of Linears freed, so callers can log it and a test
        can assert the sweep actually reached something rather than silently
        matching nothing.
        """
        from runtime.model.compressed_tensors_linear import (
            QSR_NATIVE_W8A8_FP8_CHANNEL_ENV,
            QSR_TORCH_SCALED_MM_FP8_CHANNEL_ENV,
            CompressedTensorsFP8ChannelLinear,
            fp8_channel_raw_execution_uses_all_layers,
        )

        if keep_all_raw:
            return 0

        freed = 0
        keep_scaled_mm_mlp_raw = os.environ.get(QSR_TORCH_SCALED_MM_FP8_CHANNEL_ENV) == "1"
        keep_native_w8a8_mlp_raw = os.environ.get(QSR_NATIVE_W8A8_FP8_CHANNEL_ENV) == "1"
        keep_scaled_mm_all_raw = fp8_channel_raw_execution_uses_all_layers()
        for module in self.modules():
            if isinstance(module, CompressedTensorsFP8ChannelLinear):
                # Raw W8A8 needs its checkpoint-native FP8 matrix. The
                # retained narrow ``1`` diagnostic keeps only gate/up; the
                # default/all contract retains every FP8 tensor.
                if keep_scaled_mm_all_raw or (
                    (keep_scaled_mm_mlp_raw or keep_native_w8a8_mlp_raw)
                    and module.output_size == 17408
                ):
                    continue
                module.free_fp8_raw_weight()
                freed += 1
        if freed and next(self.parameters()).device.type == "cuda":
            torch.cuda.empty_cache()
        return freed

    def warmup_attention_shapes(self, *, device: torch.device | str, dtype: torch.dtype) -> None:
        """Pay sparkinfer's one-time CuTe compile for both attention modes
        now, on throwaway buffers, instead of on whichever real request
        happens to arrive first.

        This is the Qwen3.6 analogue of ``LagunaBackend.
        warmup_paged_attention_shapes`` (``runtime/backends/laguna.py``),
        and it is only *one* forward per (layer, mode) because
        :class:`Qwen36AttentionWorkspace` now makes the compile invariant
        to prompt length -- without that, a warmup would have to guess
        which query-length bucket the first real request lands in, and
        guessing wrong would leave the stall exactly where it was. The two
        changes are complementary, not redundant.

        Runs only the attention kernels, not a whole model forward: GDN's
        recurrent state is order-dependent (B0-5), so a real forward would
        have to be careful not to leave a warmed-up state behind. Here each
        layer gets a throwaway KV cache that is dropped on return, no GDN
        state is allocated at all, and no caller-visible state is touched.
        """
        device = torch.device(device)
        for layer in self.model.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                continue
            cache = attn.new_cache(device=device, dtype=dtype)
            for mode, seq_len in (("extend", 2), ("decode", 1)):
                workspace = attn._workspace_for(mode, cache, dtype, device)
                # `dtype` (the model's BF16 compute dtype), NOT
                # `cache.dtype` -- query always stays in its own compute
                # dtype regardless of the KV cache's dtype (FP8 KV: see
                # Qwen36Attention.forward's matching comment). Using
                # `cache.dtype` here was a second, independent instance of
                # the same "cast Q to the cache's dtype" bug forward()/
                # decode_batch() had, only reachable through this warmup
                # path -- caught by an actual FP8 KV load attempt raising
                # "unsupported q dtype torch.float8_e4m3fn" from
                # sparkinfer's planner (real error, not hypothetical).
                q = torch.zeros(seq_len, attn.num_heads, attn.head_dim, dtype=dtype, device=device)
                workspace.forward(
                    q=q,
                    k_cache=cache.k_cache,
                    v_cache=cache.v_cache,
                    output=torch.empty_like(q),
                    page_table=cache.page_table,
                    cache_seqlens=torch.tensor([seq_len], dtype=torch.int32, device=device),
                    cu_seqlens_q=torch.tensor([0, seq_len], dtype=torch.int32, device=device),
                )
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def release_decode_workspaces(self) -> int:
        """Release the shared decode-mode attention arenas (plan §4.5
        P0-M2 step 3).

        Production decode is CUDA-graph-replayed; the eager decode path
        (serial control group / capture-failure fallback) is the only
        consumer of the ``mode="decode"`` workspace, and every layer also
        caches a per-layer reference that must be dropped for the arena to
        actually become collectable. A later eager decode rebuilds the
        workspace on demand. Returns the number of shared workspaces
        dropped.
        """
        dropped = release_shared_decode_workspaces()
        layers = list(self.model.layers)
        if self.mtp is not None:
            layers.extend(self.mtp.layers)
        for layer in layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                attn._decode_workspace = None
        return dropped

    def forward(
        self,
        input_ids: torch.Tensor,
        state: Qwen36GenerationState,
        *,
        capture_hidden_states: bool = False,
        capture_aux_hidden_states: bool = False,
    ):
        return self.model(
            input_ids,
            state,
            capture_hidden_states=capture_hidden_states,
            capture_aux_hidden_states=capture_aux_hidden_states,
        )

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        """Enable the selected 1-based target taps used by external DSpark."""

        self.model.set_aux_hidden_state_layers(layers)

    def decode_batch(self, batch: Qwen36DecodeBatch) -> torch.Tensor:
        """Batched decode -> ``[B, vocab_size]`` logits (B2).

        Returns logits, not hidden states, because that is the whole of
        what a decode step is for and because keeping ``lm_head`` inside
        the graphed region is what makes a captured step self-contained.
        """
        hidden_states = self.model.decode_batch(batch)
        return self.lm_head(hidden_states.reshape(hidden_states.shape[0], -1))

    def prefill_batch(
        self,
        batch: Qwen36PrefillBatch,
        *,
        capture_aux_hidden_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Pool-backed homogeneous prefill -> post-norm ``[B, Q, H]``.

        DSpark asks for the same target-layer taps that the B1 path already
        captures.  Returning them from this batch body keeps target prefill
        and draft-KV injection on the same BxQ execution boundary instead of
        silently falling back to one target forward per slot.
        """
        return self.model.prefill_batch(
            batch, capture_aux_hidden_states=capture_aux_hidden_states
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    # -- B3: MTP verify + draft head ------------------------------------

    def verify_forward(
        self,
        draft_token_ids: torch.Tensor,
        state: Qwen36GenerationState,
        *,
        spec_state_rows: dict[int, list[GdnLayerState] | list[list[GdnLayerState]]] | None = None,
        capture_aux_hidden_states: bool = False,
    ):
        """See :meth:`Qwen36TextModelSelfBuilt.verify_forward`."""
        return self.model.verify_forward(
            draft_token_ids,
            state,
            spec_state_rows=spec_state_rows,
            capture_aux_hidden_states=capture_aux_hidden_states,
        )

    def verify_batch(
        self,
        batch: Qwen36VerifyBatch,
        *,
        capture_hidden_states: bool = False,
        capture_aux_hidden_states: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, list[torch.Tensor]]
        | tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]
    ):
        """Graph-safe target verify body; see the text-model method."""
        return self.model.verify_batch(
            batch,
            capture_hidden_states=capture_hidden_states,
            capture_aux_hidden_states=capture_aux_hidden_states,
        )

    def verify_ragged_batch(
        self,
        batch: Qwen36VerifyBatch,
        *,
        capture_hidden_states: bool = False,
        capture_aux_hidden_states: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, list[torch.Tensor]]
        | tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]
    ):
        """Graph-safe compact DSpark target verify body."""
        return self.model.verify_ragged_batch(
            batch,
            capture_hidden_states=capture_hidden_states,
            capture_aux_hidden_states=capture_aux_hidden_states,
        )

    def commit_verify(
        self,
        state: Qwen36GenerationState,
        gdn_snapshots: dict[int, list[GdnLayerState]] | None,
        *,
        past_len: int,
        accepted_count: int,
    ) -> None:
        """See :meth:`Qwen36TextModelSelfBuilt.commit_verify`."""
        self.model.commit_verify(
            state, gdn_snapshots, past_len=past_len, accepted_count=accepted_count
        )

    def mtp_new_cache(
        self, *, device: torch.device, dtype: torch.dtype
    ) -> Qwen36PagedAttentionCache:
        """A fresh KV cache for :attr:`mtp`'s own self-attention layer --
        independent of any backbone layer's cache (see
        :class:`Qwen36MTPHead`'s docstring)."""
        assert self.mtp is not None, "mtp_new_cache called but enable_mtp=False at construction"
        return self.mtp.new_cache(device=device, dtype=dtype)

    def mtp_step(
        self,
        next_token_ids: torch.Tensor,
        prev_hidden: torch.Tensor,
        position: int,
        mtp_cache: Qwen36PagedAttentionCache,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One MTP draft step -> ``(draft_token_id, post_norm_hidden)``.

        ``next_token_ids``: ``[1, 1]`` int64, the token this step conditions
        on (the real just-committed token for the first step of a round,
        or this method's own previous return value's argmax for every
        later chained step -- see :class:`Qwen36MTPHead`'s module
        docstring). ``prev_hidden``: ``[1, 1, hidden_size]``, the hidden
        state from the position immediately before (target's own, for the
        first step; this method's own previous ``post_norm_hidden``
        return, for chained steps). ``position``: this step's absolute
        RoPE position (same numbering as the backbone's own -- see
        :class:`Qwen36MTPHead`'s module docstring for why matching, not
        independent, position numbers is correct here).

        Returns the greedy draft token id (``[1]`` int64) and the
        post-``norm`` hidden state (``[1, 1, hidden_size]``) this step
        produced, for the caller to feed straight back in as the next
        step's ``next_token_ids``/``prev_hidden``.
        """
        assert self.mtp is not None, "mtp_step called but enable_mtp=False at construction"
        embeds = self.model.embed_tokens(next_token_ids)
        positions = torch.tensor([position], device=next_token_ids.device, dtype=torch.long)
        cos_sin_cache = self.model.cos_sin_cache.to(embeds.device)
        hidden = self.mtp(embeds, prev_hidden, positions, cos_sin_cache, mtp_cache)
        logits = self.lm_head(hidden)
        draft_token = logits[:, -1, :].argmax(dim=-1)
        return draft_token, hidden

    def mtp_forward(
        self,
        next_token_ids: torch.Tensor,
        prev_hidden: torch.Tensor,
        start_position: int,
        mtp_cache: Qwen36PagedAttentionCache,
        *,
        logits_last_position_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-force one contiguous MTP suffix.

        Unlike :meth:`mtp_step`, this is the state-synchronisation primitive:
        every row pairs a shifted real token with the target hidden state from
        the immediately preceding target position.  It is used to build and
        refresh the MTP head's *real-prefix* KV before the cheap
        autoregressive continuation steps propose speculative tokens.

        Both tensors are ``[1, q, ...]`` and ``start_position`` must equal
        the cache's live length.  Keeping that equality explicit prevents a
        speculative tail from being accidentally treated as committed MTP
        context after a partial accept.
        """
        assert self.mtp is not None, "mtp_forward called but enable_mtp=False at construction"
        if next_token_ids.dim() != 2 or next_token_ids.shape[0] != 1:
            raise ValueError("mtp_forward next_token_ids must have shape [1, q]")
        if (
            prev_hidden.dim() != 3
            or prev_hidden.shape[0] != 1
            or prev_hidden.shape[1] != next_token_ids.shape[1]
        ):
            raise ValueError("mtp_forward prev_hidden must have shape [1, q, hidden_size]")
        if start_position != mtp_cache.seq_len:
            raise RuntimeError(
                "mtp_forward start_position must equal the MTP cache's live length "
                f"({start_position} != {mtp_cache.seq_len})"
            )
        embeds = self.model.embed_tokens(next_token_ids)
        positions = torch.arange(
            start_position,
            start_position + next_token_ids.shape[1],
            device=next_token_ids.device,
            dtype=torch.long,
        )
        cos_sin_cache = self.model.cos_sin_cache.to(embeds.device)
        hidden = self.mtp(embeds, prev_hidden, positions, cos_sin_cache, mtp_cache)
        logits_hidden = hidden[:, -1:] if logits_last_position_only else hidden
        return self.lm_head(logits_hidden), hidden

    # -- Weight loading ------------------------------------------------

    def load_weights(self, weights) -> set[str]:
        """Route every checkpoint tensor by its own dotted name.

        Most tensors remain 1:1.  GDN input projections are the deliberate
        fixed exception inherited from the historical production path: the
        checkpoint has separate ``qkv``/``z`` and ``b``/``a`` tensors, while
        this graph owns fused ``qkvz``/``ba`` parameters.  Their rows are
        copied directly by :func:`_gdn_fused_checkpoint_slice`; raw E4M3
        bytes and their channel scales are never converted or requantized.
        Top-level prefixes recognized:
        ``model.language_model.`` (backbone), ``lm_head.`` (head),
        ``mtp.`` (B3: loaded into :attr:`mtp` when ``enable_mtp=True`` was
        passed at construction -- the checkpoint's ``mtp.*`` names map
        1:1 onto :class:`Qwen36MTPHead`'s own module tree by construction
        (``self.mtp.fc``, ``self.mtp.layers.0.self_attn.q_proj``, ...), so
        no remapping is needed, unlike the backbone's
        ``model.language_model.`` -> ``model.`` rewrite below; skipped
        (B1's original behavior) when ``enable_mtp=False``, i.e. ``mtp`` is
        ``None``), ``model.visual.`` (should already be filtered out by
        the caller's ``language_model_only`` loader stage; if any slip
        through here, that is exactly the B0-1a/b guarantee failing, so
        this raises rather than silently accepting them).
        """
        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()
        fused_sources: dict[str, set[str]] = {}
        self.skipped_mtp_count = 0
        unrecognized: list[str] = []

        for name, tensor in weights:
            if name.endswith(_IGNORED_WEIGHT_SUFFIXES):
                continue
            if name.startswith("mtp."):
                if self.mtp is None:
                    self.skipped_mtp_count += 1
                    continue
                mapped = name
                if mapped not in params_dict:
                    continue
                param = params_dict[mapped]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, tensor)
                loaded.add(mapped)
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

            fused = _gdn_fused_checkpoint_slice(mapped, tensor, params_dict)
            if fused is not None:
                fused_name, source_name = fused
                fused_sources.setdefault(fused_name, set()).add(source_name)
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
        for fused_name, sources in fused_sources.items():
            if ".in_proj_qkvz." in fused_name:
                expected = {
                    fused_name.replace(".in_proj_qkvz.", ".in_proj_qkv."),
                    fused_name.replace(".in_proj_qkvz.", ".in_proj_z."),
                }
            else:
                assert ".in_proj_ba." in fused_name
                expected = {
                    fused_name.replace(".in_proj_ba.", ".in_proj_b."),
                    fused_name.replace(".in_proj_ba.", ".in_proj_a."),
                }
            missing = expected - sources
            if missing:
                raise RuntimeError(
                    f"Qwen36 fused GDN parameter {fused_name!r} is missing checkpoint "
                    f"slice(s) {sorted(missing)!r}; refusing to run with a partially "
                    "initialized input projection"
                )
            loaded.add(fused_name)
        return loaded
