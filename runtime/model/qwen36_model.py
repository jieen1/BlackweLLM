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
  correctness gate rather than guessing a default. Uses a per-layer
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

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from sparkinfer.attention.paged._forward import paged_attention_forward
from sparkinfer.attention.paged._scratch import build_paged_attention_binding
from sparkinfer.attention.paged.planner import (
    PagedPlanBudget,
    create_paged_plan,
    plan_decode_graph_capacity,
)
from sparkinfer.attention.paged.workspace import PagedAttentionWorkspace
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
    *table* itself is still trivial at batch=1 (one sequence, physical
    pages 0..num_pages-1 in order) -- the complexity this class still
    avoids is B2's real slot manager (sharing/evicting physical pages
    across concurrent sequences), not paging itself.
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
    ) -> Qwen36PagedAttentionCache:
        """Build a cache **over storage someone else owns** (B2's slot pool).

        ``k_cache``/``v_cache`` are ``[num_pages, page_size, num_kv_heads,
        head_dim]`` slices of a per-layer pool -- for slot ``s`` with
        ``P`` pages each, ``pool[s*P:(s+1)*P]``, which is contiguous, so
        this view's *local* page ids ``0..P-1`` address exactly slot
        ``s``'s own pages and nothing else. That is what lets B2's
        single-slot path (prefill, and eager per-slot decode) be the
        **same code** ``__init__``-allocated caches run in B1: identical
        page table, identical ``append`` arithmetic, identical kernel
        call. The batched decode path addresses the same bytes through
        the *global* page table instead (``slot*P + local``); both views
        are over one allocation, so they never disagree.

        Deliberately a second constructor rather than an ``__init__``
        keyword: ``__init__`` allocating is the invariant B1's callers
        rely on, and a ``storage=None`` parameter would make "who owns
        this memory" a runtime question at every call site instead of a
        choice visible in the constructor's name.
        """
        obj = cls.__new__(cls)
        obj.page_size = page_size
        obj.num_pages = int(k_cache.shape[0])
        obj.max_seq_len = obj.num_pages * page_size
        obj.num_kv_heads = int(k_cache.shape[2])
        obj.head_dim = int(k_cache.shape[3])
        obj.dtype = k_cache.dtype
        obj.device = k_cache.device
        obj.k_cache = k_cache
        obj.v_cache = v_cache
        obj.page_table = torch.arange(
            obj.num_pages, dtype=torch.int32, device=k_cache.device
        ).unsqueeze(0)
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
        self.k_cache[page_ids, offsets] = new_k.to(self.dtype)
        self.v_cache[page_ids, offsets] = new_v.to(self.dtype)
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

    @property
    def page_table(self) -> torch.Tensor:
        return self.attn.page_table

    @property
    def cache_seqlens(self) -> torch.Tensor:
        return self.attn.cache_seqlens


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

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)  # [b, conv_dim, 1]
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

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
            query, key, value, g=g, beta=beta,
            initial_state=recurrent_state, output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        recurrent_state.copy_(last_state)

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z_flat = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z_flat)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        return self.out_proj(core_attn_out)


# ---------------------------------------------------------------------------
# Full attention layer (sparkinfer paged attention).
# ---------------------------------------------------------------------------


class Qwen36AttentionWorkspace:
    """Fixed-capacity sparkinfer paged-attention workspace for one
    ``Qwen36Attention`` layer, covering both ``mode="extend"`` (prefill)
    and ``mode="decode"`` (single-token continuation) -- no ``"verify"``
    (B1 has no speculative decoding) and no ``window_left``/SWA (Qwen3.6's
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
    layers at once). This class is instead constructed **per layer**
    (``Qwen36Attention.__init__`` builds its own) -- correct (each
    instance's fixed capacity is still honored, and sparkinfer's own
    compile cache is keyed by shape parameters below the level of this
    Python object, so cross-layer compiles very likely still dedupe), but
    it allocates ``Qwen36Attention.num_layers``-times the scratch memory a
    single shared instance would. Given all 16 full-attention layers in
    this checkpoint share identical
    ``num_q_heads``/``num_kv_heads``/``head_dim``/``page_size``, sharing
    one instance across all of them the way Laguna does is a real,
    concrete follow-up -- not attempted here because it was not needed to
    fix the actual bug within this pass's time budget.

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
    ) -> None:
        if mode not in ("extend", "decode"):
            raise ValueError(f"Qwen36AttentionWorkspace: unsupported mode {mode!r}")
        self.mode = mode
        self.max_batch = max_batch
        self._descale = torch.ones(1, dtype=torch.float32, device=device)
        # eager_extend_work_items_capacity is sparkinfer's own estimator
        # for exactly this pair of modes (its name and design track
        # max_total_q, which is what extend/decode's real work-item count
        # scales with) -- see SparkinferPrefillWorkspace's docstring for
        # why this estimator specifically does NOT generalize to
        # mode="verify" (not a concern here: B1 has none).
        max_work_items = PagedAttentionWorkspace.eager_extend_work_items_capacity(
            max_total_q=max_total_q, num_q_heads=num_q_heads, num_kv_heads=num_kv_heads
        )
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
            max_partial_rows=0,  # matches PagedExtendGraphCapacity: no split-KV merge buffer
            num_cache_pages=num_cache_pages,
            use_cuda_graph=False,
        )
        # Declares this workspace's capacity to the planner so the plan's
        # own policy knobs -- above all ``cta_tile_q``, which is part of
        # sparkinfer's compile cache key -- are derived from capacity, not
        # from the live request's query length. See the class docstring.
        self._plan_budget = PagedPlanBudget(
            max_total_q=max_total_q,
            max_batch=max_batch,
            max_page_table_width=max_page_table_width,
        )
        self._prepared_metadata: object | None = None

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
    ) -> None:
        """Run attention, writing into ``output`` in place. First call
        against this instance pays sparkinfer's one-time CuTe compile
        (or hits its on-disk cache from a prior process); every later
        call at any shape within this instance's declared capacity reuses
        it -- including query lengths never seen before, which is what
        ``plan_budget`` buys (class docstring)."""
        plan = create_paged_plan(
            q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q,
            mode=self.mode, enable_cuda_graph=False, window_left=-1,
            plan_budget=self._plan_budget,
        )
        ws = self._workspace
        ws._ensure_capacity(plan)
        ws._copy_runtime_metadata(page_table, cache_seqlens, cu_seqlens_q)
        ws._copy_plan_metadata(plan)
        ws._plan = plan
        binding = build_paged_attention_binding(
            scratch=ws, q=q, k_cache=k_cache, v_cache=v_cache, output=output,
            k_descale=self._descale, v_descale=self._descale,
        )
        paged_attention_forward(binding=binding)


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
        self._descale = torch.ones(1, dtype=torch.float32, device=device)
        self.page_table = torch.zeros(
            batch, pages_per_slot, dtype=torch.int32, device=device
        )
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
    ) -> None:
        plan = create_paged_plan(
            q, k_cache, v_cache, self.page_table, self.cache_seqlens, self.cu_seqlens_q,
            mode="decode", enable_cuda_graph=False, window_left=-1,
            plan_budget=self._plan_budget,
        )
        ws = self._workspace
        ws._ensure_capacity(plan)
        ws._copy_runtime_metadata(self.page_table, self.cache_seqlens, self.cu_seqlens_q)
        ws._copy_plan_metadata(plan)
        ws._plan = plan
        binding = build_paged_attention_binding(
            scratch=ws, q=q, k_cache=k_cache, v_cache=v_cache, output=output,
            k_descale=self._descale, v_descale=self._descale,
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
        self._descale = torch.ones(1, dtype=torch.float32, device=device)
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
        self._workspace.prepare_decode_graph_replay_state(
            batch=batch,
            max_page_table_width=pages_per_slot,
            total_q_capacity=batch,
            max_cache_page_count=num_cache_pages,
            window_left=-1,
        )
        # Bind the WORST case before capture, not a representative one: the
        # graph's schedule is fixed at capture time, so a shorter context
        # captured here would silently under-serve every longer one later.
        capture_page_table = torch.arange(
            pages_per_slot, dtype=torch.int32, device=device
        ).unsqueeze(0).repeat(batch, 1)
        capture_cache_seqlens = torch.full(
            (batch,), max_seq_len, dtype=torch.int32, device=device
        )
        self.cu_seqlens_q = torch.arange(batch + 1, dtype=torch.int32, device=device)
        self._workspace._copy_runtime_metadata(
            capture_page_table, capture_cache_seqlens, self.cu_seqlens_q
        )
        # From here on these ARE the buffers the caller writes each step.
        self.page_table = self._workspace.page_table
        self.cache_seqlens = self._workspace.cache_seqlens

    def forward(
        self,
        *,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        binding = build_paged_attention_binding(
            scratch=self._workspace, q=q, k_cache=k_cache, v_cache=v_cache,
            output=output, k_descale=self._descale, v_descale=self._descale,
        )
        paged_attention_forward(binding=binding)


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

        # Built lazily on first forward() call, from the actual observed
        # cache/dtype -- safe because every Qwen36GenerationState this
        # layer instance will ever see comes from the SAME model instance
        # (same max_seq_len, same cache.num_pages every time; a second
        # model instance with a different max_seq_len gets its own
        # Qwen36Attention layers, hence its own workspaces). See
        # Qwen36AttentionWorkspace's docstring for why this exists.
        self._extend_workspace: Qwen36AttentionWorkspace | None = None
        self._decode_workspace: Qwen36AttentionWorkspace | None = None

    def _workspace_for(
        self,
        mode: str,
        cache: Qwen36PagedAttentionCache,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Qwen36AttentionWorkspace:
        attr = "_extend_workspace" if mode == "extend" else "_decode_workspace"
        existing = getattr(self, attr)
        if existing is not None:
            return existing
        workspace = Qwen36AttentionWorkspace(
            mode=mode,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            page_size=cache.page_size,
            max_total_q=self.max_seq_len,
            max_page_table_width=cache.num_pages,
            num_cache_pages=cache.num_pages,
            dtype=dtype,
            kv_dtype=cache.k_cache.dtype,
            device=device,
        )
        setattr(self, attr, workspace)
        return workspace

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

        _past_len, total_len = cache.append(key.to(cache.dtype), value.to(cache.dtype))
        mode = "decode" if seq_len == 1 else "extend"

        workspace = self._workspace_for(mode, cache, query.dtype, query.device)
        output = torch.empty(
            seq_len, self.num_heads, self.head_dim, dtype=query.dtype, device=query.device
        )
        cache_seqlens = torch.tensor([total_len], dtype=torch.int32, device=query.device)
        cu_seqlens_q = torch.tensor([0, seq_len], dtype=torch.int32, device=query.device)
        needs_cast = query.dtype != cache.k_cache.dtype
        q_for_kernel = query.to(cache.k_cache.dtype) if needs_cast else query
        workspace.forward(
            q=q_for_kernel,
            k_cache=cache.k_cache,
            v_cache=cache.v_cache,
            output=output,
            page_table=cache.page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
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

        q_and_gate = self.q_proj(hidden_states)
        q_and_gate = q_and_gate.view(batch_size, self.num_heads, self.head_dim * 2)
        query, gate = torch.chunk(q_and_gate, 2, dim=-1)
        gate = gate.reshape(batch_size, 1, -1)

        kv_shape = (batch_size, self.num_kv_heads, self.head_dim)
        query = self.q_norm(query)
        key = self.k_norm(self.k_proj(hidden_states).view(*kv_shape))
        value = self.v_proj(hidden_states).view(*kv_shape)

        query_flat = query.reshape(batch_size, self.num_heads * self.head_dim).contiguous()
        key_flat = key.reshape(batch_size, self.num_kv_heads * self.head_dim).contiguous()
        apply_rotary_embedding_inplace(positions, query_flat, self.head_dim, cos_sin_cache)
        apply_rotary_embedding_inplace(positions, key_flat, self.head_dim, cos_sin_cache)
        query = query_flat.view(batch_size, self.num_heads, self.head_dim)
        key = key_flat.view(batch_size, self.num_kv_heads, self.head_dim)

        k_flat = k_pool.view(-1, self.num_kv_heads, self.head_dim)
        v_flat = v_pool.view(-1, self.num_kv_heads, self.head_dim)
        k_flat.index_copy_(0, write_index, key.to(k_pool.dtype))
        v_flat.index_copy_(0, write_index, value.to(v_pool.dtype))

        needs_cast = query.dtype != k_pool.dtype
        q_for_kernel = query.to(k_pool.dtype) if needs_cast else query
        attn.forward(
            q=q_for_kernel,
            k_cache=k_pool,
            v_cache=v_pool,
            output=output,
        )

        attn_out = output.reshape(batch_size, 1, -1)
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

    def decode_batch(
        self,
        hidden_states: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        batch: Qwen36DecodeBatch,
    ) -> torch.Tensor:
        """:meth:`forward`'s structure with the batched, globally addressed
        decode kernels substituted for the single-sequence ones.

        The gather/scatter of the recurrent state lives here rather than
        inside :meth:`Qwen36GatedDeltaNet.decode_batch` so that method
        stays a pure function of its arguments (see its docstring): this
        is the only place in the decode path where "batch row i is slot
        ``slot_index[i]``" is known.
        """
        i = self.layer_idx
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

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
            )

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
        for layer in self.layers:
            hidden_states = layer.decode_batch(hidden_states, cos_sin_cache, batch)
        return self.norm(hidden_states)


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

    def warmup_attention_shapes(
        self, *, device: torch.device | str, dtype: torch.dtype
    ) -> None:
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
                q = torch.zeros(
                    seq_len, attn.num_heads, attn.head_dim, dtype=cache.dtype, device=device
                )
                workspace.forward(
                    q=q,
                    k_cache=cache.k_cache,
                    v_cache=cache.v_cache,
                    output=torch.empty_like(q),
                    page_table=cache.page_table,
                    cache_seqlens=torch.tensor(
                        [seq_len], dtype=torch.int32, device=device
                    ),
                    cu_seqlens_q=torch.tensor(
                        [0, seq_len], dtype=torch.int32, device=device
                    ),
                )
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def forward(
        self,
        input_ids: torch.Tensor,
        state: Qwen36GenerationState,
        *,
        capture_hidden_states: bool = False,
    ):
        return self.model(input_ids, state, capture_hidden_states=capture_hidden_states)

    def decode_batch(self, batch: Qwen36DecodeBatch) -> torch.Tensor:
        """Batched decode -> ``[B, vocab_size]`` logits (B2).

        Returns logits, not hidden states, because that is the whole of
        what a decode step is for and because keeping ``lm_head`` inside
        the graphed region is what makes a captured step self-contained.
        """
        hidden_states = self.model.decode_batch(batch)
        return self.lm_head(hidden_states.reshape(hidden_states.shape[0], -1))

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
