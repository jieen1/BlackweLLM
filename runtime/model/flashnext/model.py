"""Flash-Next full-model assembly (bring-up forward graph).

Sequence: embed_tokens -> 48 decoder layers (12 x (3x GDN-MoE + 1x QSA-MoE),
each wrapped in gated-residual hyper-connections, PLE injection at layer
index 1) -> final hyper_connection_mixer.mix -> lm_head. There is no
standalone final RMSNorm: the mixer's grouped Gemma norm closes the stream
(confirmed: no norm.weight tensor exists in the checkpoint).

Bring-up scope: single-sequence, explicit per-layer states, gather-based QSA
attention. Paging, batching, CUDA graphs and MTP land with the backend.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
from dataclasses import dataclass, field

import torch
from torch import nn

from runtime.backends.flashnext_moe import (
    FlashInferMoELayer,
    SparkinferMoELayer,
    SparkinferMoEOutputArena,
    allocate_tp_moe_workspace_pool,
    flashnext_flashinfer_moe_available,
    load_flashnext_experts,
    prepare_flashnext_cutlass_experts,
    prepare_flashnext_experts,
)
from runtime.model.flashnext.hyper_connection import GatedResidual
from runtime.model.flashnext.ple import (
    FlashNextPleHasher,
    FlashNextPLELayer,
    FlashNextPleTable,
)
from runtime.model.flashnext.qsa import (
    FlashNextQSAAttention,
    QSAIndexer,
    _normalize_rope_positions,
    _qsa_cache_is_quantized,
    load_qsa_attention,
    load_qsa_indexer,
    qsa_cache_index_copy_,
    qsa_index_cache_rows,
    qsa_kv_cache_dtype,
    quantize_qsa_kv,
)
from runtime.model.qwen36_model import GdnLayerState, Qwen36GatedDeltaNet


@dataclass(frozen=True)
class FlashNextTextConfig:
    hidden_size: int
    num_layers: int
    layer_types: tuple[str, ...]
    vocab_size: int
    eos_token_id: int
    hc_count: int
    hc_lowrank: int
    rms_norm_eps: float
    rope_theta: float
    ple_layer_ids: tuple[int, ...]
    ngram_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    mamba_ssm_dtype: str
    # Keep the architecture identifier in the runtime config.  The serving
    # backend uses it to distinguish the validated qwen4_exp numerical
    # contract from a future/third-party Flash-Next-like checkpoint before it
    # enables the large-M GDN verify projections.  Empty is conservative for
    # direct unit-test construction that does not originate from a checkpoint.
    model_type: str = ""
    # Multimodal checkpoints keep the image marker in the top-level config,
    # not in text_config. None preserves the text-only unit-test fixtures.
    image_token_id: int | None = None
    # Qwen-VL uses three-axis MRoPE for visual tokens.  Keep these trailing
    # defaults so the text-only fixtures and the MTP model retain scalar RoPE.
    mrope_section: tuple[int, ...] | None = None
    mrope_interleaved: bool = False
    vision_start_token_id: int | None = None
    vision_end_token_id: int | None = None

    @classmethod
    def from_checkpoint(cls, ckpt: pathlib.Path | str) -> FlashNextTextConfig:
        with open(pathlib.Path(ckpt) / "config.json") as f:
            full_config = json.load(f)
        tc = full_config["text_config"]
        rope_parameters = tc.get("rope_parameters", {})
        mrope_section = rope_parameters.get("mrope_section")
        return cls(
            model_type=str(full_config.get("model_type", "")),
            hidden_size=tc["hidden_size"],
            num_layers=tc["num_hidden_layers"],
            layer_types=tuple(tc["layer_types"]),
            vocab_size=tc["vocab_size"],
            eos_token_id=tc["eos_token_id"],
            hc_count=tc["hc_count"],
            hc_lowrank=tc["hc_lowrank"],
            rms_norm_eps=tc["rms_norm_eps"],
            rope_theta=float(rope_parameters["rope_theta"]),
            ple_layer_ids=tuple(tc.get("ple_layer_ids", [])),
            ngram_size=tc["ngram_size"],
            num_attention_heads=tc["num_attention_heads"],
            num_key_value_heads=tc["num_key_value_heads"],
            head_dim=tc["head_dim"],
            num_experts=tc["num_experts"],
            num_experts_per_tok=tc["num_experts_per_tok"],
            moe_intermediate_size=tc["moe_intermediate_size"],
            shared_expert_intermediate_size=tc["shared_expert_intermediate_size"],
            mamba_ssm_dtype=tc.get("mamba_ssm_dtype", tc.get("dtype", "bfloat16")),
            image_token_id=full_config.get("image_token_id"),
            mrope_section=(tuple(int(value) for value in mrope_section)
                           if mrope_section is not None else None),
            mrope_interleaved=bool(rope_parameters.get("mrope_interleaved", False)),
            vision_start_token_id=(
                int(full_config["vision_start_token_id"])
                if full_config.get("vision_start_token_id") is not None
                else None
            ),
            vision_end_token_id=(
                int(full_config["vision_end_token_id"])
                if full_config.get("vision_end_token_id") is not None
                else None
            ),
        )


class SharedExpert(nn.Module):
    """gate/up fused into ONE [2I, H] GEMV (halves launches on the decode
    critical path); weights fused at load time from the checkpoint pair."""

    def __init__(self, hidden: int, inter: int, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.inter = inter
        self.gate_up_proj = nn.Linear(hidden, 2 * inter, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(inter, hidden, bias=False, dtype=dtype)
        self.shared_gate = nn.Linear(hidden, 1, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.shared_gate(x))
        gu = self.gate_up_proj(x)
        gate, up = gu.split(self.inter, dim=-1)
        return g * self.down_proj(torch.nn.functional.silu(gate) * up)


@dataclass
class _MlpPrefillGraph:
    input: torch.Tensor
    output: torch.Tensor
    graph: torch.cuda.CUDAGraph


def _flashnext_moe_backend() -> str:
    # Keep the qualified b12x path as the production default.  The local
    # FlashInfer Python/cubin versions do not match and selecting it through
    # ``auto`` triggers a multi-minute CUTLASS JIT during model load.
    backend = os.getenv("QSR_FLASHNEXT_MOE_BACKEND", "b12x").strip().lower()
    if backend not in {"auto", "flashinfer", "b12x", "sparkinfer"}:
        raise ValueError(
            "QSR_FLASHNEXT_MOE_BACKEND must be auto, flashinfer, b12x, or sparkinfer; "
            f"got {backend!r}"
        )
    if backend == "auto":
        return "flashinfer" if flashnext_flashinfer_moe_available() else "b12x"
    if backend == "sparkinfer":
        return "b12x"
    if backend == "flashinfer" and not flashnext_flashinfer_moe_available():
        raise RuntimeError(
            "QSR_FLASHNEXT_MOE_BACKEND=flashinfer requested but FlashInfer is unavailable"
        )
    return backend


class FlashNextMlp(nn.Module):
    """Router -> routed NVFP4 experts + sigmoid-gated shared expert."""

    def __init__(
        self,
        hidden: int,
        num_experts: int,
        top_k: int,
        expert_layer: SparkinferMoELayer | FlashInferMoELayer,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(hidden, num_experts, bias=False, dtype=dtype)
        self.expert_layer = expert_layer
        self._router_weights: torch.Tensor | None = None
        self._router_ids: torch.Tensor | None = None
        self._prefill_graphs: dict[int, _MlpPrefillGraph] = {}

    def _forward_eager(
        self,
        x: torch.Tensor,
        *,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from runtime.kernels.qwen_moe_router import qwen_moe_softmax_topk

        rows = x.shape[0]
        debug_skip_router = os.environ.get("QSR_FLASHNEXT_DEBUG_SKIP_ROUTER", "0") == "1"
        router_impl = os.environ.get("QSR_FLASHNEXT_ROUTER_IMPL", "triton").strip().lower()
        if router_impl not in {"triton", "torch"}:
            raise ValueError(
                "QSR_FLASHNEXT_ROUTER_IMPL must be triton or torch, "
                f"got {router_impl!r}"
            )
        debug_skip_gate = os.environ.get("QSR_FLASHNEXT_DEBUG_SKIP_GATE", "0") == "1"
        if debug_skip_router:
            weights = torch.zeros(rows, self.top_k, dtype=x.dtype, device=x.device)
            ids = torch.zeros(rows, self.top_k, dtype=torch.int32, device=x.device)
        else:
            if debug_skip_gate:
                logits = torch.zeros(
                    rows,
                    self.gate.out_features,
                    dtype=x.dtype,
                    device=x.device,
                )
            else:
                logits = self.gate(x)
            if (
                self._router_weights is None
                or self._router_weights.shape[0] < rows
            ):
                cap = max(rows, 64)
                self._router_weights = torch.empty(cap, self.top_k, dtype=x.dtype, device=x.device)
                self._router_ids = torch.empty(cap, self.top_k, dtype=torch.int32, device=x.device)
            weights_out = self._router_weights[:rows]
            ids_out = self._router_ids[:rows]
            if router_impl == "torch":
                probs = torch.softmax(logits.float(), dim=-1)
                selected, selected_ids = torch.topk(probs, self.top_k, dim=-1)
                selected = selected / selected.sum(dim=-1, keepdim=True)
                if (
                    os.environ.get("QSR_FLASHNEXT_DEBUG_ROUTER_DIRECT", "0") == "1"
                    or torch.cuda.is_current_stream_capturing()
                ):
                    weights = selected.to(dtype=x.dtype)
                    ids = selected_ids.to(dtype=torch.int32)
                else:
                    weights_out.copy_(selected.to(dtype=x.dtype))
                    ids_out.copy_(selected_ids.to(dtype=torch.int32))
                    weights, ids = weights_out, ids_out
            else:
                if torch.cuda.is_current_stream_capturing():
                    # External router arenas are grown during the warm-up
                    # phase.  A graph capture must own the output allocations
                    # it writes; otherwise replay can retain a pointer to an
                    # arena view that another graph/capture has recycled.
                    weights, ids = qwen_moe_softmax_topk(
                        logits,
                        self.top_k,
                    )
                else:
                    weights, ids = qwen_moe_softmax_topk(
                        logits,
                        self.top_k,
                        weights_out=weights_out,
                        ids_out=ids_out,
                    )
        if os.environ.get("QSR_FLASHNEXT_DEBUG_SKIP_ROUTED", "0") == "1":
            routed = torch.zeros_like(x)
        else:
            routed = self.expert_layer.forward(x, ids, weights)
        if os.environ.get("QSR_FLASHNEXT_DEBUG_SKIP_SHARED", "0") == "1":
            shared = torch.zeros_like(routed)
        else:
            shared = self.shared(x)
        if output is None:
            return routed + shared
        torch.add(routed, shared, out=output)
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        graph = self._prefill_graphs.get(x.shape[0])
        if graph is None:
            return self._forward_eager(x)
        graph.input.copy_(x)
        graph.graph.replay()
        return graph.output

    @torch.no_grad()
    def capture_prefill_graph(self, rows: int, *, pool=None) -> None:
        """Capture the complete fixed-row prefill MLP without changing math.

        The static output is allocated outside the graph pool.  That lets all
        layer graphs share transient graph-private allocations safely: each
        graph is replayed in layer order and its output is consumed before the
        next layer reuses the pool.
        """
        if rows <= 0:
            raise ValueError(f"prefill graph rows must be positive, got {rows}")
        if rows in self._prefill_graphs:
            return
        device = self.gate.weight.device
        if device.type != "cuda":
            raise ValueError("Flash-Next prefill MLP graphs require CUDA")

        static_input = torch.empty(
            rows,
            self.gate.in_features,
            dtype=self.gate.weight.dtype,
            device=device,
        )
        static_input.normal_(mean=0.0, std=0.02)
        static_output = torch.empty_like(static_input)

        warmup_stream = torch.cuda.Stream(device=device)
        warmup_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(warmup_stream):
            reference = self._forward_eager(static_input).clone()
            self._forward_eager(static_input, output=static_output)
        torch.cuda.current_stream(device).wait_stream(warmup_stream)
        torch.cuda.synchronize(device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            self._forward_eager(static_input, output=static_output)
        torch.cuda.synchronize(device)
        if not torch.equal(static_output, reference):
            max_abs = float((static_output.float() - reference.float()).abs().max())
            raise RuntimeError(
                "Flash-Next prefill MLP graph failed its capture-time bitwise gate: "
                f"rows={rows}, max_abs={max_abs}"
            )
        self._prefill_graphs[rows] = _MlpPrefillGraph(
            input=static_input,
            output=static_output,
            graph=graph,
        )


class QsaLayerBundle(nn.Module):
    """Indexer + main attention with bring-up gather attention."""

    def __init__(self, indexer: QSAIndexer, attn: FlashNextQSAAttention) -> None:
        super().__init__()
        self.indexer = indexer
        self.attn = attn

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        t = x.shape[0]
        qi, ki = self.indexer.project_qk(x, positions)
        pooled = self.indexer.pool_keys(ki)
        ends = torch.tensor(
            [max(0, (int(p) - 3) // self.indexer.compress_ratio + 1) for p in positions.tolist()],
            device=x.device,
        )
        logits = self.indexer.score_blocks(qi, pooled, ends)
        blocks = self.indexer.select_blocks(logits, ends)  # [T, <=512]
        q, k, v, gate = self.attn.project(x, positions)
        # materialize per-layer KV over the full sequence (bring-up)
        k_all, v_all = k, v
        selected, selected_valid = self.indexer.batch_decode_gather_indices(
            blocks,
            positions,
            self.indexer.block_topk * self.indexer.compress_ratio + self.indexer.compress_ratio - 1,
        )
        out_rows = []
        for i in range(t):
            idx = selected[i, selected_valid[i]]
            ksel = k_all[idx].repeat_interleave(self.attn.repeat, dim=1)
            vsel = v_all[idx].repeat_interleave(self.attn.repeat, dim=1)
            scale = 1.0 / math.sqrt(self.attn.head_dim)
            scores = torch.einsum("hd,shd->hs", q[i].float(), ksel.float()) * scale
            a = torch.softmax(scores, dim=-1)
            o = torch.einsum("hs,shd->hd", a, vsel.float()).to(q.dtype)
            o = o * torch.sigmoid(gate[i].float()).to(q.dtype)
            out_rows.append(self.attn.o_proj(o.reshape(-1)))
        return torch.stack(out_rows, dim=0)


class FlashNextLayer(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        cfg: FlashNextTextConfig,
        attn_module,
        is_qsa: bool,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.cfg = cfg
        self.is_qsa = is_qsa
        self.attn = attn_module
        hc_kw = dict(
            hc_count=cfg.hc_count,
            hidden_size=cfg.hidden_size,
            lowrank=cfg.hc_lowrank,
            eps=cfg.rms_norm_eps,
        )
        self.attn_hc = GatedResidual(use_mix=True, use_combine=True, **hc_kw)
        self.mlp_hc = GatedResidual(use_mix=True, use_combine=True, **hc_kw)
        self.mlp: FlashNextMlp | None = None
        self.ple: FlashNextPLELayer | None = None
        self.ple_hasher: FlashNextPleHasher | None = None

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        states: dict,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cfg = self.cfg
        if x.shape[-1] == cfg.hidden_size:
            x = torch.cat([x] * cfg.hc_count, dim=-1)
        if self.ple is not None and input_ids is not None:
            ids = self.ple_hasher.sequence_ids(input_ids)
            emb = self.ple.embed(ids, device=x.device)
            gated_flat, normed_flat = self.ple.inject(emb, x)
            x = x + gated_flat + self.ple.prefill_conv(normed_flat, [x.shape[0]])
        mixed, residuals = self.attn_hc.mix(x)
        if self.is_qsa:
            attn_out = self.attn(mixed, positions)
        else:
            gdn_state = states[f"gdn_{self.layer_idx}"]
            attn_out = self.attn(mixed.unsqueeze(0), gdn_state).squeeze(0)
        x = self.attn_hc.combine(attn_out, residuals)
        mixed2, res2 = self.mlp_hc.mix(x)
        x = self.mlp_hc.combine(self.mlp(mixed2), res2)
        return x


class FlashNextModel(nn.Module):
    def __init__(self, cfg: FlashNextTextConfig, vision_tower: nn.Module | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.vision_tower = vision_tower
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList()
        self.ple_table: FlashNextPleTable | None = None
        # CUDA Graphs capture raw addresses for the b12x MoE scratch and
        # routed-output arena.  The eager prefill path is allowed to grow
        # those capacity caches, which replaces the backing tensors.  Keep
        # every allocation observed by a live graph strongly referenced so a
        # later ``torch.cuda.empty_cache()`` cannot return it to the allocator
        # while the graph still contains its address.
        self._graph_moe_allocations: list[object] = []
        self.final_mixer = GatedResidual(
            hc_count=cfg.hc_count,
            hidden_size=cfg.hidden_size,
            lowrank=cfg.hc_lowrank,
            eps=cfg.rms_norm_eps,
            use_mix=True,
            use_combine=False,
        )
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def _retain_graph_moe_allocations(self) -> None:
        """Pin b12x/arena tensors used by captured target graphs.

        b12x's pool is intentionally grow-only, but growth replaces the
        Python-visible workspace object.  CUDA Graphs do not retain the
        caller-owned scratch tensors for us, so the old object could become
        unreachable after a large prefill and ``empty_cache`` could free its
        address.  Graph replay then reports an asynchronous illegal access.
        Retaining the small set of graph-era objects is cheaper and safer than
        disabling the memory trim for the whole request.
        """
        seen = {id(value) for value in self._graph_moe_allocations}
        for layer in self.layers:
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                continue
            expert_layer = getattr(mlp, "expert_layer", None)
            if expert_layer is None:
                continue
            arena = getattr(expert_layer, "_output_arena", None)
            arena_buffer = getattr(arena, "buffer", None)
            if arena_buffer is not None and id(arena_buffer) not in seen:
                self._graph_moe_allocations.append(arena_buffer)
                seen.add(id(arena_buffer))
            workspace = getattr(expert_layer, "workspace", None)
            for value in getattr(workspace, "workspaces", {}).values():
                if id(value) not in seen:
                    self._graph_moe_allocations.append(value)
                    seen.add(id(value))

    def close(self) -> None:
        if self.ple_table is not None:
            self.ple_table.close()
            self.ple_table = None

    @torch.no_grad()
    def encode_multimodal(self, input_ids: list[int] | torch.Tensor, prepared) -> torch.Tensor:
        """Fuse compressed Qwen3-VL features into text token embeddings.

        The server expands every single template image marker to the exact
        number of patch-merged visual tokens before calling this method. The
        strict shape check here turns a tokenizer/processor mismatch into a
        request error instead of silently shifting all following text.
        """

        if self.vision_tower is None:
            raise RuntimeError(
                "Flash-Next vision is disabled; set QSR_FLASHNEXT_VISION=1 and restart"
            )
        device = next(self.parameters()).device
        tokens = torch.as_tensor(input_ids, dtype=torch.long, device=device)
        image_token_id = self.cfg.image_token_id
        if image_token_id is None:
            raise RuntimeError("Flash-Next config does not define image_token_id")
        pixel_values = torch.as_tensor(
            prepared.pixel_values,
            dtype=torch.bfloat16,
            device=device,
        )
        image_grid_thw = torch.as_tensor(
            prepared.image_grid_thw,
            dtype=torch.long,
            device=device,
        )
        vision_output = self.vision_tower(
            pixel_values,
            grid_thw=image_grid_thw,
            return_dict=True,
        )
        features = getattr(vision_output, "pooler_output", None)
        if features is None:
            features = vision_output[0]
        features = features.to(device=device, dtype=self.embed_tokens.weight.dtype)
        mask = tokens == int(image_token_id)
        expected = int(mask.sum().item())
        actual = int(features.shape[0])
        if expected != actual:
            raise RuntimeError(
                "Flash-Next vision/text token mismatch: "
                f"image_slots={expected}, vision_features={actual}"
            )
        embeds = self.embed_tokens(tokens)
        if expected:
            embeds[mask] = features
        return embeds

    def forward(self, input_ids: torch.Tensor, states: dict, pos_offset: int = 0) -> torch.Tensor:
        positions = torch.arange(
            pos_offset, pos_offset + input_ids.shape[0], device=input_ids.device
        )
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, positions, states, input_ids=input_ids)
        x, _ = self.final_mixer.mix(x)
        return self.lm_head(x).float()


def _gdn_config_dict(tc: dict) -> dict:
    return {
        "hidden_size": tc["hidden_size"],
        "linear_num_value_heads": tc["linear_num_value_heads"],
        "linear_num_key_heads": tc["linear_num_key_heads"],
        "linear_key_head_dim": tc["linear_key_head_dim"],
        "linear_value_head_dim": tc["linear_value_head_dim"],
        "linear_conv_kernel_dim": tc["linear_conv_kernel_dim"],
        "rms_norm_eps": tc["rms_norm_eps"],
        "hidden_act": tc["hidden_act"],
        "output_gate_type": tc.get("output_gate_type", "silu"),
    }


def new_layer_states(model: FlashNextModel, device) -> dict:
    recurrent_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }.get(model.cfg.mamba_ssm_dtype)
    if recurrent_dtype is None:
        raise ValueError(
            f"unsupported Flash-Next mamba_ssm_dtype={model.cfg.mamba_ssm_dtype!r}"
        )
    states: dict = {}
    for layer in model.layers:
        if not layer.is_qsa:
            gdn = layer.attn
            states[f"gdn_{layer.layer_idx}"] = GdnLayerState(
                conv_state=torch.zeros(
                    1,
                    gdn.conv_dim,
                    gdn.conv_kernel_size,
                    dtype=torch.bfloat16,
                    device=device,
                ),
                recurrent_state=torch.zeros(
                    1,
                    gdn.num_v_heads,
                    gdn.head_k_dim,
                    gdn.head_v_dim,
                    dtype=recurrent_dtype,
                    device=device,
                ),
            )
    return states


@dataclass
class FlashNextSession:
    """Persistent per-sequence state for incremental decode."""

    gdn: dict
    qsa_k: dict
    qsa_v: dict
    qsa_idx_k: dict
    qsa_attn: dict
    qsa_pad: int
    ple_conv_state: torch.Tensor | None
    window: list
    # Keep this optional for callers that construct a session fixture
    # positionally (the pre-MRoPE ABI did not expose a rope-history map).
    # Production sessions still pass the map explicitly from ``new_session``.
    qsa_idx_rope: dict = field(default_factory=dict)
    pos: int = 0
    # graph-mode fixed pools (allocated lazily by prepare_graph_buffers)
    qsa_k_pool: dict | None = None
    qsa_v_pool: dict | None = None
    qsa_idx_k_pool: dict | None = None
    qsa_pooled_k_pool: dict | None = None
    qsa_k_scale_pool: dict | None = None
    qsa_v_scale_pool: dict | None = None
    qsa_idx_rope_pool: dict | None = None
    token_buf: torch.Tensor | None = None
    pos_buf: torch.Tensor | None = None
    rope_pos_buf: torch.Tensor | None = None
    rope_next: torch.Tensor | None = None
    ends_buf: dict | None = None
    ple_emb_buf: torch.Tensor | None = None
    # One graph-owned candidate row per QSA layer.  ``pool[pos]`` is an
    # advanced-indexing copy, so quantizing into it never updates the cache.
    # Decode uses these rows before committing with ``index_copy_``.
    qsa_k_row: dict[int, torch.Tensor] | None = None
    qsa_v_row: dict[int, torch.Tensor] | None = None
    qsa_k_scale_row: dict[int, torch.Tensor] | None = None
    qsa_v_scale_row: dict[int, torch.Tensor] | None = None


def prepare_graph_buffers(
    model: FlashNextModel,
    sess: FlashNextSession,
    device,
    max_seq: int = 4096,
    *,
    fixed_index_rows: int = 4,
) -> None:
    """Allocate fixed-address pools so the decode body is graph-capturable."""
    cfg = model.cfg
    sess.qsa_k_pool = {}
    sess.qsa_v_pool = {}
    sess.qsa_idx_k_pool = {}
    sess.qsa_pooled_k_pool = {}
    sess.qsa_k_scale_pool = {}
    sess.qsa_v_scale_pool = {}
    sess.qsa_idx_rope_pool = {}
    sess.ends_buf = {}
    sess.qsa_k_row = {}
    sess.qsa_v_row = {}
    sess.qsa_k_scale_row = {}
    sess.qsa_v_scale_row = {}
    kv_dtype = qsa_kv_cache_dtype()
    for layer in model.layers:
        if layer.is_qsa:
            attn = layer.attn.attn
            sess.qsa_k_pool[layer.layer_idx] = torch.zeros(
                max_seq,
                attn.num_kv_heads,
                attn.head_dim,
                dtype=kv_dtype,
                device=device,
            )
            sess.qsa_v_pool[layer.layer_idx] = torch.zeros_like(sess.qsa_k_pool[layer.layer_idx])
            # FP8 K/V uses one FP16 scale per token/KV-head.  Keep the planes
            # allocated for BF16 too so every decode/prefill call has a
            # graph-stable pointer and can switch dtype without rebuilding
            # the session object.
            sess.qsa_k_scale_pool[layer.layer_idx] = torch.ones(
                max_seq,
                attn.num_kv_heads,
                dtype=torch.float16,
                device=device,
            )
            sess.qsa_v_scale_pool[layer.layer_idx] = torch.ones_like(
                sess.qsa_k_scale_pool[layer.layer_idx]
            )
            sess.qsa_k_row[layer.layer_idx] = torch.empty(
                1,
                attn.num_kv_heads,
                attn.head_dim,
                dtype=kv_dtype,
                device=device,
            )
            sess.qsa_v_row[layer.layer_idx] = torch.empty_like(
                sess.qsa_k_row[layer.layer_idx]
            )
            sess.qsa_k_scale_row[layer.layer_idx] = torch.empty(
                1,
                attn.num_kv_heads,
                dtype=torch.float16,
                device=device,
            )
            sess.qsa_v_scale_row[layer.layer_idx] = torch.empty_like(
                sess.qsa_k_scale_row[layer.layer_idx]
            )
            sess.qsa_idx_k_pool[layer.layer_idx] = torch.zeros(
                qsa_index_cache_rows(
                    max_seq,
                    layer.attn.indexer.compress_ratio,
                    fixed_rows=fixed_index_rows,
                ),
                layer.attn.indexer.head_dim,
                dtype=torch.bfloat16,
                device=device,
            )
            if layer.attn.indexer.mrope_section:
                sess.qsa_idx_rope_pool[layer.layer_idx] = torch.zeros(
                    sess.qsa_idx_k_pool[layer.layer_idx].shape[0],
                    3,
                    dtype=torch.long,
                    device=device,
                )
            sess.qsa_pooled_k_pool[layer.layer_idx] = torch.zeros(
                (max_seq + layer.attn.indexer.compress_ratio - 1)
                // layer.attn.indexer.compress_ratio,
                layer.attn.indexer.head_dim,
                dtype=torch.bfloat16,
                device=device,
            )
            sess.ends_buf[layer.layer_idx] = torch.zeros(1, dtype=torch.long, device=device)
    sess.token_buf = torch.zeros(1, dtype=torch.long, device=device)
    sess.pos_buf = torch.zeros(1, dtype=torch.long, device=device)
    if any(
        layer.is_qsa and layer.attn.indexer.mrope_section
        for layer in model.layers
    ):
        sess.rope_pos_buf = torch.zeros(3, 1, dtype=torch.long, device=device)
    else:
        sess.rope_pos_buf = None
    sess.hc_hidden_buf = torch.zeros(
        cfg.hc_count * cfg.hidden_size, dtype=torch.bfloat16, device=device
    )
    sess.want_hc_hidden = False
    sess.ple_emb_buf = torch.zeros(1, cfg.hidden_size, dtype=torch.bfloat16, device=device)
    for layer in model.layers:
        if layer.ple is not None:
            sess.ple_conv_state = torch.zeros(
                1,
                cfg.hc_count * cfg.hidden_size,
                layer.ple.state_len,
                dtype=torch.bfloat16,
                device=device,
            )
            break


def _prefill_embeddings(
    model: FlashNextModel,
    tokens: torch.Tensor,
    input_embeds: torch.Tensor | None,
) -> torch.Tensor:
    """Return the target prefill stream, optionally with vision rows fused."""

    if input_embeds is None:
        return model.embed_tokens(tokens)
    if input_embeds.ndim != 2 or tuple(input_embeds.shape) != (
        int(tokens.shape[0]),
        model.cfg.hidden_size,
    ):
        raise ValueError(
            "Flash-Next input_embeds must have shape "
            f"[{tokens.shape[0]}, {model.cfg.hidden_size}], got {tuple(input_embeds.shape)}"
        )
    return input_embeds.to(
        device=tokens.device,
        dtype=model.embed_tokens.weight.dtype,
        non_blocking=True,
    )


def _prefill_rope_positions(
    positions: object | None,
    token_count: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Normalize an optional CPU/NumPy MRoPE matrix for one prefill call."""

    if positions is None:
        return None
    tensor = torch.as_tensor(positions, dtype=torch.long, device=device)
    return _normalize_rope_positions(tensor, token_count).contiguous()


def _prefill_rope_slice(
    positions: torch.Tensor | object | None,
    start: int,
    end: int,
) -> torch.Tensor | None:
    if positions is None:
        return None
    tensor = torch.as_tensor(positions)
    if tensor.ndim == 2 and tensor.shape[1] == 3 and tensor.shape[0] != 3:
        return tensor[start:end]
    if tensor.ndim == 2 and tensor.shape[0] in {1, 3}:
        return tensor[:, start:end]
    return tensor[start:end]


@torch.no_grad()
def prefill_session(
    model: FlashNextModel,
    input_ids: torch.Tensor,
    sess: FlashNextSession,
    *,
    input_embeds: torch.Tensor | None = None,
    rope_positions: torch.Tensor | None = None,
    rope_next_position: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prefill one sequence chunk and seed every incremental-decode state.

    The bring-up driver originally replayed the M=1 decode graph once per
    prompt token.  That made TTFT scale with decode latency and discarded the
    large-M paths already present in GDN, MoE, hyper-connections, and PLE.
    This entry point runs the prompt as one sequence while writing the same
    fixed QSA pools and final recurrent/conv states consumed by graph decode.

    Returns ``(last_logits, hc_hidden_rows)``.  The complete HC stream is the
    target-side input required by the initial NEXTN teacher sync; materialising
    vocabulary logits for every prompt row would only waste memory and GEMM
    work, so the LM head runs on the final row only.
    """
    if input_ids.ndim != 1 or input_ids.numel() == 0:
        raise ValueError(
            f"Flash-Next prefill expects non-empty input_ids [T], got {tuple(input_ids.shape)}"
        )
    if (
        sess.qsa_k_pool is None
        or sess.qsa_v_pool is None
        or sess.qsa_idx_k_pool is None
        or sess.qsa_pooled_k_pool is None
    ):
        raise ValueError("prepare_graph_buffers must run before Flash-Next prefill")

    device = next(model.parameters()).device
    cpu_tokens = input_ids.detach().to(device="cpu", dtype=torch.long)
    tokens = cpu_tokens.to(device=device)
    seq_len = int(tokens.shape[0])
    start = sess.pos
    end = start + seq_len
    positions = torch.arange(start, end, dtype=torch.long, device=device)
    rope = _prefill_rope_positions(rope_positions, seq_len, device)
    if end > next(iter(sess.qsa_k_pool.values())).shape[0]:
        raise ValueError(f"Flash-Next prefill end {end} exceeds allocated graph capacity")
    history = None
    x = _prefill_embeddings(model, tokens, input_embeds)
    x = torch.cat([x] * model.cfg.hc_count, dim=-1)

    # Graph capture deliberately leaves the fresh GDN state marked as a
    # decode continuation.  Restore the initial-sequence contract before the
    # large-M prefill kernels inspect has_previous_state.
    if start == 0:
        for state in sess.gdn.values():
            state.conv_state.zero_()
            state.recurrent_state.zero_()
            state.has_previous_state = False

    for layer in model.layers:
        if layer.ple is not None:
            if history is None:
                history = torch.tensor(
                    [*sess.window, *cpu_tokens.tolist()], dtype=torch.long
                )
            ids = layer.ple_hasher.sequence_ids(history)[-seq_len:]
            embeddings = layer.ple.table.gather(ids, device=device).flatten(start_dim=-2)
            gated_flat, normed_flat = layer.ple.inject(embeddings, x)
            state = sess.ple_conv_state
            if state is None:
                raise RuntimeError("PLE graph state was not allocated")
            conv_out = layer.ple.prefill_conv_with_state(normed_flat, state)
            x = x + gated_flat + conv_out

        mixed, residuals = layer.attn_hc.mix(x)
        if layer.is_qsa:
            bundle = layer.attn
            qsa_positions = positions if rope is None else rope
            qi, ki = bundle.indexer.project_qk(mixed, qsa_positions)
            idx_pool = sess.qsa_idx_k_pool[layer.layer_idx]
            pooled_pool = sess.qsa_pooled_k_pool[layer.layer_idx]
            if rope is None:
                bundle.indexer.update_index_cache_eager(
                    idx_pool,
                    pooled_pool,
                    ki,
                    start=start,
                )
            else:
                rope_cache = (
                    sess.qsa_idx_rope_pool.get(layer.layer_idx)
                    if sess.qsa_idx_rope_pool is not None
                    else None
                )
                bundle.indexer.update_index_cache_eager(
                    idx_pool,
                    pooled_pool,
                    ki,
                    start=start,
                    rope_cache=rope_cache,
                    rope_positions=rope,
                )
            q, k, v, gate = bundle.attn.project(mixed, qsa_positions)
            k_pool = sess.qsa_k_pool[layer.layer_idx]
            v_pool = sess.qsa_v_pool[layer.layer_idx]
            quantize_qsa_kv(
                k,
                k_pool[start:end],
                sess.qsa_k_scale_pool[layer.layer_idx][start:end],
            )
            quantize_qsa_kv(
                v,
                v_pool[start:end],
                sess.qsa_v_scale_pool[layer.layer_idx][start:end],
            )

            sparse_budget = bundle.indexer.block_topk * bundle.indexer.compress_ratio
            if end <= sparse_budget:
                # Every visible key is selected below the QSA budget.  The
                # grouped-GQA implementation avoids expanding 2 KV heads to
                # 24 and is both the fastest and lowest-memory exact form.
                attn_out = sess.qsa_attn[layer.layer_idx].causal_prefix(
                    q,
                    gate,
                    k_pool,
                    v_pool,
                    positions,
                    sess.qsa_k_scale_pool[layer.layer_idx],
                    sess.qsa_v_scale_pool[layer.layer_idx],
                )
            else:
                attn_out = sess.qsa_attn[layer.layer_idx].sparse_prefill(
                    bundle.indexer,
                    qi,
                    q,
                    gate,
                    k_pool,
                    v_pool,
                    idx_pool,
                    positions,
                    pooled_k_cache=pooled_pool,
                    k_scales=sess.qsa_k_scale_pool[layer.layer_idx],
                    v_scales=sess.qsa_v_scale_pool[layer.layer_idx],
                )
        else:
            state = sess.gdn[f"gdn_{layer.layer_idx}"]
            attn_out = layer.attn(mixed.unsqueeze(0), state).squeeze(0)

        x = layer.attn_hc.combine(attn_out, residuals)
        mixed2, residuals2 = layer.mlp_hc.mix(x)
        x = layer.mlp_hc.combine(layer.mlp(mixed2), residuals2)

    sess.window.extend(int(token) for token in cpu_tokens.tolist())
    sess.window = sess.window[-model.cfg.ngram_size :]
    sess.pos = end
    if rope is not None:
        next_position = (
            int(rope_next_position)
            if rope_next_position is not None
            else int(rope.max().item()) + 1
        )
        sess.rope_next = torch.full(
            (3,), next_position, dtype=torch.long, device=device
        )
    mixed, _ = model.final_mixer.mix(x[-1:])
    return model.lm_head(mixed).float().squeeze(0), x


@torch.no_grad()
def prefill_session_layer_major(
    model: FlashNextModel,
    input_ids: torch.Tensor,
    sess: FlashNextSession,
    *,
    attention_chunk_size: int = 512,
    input_embeds: torch.Tensor | None = None,
    rope_positions: torch.Tensor | None = None,
    rope_next_position: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prefill one sequence with chunked state updates and full-row MLPs.

    GDN, QSA, and PLE remain on the qualified fixed-size chunk shape so their
    recurrent/cache math is unchanged.  The layer output is accumulated before
    its token-independent MLP, allowing router, routed experts, and shared
    expert GEMMs to consume the whole prompt in one launch family instead of
    once per attention chunk.
    """
    if input_ids.ndim != 1 or input_ids.numel() == 0:
        raise ValueError(
            f"Flash-Next prefill expects non-empty input_ids [T], got {tuple(input_ids.shape)}"
        )
    if attention_chunk_size <= 0:
        raise ValueError(
            f"attention_chunk_size must be positive, got {attention_chunk_size}"
        )
    if (
        sess.qsa_k_pool is None
        or sess.qsa_v_pool is None
        or sess.qsa_idx_k_pool is None
        or sess.qsa_pooled_k_pool is None
    ):
        raise ValueError("prepare_graph_buffers must run before Flash-Next prefill")

    device = next(model.parameters()).device
    cpu_tokens = input_ids.detach().to(device="cpu", dtype=torch.long)
    tokens = cpu_tokens.to(device=device)
    seq_len = int(tokens.shape[0])
    sequence_start = sess.pos
    sequence_end = sequence_start + seq_len
    rope = _prefill_rope_positions(rope_positions, seq_len, device)
    if sequence_end > next(iter(sess.qsa_k_pool.values())).shape[0]:
        raise ValueError(
            f"Flash-Next prefill end {sequence_end} exceeds allocated graph capacity"
        )

    history = None
    x = _prefill_embeddings(model, tokens, input_embeds)
    x = torch.cat([x] * model.cfg.hc_count, dim=-1)
    if sequence_start == 0:
        for state in sess.gdn.values():
            state.conv_state.zero_()
            state.recurrent_state.zero_()
            state.has_previous_state = False

    for layer in model.layers:
        ple_embeddings = None
        if layer.ple is not None:
            if history is None:
                history = torch.tensor([*sess.window, *cpu_tokens.tolist()], dtype=torch.long)
            ids = layer.ple_hasher.sequence_ids(history)[-seq_len:]
            ple_embeddings = layer.ple.table.gather(ids, device=device).flatten(start_dim=-2)

        attention_hidden = torch.empty_like(x)
        for offset in range(0, seq_len, attention_chunk_size):
            chunk_end = min(offset + attention_chunk_size, seq_len)
            start = sequence_start + offset
            end = sequence_start + chunk_end
            positions = torch.arange(start, end, dtype=torch.long, device=device)
            chunk_rope = _prefill_rope_slice(rope, offset, chunk_end)
            chunk = x[offset:chunk_end]

            if layer.ple is not None:
                assert ple_embeddings is not None
                gated_flat, normed_flat = layer.ple.inject(
                    ple_embeddings[offset:chunk_end], chunk
                )
                state = sess.ple_conv_state
                if state is None:
                    raise RuntimeError("PLE graph state was not allocated")
                conv_out = layer.ple.prefill_conv_with_state(normed_flat, state)
                chunk = chunk + gated_flat + conv_out

            mixed, residuals = layer.attn_hc.mix(chunk)
            if layer.is_qsa:
                bundle = layer.attn
                qsa_positions = positions if chunk_rope is None else chunk_rope
                qi, ki = bundle.indexer.project_qk(mixed, qsa_positions)
                idx_pool = sess.qsa_idx_k_pool[layer.layer_idx]
                pooled_pool = sess.qsa_pooled_k_pool[layer.layer_idx]
                if chunk_rope is None:
                    bundle.indexer.update_index_cache_eager(
                        idx_pool,
                        pooled_pool,
                        ki,
                        start=start,
                    )
                else:
                    rope_cache = (
                        sess.qsa_idx_rope_pool.get(layer.layer_idx)
                        if sess.qsa_idx_rope_pool is not None
                        else None
                    )
                    bundle.indexer.update_index_cache_eager(
                        idx_pool,
                        pooled_pool,
                        ki,
                        start=start,
                        rope_cache=rope_cache,
                        rope_positions=chunk_rope,
                    )
                q, k, v, gate = bundle.attn.project(mixed, qsa_positions)
                k_pool = sess.qsa_k_pool[layer.layer_idx]
                v_pool = sess.qsa_v_pool[layer.layer_idx]
                quantize_qsa_kv(
                    k,
                    k_pool[start:end],
                    sess.qsa_k_scale_pool[layer.layer_idx][start:end],
                )
                quantize_qsa_kv(
                    v,
                    v_pool[start:end],
                    sess.qsa_v_scale_pool[layer.layer_idx][start:end],
                )
                sparse_budget = (
                    bundle.indexer.block_topk * bundle.indexer.compress_ratio
                )
                if end <= sparse_budget:
                    attn_out = sess.qsa_attn[layer.layer_idx].causal_prefix(
                        q,
                        gate,
                        k_pool,
                        v_pool,
                        positions,
                        sess.qsa_k_scale_pool[layer.layer_idx],
                        sess.qsa_v_scale_pool[layer.layer_idx],
                    )
                else:
                    attn_out = sess.qsa_attn[layer.layer_idx].sparse_prefill(
                        bundle.indexer,
                        qi,
                        q,
                        gate,
                        k_pool,
                        v_pool,
                        idx_pool,
                        positions,
                        pooled_k_cache=pooled_pool,
                        k_scales=sess.qsa_k_scale_pool[layer.layer_idx],
                        v_scales=sess.qsa_v_scale_pool[layer.layer_idx],
                    )
            else:
                state = sess.gdn[f"gdn_{layer.layer_idx}"]
                attn_out = layer.attn(mixed.unsqueeze(0), state).squeeze(0)

            attention_hidden[offset:chunk_end].copy_(
                layer.attn_hc.combine(attn_out, residuals)
            )

        mixed, residuals = layer.mlp_hc.mix(attention_hidden)
        x = layer.mlp_hc.combine(layer.mlp(mixed), residuals)

    sess.window.extend(int(token) for token in cpu_tokens.tolist())
    sess.window = sess.window[-model.cfg.ngram_size :]
    sess.pos = sequence_end
    if rope is not None:
        next_position = (
            int(rope_next_position)
            if rope_next_position is not None
            else int(rope.max().item()) + 1
        )
        sess.rope_next = torch.full(
            (3,), next_position, dtype=torch.long, device=device
        )
    mixed, _ = model.final_mixer.mix(x[-1:])
    return model.lm_head(mixed).float().squeeze(0), x


def new_session(model: FlashNextModel, device) -> FlashNextSession:
    from runtime.model.flashnext.qsa import QsaDecodeAttention

    qsa_attn = {}
    pad = 0
    for layer in model.layers:
        if layer.is_qsa:
            pad = (
                layer.attn.indexer.block_topk * layer.attn.indexer.compress_ratio
                + layer.attn.indexer.compress_ratio
                - 1
            )
            qsa_attn[layer.layer_idx] = QsaDecodeAttention(layer.attn.attn, pad)
    return FlashNextSession(
        gdn=new_layer_states(model, device),
        qsa_k={},
        qsa_v={},
        qsa_idx_k={},
        qsa_idx_rope={},
        qsa_attn=qsa_attn,
        qsa_pad=max(pad, 1),
        ple_conv_state=None,
        window=[],
    )


def _session_rope_position(sess: FlashNextSession, device: torch.device) -> torch.Tensor:
    """Return the current scalar or three-axis decode position."""

    if sess.rope_next is None:
        return torch.tensor([sess.pos], dtype=torch.long, device=device)
    return sess.rope_next.to(device=device, dtype=torch.long).reshape(3, 1)


@torch.no_grad()
def decode_step(model: FlashNextModel, token_id: int, sess: FlashNextSession) -> torch.Tensor:
    """One incremental token through all 48 layers; returns final logits
    ``[vocab]``. Every tensor stays 2-D ``[1, ...]`` (the stateless layer
    path's contract). GDN recurrent states, QSA KV caches and the PLE conv
    state persist in ``sess``."""
    cfg = model.cfg
    device = next(model.parameters()).device

    # ``prepare_graph_buffers`` is intentionally used even when CUDA Graphs
    # are disabled: the fixed pools avoid retaining a second full BF16 K/V
    # copy for a long context.  Reuse the graph-shaped body for eager decode
    # as well, otherwise a no-CUDA-Graph deployment would look at the legacy
    # append-only dictionaries (which are empty after true prefill) and lose
    # the prompt's attention history.
    if (
        sess.qsa_k_pool is not None
        and sess.qsa_v_pool is not None
        and sess.qsa_idx_k_pool is not None
        and sess.qsa_pooled_k_pool is not None
        and sess.token_buf is not None
        and sess.pos_buf is not None
    ):
        ple_prelude(model, sess, token_id)
        sess.token_buf.fill_(int(token_id))
        sess.pos_buf.fill_(sess.pos)
        if sess.rope_pos_buf is not None:
            if sess.rope_next is None:
                sess.rope_pos_buf.fill_(sess.pos)
            else:
                sess.rope_pos_buf.copy_(sess.rope_next.reshape(3, 1))
        logits = decode_body(model, sess)
        sess.pos += 1
        if sess.rope_next is not None:
            sess.rope_next.add_(1)
        return logits

    rope_pos = _session_rope_position(sess, device)
    x = model.embed_tokens(torch.tensor([token_id], dtype=torch.long, device=device))  # [1, hidden]
    x = torch.cat([x] * cfg.hc_count, dim=-1)  # [1, hc*hidden]

    for layer in model.layers:
        if layer.ple is not None:
            sess.window.append(token_id)
            if len(sess.window) > cfg.ngram_size:
                del sess.window[0]
            ids = layer.ple_hasher.decode_ids(sess.window)
            if sess.ple_emb_buf is None:
                sess.ple_emb_buf = torch.zeros(
                    1, cfg.hidden_size, dtype=torch.bfloat16, device=device
                )
            emb = layer.ple.embed(ids, device=device, out=sess.ple_emb_buf)
            gated_flat, normed_flat = layer.ple.inject(emb, x)
            if sess.ple_conv_state is None:
                sess.ple_conv_state = torch.zeros(
                    1,
                    cfg.hc_count * cfg.hidden_size,
                    layer.ple.state_len,
                    dtype=x.dtype,
                    device=device,
                )
            conv_out, sess.ple_conv_state = layer.ple.decode_conv(normed_flat, sess.ple_conv_state)
            x = x + gated_flat + conv_out

        mixed, residuals = layer.attn_hc.mix(x)  # mixed [1, hidden]
        if layer.is_qsa:
            bundle = layer.attn
            qi, ki = bundle.indexer.project_qk(mixed, rope_pos)
            prev = sess.qsa_idx_k.get(layer.layer_idx)
            sess.qsa_idx_k[layer.layer_idx] = ki if prev is None else torch.cat([prev, ki], dim=0)
            if sess.rope_next is None:
                pooled = bundle.indexer.pool_keys(sess.qsa_idx_k[layer.layer_idx])
            else:
                rope_history = sess.qsa_idx_rope.get(layer.layer_idx)
                rope_row = rope_pos.transpose(0, 1)
                sess.qsa_idx_rope[layer.layer_idx] = (
                    rope_row
                    if rope_history is None
                    else torch.cat([rope_history, rope_row], dim=0)
                )
                pooled = bundle.indexer.pool_keys(
                    sess.qsa_idx_k[layer.layer_idx],
                    group_positions=sess.qsa_idx_rope[layer.layer_idx],
                )
            ends = torch.tensor(
                [max(0, (sess.pos - 3) // bundle.indexer.compress_ratio + 1)],
                device=device,
            )
            logits = bundle.indexer.score_blocks(qi, pooled, ends)
            blocks = bundle.indexer.select_blocks(logits, ends)
            q, k, v, gate = bundle.attn.project(mixed, rope_pos)
            pk = sess.qsa_k.get(layer.layer_idx)
            pv = sess.qsa_v.get(layer.layer_idx)
            sess.qsa_k[layer.layer_idx] = k if pk is None else torch.cat([pk, k], dim=0)
            sess.qsa_v[layer.layer_idx] = v if pv is None else torch.cat([pv, v], dim=0)
            idx, valid = bundle.indexer.decode_gather_indices(
                blocks[0], sess.pos, device, sess.qsa_pad
            )
            attn_out = sess.qsa_attn[layer.layer_idx](
                q,
                gate,
                sess.qsa_k[layer.layer_idx],
                sess.qsa_v[layer.layer_idx],
                idx,
                valid,
            )
        else:
            gdn_state = sess.gdn[f"gdn_{layer.layer_idx}"]
            attn_out = layer.attn(mixed.unsqueeze(1), gdn_state).squeeze(1)

        x = layer.attn_hc.combine(attn_out, residuals)
        mixed2, res2 = layer.mlp_hc.mix(x)
        x = layer.mlp_hc.combine(layer.mlp(mixed2), res2)

    sess.pos += 1
    if sess.rope_next is not None:
        sess.rope_next.add_(1)
    x, _ = model.final_mixer.mix(x)
    return model.lm_head(x).float().squeeze(0)


def load_flashnext_model(
    ckpt: pathlib.Path | str,
    device: str = "cuda",
    *,
    enable_vision: bool | None = None,
    ple_resident: bool = False,
    ple_cache_rows: int = 131_072,
    ple_cache_pages: int = 0,
    ple_io_workers: int = 32,
    progress=None,
) -> FlashNextModel:
    """Build the Flash-Next model with text and optional vision weights.

    Vision is enabled by default for this multimodal checkpoint and can be
    disabled with ``QSR_FLASHNEXT_VISION=0`` when a text-only deployment needs
    the extra ~0.84 GiB of visual BF16 weights back.
    """
    from safetensors import safe_open

    ckpt = pathlib.Path(ckpt)
    with open(ckpt / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]
    with open(ckpt / "config.json") as f:
        full_config = json.load(f)
    tc = full_config["text_config"]
    cfg = FlashNextTextConfig.from_checkpoint(ckpt)

    if enable_vision is None:
        vision_env = os.environ.get("QSR_FLASHNEXT_VISION", "1").strip().lower()
        if vision_env not in {"0", "1", "false", "true", "off", "on"}:
            raise ValueError(
                "QSR_FLASHNEXT_VISION must be 0 or 1, "
                f"got {vision_env!r}"
            )
        enable_vision = vision_env in {"1", "true", "on"}

    def load(name: str) -> torch.Tensor:
        with safe_open(str(ckpt / weight_map[name]), framework="pt", device="cpu") as f:
            return f.get_tensor(name)

    def copy_into(param: torch.Tensor, name: str) -> None:
        with torch.no_grad():
            param.copy_(load(name))

    model = FlashNextModel(cfg)
    copy_into(model.embed_tokens.weight, "model.language_model.embed_tokens.weight")
    copy_into(model.lm_head.weight, "lm_head.weight")
    pfx = "model.language_model.hyper_connection_mixer"
    copy_into(model.final_mixer.hc_norm.weight, f"{pfx}.hc_norm.weight")
    copy_into(model.final_mixer.input_mix_weight_down.weight, f"{pfx}.input_mix_weight_down.weight")
    copy_into(model.final_mixer.input_mix_weight_up.weight, f"{pfx}.input_mix_weight_up.weight")
    model.embed_tokens.to(device, torch.bfloat16)
    model.lm_head.to(device, torch.bfloat16)
    model.final_mixer.to(device)

    ple_table: FlashNextPleTable | None = None
    moe_backend = _flashnext_moe_backend()
    top_k = tc["num_experts_per_tok"]
    num_experts = tc["num_experts"]

    # b12x's workspace pool and routed-output arena are capacity caches for a
    # single sequential execution lane.  Keep one instance for the whole
    # model so every layer reuses the same warmed shape allocations; the
    # graph body executes MoE layers in order on one stream, so there is no
    # overlap requiring per-layer scratch.  Splitting these into 48 pools
    # needlessly multiplies allocator state and, more importantly, changes
    # the workspace identity between the warm-up and graph-capture paths.
    workspace = allocate_tp_moe_workspace_pool()
    moe_output_arena = SparkinferMoEOutputArena(cfg.hidden_size)

    for i in range(cfg.num_layers):
        lp = f"model.language_model.layers.{i}"
        is_qsa = cfg.layer_types[i] == "full_attention"
        if is_qsa:
            attn_module = QsaLayerBundle(
                load_qsa_indexer(
                    ckpt,
                    i,
                    rope_theta=cfg.rope_theta,
                    mrope_section=cfg.mrope_section,
                    mrope_interleaved=cfg.mrope_interleaved,
                ),
                load_qsa_attention(
                    ckpt,
                    i,
                    rope_theta=cfg.rope_theta,
                    mrope_section=cfg.mrope_section,
                    mrope_interleaved=cfg.mrope_interleaved,
                ),
            )
            attn_module.to(device)
        else:
            gdn = Qwen36GatedDeltaNet(_gdn_config_dict(tc), layer_idx=i, quantized={})
            with torch.no_grad():
                qkv = load(f"{lp}.linear_attn.in_proj_qkv.weight")
                z = load(f"{lp}.linear_attn.in_proj_z.weight")
                gdn.in_proj_qkvz.weight.copy_(torch.cat([qkv, z], dim=0))
                b = load(f"{lp}.linear_attn.in_proj_b.weight")
                a = load(f"{lp}.linear_attn.in_proj_a.weight")
                gdn.in_proj_ba.weight.copy_(torch.cat([b, a], dim=0))
                gdn.conv1d.weight.copy_(load(f"{lp}.linear_attn.conv1d.weight"))
                gdn.dt_bias.copy_(load(f"{lp}.linear_attn.dt_bias"))
                gdn.A_log.copy_(load(f"{lp}.linear_attn.A_log"))
                gdn.norm.weight.copy_(load(f"{lp}.linear_attn.norm.weight"))
                gdn.out_proj.weight.copy_(load(f"{lp}.linear_attn.out_proj.weight"))
            attn_module = gdn.to(device, torch.bfloat16)
        layer = FlashNextLayer(i, cfg, attn_module, is_qsa)

        hc_pairs = (
            (layer.attn_hc, "attn_hyper_connection"),
            (layer.mlp_hc, "mlp_hyper_connection"),
        )
        for hc, name in hc_pairs:
            copy_into(hc.hc_norm.weight, f"{lp}.{name}.hc_norm.weight")
            copy_into(hc.input_mix_weight_down.weight, f"{lp}.{name}.input_mix_weight_down.weight")
            copy_into(hc.input_mix_weight_up.weight, f"{lp}.{name}.input_mix_weight_up.weight")
            copy_into(hc.block_inject_weight.weight, f"{lp}.{name}.block_inject_weight.weight")
            hc.to(device)

        raw = load_flashnext_experts(ckpt, i, _expert_geometry(tc), device)
        if moe_backend == "flashinfer":
            experts = prepare_flashnext_cutlass_experts(raw, _expert_geometry(tc), device)
            expert_layer = FlashInferMoELayer(
                experts,
                device,
                output_arena=moe_output_arena,
            )
        else:
            experts = prepare_flashnext_experts(raw, _expert_geometry(tc), device)
            expert_layer = SparkinferMoELayer(
                experts,
                workspace,
                device,
                output_arena=moe_output_arena,
            )
        del raw
        torch.cuda.empty_cache()
        mlp = FlashNextMlp(
            cfg.hidden_size,
            num_experts,
            top_k,
            expert_layer,
        )
        copy_into(mlp.gate.weight, f"{lp}.mlp.gate.weight")
        mlp.shared = SharedExpert(cfg.hidden_size, tc["shared_expert_intermediate_size"])
        with torch.no_grad():
            gw = load(f"{lp}.mlp.shared_expert.gate_proj.weight")
            uw = load(f"{lp}.mlp.shared_expert.up_proj.weight")
            mlp.shared.gate_up_proj.weight.copy_(torch.cat([gw, uw], dim=0))
        copy_into(mlp.shared.down_proj.weight, f"{lp}.mlp.shared_expert.down_proj.weight")
        copy_into(mlp.shared.shared_gate.weight, f"{lp}.mlp.shared_expert_gate.weight")
        mlp.to(device)
        layer.mlp = mlp

        if (i + 1) in cfg.ple_layer_ids:
            if ple_table is None:
                ple_table = FlashNextPleTable(
                    ckpt,
                    layer_idx=i,
                    resident=ple_resident,
                    cache_rows=ple_cache_rows,
                    cache_pages=ple_cache_pages,
                    io_workers=ple_io_workers,
                )
            ple_layer = FlashNextPLELayer(ckpt, ple_table, layer_idx=i, eps=cfg.rms_norm_eps)
            layer.ple = ple_layer.to(device)
            layer.ple_hasher = FlashNextPleHasher(ple_table, cfg.eos_token_id, cfg.ngram_size)

        model.layers.append(layer)
        if progress is not None:
            progress(i + 1, cfg.num_layers)
    if enable_vision:
        from runtime.model.flashnext.vision import load_vision_tower

        model.vision_tower = load_vision_tower(
            ckpt,
            full_config,
            weight_map,
            device=device,
        )
    model.ple_table = ple_table
    return model.to(device)


def _expert_geometry(tc: dict):
    from runtime.model.qwen38_moe import QwenMoeGeometry

    return QwenMoeGeometry(
        num_experts=tc["num_experts"],
        top_k=tc["num_experts_per_tok"],
        hidden_size=tc["hidden_size"],
        moe_intermediate_size=tc["moe_intermediate_size"],
        shared_expert_intermediate_size=tc["shared_expert_intermediate_size"],
    )


@torch.no_grad()
def ple_prelude(model: FlashNextModel, sess: FlashNextSession, token_id: int) -> None:
    """Eager (graph-external) PLE table gather into the fixed buffer."""
    cfg = model.cfg
    if sess.ple_emb_buf is None:
        device = next(model.parameters()).device
        sess.ple_emb_buf = torch.zeros(1, cfg.hidden_size, dtype=torch.bfloat16, device=device)
    for layer in model.layers:
        if layer.ple is None:
            continue
        sess.window.append(token_id)
        if len(sess.window) > cfg.ngram_size:
            del sess.window[0]
        ids = layer.ple_hasher.decode_ids(sess.window)
        layer.ple.embed(ids, device=sess.ple_emb_buf.device, out=sess.ple_emb_buf)
        break


@torch.no_grad()
def decode_body(model: FlashNextModel, sess: FlashNextSession) -> torch.Tensor:
    """Fixed-address decode body (graph-capturable). Reads ``sess.token_buf``
    and the pooled QSA/PLE buffers; returns logits ``[vocab]``."""
    cfg = model.cfg
    pos = sess.pos_buf
    pos_int = pos  # [1] tensor
    rope_pos = sess.rope_pos_buf if sess.rope_pos_buf is not None else pos_int
    x = model.embed_tokens(sess.token_buf)
    x = torch.cat([x] * cfg.hc_count, dim=-1)

    for layer in model.layers:
        if layer.ple is not None:
            emb = sess.ple_emb_buf
            gated_flat, normed_flat = layer.ple.inject(emb, x)
            conv_out = layer.ple.decode_conv_inplace(normed_flat, sess.ple_conv_state)
            x = x + gated_flat + conv_out

        mixed, residuals = layer.attn_hc.mix(x)
        if layer.is_qsa:
            bundle = layer.attn
            qi, ki = bundle.indexer.project_qk(mixed, rope_pos)
            idx_pool = sess.qsa_idx_k_pool[layer.layer_idx]
            pooled_pool = sess.qsa_pooled_k_pool[layer.layer_idx]
            rope_cache = (
                sess.qsa_idx_rope_pool.get(layer.layer_idx)
                if sess.qsa_idx_rope_pool is not None
                else None
            )
            if rope_cache is None:
                bundle.indexer.update_index_cache_fixed(
                    idx_pool,
                    pooled_pool,
                    ki,
                    pos_int,
                )
            else:
                bundle.indexer.update_index_cache_fixed(
                    idx_pool,
                    pooled_pool,
                    ki,
                    pos_int,
                    rope_cache=rope_cache,
                    rope_positions=rope_pos,
                )
            ends = torch.clamp((pos - 3) // bundle.indexer.compress_ratio + 1, min=0)
            logits = bundle.indexer.score_blocks(qi, pooled_pool, ends)
            blocks = bundle.indexer.select_blocks(logits, ends)
            q, k, v, gate = bundle.attn.project(mixed, rope_pos)
            layer_idx = layer.layer_idx
            k_pool = sess.qsa_k_pool[layer_idx]
            v_pool = sess.qsa_v_pool[layer_idx]
            k_scale_pool = sess.qsa_k_scale_pool[layer_idx]
            v_scale_pool = sess.qsa_v_scale_pool[layer_idx]
            if _qsa_cache_is_quantized(k_pool.dtype):
                k_row = (sess.qsa_k_row or {}).get(layer_idx)
                v_row = (sess.qsa_v_row or {}).get(layer_idx)
                k_scale_row = (sess.qsa_k_scale_row or {}).get(layer_idx)
                v_scale_row = (sess.qsa_v_scale_row or {}).get(layer_idx)
                if any(
                    value is None
                    for value in (k_row, v_row, k_scale_row, v_scale_row)
                ):
                    raise RuntimeError(
                        "quantized QSA decode requires graph-owned row scratch; "
                        f"layer={layer_idx}"
                    )
                quantize_qsa_kv(k, k_row, k_scale_row)
                quantize_qsa_kv(v, v_row, v_scale_row)
                qsa_cache_index_copy_(k_pool, pos, k_row)
                qsa_cache_index_copy_(v_pool, pos, v_row)
                k_scale_pool.index_copy_(0, pos, k_scale_row)
                v_scale_pool.index_copy_(0, pos, v_scale_row)
            else:
                k_pool.index_copy_(0, pos, k)
                v_pool.index_copy_(0, pos, v)
            idx, valid = bundle.indexer.decode_gather_indices(
                blocks[0], pos, idx_pool.device, sess.qsa_pad
            )
            # ``decode_gather_indices`` packs valid block lanes and the
            # partial causal tail at the front of its fixed row.  Count the
            # packed prefix rather than using the absolute position (which
            # would be wrong once sparse top-k selection omits blocks).
            selected_counts = valid.to(torch.int32).sum().reshape(1)
            attn_out = sess.qsa_attn[layer.layer_idx](
                q,
                gate,
                k_pool,
                v_pool,
                idx,
                valid,
                k_scale_pool,
                v_scale_pool,
                selected_counts,
            )
        else:
            gdn_state = sess.gdn[f"gdn_{layer.layer_idx}"]
            attn_out = layer.attn(mixed.unsqueeze(1), gdn_state).squeeze(1)

        x = layer.attn_hc.combine(attn_out, residuals)
        mixed2, res2 = layer.mlp_hc.mix(x)
        x = layer.mlp_hc.combine(layer.mlp(mixed2), res2)

    mixed, _ = model.final_mixer.mix(x)
    logits = model.lm_head(mixed).float().squeeze(0)
    if getattr(sess, "want_hc_hidden", False):
        sess.hc_hidden_buf.copy_(x.squeeze(0))
    return logits


class FlashNextGraphEngine:
    """CUDA-graph decode: eager PLE prelude + captured body replay."""

    def __init__(self, model: FlashNextModel, sess: FlashNextSession, device) -> None:
        self.model = model
        self.sess = sess
        self.device = device
        self.graph: torch.cuda.CUDAGraph | None = None
        self._logits: torch.Tensor | None = None

    def _zero_state(self, *, clear_kv: bool = True) -> None:
        """Reset recurrent/decode metadata, optionally retaining QSA KV.

        Prefix reuse keeps the fixed-address QSA pools in place and restores
        the co-keyed recurrent checkpoint separately.  CUDA-graph capture and
        a true cold reset still use the historical ``clear_kv=True`` path;
        only the prefix-cache lifecycle asks for ``False``.
        """
        sess = self.sess
        for st in sess.gdn.values():
            st.conv_state.zero_()
            st.recurrent_state.zero_()
            st.has_previous_state = True  # capture the decode path
        if sess.ple_conv_state is not None:
            sess.ple_conv_state.zero_()
        if clear_kv:
            for pool in list(sess.qsa_k_pool.values()) + list(sess.qsa_v_pool.values()):
                pool.zero_()
            for pool in list(sess.qsa_k_scale_pool.values()) + list(sess.qsa_v_scale_pool.values()):
                pool.fill_(1.0)
            for pool in sess.qsa_idx_k_pool.values():
                pool.zero_()
            for pool in sess.qsa_pooled_k_pool.values():
                pool.zero_()
            if sess.qsa_idx_rope_pool is not None:
                for pool in sess.qsa_idx_rope_pool.values():
                    pool.zero_()
        sess.window = []
        sess.pos = 0
        sess.rope_next = None
        if sess.rope_pos_buf is not None:
            sess.rope_pos_buf.zero_()

    @torch.no_grad()
    def capture_prefill_mlp_graphs(self, rows: int = 512) -> None:
        """Capture fixed-row MLP graphs for every transformer layer."""
        pool = torch.cuda.graph_pool_handle()
        for layer in self.model.layers:
            if layer.mlp is None:
                raise RuntimeError(f"Flash-Next layer {layer.layer_idx} has no MLP")
            layer.mlp.capture_prefill_graph(rows, pool=pool)

    def capture(self) -> None:
        sess = self.sess
        self._zero_state()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                decode_body(self.model, sess)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._logits = decode_body(self.model, sess)
        self.graph = g
        self.model._retain_graph_moe_allocations()
        self._zero_state()  # warmup polluted the states; start clean

    def prefill(
        self,
        token_ids,
        *,
        chunk_size: int = 0,
        layer_major: bool | None = None,
        input_embeds: torch.Tensor | None = None,
        rope_positions: torch.Tensor | None = None,
        rope_next_position: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one true large-M prompt prefill and seed graph-decode state."""
        tokens = torch.as_tensor(token_ids, dtype=torch.long)
        if layer_major is None:
            # Layer-major changes the row shape of HC/router/shared GEMMs and
            # has not passed the long-prompt numerical gate.  Keep the
            # qualified token-major path as the implicit production default;
            # experiments must opt in explicitly.
            layer_major = False
        if layer_major:
            kwargs = {"attention_chunk_size": chunk_size or 512}
            if input_embeds is not None:
                kwargs["input_embeds"] = input_embeds
            if rope_positions is not None:
                kwargs["rope_positions"] = rope_positions
            if rope_next_position is not None:
                kwargs["rope_next_position"] = rope_next_position
            return prefill_session_layer_major(self.model, tokens, self.sess, **kwargs)
        if chunk_size <= 0 or chunk_size >= tokens.numel():
            if input_embeds is None and rope_positions is None:
                return prefill_session(self.model, tokens, self.sess)
            kwargs = {}
            if input_embeds is not None:
                kwargs["input_embeds"] = input_embeds
            if rope_positions is not None:
                kwargs["rope_positions"] = rope_positions
            if rope_next_position is not None:
                kwargs["rope_next_position"] = rope_next_position
            return prefill_session(self.model, tokens, self.sess, **kwargs)
        hidden_rows = []
        logits = None
        for start in range(0, tokens.numel(), chunk_size):
            chunk_tokens = tokens[start : start + chunk_size]
            if input_embeds is None:
                kwargs = {}
            else:
                kwargs = {
                    "input_embeds": input_embeds[start : start + chunk_size]
                }
            if rope_positions is not None:
                kwargs["rope_positions"] = _prefill_rope_slice(
                    torch.as_tensor(rope_positions), start, start + chunk_size
                )
            if rope_next_position is not None:
                kwargs["rope_next_position"] = rope_next_position
            logits, hidden = prefill_session(
                self.model,
                chunk_tokens,
                self.sess,
                **kwargs,
            )
            hidden_rows.append(hidden)
        return logits, torch.cat(hidden_rows)

    def step(self, token_id: int) -> torch.Tensor:
        sess = self.sess
        ple_prelude(self.model, sess, token_id)
        sess.token_buf.fill_(token_id)
        sess.pos_buf.fill_(sess.pos)
        if sess.rope_pos_buf is not None:
            if sess.rope_next is None:
                sess.rope_pos_buf.fill_(sess.pos)
            else:
                sess.rope_pos_buf[:, 0].copy_(sess.rope_next)
        for layer in self.model.layers:
            if layer.is_qsa:
                sess.ends_buf[layer.layer_idx].fill_(
                    max(0, (sess.pos - 3) // layer.attn.indexer.compress_ratio + 1)
                )
        self.graph.replay()
        sess.pos += 1
        if sess.rope_next is not None:
            sess.rope_next.add_(1)
        return self._logits
