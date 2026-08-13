"""DeepSeek-V4-Flash backend: multi-slot kernel-path serving (Phase 4).

The serving path runs one slot-aware kernel-path attention stack
(``Dsv4AttnKernelLayer`` per model layer, each with a leading slot arena for
its page buffers, compressor decode state, and indexer scoring caches),
weights shared from the eager ``Dsv4Transformer`` that owns the checkpoint.
Packed FP8 KV pages + the fork compressed_mla kernel stay on the serving
path; the eager graph remains the load-time weight holder and oracle only.
``reset_slot`` zeroes one slot's recursive compressor/indexer state while
leaving its KV bytes in place, same rule as the slot pool.

Phase 3's single-slot eager backend is superseded; the protocol surface is
unchanged (``ModelBackend``'s six unconditional members).  Decode CUDA Graph
is available when capture succeeds; chunked-prefill scheduling and prefix
cache remain conservative follow-ups with their own gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from bfdiag.trace import events as bfdiag_events
from bfdiag.trace import ring as bfdiag_trace
from runtime.backends.protocol import (
    BackendCapabilities,
    BackendSnapshot,
    PrefixHit,
    PrefixSnapshot,
    SlotSnapshot,
)
from runtime.block_pool import ChunkedPrefillState
from runtime.kernels.iq2_mma16_tc import (
    Dsv4PrefillMoEWorkspace,
    grouped_moe_prefill_k32_graph,
)
from runtime.logprobs import compute_logprobs
from runtime.model.dsv4_attn_kernel import Dsv4AttnKernelLayer
from runtime.model.dsv4_config import Dsv4Config
from runtime.model.dsv4_model import (
    Dsv4Transformer,
    load_dsv4_from_gguf,
    rms_norm,
)
from runtime.sampling import SamplingParams, make_generator, sample_from_logits

EOS_TOKENS: tuple[int, ...] = (1,)
DSV4_PREFIX_BLOCK_SIZE = 256


def _enable_serving_q8_kernels(model: Dsv4Transformer) -> None:
    """Enable packed Q8 kernels on numerically qualified projections.

    BF16-declared projections already use the tensor-core path.  The output
    head is FP32-declared by the reference graph, so it was previously omitted
    and every decode graph materialized a 2.02 GiB FP32 weight before GEMV.
    The packed kernel still returns FP32 logits; only its weight tile is
    rounded to BF16 for tensor-core accumulation.  Real-weight qualification
    is required before extending this exception to other FP32 projections.
    """
    from runtime.model.dsv4_model import PackedQ8_0Weight

    for mod in model.modules():
        if not isinstance(mod, PackedQ8_0Weight):
            continue
        if getattr(mod, "weight_dtype", torch.bfloat16) is torch.bfloat16:
            mod.fused_q8 = True
    model.lm_head.fused_q8 = True
    for block in model.blocks:
        block.moe.shared_w1.fused_q8_fp32 = True
        block.moe.shared_w3.fused_q8_fp32 = True
        block.moe.shared_w2.fused_q8_fp32 = True


def _share_mla_scratch_across_layers(
    kernel_layers: list[Dsv4AttnKernelLayer],
) -> torch.Tensor | None:
    """Allocate one backend-owned MLA scratch arena and bind every layer to it."""
    if not kernel_layers:
        return None

    first_spec = kernel_layers[0].mla_scratch_spec()
    dtype = first_spec.dtype
    device = first_spec.device
    max_nbytes = int(first_spec.shape[0])
    for layer in kernel_layers[1:]:
        spec = layer.mla_scratch_spec()
        if spec.dtype != dtype:
            raise RuntimeError(
                "DSV4 shared MLA scratch requires one dtype across layers, got "
                f"{dtype} and {spec.dtype}"
            )
        if spec.device != device:
            raise RuntimeError(
                "DSV4 shared MLA scratch requires one device across layers, got "
                f"{device} and {spec.device}"
            )
        max_nbytes = max(max_nbytes, int(spec.shape[0]))

    scratch = torch.empty((max_nbytes,), dtype=dtype, device=device)
    for layer in kernel_layers:
        layer.set_mla_scratch(scratch)
    return scratch


@dataclass(frozen=True)
class Dsv4SlotStateView:
    """Read-only server view of one DSV4 slot (protocol shape)."""

    kv_len: int
    committed_tokens: tuple[int, ...]

    @property
    def is_fresh(self) -> bool:
        return self.kv_len == 0


def _decode_batch_chunks(count: int) -> tuple[int, ...]:
    """Cover ``count`` rows with the native fixed graph buckets in order."""
    if count < 0:
        raise ValueError(f"decode batch count must be >= 0, got {count}")
    chunks: list[int] = []
    remaining = count
    for size in (4, 2, 1):
        while remaining >= size:
            chunks.append(size)
            remaining -= size
    return tuple(chunks)


class DeepseekV4Backend:
    """Multi-slot kernel-path serving of the DSV4-Flash graph.

    ``forward_fn`` is an injection point for tests: the production default
    is the kernel-path stack forward (``_forward``), which requires the
    real 512-dim DSV4 shapes; tests stub it with a tiny zeroed-graph
    forward to exercise the serving contract without weights.
    """

    def __init__(
        self,
        model: Dsv4Transformer,
        config: Dsv4Config,
        *,
        num_slots: int = 1,
        max_seq_len: int = 4096,
        max_q_rows: int = 1,
        device: str = "cuda",
        forward_fn: Any | None = None,
    ) -> None:
        if num_slots < 1:
            raise ValueError(f"num_slots must be >= 1, got {num_slots}")
        if max_q_rows < 1:
            raise ValueError(f"max_q_rows must be >= 1, got {max_q_rows}")
        self.model = model
        self.config = config
        self.num_slots = num_slots
        self.max_seq_len = max_seq_len
        self.max_q_rows = max_q_rows
        # Prefill may intentionally use one-row chunks, but native decode must
        # still plan enough MLA rows for the largest reachable B=1/2/4 bucket.
        self._kernel_max_q_rows = max(max_q_rows, min(num_slots, 4))
        self.device = device
        self._forward_fn = forward_fn

        # One kernel-path stack shared across slots, with slot-indexed state
        # inside each attention layer. Built lazily: the real shapes (512
        # head dim, 256-page window) are DSV4-specific, so tests with a tiny
        # config pass ``forward_fn`` and never build the stack.
        self.slot_layers: list[Dsv4AttnKernelLayer] = []
        self._shared_mla_scratch: torch.Tensor | None = None
        if forward_fn is None:
            self.slot_layers = [
                Dsv4AttnKernelLayer(
                    config,
                    layer_id,
                    num_slots=num_slots,
                    max_seq_len=max_seq_len,
                    max_q_rows=self._kernel_max_q_rows,
                    device=device,
                    shared_from=model.blocks[layer_id].attn,
                    allocate_mla_scratch=False,
                )
                for layer_id in range(config.num_layers)
            ]
            self._shared_mla_scratch = _share_mla_scratch_across_layers(self.slot_layers)
            # Route bf16 Q8_0 projections through the fused tensor-core
            # dequant-GEMM on the serving path (more accurate than cuBLAS
            # bf16, zero dequant cache).  The eager graph stays untouched as
            # the official-reference oracle.
            _enable_serving_q8_kernels(model)
        self._native_decode_batch_available = self._native_decode_batch_contract_supported()
        self._freed_eager_oracle_kv = 0
        self._freed_eager_freqs = 0
        self._prefill_graph: Dsv4PrefillGraphDriver | None = None
        self._kv_len = [0] * num_slots
        self._committed: list[list[int]] = [[] for _ in range(num_slots)]
        self._cg_status: dict[str, str] = {}
        self._decode_graphs: dict[int, Any] = {}
        self._prefix_cache_tokens: list[tuple[int, ...] | None] = [None] * num_slots
        self._prefix_cache_kv_len = [0] * num_slots
        self._prefix_checkpoint_tensors: list[tuple[tuple[torch.Tensor, ...], ...] | None] = [
            None
        ] * num_slots
        self._prefix_window_tensors: list[tuple[torch.Tensor, ...] | None] = [None] * num_slots
        self._prefix_anchor_logits: list[torch.Tensor | None] = [None] * num_slots
        self._pending_prefix_source: dict[int, tuple[int, int]] = {}
        self._trace_compressor_ratios = tuple(sorted(set(config.compress_ratios)))
        self.stats: dict[str, int] = {
            "prefill_calls": 0,
            "prefill_chunks": 0,
            "prefill_tokens": 0,
            "decode_rounds": 0,
            "decode_tokens": 0,
            "decode_graph_capture_attempts": 0,
            "decode_graph_capture_successes": 0,
            "decode_graph_capture_failures": 0,
            "decode_graph_replays": 0,
            "decode_eager_fallbacks": 0,
            "prefix_kv_hit_tokens": 0,
            "prefix_state_hit_tokens": 0,
            "prefix_same_slot_restores": 0,
            "prefix_cross_slot_restores": 0,
            "prefix_restore_failures": 0,
        }
        self.cg_fallback_reasons: dict[str, int] = {}

    def _share_rope_freqs(self) -> dict[str, int]:
        """Collapse the per-layer RoPE tables into two regime-shared tables.

        Every layer computes ``freqs_cis`` with parameters that depend only on
        ``layer_ratio``: ratio-4 and ratio-128 layers both use the compressed
        regime (``compress_rope_theta`` + YaRN), and the two window-only layers
        use the base regime.  The 43 per-layer tables are therefore 41
        bit-identical copies of one compressed table plus 2 copies of one
        window table (verified 2026-08-13: ``torch.equal`` across all layers in
        each regime).  The eager graph and the kernel-path layers each hold
        their own copies, so this collapses 43 + 43 = 86 tables to 2.

        The kernel-path layer keeps the buffer registered (its
        ``named_buffers`` still enumerate it) so capture and forward are
        untouched; only the backing storage is shared.  This MUST run before
        decode-graph capture: the capture bakes in the shared buffer address,
        and sharing afterward would leave the graph pointing at storage that
        gets GC'd (measured 2026-08-13: graph replay returns garbage tokens).
        """
        freed = 0
        if not self.slot_layers or not self.model.blocks:
            return {"kernel_freqs": 0}

        # -- resolve the two regime-canonical tables (eager graph copies) ---
        def regime(layer) -> str:
            return "window" if getattr(layer, "ratio", 0) == 0 else "compressed"

        canonical: dict[str, torch.Tensor] = {}
        for block in self.model.blocks:
            attn = getattr(block, "attn", None)
            freqs = getattr(attn, "freqs_cis", None)
            if freqs is None or not isinstance(freqs, torch.Tensor):
                continue
            key = regime(attn)
            if key not in canonical:
                canonical[key] = freqs

        # -- re-point every eager layer's table at its regime canonical -----
        for block in self.model.blocks:
            attn = getattr(block, "attn", None)
            freqs = getattr(attn, "freqs_cis", None)
            if freqs is None or not isinstance(freqs, torch.Tensor):
                continue
            key = regime(attn)
            target = canonical[key]
            if freqs.data_ptr() == target.data_ptr():
                continue
            if freqs.shape != target.shape:
                raise RuntimeError(
                    f"layer {attn.layer_id} RoPE shape mismatch in regime "
                    f"{key!r}: {tuple(freqs.shape)} vs {tuple(target.shape)}"
                )
            freed += freqs.numel() * freqs.element_size()
            attn.register_buffer("freqs_cis", target, persistent=False)
            # The eager layer's compressor/indexer hold a reference to the old
            # table (Dsv4Attention._wire_subcaches); re-point them too or the
            # old storage never releases.
            for sub in (
                getattr(attn, "compressor", None),
                getattr(attn, "indexer", None),
            ):
                if sub is not None and getattr(sub, "freqs_cis", None) is not None:
                    sub.freqs_cis = target

        # -- point the kernel-path layers + their subcaches at the same ------
        for layer in self.slot_layers:
            eager_freqs = getattr(self.model.blocks[layer.layer_id].attn, "freqs_cis", None)
            kernel_freqs = getattr(layer, "freqs_cis", None)
            if eager_freqs is None or kernel_freqs is None:
                continue
            if kernel_freqs.data_ptr() == eager_freqs.data_ptr():
                continue
            if kernel_freqs.shape != eager_freqs.shape:
                raise RuntimeError(
                    f"layer {layer.layer_id} RoPE shape mismatch: kernel "
                    f"{tuple(kernel_freqs.shape)} vs eager {tuple(eager_freqs.shape)}"
                )
            freed += kernel_freqs.numel() * kernel_freqs.element_size()
            layer.register_buffer("freqs_cis", eager_freqs, persistent=False)
            # The layer's compressor/indexer captured the OLD kernel table by
            # reference at construction (dsv4_attn_kernel.py sets
            # ``compressor.freqs_cis = self.freqs_cis``); re-point them at the
            # shared eager table so they do not dangle after the old storage
            # is released by the register_buffer swap.
            for sub in (
                getattr(layer, "compressor", None),
                getattr(layer, "indexer", None),
            ):
                if sub is not None and getattr(sub, "freqs_cis", None) is not None:
                    sub.freqs_cis = eager_freqs
            if getattr(layer, "indexer", None) is not None:
                icomp = getattr(layer.indexer, "compressor", None)
                if icomp is not None and getattr(icomp, "freqs_cis", None) is not None:
                    icomp.freqs_cis = eager_freqs
        self._freed_eager_freqs = freed
        return {"kernel_freqs": freed}

    def _free_eager_oracle_caches(self) -> dict[str, int]:
        """Release the eager graph's per-layer KV arenas (oracle-only).

        The eager ``Dsv4Transformer`` exists as the load-time weight holder
        and the parity oracle; the serving path computes attention through
        ``slot_layers`` (kernel-path ``Dsv4AttnKernelLayer``) with its own
        page buffers and compressor arenas.  The eager graph's own
        ``kv_cache`` / ``kv_state`` / ``score_state`` buffers are therefore
        dead weight during serving -- freed here, after CG capture and any
        eager warmup, so the freed bytes become available to prefill
        scratch.  Returns bytes freed per category for the memory probe.

        Any caller that later invokes ``model.forward()`` (oracle/parity
        scripts) must re-create these buffers; the serving backend never
        does.
        """
        freed: dict[str, int] = {"eager_oracle_kv": 0}
        for block in self.model.blocks:
            attn = getattr(block, "attn", None)
            if attn is None:
                continue
            compressor = getattr(attn, "compressor", None)
            if compressor is not None:
                # The compressor's kv_cache is a view of attn.kv_cache; drop
                # the reference first so the storage actually releases.
                kv_view = getattr(compressor, "kv_cache", None)
                if isinstance(kv_view, torch.Tensor):
                    compressor.kv_cache = None
                for name in ("kv_state", "score_state"):
                    tensor = getattr(compressor, name, None)
                    if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                        freed["eager_oracle_kv"] += tensor.numel() * tensor.element_size()
                        delattr(compressor, name)
            for name in ("kv_cache",):
                tensor = getattr(attn, name, None)
                if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                    freed["eager_oracle_kv"] += tensor.numel() * tensor.element_size()
                    delattr(attn, name)
            indexer = getattr(attn, "indexer", None)
            if indexer is not None:
                icomp = getattr(indexer, "compressor", None)
                if icomp is not None:
                    kv_view = getattr(icomp, "kv_cache", None)
                    if isinstance(kv_view, torch.Tensor):
                        icomp.kv_cache = None
                    for name in ("kv_state", "score_state"):
                        tensor = getattr(icomp, name, None)
                        if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                            freed["eager_oracle_kv"] += tensor.numel() * tensor.element_size()
                            delattr(icomp, name)
                for name in ("kv_cache",):
                    tensor = getattr(indexer, name, None)
                    if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                        freed["eager_oracle_kv"] += tensor.numel() * tensor.element_size()
                        delattr(indexer, name)
        self._freed_eager_oracle_kv = freed["eager_oracle_kv"]
        return freed

    def _native_decode_batch_contract_supported(self) -> bool:
        if self._forward_fn is not None or not self.slot_layers:
            return False
        from runtime.kernels.dsv4_compressor import (
            supports_fused_decode_postgemv_batch,
            supports_fused_indexer_decode_postgemv_batch,
        )

        batch_size = min(self.num_slots, 4)
        for layer in self.slot_layers:
            compressor = layer.compressor
            if compressor is None:
                continue
            if not supports_fused_decode_postgemv_batch(
                ratio=compressor.ratio,
                rotate=compressor.rotate,
                quantize=compressor.quantize,
                device=torch.device(self.device),
                batch_size=batch_size,
                seq_len=1,
                head_dim=compressor.head_dim,
                rope_head_dim=compressor.rope_head_dim,
            ):
                return False
            if layer.indexer is not None:
                index_compressor = layer.indexer.compressor
                if not supports_fused_indexer_decode_postgemv_batch(
                    ratio=index_compressor.ratio,
                    rotate=index_compressor.rotate,
                    quantize=index_compressor.quantize,
                    device=torch.device(self.device),
                    batch_size=batch_size,
                    seq_len=1,
                    head_dim=index_compressor.head_dim,
                    rope_head_dim=index_compressor.rope_head_dim,
                ):
                    return False
        return True

    # -- protocol ------------------------------------------------------------

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            speculative_decode=False,
            prefix_cache=True,
            cuda_graph=True,
            chunked_prefill=False,
            warm_continue=False,
        )

    def reset_slot(self, slot: int) -> None:
        """Release the slot: zero the recursive compressor state.

        KV bytes are left in place (never read past the slot's length, and
        same-slot prefix reuse wants them kept) -- the same rule as the
        slot pool and qwen36_slots.
        """
        if not 0 <= slot < self.num_slots:
            raise IndexError(f"slot {slot} out of range ({self.num_slots} slots)")
        self._pending_prefix_source.pop(slot, None)
        if self.slot_layers:
            for layer in self.slot_layers:
                layer.reset_caches(slot)
        self._kv_len[slot] = 0
        self._committed[slot] = []

    def _invalidate_slots_after_decode_failure(self, slots: list[int]) -> None:
        """Make partially mutated device state unreachable after a failed step."""
        for slot in slots:
            for layer in self.slot_layers:
                hard_clear = getattr(layer, "hard_clear_slot", None)
                if hard_clear is not None:
                    hard_clear(slot)
                else:
                    layer.reset_caches(slot)
            self._kv_len[slot] = 0
            self._committed[slot] = []
            self._drop_prefix_cache(slot)
            self._pending_prefix_source.pop(slot, None)

    def slot_state(self, slot: int) -> Dsv4SlotStateView:
        if not 0 <= slot < self.num_slots:
            raise IndexError(f"slot {slot} out of range ({self.num_slots} slots)")
        return Dsv4SlotStateView(
            kv_len=self._kv_len[slot],
            committed_tokens=tuple(self._committed[slot]),
        )

    def snapshot(self) -> BackendSnapshot:
        return BackendSnapshot(
            slots=tuple(
                SlotSnapshot(slot=s, kv_len=self._kv_len[s], is_fresh=self._kv_len[s] == 0)
                for s in range(self.num_slots)
            ),
            prefix=tuple(
                PrefixSnapshot(
                    slot=s,
                    cached_kv_len=self._prefix_cache_kv_len[s],
                    cached_tokens=(
                        len(self._prefix_cache_tokens[s])
                        if self._prefix_cache_tokens[s] is not None
                        else 0
                    ),
                    head=(self._prefix_cache_tokens[s] or ())[:8],
                )
                for s in range(self.num_slots)
            ),
            dflash_cg_status=tuple(sorted(self._cg_status.items())),
            runtime_stats=tuple(sorted(self.stats.items())),
            cg_fallback_reasons=tuple(sorted(self.cg_fallback_reasons.items())),
        )

    # -- memory probe ------------------------------------------------------

    def memory_breakdown(self) -> dict[str, int]:
        """Byte-level accounting of every CUDA tensor this backend holds.

        Categories (all bytes, ``torch`` storage accounting -- views share
        their base storage and are counted once via ``data_ptr``):
        ``weights``: all eager-graph parameters/buffers (the packed IQ2_XS /
        Q8_0 checkpoint); ``eager_oracle_kv``: the eager attention graph's
        own window+compressed KV arena and indexer scoring caches (kept as
        the load-time oracle; the serving path does NOT read them);
        ``kernel_kv_pages``: per-slot FP8 page buffers in ``slot_layers``
        (window/prefill/csa/hca); ``kernel_compressor_kv``: per-slot bf16
        compressor + indexer raw KV arenas; ``kernel_recursive_state``:
        fp32 compressor/indexer decode state; ``mla_scratch``: the shared
        backend-owned MLA arena; ``prefix_checkpoint``: retained prefix
        checkpoint clones; ``graph_buffers``: decode CUDA-Graph driver
        input/logits scratch (driver-owned, outside the graph pool).
        """
        if torch.device(self.device).type != "cuda":
            return {}
        # Compare device TYPE (not full device): tensors live on ``cuda:0``
        # while ``torch.device("cuda")`` carries index None, so a full
        # equality check silently excludes every tensor.
        device_type = torch.device(self.device).type

        # One storage is counted once, no matter how many views name it.
        seen: set[int] = set()
        totals: dict[str, int] = {
            "weights": 0,
            "eager_oracle_kv": 0,
            "kernel_kv_pages": 0,
            "kernel_compressor_kv": 0,
            "kernel_recursive_state": 0,
            "mla_scratch": 0,
            "prefix_checkpoint": 0,
            "graph_buffers": 0,
            "rope_freqs": 0,
        }

        def add(target: str, tensor: torch.Tensor | None) -> None:
            if tensor is None:
                return
            if tensor.device.type != device_type:
                return
            key = tensor.data_ptr()
            if key in seen:
                return
            seen.add(key)
            totals[target] += tensor.numel() * tensor.element_size()

        # -- weights: eager graph owns every checkpoint tensor --------------
        for name, tensor in self.model.named_parameters():
            add("weights", tensor)
        for name, tensor in self.model.named_buffers():
            key = tensor.data_ptr()
            if key in seen:
                continue
            if ".kv_cache" in name or ".kv_state" in name or ".score_state" in name:
                add("eager_oracle_kv", tensor)
            elif name.endswith("freqs_cis") or ".freqs_cis" in name:
                add("rope_freqs", tensor)
            else:
                add("weights", tensor)

        # -- kernel-path slot layers ----------------------------------------
        for layer in self.slot_layers:
            for name, tensor in layer.named_buffers():
                if name.endswith("_pages") or "pages" in name:
                    add("kernel_kv_pages", tensor)
                elif "kv_cache" in name:
                    add("kernel_compressor_kv", tensor)
                elif "kv_state" in name or "score_state" in name:
                    add("kernel_recursive_state", tensor)
                elif name == "freqs_cis" or name.endswith("freqs_cis"):
                    # Kernel-layer RoPE table duplicates the eager graph's --
                    # candidates for sharing.
                    add("rope_freqs", tensor)
                else:
                    # norm weights, sink, etc. -- real allocations but tiny.
                    add("weights", tensor)
            # compressor.kv_cache is a plain attribute (not a buffer).
            compressor = getattr(layer, "compressor", None)
            if compressor is not None:
                add("kernel_compressor_kv", getattr(compressor, "kv_cache", None))
            indexer = getattr(layer, "indexer", None)
            if indexer is not None:
                add("kernel_compressor_kv", getattr(indexer, "kv_cache", None))

        add("mla_scratch", self._shared_mla_scratch)

        # -- prefix checkpoint clones ---------------------------------------
        for checkpoint in self._prefix_checkpoint_tensors:
            if checkpoint is None:
                continue
            for row in checkpoint:
                for tensor in row:
                    add("prefix_checkpoint", tensor)
        for windows in self._prefix_window_tensors:
            if windows is None:
                continue
            for tensor in windows:
                add("prefix_checkpoint", tensor)
        for anchor in self._prefix_anchor_logits:
            add("prefix_checkpoint", anchor)

        # -- decode graph driver scratch (outside the graph pool) -----------
        for driver in self._decode_graphs.values():
            for name, tensor in vars(driver).items():
                if isinstance(tensor, torch.Tensor) and tensor.device.type == device_type:
                    add("graph_buffers", tensor)

        totals["torch_allocated"] = torch.cuda.memory_allocated(device_type)
        totals["torch_reserved"] = torch.cuda.memory_reserved(device_type)
        totals["freed_eager_oracle_kv"] = getattr(self, "_freed_eager_oracle_kv", 0)
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_type)
            totals["driver_free_bytes"] = int(free_bytes)
            totals["driver_total_bytes"] = int(total_bytes)
            totals["driver_used_bytes"] = int(total_bytes - free_bytes)
        except Exception:  # pragma: no cover - best-effort
            pass
        return totals

    # -- retained-slot prefix cache -----------------------------------------

    def _prefix_hit_for_slot(self, token_ids: list[int], slot: int) -> PrefixHit:
        cached = self._prefix_cache_tokens[slot]
        checkpoint_len = self._prefix_cache_kv_len[slot]
        if cached is None or checkpoint_len <= 0 or not token_ids:
            return PrefixHit(kv_hit=0, state_hit=0)
        limit = min(len(token_ids), len(cached), checkpoint_len)
        match = 0
        for index in range(limit):
            if token_ids[index] != cached[index]:
                break
            match += 1
        kv_hit = match // DSV4_PREFIX_BLOCK_SIZE * DSV4_PREFIX_BLOCK_SIZE
        state_hit = checkpoint_len if checkpoint_len <= kv_hit else 0
        if (
            self._prefix_checkpoint_tensors[slot] is None
            or self._prefix_window_tensors[slot] is None
        ):
            state_hit = 0
        return PrefixHit(kv_hit=kv_hit, state_hit=state_hit)

    @staticmethod
    def _layer_recurrent_tensors(
        layer: Dsv4AttnKernelLayer, slot: int
    ) -> tuple[torch.Tensor, ...]:
        tensors: list[torch.Tensor] = []
        if layer.compressor is not None:
            tensors.extend((layer.compressor.kv_state[slot], layer.compressor.score_state[slot]))
        if layer.indexer is not None:
            compressor = layer.indexer.compressor
            tensors.extend((compressor.kv_state[slot], compressor.score_state[slot]))
        return tuple(tensors)

    def _capture_prefix_checkpoint(
        self,
        slot: int,
        length: int,
        token_ids: list[int],
        anchor_logits: torch.Tensor,
    ) -> None:
        if length <= 0 or length % DSV4_PREFIX_BLOCK_SIZE:
            return
        live = tuple(self._layer_recurrent_tensors(layer, slot) for layer in self.slot_layers)
        cached = self._prefix_checkpoint_tensors[slot]
        if cached is None or tuple(tuple(t.shape for t in row) for row in cached) != tuple(
            tuple(t.shape for t in row) for row in live
        ):
            cached = tuple(tuple(t.detach().clone() for t in row) for row in live)
            self._prefix_checkpoint_tensors[slot] = cached
        else:
            for cached_row, live_row in zip(cached, live):
                for destination, source in zip(cached_row, live_row):
                    destination.copy_(source)
        live_windows = tuple(layer.window_pages[slot] for layer in self.slot_layers)
        cached_windows = self._prefix_window_tensors[slot]
        if cached_windows is None or tuple(t.shape for t in cached_windows) != tuple(
            t.shape for t in live_windows
        ):
            self._prefix_window_tensors[slot] = tuple(
                tensor.detach().clone() for tensor in live_windows
            )
        else:
            for destination, source in zip(cached_windows, live_windows):
                destination.copy_(source)
        anchor = anchor_logits[:, -1:].detach()
        cached_anchor = self._prefix_anchor_logits[slot]
        if cached_anchor is None or cached_anchor.shape != anchor.shape:
            self._prefix_anchor_logits[slot] = anchor.clone()
        else:
            cached_anchor.copy_(anchor)
        self._prefix_cache_tokens[slot] = tuple(token_ids[:length])
        self._prefix_cache_kv_len[slot] = length

    def _drop_prefix_cache(self, slot: int) -> None:
        self._prefix_cache_tokens[slot] = None
        self._prefix_cache_kv_len[slot] = 0
        self._prefix_checkpoint_tensors[slot] = None
        self._prefix_window_tensors[slot] = None
        self._prefix_anchor_logits[slot] = None

    @staticmethod
    def _checkpoint_shapes(
        rows: tuple[tuple[torch.Tensor, ...], ...]
    ) -> tuple[tuple[torch.Size, ...], ...]:
        return tuple(tuple(tensor.shape for tensor in row) for row in rows)

    def _restore_recurrent_checkpoint(
        self,
        slot: int,
        checkpoint: tuple[tuple[torch.Tensor, ...], ...],
    ) -> bool:
        live = tuple(self._layer_recurrent_tensors(layer, slot) for layer in self.slot_layers)
        if self._checkpoint_shapes(live) != self._checkpoint_shapes(checkpoint):
            return False
        for live_row, cached_row in zip(live, checkpoint):
            for destination, source in zip(live_row, cached_row):
                destination.copy_(source)
        return True

    def _apply_same_slot_prefix(
        self, slot: int, token_ids: list[int]
    ) -> tuple[int, torch.Tensor | None]:
        pending = self._pending_prefix_source.pop(slot, None)
        source_slot = slot if pending is None else pending[0]
        hit = self._prefix_hit_for_slot(token_ids, source_slot)
        if pending is not None and hit.effective != pending[1]:
            hit = PrefixHit(kv_hit=0, state_hit=0)
        if pending is None and hit.effective <= 0:
            for candidate in range(self.num_slots):
                if candidate == slot:
                    continue
                remote = self._prefix_hit_for_slot(token_ids, candidate)
                if (remote.effective, remote.kv_hit) > (hit.effective, hit.kv_hit):
                    source_slot, hit = candidate, remote
        length = hit.effective
        checkpoint = self._prefix_checkpoint_tensors[source_slot]
        window_checkpoint = self._prefix_window_tensors[source_slot]
        if length <= 0 or checkpoint is None or window_checkpoint is None:
            self._drop_prefix_cache(slot)
            return 0, None
        source_anchor = self._prefix_anchor_logits[source_slot]
        try:
            if source_slot != slot:
                for layer in self.slot_layers:
                    layer.hard_clear_slot(slot)
                for layer in self.slot_layers:
                    layer.copy_prefix(source_slot, slot, length)
            if len(window_checkpoint) != len(self.slot_layers):
                raise RuntimeError("prefix window checkpoint shape mismatch")
            for layer, cached_window in zip(self.slot_layers, window_checkpoint):
                if layer.window_pages[slot].shape != cached_window.shape:
                    raise RuntimeError("prefix window checkpoint shape mismatch")
                layer.window_pages[slot].copy_(cached_window)
                if source_slot == slot:
                    layer.clear_after_prefix(slot, length)
            if not self._restore_recurrent_checkpoint(slot, checkpoint):
                raise RuntimeError("prefix recurrent checkpoint shape mismatch")
        except Exception:
            for layer in self.slot_layers:
                layer.hard_clear_slot(slot)
            self._kv_len[slot] = 0
            self._committed[slot] = []
            self._drop_prefix_cache(slot)
            self.stats["prefix_restore_failures"] += 1
            return 0, None
        self._kv_len[slot] = length
        self._committed[slot] = list(token_ids[:length])
        self.stats["prefix_kv_hit_tokens"] += hit.kv_hit
        self.stats["prefix_state_hit_tokens"] += length
        if source_slot == slot:
            self.stats["prefix_same_slot_restores"] += 1
        else:
            self.stats["prefix_cross_slot_restores"] += 1
            assert source_anchor is not None
            self._capture_prefix_checkpoint(slot, length, token_ids, source_anchor)
        anchor = self._prefix_anchor_logits[slot]
        return length, anchor if length == len(token_ids) else None

    def reconcile_prefix_hit(self, token_ids: list[int]) -> PrefixHit:
        """Return the deepest complete same-slot retained checkpoint."""
        best = PrefixHit(kv_hit=0, state_hit=0)
        for slot in range(self.num_slots):
            if self._kv_len[slot] != 0:
                continue
            hit = self._prefix_hit_for_slot(token_ids, slot)
            if (hit.effective, hit.kv_hit) > (best.effective, best.kv_hit):
                best = hit
        return best

    def find_best_slot_for_prompt(
        self,
        token_ids: list[int],
        free_slots: list[int],
    ) -> tuple[int, int]:
        """Prefer the free slot retaining the deepest complete checkpoint."""
        if not free_slots:
            raise IndexError("no free slots")
        best_slot = free_slots[0]
        best = PrefixHit(kv_hit=0, state_hit=0)
        for slot in free_slots:
            hit = self._prefix_hit_for_slot(token_ids, slot)
            if (hit.effective, hit.kv_hit) > (best.effective, best.kv_hit):
                best_slot, best = slot, hit
        if best.effective > 0:
            self._pending_prefix_source.pop(best_slot, None)
            return best_slot, best.effective
        source_slot = -1
        remote = PrefixHit(kv_hit=0, state_hit=0)
        free = set(free_slots)
        for slot in range(self.num_slots):
            if slot in free:
                continue
            hit = self._prefix_hit_for_slot(token_ids, slot)
            if (hit.effective, hit.kv_hit) > (remote.effective, remote.kv_hit):
                source_slot, remote = slot, hit
        if remote.effective > 0:
            destination = free_slots[0]
            self._pending_prefix_source[destination] = (source_slot, remote.effective)
            return destination, remote.effective
        return best_slot, 0

    def prefix_hit_for_slot(self, token_ids: list[int], slot: int) -> PrefixHit:
        return self._prefix_hit_for_slot(token_ids, slot)

    def cross_slot_prefix_hit(self, token_ids: list[int]) -> PrefixHit:
        best = PrefixHit(kv_hit=0, state_hit=0)
        for slot in range(self.num_slots):
            hit = self._prefix_hit_for_slot(token_ids, slot)
            if (hit.effective, hit.kv_hit) > (best.effective, best.kv_hit):
                best = hit
        return best

    @property
    def has_speculative_decode(self) -> bool:
        return False

    # -- chunked prefill surface (one-shot: DSV4 prefill is never chunked) --

    def prefill_chunked_begin(
        self,
        slots: list[int],
        prompts_per_slot: list[list[int]],
        chunk_size: int = 512,
        *,
        params_per_slot: dict[int, SamplingParams] | None = None,
    ) -> ChunkedPrefillState:
        """One-shot prefill: every prompt fully prefilled, done immediately."""
        if len(slots) != len(prompts_per_slot):
            raise ValueError("slots and prompts_per_slot must have equal length")
        result: dict[int, dict] = {}
        for slot, prompt in zip(slots, prompts_per_slot):
            params = params_per_slot.get(slot) if params_per_slot else None
            if params is not None and params.temperature != 0.0:
                # E2-b contract: the anchor of a non-greedy request must be
                # sampled, not argmax'd.  Sample from the prefill logits.
                logits = self._prefill_logits(slot, prompt)
                gen = make_generator(params.seed)
                anchor = int(
                    sample_from_logits(logits[0, -1].unsqueeze(0), params, generator=gen).item()
                )
            else:
                anchor = self.prefill(slot, prompt)
            result[slot] = {"anchor": anchor, "draft_tokens": []}
        return ChunkedPrefillState(done=True, result=result)

    def prefill_chunked_step(self, state: ChunkedPrefillState) -> bool:
        """DSV4 prefill is never incremental; state is always already done."""
        return state.done

    def _prefill_logits(self, slot: int, prompt_ids: list[int]) -> torch.Tensor:
        """Prefill the whole prompt in chunks; return the final logits.

        The MLA scratch is planned for ``max_q_rows`` rows per forward
        (bounded, not the full context -- a full-context plan OOMs: the
        gate measured ~11 GB of scratch per 128 rows across the ratio-4
        layers).  Each chunk is one multi-row forward at its absolute
        start position; the compressor/indexer state machines step per
        token inside, and only the last chunk's logits matter (the
        anchor).
        """
        if not 0 <= slot < self.num_slots:
            raise IndexError(f"slot {slot} out of range ({self.num_slots} slots)")
        if not prompt_ids:
            raise ValueError("prefill needs at least one token")
        if len(prompt_ids) > self.max_seq_len:
            raise IndexError(
                f"prefill length {len(prompt_ids)} exceeds max_seq_len={self.max_seq_len}"
            )
        if self._kv_len[slot] != 0:
            raise RuntimeError(
                f"slot {slot} is at kv_len={self._kv_len[slot]}; the caller must reset_slot first"
            )
        prefix_len, cached_anchor = self._apply_same_slot_prefix(slot, prompt_ids)
        if cached_anchor is not None:
            self.stats["prefill_calls"] += 1
            return cached_anchor
        if self._forward_fn is not None or not self.slot_layers:
            # Test-injected forward stub (no kernel-path stack): keep the
            # chunk-major loop, whose ``_forward`` short-circuits to the stub.
            chunk = max(1, min(self.max_q_rows, 128))
            logits = None
            chunks = 0
            for start in range(prefix_len, len(prompt_ids), chunk):
                ids = prompt_ids[start : start + chunk]
                trace_row = (
                    bfdiag_trace.begin_round(slot, start) if bfdiag_trace.TRACE_ENABLED else -1
                )
                logits = self._forward(
                    slot,
                    torch.tensor([ids], dtype=torch.long, device=self.device),
                    start,
                )
                if bfdiag_trace.TRACE_ENABLED:
                    end = start + len(ids)
                    bfdiag_trace.finish_dsv4_prefill_chunk(
                        trace_row,
                        position=start,
                        row_count=len(ids),
                        compressor_ratio=-1,
                        window_entries=min(end, self.config.window_size),
                        ratio4_entries=(end // 4 if 4 in self._trace_compressor_ratios else 0),
                        ratio128_entries=(
                            end // 128 if 128 in self._trace_compressor_ratios else 0
                        ),
                    )
                if (start + len(ids)) % DSV4_PREFIX_BLOCK_SIZE == 0:
                    self._capture_prefix_checkpoint(slot, start + len(ids), prompt_ids, logits)
                chunks += 1
            assert logits is not None
            self._kv_len[slot] = len(prompt_ids)
            self._committed[slot] = list(prompt_ids)
            self.stats["prefill_calls"] += 1
            self.stats["prefill_chunks"] += chunks
            self.stats["prefill_tokens"] += len(prompt_ids) - prefix_len
            return logits
        # Layer-major superchunk: attention stepped in GPU-position tiles,
        # MoE eager over the whole suffix.  This replaces the old chunk-major
        # 43-layer loop (its 64-token MoE chunks under-utilised the GEMMs and
        # the ratio-4 compressor's Python loop dominated the CPU).
        logits = self._prefill_superchunk_logits(
            slot, prompt_ids, tile=self.max_q_rows, prefix_len=prefix_len
        )
        self._kv_len[slot] = len(prompt_ids)
        self._committed[slot] = list(prompt_ids)
        self.stats["prefill_calls"] += 1
        self.stats["prefill_chunks"] += 1
        self.stats["prefill_tokens"] += len(prompt_ids) - prefix_len
        return logits

    # -- serving forward -----------------------------------------------------

    def _forward(self, slot: int, input_ids: torch.Tensor, start_pos: int) -> torch.Tensor:
        """One forward on the slot's kernel-path stack.

        Mirrors the Phase-3 gate's kernel-path run: eager HC/FFN/moe, kernel
        attention, sharing every weight module with the eager graph.
        ``input_ids`` is a [1, seq] long tensor; returns fp32 logits.

        With a test-injected ``forward_fn`` this is replaced entirely
        (``forward_fn(slot, input_ids, start_pos)``).
        """
        if self._forward_fn is not None:
            return self._forward_fn(slot, input_ids, start_pos)
        model, layers = self.model, self.slot_layers
        h = model.embed(input_ids)
        h = h.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
        for i, block in enumerate(model.blocks):
            residual = h
            x, post, comb = block.hc_pre(
                h, block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base
            )
            x = rms_norm(x, block.attn_norm_weight, block.eps)
            x = layers[i](x, start_pos, slot=slot)
            x = block.hc_post(x, residual, post, comb)
            residual = x
            x, post, comb = block.hc_pre(x, block.hc_ffn_fn, block.hc_ffn_scale, block.hc_ffn_base)
            x = rms_norm(x, block.ffn_norm_weight, block.eps)
            if self._prefill_graph is not None and input_ids.shape[1] == self._prefill_graph.m:
                # CUDA-graph MoE: flat [M, H] -> routed+shared [M, H] bf16,
                # then reshape back to [1, M, H] for hc_post.  Only exact-M
                # rows go through the graph: a shorter tail chunk (or the
                # M=1 decode fallback, which shares this forward) would trip
                # the graph's fixed (M, H) replay contract, so those fall
                # back to eager ``block.moe`` -- correctness first, the tail
                # is one chunk out of many.
                x_flat = x.reshape(-1, x.shape[-1])
                ids_flat = input_ids.reshape(-1)
                out = self._prefill_graph.replay_layer(i, x_flat, ids_flat)
                x = out.to(x.dtype).reshape(x.shape)
            else:
                x = block.moe(x, input_ids)
            x = block.hc_post(x, residual, post, comb)
            h = x
        h = model.hc_head(h)
        return model.lm_head(rms_norm(h, model.norm_weight, model.eps))

    def _prefill_superchunk_logits(
        self, slot: int, prompt_ids: list[int], *, tile: int = 64, prefix_len: int = 0
    ) -> torch.Tensor:
        """Layer-major superchunk prefill (fast path).

        Processes the whole prompt layer-by-layer: HC/norm/MoE batched over
        all rows, causal attention stepped in ``tile``-row tiles through the
        GPU-position ``forward_graph_prefill`` (which removes the ratio-4
        compressor's per-token Python loop -- the dominant prefill CPU cost).
        MoE stays eager over the full prompt (its dynamic batch-2 split is the
        only correct handling of DSV4's skewed routing; a fixed-bucket graph
        cannot cover max route ~384).  ``prefix_len`` continues a restored
        prefix: rows are processed starting at that absolute position.  Only
        the final token's logits are returned.
        """
        if self._kv_len[slot] != 0 and prefix_len == 0:
            raise RuntimeError(
                f"slot {slot} is at kv_len={self._kv_len[slot]}; the caller must reset_slot first"
            )
        suffix = prompt_ids[prefix_len:]
        n = len(suffix)
        if n == 0:
            raise ValueError("superchunk prefill needs a non-empty suffix")
        model, layers = self.model, self.slot_layers
        ids = torch.tensor([suffix], dtype=torch.long, device=self.device)
        h = model.embed(ids)
        h = h.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
        for i, block in enumerate(model.blocks):
            residual = h
            x, post, comb = block.hc_pre(
                h, block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base
            )
            x = rms_norm(x, block.attn_norm_weight, block.eps)
            outs = []
            for start in range(0, n, tile):
                end = min(start + tile, n)
                outs.append(layers[i](x[:, start:end], prefix_len + start, slot=slot))
            x = torch.cat(outs, dim=1)
            x = block.hc_post(x, residual, post, comb)
            residual = x
            x, post, comb = block.hc_pre(
                x, block.hc_ffn_fn, block.hc_ffn_scale, block.hc_ffn_base
            )
            x = rms_norm(x, block.ffn_norm_weight, block.eps)
            x = block.moe(x, ids)
            x = block.hc_post(x, residual, post, comb)
            h = x
        h = model.hc_head(h)
        return model.lm_head(rms_norm(h, model.norm_weight, model.eps))

    def _forward_decode_batch(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        slot_ids: torch.Tensor,
        *,
        max_index_entries: int | None,
    ) -> torch.Tensor:
        """One native B=1/2/4 decode forward over the shared slot arena."""
        if self._forward_fn is not None:
            raise RuntimeError("native decode batching is unavailable with an injected forward_fn")
        bsz = int(input_ids.shape[0])
        if input_ids.shape != (bsz, 1) or bsz not in (1, 2, 4):
            raise ValueError(
                f"input_ids must be [B, 1] with B in (1, 2, 4), got {tuple(input_ids.shape)}"
            )
        if positions.shape != (bsz,) or slot_ids.shape != (bsz,):
            raise ValueError(
                f"positions and slot_ids must both be [{bsz}], got "
                f"{tuple(positions.shape)} and {tuple(slot_ids.shape)}"
            )

        model = self.model
        h = model.embed(input_ids)
        h = h.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
        for index, block in enumerate(model.blocks):
            residual = h
            x, post, comb = block.hc_pre(
                h,
                block.hc_attn_fn,
                block.hc_attn_scale,
                block.hc_attn_base,
            )
            x = rms_norm(x, block.attn_norm_weight, block.eps)
            x = self.slot_layers[index].forward_decode_batch(
                x,
                positions,
                slot_ids,
                graph_max_index_entries=max_index_entries,
            )
            x = block.hc_post(x, residual, post, comb)

            residual = x
            x, post, comb = block.hc_pre(
                x,
                block.hc_ffn_fn,
                block.hc_ffn_scale,
                block.hc_ffn_base,
            )
            x = rms_norm(x, block.ffn_norm_weight, block.eps)
            x = block.moe.forward_decode_batch(x, input_ids)
            h = block.hc_post(x, residual, post, comb)
        h = model.hc_head(h)
        return model.lm_head(rms_norm(h, model.norm_weight, model.eps))

    def _decode_index_bucket(self, positions: list[int]) -> int | None:
        """Smallest ratio-4 graph capacity covering every batch row."""
        from runtime.backends.dsv4_cudagraph import _index_entry_buckets

        caps = _index_entry_buckets(self.slot_layers)
        if not caps:
            return None
        needed = max(max(1, (position + 1) // 4) for position in positions)
        for cap in caps:
            if needed <= cap:
                return cap
        return caps[-1]

    def capture_decode_cuda_graph(self) -> int | None:
        """Capture native B=1/2/4 decode graphs over the shared slot arena.

        One bucketed driver owns every supported fixed batch size and ratio-4
        history-capacity bucket.  Slot ids and positions remain persistent
        tensor inputs, so a B=1 graph is no longer bound to one concrete slot.
        Capture is atomic and load-time only: failure drops the whole set and
        leaves the already verified eager batch path available.
        """
        supported_batches = tuple(size for size in (1, 2, 4) if size <= self.num_slots)
        if self._decode_graphs:
            if tuple(self._decode_graphs) != supported_batches:
                raise RuntimeError("partial DSV4 decode graph set is not valid")
            return len(self._decode_graphs)
        if not self._native_decode_batch_available:
            self._cg_status["decode"] = "unsupported"
            return None
        if self._forward_fn is not None or not self.slot_layers:
            return None
        if torch.device(self.device).type != "cuda":
            return None
        busy_slots = [
            slot
            for slot in range(self.num_slots)
            if self._kv_len[slot] != 0 or self._committed[slot]
        ]
        if busy_slots:
            raise RuntimeError(
                "DSV4 decode CUDA Graph capture must run before slot admission; "
                f"active slots: {busy_slots}"
            )
        # Capture warmup writes every slot arena. Retained prefixes are not
        # live admissions, but their backing bytes are about to be replaced.
        for slot in range(self.num_slots):
            self._drop_prefix_cache(slot)
        self._pending_prefix_source.clear()
        self.stats["decode_graph_capture_attempts"] += 1
        try:
            from runtime.backends.dsv4_cudagraph import build_batched_decode_graph_driver

            driver = build_batched_decode_graph_driver(
                backend=self,
                device=self.device,
            )
            driver.capture()
            self._decode_graphs = {size: driver for size in supported_batches}
            self._cg_status["decode"] = "captured"
            self.stats["decode_graph_capture_successes"] += 1
            return len(self._decode_graphs)
        except Exception:
            import logging

            logging.getLogger("qwen_sm120_runtime.dsv4").exception(
                "DSV4 decode CUDA Graph capture failed; falling back to eager"
            )
            self._decode_graphs.clear()
            # Warmup/capture exercised the complete native batch body before
            # any request was admitted.  If that qualification fails, serving
            # falls back to the verified serial B1 path instead of retrying a
            # partially mutating batch forward on live slot state.
            self._native_decode_batch_available = False
            self._cg_status["decode"] = "failed"
            self.stats["decode_graph_capture_failures"] += 1
            return None
        finally:
            for slot in range(self.num_slots):
                self.reset_slot(slot)

    def capture_prefill_cuda_graph(self) -> bool:
        """Capture the 43-layer K32 MoE prefill CUDA graphs (Phase 1K).

        Must run before slot admission and after ``capture_decode_cuda_graph``.
        Failure is non-fatal: falls back to eager ``block.moe`` prefill.

        Memory guard: capture freezes every temporary tensor of the 43-layer
        body into CUDA-graph pool memory (measured ~10 GiB on top of the 82
        GiB weights + KV envelope), and a failed capture leaves that pool
        reserved.  We check free memory before starting and call
        ``empty_cache`` after a failure so a thin-margin card falls back to
        eager MoE cleanly instead of wedging every later request with a dead
        reserved pool (2026-08-13: 2x64K service OOM'd on reset_slot because
        of exactly this residue).
        """
        if self._prefill_graph is not None:
            return True
        if self._forward_fn is not None or not self.slot_layers:
            return False
        if torch.device(self.device).type != "cuda":
            return False
        busy = any(self._kv_len[s] != 0 or self._committed[s] for s in range(self.num_slots))
        if busy:
            raise RuntimeError(
                "DSV4 prefill CUDA Graph capture must run before slot admission"
            )
        # Estimate the capture pool delta before allocating: the graph body's
        # intermediates are frozen into graph memory per layer.  If we do not
        # have a comfortable margin over the resident weights+KV envelope,
        # skip capture now -- eager MoE is the safe path and the failed pool
        # would otherwise eat the headroom every later request needs.
        try:
            free_bytes, _total = torch.cuda.mem_get_info(self.device)
        except Exception:  # pragma: no cover - best-effort guard
            free_bytes = 0
        # The 43 layers share ONE CUDA graph memory pool (Dsv4PrefillGraphDriver
        # reuses the first layer's pool), so the frozen intermediate delta is
        # roughly one layer's worth, not 43x.  Measured 2026-08-13 at 128K:
        # reserved delta 1.14 GiB, free delta 1.31 GiB.  Reserve a 2 GiB
        # envelope plus slack for driver/allocator overhead.
        ENVELOPE = 2.5 * 2**30
        if free_bytes < ENVELOPE:
            import logging

            logging.getLogger("qwen_sm120_runtime.dsv4").warning(
                "DSV4 prefill CUDA Graph capture skipped: %.1f GiB free < "
                "%.1f GiB needed; falling back to eager MoE prefill",
                free_bytes / 2**30,
                ENVELOPE / 2**30,
            )
            self._cg_status["prefill"] = "skipped-low-memory"
            return False
        try:
            driver = Dsv4PrefillGraphDriver(self, m=self.max_q_rows)
            driver.capture()
            self._prefill_graph = driver
            self._cg_status["prefill"] = "captured"
            return True
        except Exception:
            import logging

            logging.getLogger("qwen_sm120_runtime.dsv4").exception(
                "DSV4 prefill CUDA Graph capture failed; falling back to eager MoE"
            )
            self._prefill_graph = None
            self._cg_status["prefill"] = "failed"
            # The failed capture may have grown a CUDA-graph pool that torch
            # keeps reserved but idle.  Release it so the serving path does
            # not inherit a dead reservation it can never use.
            torch.cuda.empty_cache()
            return False

    def prefill(self, slot: int, prompt_ids: list[int]) -> int:
        """Cold prefill of ``prompt_ids``; greedy first token (server contract)."""
        logits = self._prefill_logits(slot, prompt_ids)
        return int(logits[0, -1].argmax(dim=-1).item())

    def decode_batch_sampled(
        self,
        slot_ids: list[int],
        token_ids: list[int],
        kv_lengths: list[int],
        params_list: list[SamplingParams],
        *,
        return_logprobs: bool = False,
        top_logprobs: int = 0,
    ) -> list[int] | tuple[list[int], list[dict]]:
        """Decode one token per distinct slot through native B=1/2/4 buckets."""
        if not (len(slot_ids) == len(token_ids) == len(kv_lengths) == len(params_list)):
            raise ValueError("slot_ids/token_ids/kv_lengths/params_list must be equal length")
        if not slot_ids:
            return ([], []) if return_logprobs else []
        if len(set(slot_ids)) != len(slot_ids):
            raise ValueError("one decode batch cannot contain the same slot more than once")

        # Validate the complete host request before any graph/eager state is
        # mutated. This is the graph-safety boundary: device-side slot tensors
        # never need .item(), .tolist(), or uniqueness checks.
        for slot, kv_len in zip(slot_ids, kv_lengths):
            if not 0 <= slot < self.num_slots:
                raise IndexError(f"slot {slot} out of range ({self.num_slots} slots)")
            if kv_len != self._kv_len[slot]:
                raise RuntimeError(
                    f"slot {slot} is at kv_len={self._kv_len[slot]}, caller says {kv_len}; "
                    "the caller must reset_slot first"
                )
            if not 0 <= kv_len < self.max_seq_len:
                raise IndexError(
                    f"decode position {kv_len} out of range for max_seq_len={self.max_seq_len}"
                )

        outs: list[int] = []
        logprob_rows: list[dict] = []
        serial_decode = self._forward_fn is not None or not self._native_decode_batch_available
        chunks = (1,) * len(slot_ids) if serial_decode else _decode_batch_chunks(len(slot_ids))
        offset = 0
        for batch_size in chunks:
            end_offset = offset + batch_size
            chunk_slots = slot_ids[offset:end_offset]
            chunk_tokens = token_ids[offset:end_offset]
            chunk_positions = kv_lengths[offset:end_offset]
            chunk_params = params_list[offset:end_offset]
            trace_rows = [
                bfdiag_trace.begin_round(slot, position) if bfdiag_trace.TRACE_ENABLED else -1
                for slot, position in zip(chunk_slots, chunk_positions)
            ]

            graph_key = chunk_slots[0] if serial_decode else batch_size
            graph = self._decode_graphs.get(graph_key)
            try:
                if graph is not None:
                    if serial_decode:
                        logits = graph.replay(chunk_slots[0], chunk_tokens[0], chunk_positions[0])
                    else:
                        logits = graph.replay_host(
                            chunk_tokens,
                            chunk_positions,
                            chunk_slots,
                            max_index_entries=self._decode_index_bucket(chunk_positions),
                        )
                    self.stats["decode_graph_replays"] += 1
                    trace_path = bfdiag_events.Path.CG_REPLAY
                    trace_reason = bfdiag_events.CgMissReason.NONE
                else:
                    if serial_decode:
                        logits = self._forward(
                            chunk_slots[0],
                            torch.tensor(
                                [[chunk_tokens[0]]], dtype=torch.long, device=self.device
                            ),
                            chunk_positions[0],
                        )
                    else:
                        input_tensor = torch.tensor(
                            chunk_tokens, dtype=torch.long, device=self.device
                        ).reshape(batch_size, 1)
                        position_tensor = torch.tensor(
                            chunk_positions, dtype=torch.long, device=self.device
                        )
                        slot_tensor = torch.tensor(
                            chunk_slots, dtype=torch.long, device=self.device
                        )
                        logits = self._forward_decode_batch(
                            input_tensor,
                            position_tensor,
                            slot_tensor,
                            max_index_entries=self._decode_index_bucket(chunk_positions),
                        )
                    self.stats["decode_eager_fallbacks"] += 1
                    reason = (
                        "capture_failed"
                        if self._cg_status.get("decode") == "failed"
                        else "not_captured"
                    )
                    self.cg_fallback_reasons[reason] = (
                        self.cg_fallback_reasons.get(reason, 0) + 1
                    )
                    trace_path = bfdiag_events.Path.EAGER
                    trace_reason = (
                        bfdiag_events.CgMissReason.CAPTURE_FAILED
                        if reason == "capture_failed"
                        else bfdiag_events.CgMissReason.NOT_CAPTURED
                    )
            except Exception:
                self._invalidate_slots_after_decode_failure(chunk_slots)
                raise

            if all(params.temperature == 0.0 for params in chunk_params):
                if graph is not None and getattr(graph, "greedy", False):
                    # The graph baked argmax into the input buffer; replay
                    # already returned the next-token ids for the batch.
                    chunk_outs = [int(t) for t in logits.tolist()]
                else:
                    # One result synchronization for the whole greedy batch.
                    chunk_outs = [int(token) for token in logits[:, 0].argmax(dim=-1).tolist()]
            else:
                chunk_outs = []
                for row, params in enumerate(chunk_params):
                    if params.temperature == 0.0:
                        out = int(logits[row, 0].argmax(dim=-1).item())
                    else:
                        generator = make_generator(params.seed)
                        out = int(
                            sample_from_logits(
                                logits[row, 0].unsqueeze(0), params, generator=generator
                            ).item()
                        )
                    chunk_outs.append(out)

            for row, (slot, token, kv_len, out) in enumerate(
                zip(chunk_slots, chunk_tokens, chunk_positions, chunk_outs)
            ):
                self._kv_len[slot] += 1
                # This forward wrote ``token`` at ``kv_len``. ``out`` is the
                # pending next token and enters KV on the following round.
                self._committed[slot].append(token)
                if self._kv_len[slot] % DSV4_PREFIX_BLOCK_SIZE == 0:
                    self._capture_prefix_checkpoint(
                        slot,
                        self._kv_len[slot],
                        self._committed[slot],
                        logits[row : row + 1],
                    )
                outs.append(out)
                if return_logprobs:
                    logprob_rows.append(compute_logprobs(logits[row], [out], top_k=top_logprobs)[0])
                if bfdiag_trace.TRACE_ENABLED:
                    trace_end = kv_len + 1
                    bfdiag_trace.finish_dsv4_decode_round(
                        trace_rows[row],
                        position=kv_len,
                        row_count=1,
                        path=trace_path,
                        cg_miss_reason=trace_reason,
                        window_entries=min(trace_end, self.config.window_size),
                        ratio4_entries=(
                            trace_end // 4 if 4 in self._trace_compressor_ratios else 0
                        ),
                        ratio128_entries=(
                            trace_end // 128 if 128 in self._trace_compressor_ratios else 0
                        ),
                    )
            offset = end_offset
        self.stats["decode_rounds"] += 1
        self.stats["decode_tokens"] += len(slot_ids)
        if return_logprobs:
            return outs, logprob_rows
        return outs


def load_deepseek_v4_backend(
    gguf_path: str | torch,  # torch re-export guard for the engine import path
    *,
    num_slots: int = 1,
    max_seq_len: int = 4096,
    max_q_rows: int = 1,
    device: str = "cuda",
) -> DeepseekV4Backend:
    """Load the GGUF and build the serving backend (engine-thread entry)."""
    model, count = load_dsv4_from_gguf(gguf_path, max_seq_len=max_seq_len, device=device)
    backend = DeepseekV4Backend(
        model,
        model.config,
        num_slots=num_slots,
        max_seq_len=max_seq_len,
        max_q_rows=max_q_rows,
        device=device,
    )
    return backend


def _prefill_bucket_for_rows(m: int, top_k: int, n_experts: int) -> int:
    """Pick the MoE tile bucket for a prefill chunk of ``m`` rows.

    The graph now runs TWO fixed batches (bucket + overflow bucket), covering
    up to ``2 * bucket`` routes per expert, so the base bucket stays small
    (32) for fast tile GEMMs while the overflow batch absorbs the hash layers'
    skewed route distribution (max route ~51 at 128 tokens).  bucket=32 is
    ~35% faster than bucket=64 for the same chunk (measured 14.9 vs 22.8 ms).
    """
    return 32


class Dsv4PrefillGraphDriver:
    """CUDA-graph-captured 43-layer K32 MoE prefill (Phase 1K service path).

    Each of the 43 layers gets its OWN graph capturing ``router -> K32
    grouped MoE`` for a fixed 64-token chunk (the service prefill chunk
    size).  The surrounding HC/norm/attention stay eager; the MoE was the
    dominant CPU cost (profile 2026-08-12: ~2.4 s CPU vs ~270 ms GPU per
    chunk, most of it MoE Python glue), so removing the 43-layer MoE launch
    storm is the largest single lever without re-architecting attention.

    Replay contract:
      - ``capture()`` must run on the engine thread after slot layers exist
        and before any request; it warmups each layer's kernels eagerly.
      - ``replay_layer(i, flat, ids)`` writes ``flat`` [M, H] and ``ids``
        [M] into layer ``i``'s workspace (contents only), replays its graph,
        and returns the routed MoE output [M, H].
      - Workspaces are caller-owned; no allocation inside a graph body.
    """

    def __init__(
        self,
        backend: DeepseekV4Backend,
        *,
        m: int = 64,
        bucket: int | None = None,
    ) -> None:
        if not backend.slot_layers:
            raise RuntimeError("Dsv4PrefillGraphDriver needs kernel-path slot layers")
        self.backend = backend
        self.model = backend.model
        self.layers = backend.slot_layers
        self.m = m
        self.top_k = backend.model.config.n_activated_experts
        self.hidden = backend.model.config.hidden_size
        self.inter = backend.model.config.moe_intermediate_size
        self.n_experts = backend.model.config.n_routed_experts
        self.device = backend.device
        # The graph's tile is [n_experts, bucket, ...]; bucket must cover the
        # max routes a single expert receives in one chunk, but a fixed 64
        # wastes the tile when the chunk is small.  Measured 2026-08-13 at
        # m=64: max route 14 (p99 13) over 300 samples, so bucket=32 gives a
        # 2x margin and is 35% faster than bucket=64 (14.9 vs 22.8 ms).  A
        # 64-token chunk never needs bucket=64 -- that sizing came from the
        # 1024-token superchunk distribution (max route ~83), wrongly applied
        # to the 64-token chunk in the first wiring.
        if bucket is None:
            bucket = _prefill_bucket_for_rows(m, self.top_k, self.n_experts)
        self.bucket = bucket
        # ONE shared workspace: layers run serially, so every layer graph may
        # capture the same buffer addresses and overwrite their contents
        # between replays.  Allocating per-layer workspaces would cost
        # ~43 x 1 GiB and OOM the 96 GB card alongside the 82 GiB weights.
        self.workspace = Dsv4PrefillMoEWorkspace(
            device=self.device,
            hidden=self.hidden,
            inter=self.inter,
            m=m,
            top_k=self.top_k,
            n_experts=self.n_experts,
            bucket=self.bucket,
        )
        self.graphs: list[torch.cuda.CUDAGraph | None] = [None] * len(self.layers)
        self._captured = False
        # per-layer input staging (written by replay_layer, read inside graph)
        self._flat_in = torch.empty(m, self.hidden, dtype=torch.bfloat16, device=self.device)
        self._ids_in = torch.empty(m, dtype=torch.long, device=self.device)

    def _layer_router(self, layer_id: int) -> Any:
        return self.model.blocks[layer_id].moe.gate

    def _layer_packed(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        moe = self.model.blocks[layer_id].moe
        grid, ksigns, _ = moe.gate_exps.tables()
        return moe.gate_exps.packed, moe.up_exps.packed, moe.down_exps.packed, grid, ksigns

    def _layer_body(self, layer_id: int, ws: Dsv4PrefillMoEWorkspace) -> torch.Tensor:
        """One layer's graph body: router -> K32 grouped MoE (single chunk).

        Reads ONLY ``ws.flat`` (caller writes it before replay) plus the
        driver's ``_ids_in`` (copied per replay); returns the routed output
        written into ``ws.out``.
        """
        gate = self._layer_router(layer_id)
        flat = ws.flat
        ids = self._ids_in
        # router: [M, H] -> weights [M, top_k], indices [M, top_k]
        weights, indices = gate(flat, ids)
        ws.weights.copy_(weights)
        ws.indices.copy_(indices)
        gate_packed, up_packed, down_packed, grid, ksigns = self._layer_packed(layer_id)
        # Routed MoE ONLY inside the graph; the shared Q8_0 expert stays
        # eager in replay_layer (adding it here aliases ws.out and the
        # read-write on the same buffer is not graph-stable).
        return grouped_moe_prefill_k32_graph(
            ws,
            flat,
            ws.indices,
            ws.weights,
            gate_packed,
            up_packed,
            down_packed,
            grid,
            ksigns,
            inter=self.inter,
            hidden=self.hidden,
            swiglu_limit=self.model.config.swiglu_limit,
        )

    def capture(self) -> None:
        if self._captured:
            return
        ws = self.workspace
        # ONE shared CUDA graph memory pool across all 43 layers.  Without it
        # every layer's graph freezes its own copy of the body's intermediate
        # tensors into graph-pool memory (~10 GiB measured at 2x64K, which
        # OOM'd the 96 GB card alongside the 82 GiB weights).  With a shared
        # pool the layers run serially and reuse the same addresses, so the
        # frozen delta collapses to roughly one layer's worth.
        pool = None
        # Warmup inputs must look like real routing: an all-zeros ``flat``
        # collapses the router's softmax and can send every route to one
        # expert (``within`` overflows the bucket and trips a device assert in
        # the tile fill).  Use a seeded random batch so the warmup exercises a
        # representative route spread; the captured graph reads whatever
        # contents ``replay_layer`` writes, so warmup values never leak into
        # serving output.
        warm_gen = torch.Generator(device=self.device).manual_seed(20260813)
        warm_flat = torch.randn(
            self.m, self.hidden, dtype=torch.bfloat16, device=self.device, generator=warm_gen
        ) * 0.3
        warm_ids = torch.randint(
            0, 1000, (self.m,), dtype=torch.long, device=self.device, generator=warm_gen
        )
        for layer_id in range(len(self.layers)):
            # warmup on a side stream (JIT kernels are not capturable)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                ws.flat.copy_(warm_flat)
                self._ids_in.copy_(warm_ids)
                for _ in range(3):
                    self._layer_body(layer_id, ws)
            torch.cuda.current_stream().wait_stream(side)
            # capture on the current stream, into the shared pool
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                self._layer_body(layer_id, ws)
            if pool is None:
                pool = graph.pool()
            self.graphs[layer_id] = graph
        self._captured = True

    def replay_layer(self, layer_id: int, flat: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """Run layer ``layer_id``'s graph on ``flat`` [M, H] and ``ids`` [M]."""
        if not self._captured:
            raise RuntimeError("prefill graph replay before capture")
        if tuple(flat.shape) != (self.m, self.hidden):
            raise ValueError(
                f"prefill graph expects flat {(self.m, self.hidden)}, got {tuple(flat.shape)}"
            )
        self.workspace.flat.copy_(flat)
        self._ids_in.copy_(ids)
        self.graphs[layer_id].replay()
        # shared Q8_0 expert (eager, cheap 3 GEMMs) -- the routed result in
        # workspace.out gets the shared expert added, matching Dsv4MoE.forward.
        moe = self.model.blocks[layer_id].moe
        return self.workspace.out + moe._shared_forward(self.workspace.flat)  # noqa: SLF001
