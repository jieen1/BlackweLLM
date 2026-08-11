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

    def _native_decode_batch_contract_supported(self) -> bool:
        """Preflight every production compressor shape before state mutation."""
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
        chunk = max(1, min(self.max_q_rows, 128))
        logits = None
        chunks = 0
        for start in range(prefix_len, len(prompt_ids), chunk):
            ids = prompt_ids[start : start + chunk]
            trace_row = bfdiag_trace.begin_round(slot, start) if bfdiag_trace.TRACE_ENABLED else -1
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
                    ratio128_entries=(end // 128 if 128 in self._trace_compressor_ratios else 0),
                )
            end = start + len(ids)
            if end % DSV4_PREFIX_BLOCK_SIZE == 0:
                self._capture_prefix_checkpoint(slot, end, prompt_ids, logits)
            chunks += 1
        assert logits is not None
        self._kv_len[slot] = len(prompt_ids)
        self._committed[slot] = list(prompt_ids)
        self.stats["prefill_calls"] += 1
        self.stats["prefill_chunks"] += chunks
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
            x = block.moe(x, input_ids)
            x = block.hc_post(x, residual, post, comb)
            h = x
        h = model.hc_head(h)
        return model.lm_head(rms_norm(h, model.norm_weight, model.eps))

    def _prefill_superchunk_logits(
        self, slot: int, prompt_ids: list[int], *, tile: int = 64
    ) -> torch.Tensor:
        """Phase 1: layer-major superchunk prefill (correctness-first).

        Processes the whole prompt layer-by-layer with HC/norm/MoE batched
        over all rows; the causal attention is still stepped in ``tile``-row
        tiles so compressor/indexer/window state advances exactly as the
        chunk-major path.  Only the final token's logits are returned; the
        last tile carries it.
        """
        if self._kv_len[slot] != 0:
            raise RuntimeError(
                f"slot {slot} is at kv_len={self._kv_len[slot]}; the caller must reset_slot first"
            )
        n = len(prompt_ids)
        model, layers = self.model, self.slot_layers
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
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
                outs.append(layers[i](x[:, start:end], start, slot=slot))
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
