"""DeepSeek-V4-Flash backend: multi-slot kernel-path serving (Phase 4).

The serving path runs the kernel-path attention stack (``Dsv4AttnKernelLayer``
per slot, weights shared from the eager ``Dsv4Transformer`` that owns the
checkpoint) -- packed FP8 KV pages + the fork compressed_mla kernel -- with
the eager graph kept as the load-time weight holder and oracle only.  Each
slot owns a full 43-layer kernel stack (its own page buffers, compressor
decode state, and indexer scoring caches); ``reset_slot`` zeroes the
recursive compressor state (KV bytes survive, same rule as the slot pool).

Phase 3's single-slot eager backend is superseded; the protocol surface is
unchanged (``ModelBackend``'s six unconditional members).  Capabilities are
still conservative: no CUDA Graph, no chunked prefill, no prefix cache --
those land incrementally, each with its own gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from runtime.backends.protocol import (
    BackendCapabilities,
    BackendSnapshot,
    PrefixHit,
    PrefixSnapshot,
    SlotSnapshot,
)
from runtime.block_pool import ChunkedPrefillState
from runtime.model.dsv4_attn_kernel import Dsv4AttnKernelLayer
from runtime.model.dsv4_config import Dsv4Config
from runtime.model.dsv4_model import (
    Dsv4Transformer,
    load_dsv4_from_gguf,
    rms_norm,
)
from runtime.sampling import SamplingParams, make_generator, sample_from_logits

EOS_TOKENS: tuple[int, ...] = (1,)


@dataclass(frozen=True)
class Dsv4SlotStateView:
    """Read-only server view of one DSV4 slot (protocol shape)."""

    kv_len: int
    committed_tokens: tuple[int, ...]

    @property
    def is_fresh(self) -> bool:
        return self.kv_len == 0


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
        self.model = model
        self.config = config
        self.num_slots = num_slots
        self.max_seq_len = max_seq_len
        self.max_q_rows = max_q_rows
        self.device = device
        self._forward_fn = forward_fn

        # One kernel-path attention stack per slot, weights shared from the
        # eager model (which stays the weight owner and oracle).  Each slot's
        # stack owns its page buffers, compressor state and indexer caches.
        # Built lazily: the real shapes (512 head dim, 256-page window) are
        # DSV4-specific, so tests with a tiny config pass ``forward_fn`` and
        # never build the stacks.
        self.slot_layers: list[list[Dsv4AttnKernelLayer]] = []
        if forward_fn is None:
            for _ in range(num_slots):
                self.slot_layers.append(
                    [
                        Dsv4AttnKernelLayer(
                            config,
                            layer_id,
                            max_seq_len=max_seq_len,
                            max_q_rows=max_q_rows,
                            device=device,
                            shared_from=model.blocks[layer_id].attn,
                        )
                        for layer_id in range(config.num_layers)
                    ]
                )
            # Route bf16 Q8_0 projections through the fused tensor-core
            # dequant-GEMM on the serving path (more accurate than cuBLAS
            # bf16, zero dequant cache).  The eager graph stays untouched as
            # the official-reference oracle.
            from runtime.model.dsv4_model import PackedQ8_0Weight

            for mod in model.modules():
                if isinstance(mod, PackedQ8_0Weight) and getattr(
                    mod, "weight_dtype", torch.bfloat16
                ) is torch.bfloat16:
                    mod.fused_q8 = True
        self._kv_len = [0] * num_slots
        self._committed: list[list[int]] = [[] for _ in range(num_slots)]
        self._cg_status: dict[str, str] = {}

    # -- protocol ------------------------------------------------------------

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            speculative_decode=False,
            prefix_cache=False,
            cuda_graph=False,
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
        if self.slot_layers:
            for layer in self.slot_layers[slot]:
                layer.reset_caches()
        self._kv_len[slot] = 0
        self._committed[slot] = []
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
                PrefixSnapshot(slot=s, cached_kv_len=0, cached_tokens=0, head=())
                for s in range(self.num_slots)
            ),
            dflash_cg_status=tuple(sorted(self._cg_status.items())),
        )

    # -- prefix-cache surface (no-op: Phase 4 serves without prefix cache) --

    def reconcile_prefix_hit(self, token_ids: list[int]) -> PrefixHit:
        """No warm prefix: ``PrefixHit(0, 0)``, the no-cache contract."""
        return PrefixHit(kv_hit=0, state_hit=0)

    def find_best_slot_for_prompt(
        self,
        token_ids: list[int],
        free_slots: list[int],
    ) -> tuple[int, int]:
        """No prefix cache: pick the first free slot, zero hit depth."""
        if not free_slots:
            raise IndexError("no free slots")
        return free_slots[0], 0

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
                    sample_from_logits(
                        logits[0, -1].unsqueeze(0), params, generator=gen
                    ).item()
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
        if self._kv_len[slot] != 0:
            raise RuntimeError(
                f"slot {slot} is at kv_len={self._kv_len[slot]}; the caller must reset_slot first"
            )
        chunk = max(1, min(self.max_q_rows, 128))
        logits = None
        for start in range(0, len(prompt_ids), chunk):
            ids = prompt_ids[start : start + chunk]
            logits = self._forward(
                slot,
                torch.tensor([ids], dtype=torch.long, device=self.device),
                start,
            )
        assert logits is not None
        self._kv_len[slot] = len(prompt_ids)
        self._committed[slot] = list(prompt_ids)
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
        model, layers = self.model, self.slot_layers[slot]
        h = model.embed(input_ids)
        h = h.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
        for i, block in enumerate(model.blocks):
            residual = h
            x, post, comb = block.hc_pre(
                h, block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base
            )
            x = rms_norm(x, block.attn_norm_weight, block.eps)
            x = layers[i](x, start_pos)
            x = block.hc_post(x, residual, post, comb)
            residual = x
            x, post, comb = block.hc_pre(
                x, block.hc_ffn_fn, block.hc_ffn_scale, block.hc_ffn_base
            )
            x = rms_norm(x, block.ffn_norm_weight, block.eps)
            x = block.moe(x, input_ids)
            x = block.hc_post(x, residual, post, comb)
            h = x
        h = model.hc_head(h)
        return model.lm_head(rms_norm(h, model.norm_weight, model.eps))

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
        """One decode step per slot through the kernel-path stack."""
        if return_logprobs:
            raise NotImplementedError(
                "logprobs are Phase 4 follow-up work; decode returns token ids only"
            )
        if len(slot_ids) != len(token_ids) != len(kv_lengths) != len(params_list):
            raise ValueError("slot_ids/token_ids/kv_lengths/params_list must be equal length")
        outs: list[int] = []
        for slot, token, kv_len, params in zip(slot_ids, token_ids, kv_lengths, params_list):
            if not 0 <= slot < self.num_slots:
                raise IndexError(f"slot {slot} out of range ({self.num_slots} slots)")
            if kv_len != self._kv_len[slot]:
                raise RuntimeError(
                    f"slot {slot} is at kv_len={self._kv_len[slot]}, caller says {kv_len}; "
                    "the caller must reset_slot first"
                )
            logits = self._forward(
                slot, torch.tensor([[token]], dtype=torch.long, device=self.device), kv_len
            )
            if params.temperature == 0.0:
                out = int(logits[0, 0].argmax(dim=-1).item())
            else:
                gen = make_generator(params.seed)
                out = int(
                    sample_from_logits(
                        logits[0, 0].unsqueeze(0), params, generator=gen
                    ).item()
                )
            self._kv_len[slot] += 1
            self._committed[slot].append(out)
            outs.append(out)
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
