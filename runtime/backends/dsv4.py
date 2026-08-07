"""DeepSeek-V4-Flash backend: eager, single-slot (Phase 3).

Phase 3's contract is correctness, not concurrency: eager forward, one
slot, no CUDA Graph, no speculative decode, no prefix cache. The eager
``Dsv4Transformer`` owns exactly one set of per-layer caches (attention
KV, compressor decode state, indexer scoring caches), so ``num_slots``
must be 1 here; the multi-slot slot-pool integration is the next phase's
work (``runtime/model/dsv4_slots.py`` + ``Dsv4AttnKernelLayer`` exist for
it, and ``reset_slot`` already implements the pool's semantics: KV bytes
survive, recursive compressor state is zeroed).

Conforms to ``ModelBackend`` with every capability flag False, i.e. the
six unconditionally required members only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from runtime.backends.protocol import (
    BackendCapabilities,
    BackendSnapshot,
    PrefixSnapshot,
    SlotSnapshot,
)
from runtime.model.dsv4_config import Dsv4Config
from runtime.model.dsv4_model import Dsv4Transformer
from runtime.sampling import SamplingParams, make_generator, sample_from_logits

EOS_TOKENS: tuple[int, ...] = (1,)


@dataclass(frozen=True)
class Dsv4SlotStateView:
    """Read-only server view of the single DSV4 slot (protocol shape)."""

    kv_len: int
    committed_tokens: tuple[int, ...]

    @property
    def is_fresh(self) -> bool:
        return self.kv_len == 0


class DeepseekV4Backend:
    """Eager single-slot serving of the DSV4-Flash graph."""

    def __init__(
        self,
        model: Dsv4Transformer,
        config: Dsv4Config,
        *,
        num_slots: int = 1,
        max_seq_len: int = 4096,
        device: str = "cuda",
    ) -> None:
        if num_slots != 1:
            raise NotImplementedError(
                "Phase 3 serves one slot: the eager graph owns one cache set; "
                "multi-slot wiring lands with the slot-pool integration"
            )
        self.model = model
        self.config = config
        self.num_slots = num_slots
        self.max_seq_len = max_seq_len
        self.device = device
        self._kv_len = 0
        self._committed: list[int] = []
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
        if slot != 0:
            raise IndexError(f"slot {slot} out of range (Phase 3 has 1 slot)")
        self.model.reset_caches()
        self._kv_len = 0
        self._committed = []

    def slot_state(self, slot: int) -> Dsv4SlotStateView:
        if slot != 0:
            raise IndexError(f"slot {slot} out of range (Phase 3 has 1 slot)")
        return Dsv4SlotStateView(
            kv_len=self._kv_len,
            committed_tokens=tuple(self._committed),
        )

    def snapshot(self) -> BackendSnapshot:
        return BackendSnapshot(
            slots=tuple(
                SlotSnapshot(slot=s, kv_len=self._kv_len, is_fresh=self._kv_len == 0)
                for s in range(self.num_slots)
            ),
            prefix=tuple(
                PrefixSnapshot(slot=s, cached_kv_len=0, cached_tokens=0, head=())
                for s in range(self.num_slots)
            ),
            dflash_cg_status=tuple(sorted(self._cg_status.items())),
        )

    def prefill(self, slot: int, prompt_ids: list[int]) -> int:
        """Cold prefill of ``prompt_ids``; greedy first token (Laguna contract)."""
        if slot != 0:
            raise IndexError(f"slot {slot} out of range (Phase 3 has 1 slot)")
        if not prompt_ids:
            raise ValueError("prefill needs at least one token")
        if self._kv_len != 0:
            raise RuntimeError(
                f"slot {slot} is at kv_len={self._kv_len}; the caller must reset_slot first"
            )
        logits = self.model.forward(
            torch.tensor([prompt_ids], dtype=torch.long, device=self.device), 0
        )
        first_token = int(logits[0, -1].argmax(dim=-1).item())
        self._kv_len = len(prompt_ids)
        self._committed = list(prompt_ids)
        return first_token

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
        """One decode step per slot (Phase 3: exactly one slot)."""
        if return_logprobs:
            raise NotImplementedError(
                "logprobs are Phase 4 work; the Phase 3 gate is greedy/sampled token identity"
            )
        if len(slot_ids) != 1:
            raise NotImplementedError(f"Phase 3 decodes one slot at a time, got {len(slot_ids)}")
        (slot,) = slot_ids
        (token,), (kv_len,) = token_ids, kv_lengths
        if slot != 0:
            raise IndexError(f"slot {slot} out of range (Phase 3 has 1 slot)")
        if kv_len != self._kv_len:
            raise RuntimeError(
                f"slot {slot} is at kv_len={self._kv_len}, caller says {kv_len}; "
                "the caller must reset_slot first"
            )
        logits = self.model.forward(
            torch.tensor([[token]], dtype=torch.long, device=self.device), kv_len
        )
        params = params_list[0]
        if params.temperature == 0.0:
            out = int(logits[0, 0].argmax(dim=-1).item())
        else:
            gen = make_generator(params.seed)
            out = int(sample_from_logits(logits[0, 0].unsqueeze(0), params, generator=gen).item())
        self._kv_len += 1
        self._committed.append(out)
        return [out]
