"""Fixed-slot three-region cache pool for DeepSeek-V4-Flash (Phase 3).

DSV4's MLA variant keeps ONE latent KV entry per token (512 dims), and the
per-layer compressors turn the sequence into a second, shorter stream that
the attention of ratio-4/128 layers gathers from. So each slot owns three
cache regions plus the compressor decode state:

- window ring:  43 layers x 128 entries of latent KV (every layer);
- csa_comp:     21 ratio-4 layers x seq/4 compressed entries;
- hca_comp:     20 ratio-128 layers x seq/128 compressed entries;
- idx_k:        21 ratio-4 layers x seq/4 indexer-K entries (128 dims);
- comp_state:   fixed fp32 compressor windows per layer (recursive state).

The regions have different shapes, so they are five separate allocations
with static per-slot slicing -- the same "separate allocators, coordinated
layer" conclusion as ``qwen36_slots.py``. Allocate once, never rebind;
``reset_slot`` zeroes the recursive compressor state (it IS read on the
first step of the next sequence, like the GDN state) but leaves the KV
regions alone (their stale bytes are never read past the slot's length,
and same-slot prefix reuse wants them kept).

Design memo: notes/2026-08-07-dsv4-phase3-slot-pool-design.md
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from runtime.model.dsv4_config import Dsv4Config


@dataclass(frozen=True)
class Dsv4SlotPoolGeometry:
    """Shapes and byte counts this pool was built with -- reported, not
    re-derived (same discipline as qwen36_slots.SlotPoolGeometry)."""

    num_slots: int
    max_seq_len: int
    layout: str
    window_entries_per_layer: int
    csa_entries: int
    hca_entries: int
    idx_entries: int
    bytes_per_slot: int
    total_bytes: int


class Dsv4SlotPool:
    """All DSV4 cache regions for ``num_slots`` fixed slots.

    ``layout="bf16"`` stores latent entries as 512 bf16 values -- the
    bit-exact reference layout used to bring kernels up. The production
    FP8-hybrid layout (nope e4m3 block-64 ue8m0 + bf16 rope, 583 B/entry)
    lands with the KV-write kernel and will be a second layout value here.
    """

    def __init__(
        self,
        config: Dsv4Config,
        num_slots: int,
        max_seq_len: int,
        *,
        layout: str = "bf16",
        device: torch.device | str | None = None,
    ) -> None:
        if layout != "bf16":
            raise ValueError(f"unsupported slot-pool layout: {layout!r}")
        if num_slots < 1:
            raise ValueError(f"num_slots must be >= 1, got {num_slots}")
        self.config = config
        self.num_slots = num_slots
        self.max_seq_len = max_seq_len
        self.layout = layout
        self.window = config.window_size
        self.head_dim = config.head_dim
        self.index_head_dim = config.index_head_dim

        self.csa_layer_ids = tuple(
            i for i in range(config.num_layers) if config.layer_ratio(i) == 4
        )
        self.hca_layer_ids = tuple(
            i for i in range(config.num_layers) if config.layer_ratio(i) == 128
        )
        self.csa_entries = max_seq_len // 4
        self.hca_entries = max_seq_len // 128
        if self.csa_layer_ids and self.csa_entries < 1:
            raise ValueError(f"max_seq_len {max_seq_len} too small for ratio-4 layers")

        s, d = num_slots, self.head_dim
        self.window_pool = torch.empty(
            s, config.num_layers, self.window, d, dtype=torch.bfloat16, device=device
        )
        self.csa_comp_pool = torch.empty(
            s, len(self.csa_layer_ids), self.csa_entries, d, dtype=torch.bfloat16, device=device
        )
        self.hca_comp_pool = torch.empty(
            s, len(self.hca_layer_ids), self.hca_entries, d, dtype=torch.bfloat16, device=device
        )
        self.idx_k_pool = torch.empty(
            s,
            len(self.csa_layer_ids),
            self.csa_entries,
            self.index_head_dim,
            dtype=torch.bfloat16,
            device=device,
        )

        # compressor decode state (recursive; reset_slot must zero it).
        # ratio-4 attn compressors: overlap, coff=2 -> [2*ratio, 2*d] = [8, 1024]
        self.csa_kv_state = torch.zeros(
            s, len(self.csa_layer_ids), 8, 2 * d, dtype=torch.float32, device=device
        )
        self.csa_score_state = torch.empty(
            s, len(self.csa_layer_ids), 8, 2 * d, dtype=torch.float32, device=device
        )
        # ratio-128 attn compressors: coff=1 -> [128, 512]
        self.hca_kv_state = torch.zeros(
            s, len(self.hca_layer_ids), 128, d, dtype=torch.float32, device=device
        )
        self.hca_score_state = torch.empty(
            s, len(self.hca_layer_ids), 128, d, dtype=torch.float32, device=device
        )
        # indexer compressors: coff=2, head_dim 128 -> [8, 256]
        self.idx_kv_state = torch.zeros(
            s,
            len(self.csa_layer_ids),
            8,
            2 * self.index_head_dim,
            dtype=torch.float32,
            device=device,
        )
        self.idx_score_state = torch.empty(
            s,
            len(self.csa_layer_ids),
            8,
            2 * self.index_head_dim,
            dtype=torch.float32,
            device=device,
        )
        for state in (
            self.csa_score_state,
            self.hca_score_state,
            self.idx_score_state,
        ):
            state.fill_(float("-inf"))

    # -- per-slot views (narrow, never new storage) ----------------------

    def slot_window(self, slot: int) -> torch.Tensor:
        return self.window_pool[slot]

    def slot_csa_comp(self, slot: int) -> torch.Tensor:
        return self.csa_comp_pool[slot]

    def slot_hca_comp(self, slot: int) -> torch.Tensor:
        return self.hca_comp_pool[slot]

    def slot_idx_k(self, slot: int) -> torch.Tensor:
        return self.idx_k_pool[slot]

    def slot_csa_state(self, slot: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.csa_kv_state[slot], self.csa_score_state[slot]

    def slot_hca_state(self, slot: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.hca_kv_state[slot], self.hca_score_state[slot]

    def slot_idx_state(self, slot: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.idx_kv_state[slot], self.idx_score_state[slot]

    # -- lifecycle --------------------------------------------------------

    def reset_slot(self, slot: int) -> None:
        """Zero the recursive compressor state of one slot.

        The KV regions are deliberately left untouched: bytes past the
        slot's sequence length are never read, and keeping them is what
        same-slot prefix reuse will exploit (same rule as qwen36_slots).
        """
        self.csa_kv_state[slot].zero_()
        self.hca_kv_state[slot].zero_()
        self.idx_kv_state[slot].zero_()
        self.csa_score_state[slot].fill_(float("-inf"))
        self.hca_score_state[slot].fill_(float("-inf"))
        self.idx_score_state[slot].fill_(float("-inf"))

    def reset_all(self) -> None:
        for slot in range(self.num_slots):
            self.reset_slot(slot)

    def geometry(self) -> Dsv4SlotPoolGeometry:
        entry = self.head_dim * 2  # bf16 layout
        per_slot = (
            self.config.num_layers * self.window * entry
            + len(self.csa_layer_ids) * self.csa_entries * entry
            + len(self.hca_layer_ids) * self.hca_entries * entry
            + len(self.csa_layer_ids) * self.csa_entries * self.index_head_dim * 2
            + len(self.csa_layer_ids) * 8 * 2 * self.head_dim * 4 * 2
            + len(self.hca_layer_ids) * 128 * self.head_dim * 4 * 2
            + len(self.csa_layer_ids) * 8 * 2 * self.index_head_dim * 4 * 2
        )
        return Dsv4SlotPoolGeometry(
            num_slots=self.num_slots,
            max_seq_len=self.max_seq_len,
            layout=self.layout,
            window_entries_per_layer=self.window,
            csa_entries=self.csa_entries,
            hca_entries=self.hca_entries,
            idx_entries=self.csa_entries,
            bytes_per_slot=per_slot,
            total_bytes=per_slot * self.num_slots,
        )
