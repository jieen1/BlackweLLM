"""``FakeBackend`` / ``FakeDFlashEngine``: pure-Python, CPU-tensor stand-ins
for ``runtime.backends.laguna.LagunaBackend`` / ``laguna_dflash.DFlashEngine``.

Mirrors the existing repo convention in ``bfdiag/daemon/provider.py``, where
``FakeEngineProvider`` lives alongside the real (never-executed)
``LagunaEngineProvider`` in production code specifically so tests can import
it. Same idea here: every test in ``tests/test_bfdiag_checkpoint*.py``
exercises the real ``bfdiag/checkpoint/{state,store,restore,verify}.py``
code paths against these fakes, never against a real GPU (hard constraint
for this task).

Design goal: these fakes must be deterministic functions of their OWN raw
tensor bytes plus bookkeeping ints -- not just "produce plausible-looking
output" -- so that a test which corrupts or omits exactly one checkpoint
item (see :mod:`bfdiag.checkpoint.state`'s ``SLOT_STATE_ITEMS``) reliably
changes the fakes' subsequent output. Concretely:

* ``prefill``/``dflash_prefill_bootstrap`` write deterministic byte patterns
  (derived from a SHA-256 of the prompt) into the full-attention, SWA-ring,
  and draft-ring block ranges for the prompt's token positions, using the
  SAME modulo/ring addressing formulas as the real code
  (``runtime/backends/laguna.py``'s ``pos % ring_slots`` pattern -- see
  :func:`bfdiag.checkpoint.state.ring_blocks_for_window`).
* ``dflash_round`` derives its output tokens from a SHA-256 digest of the
  CURRENT live bytes in all three regions (plus kv_len/anchor/draft_tokens),
  then writes new derived bytes into the newly-extended positions -- a
  causal chain exactly like a real forward pass depending on cached KV.
  Restoring the wrong bytes into ANY region changes every subsequent
  round's output, which is exactly what the "drop one item" parametrized
  test in ``tests/test_bfdiag_checkpoint_restore.py`` relies on.
"""

from __future__ import annotations

import hashlib
from typing import Any

import torch

from bfdiag.checkpoint.state import ring_blocks_for_window

DEFAULT_QO_MAX = 16


def _digest(*parts: Any) -> bytes:
    """Deterministic digest of arbitrary ints/bytes/strings/tuples."""
    return hashlib.sha256(repr(parts).encode()).digest()


def _tensor_digest(tensor: torch.Tensor, start: int, end: int) -> bytes:
    """Byte-exact digest of ``tensor[:, start:end]`` -- any single-byte
    difference anywhere in that range changes the result. Used so the fake
    forward passes are sensitive to every byte of restored KV content, not
    just to the bookkeeping ints."""
    if end <= start:
        return b""
    chunk = tensor[:, start:end].contiguous().numpy().tobytes()
    return hashlib.sha256(chunk).digest()


def _byte_at(h: bytes, name: str, pos: int) -> int:
    return hashlib.sha256(h + name.encode() + pos.to_bytes(8, "big")).digest()[0]


def _int_from(h: bytes, index: int, modulo: int) -> int:
    lo = (index * 2) % (len(h) - 1)
    return int.from_bytes(h[lo : lo + 2], "big") % modulo


class FakeBackend:
    """CPU-only stand-in for ``LagunaBackend``. Attribute names match the
    real backend exactly (see ``bfdiag/checkpoint/state.py::slot_geometry``'s
    duck-typed reads) so ``state.slot_geometry``/``store.save_checkpoint``/
    ``restore.restore_checkpoint`` run unmodified against either object.
    """

    RESERVED_PHYSICAL_SLOTS = 0

    def __init__(
        self,
        *,
        num_slots: int = 2,
        block_size: int = 16,
        blocks_per_slot: int = 8,
        swa_window: int = 40,
        num_full_layers: int = 2,
        num_swa_layers: int = 2,
        num_kv_heads: int = 2,
        head_dim: int = 4,
        qo_max: int = DEFAULT_QO_MAX,
    ) -> None:
        self.num_slots = num_slots
        self.block_size = block_size
        self.blocks_per_slot = blocks_per_slot
        self._swa_window = swa_window
        self._full_layer_names = [f"full.layer{i}" for i in range(num_full_layers)]
        self._swa_layer_names = [f"swa.layer{i}" for i in range(num_swa_layers)]
        self._ring_blocks_per_slot = ring_blocks_for_window(swa_window, block_size, qo_max)
        self._ring_slots_per_slot = self._ring_blocks_per_slot * block_size
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self.device = "cpu"

        num_phys = num_slots + self.RESERVED_PHYSICAL_SLOTS
        full_num_blocks = num_phys * blocks_per_slot
        ring_num_blocks = num_phys * self._ring_blocks_per_slot
        self.kv_caches: dict[str, torch.Tensor] = {}
        shape_kwargs = dict(dtype=torch.uint8, device=self.device)
        for name in self._full_layer_names:
            self.kv_caches[name] = torch.zeros(
                2, full_num_blocks, block_size, num_kv_heads, head_dim, **shape_kwargs
            )
        for name in self._swa_layer_names:
            self.kv_caches[name] = torch.zeros(
                2, ring_num_blocks, block_size, num_kv_heads, head_dim, **shape_kwargs
            )

        self.slot_kv_len: list[int] = [0] * num_slots
        self.slot_committed_tokens: list[list[int]] = [[] for _ in range(num_slots)]

    # ── addressing, mirrors runtime/backends/laguna.py exactly ──────────

    def _physical_slot(self, slot: int) -> int:
        return slot + self.RESERVED_PHYSICAL_SLOTS

    def _full_address(self, slot: int, pos: int) -> tuple[int, int]:
        phys = self._physical_slot(slot)
        base = phys * self.blocks_per_slot
        bs = self.block_size
        return base + pos // bs, pos % bs

    def _ring_address(self, slot: int, pos: int) -> tuple[int, int]:
        phys = self._physical_slot(slot)
        base = phys * self._ring_blocks_per_slot
        bs = self.block_size
        ring_slot_idx = pos % self._ring_slots_per_slot
        return base + ring_slot_idx // bs, ring_slot_idx % bs

    def _write_positions(self, slot: int, start_pos: int, count: int, h: bytes) -> None:
        for i in range(count):
            pos = start_pos + i
            for name in self._full_layer_names:
                block, off = self._full_address(slot, pos)
                self.kv_caches[name][:, block, off, :, :] = _byte_at(h, name, pos)
            for name in self._swa_layer_names:
                block, off = self._ring_address(slot, pos)
                self.kv_caches[name][:, block, off, :, :] = _byte_at(h, name, pos)

    def reset_slot(self, slot: int) -> None:
        """Mirrors ``LagunaBackend.reset_slot`` (laguna.py:1639-1653)
        exactly: zero this slot's full-attention blocks and SWA ring
        blocks, clear kv_len/committed_tokens."""
        self.slot_kv_len[slot] = 0
        self.slot_committed_tokens[slot] = []
        phys = self._physical_slot(slot)
        full_start = phys * self.blocks_per_slot
        full_end = full_start + self.blocks_per_slot
        for name in self._full_layer_names:
            self.kv_caches[name][:, full_start:full_end].zero_()
        if self._ring_blocks_per_slot > 0:
            ring_start = phys * self._ring_blocks_per_slot
            ring_end = ring_start + self._ring_blocks_per_slot
            for name in self._swa_layer_names:
                self.kv_caches[name][:, ring_start:ring_end].zero_()

    def prefill(self, slot: int, prompt_ids: list[int]) -> int:
        """Mirrors ``LagunaBackend.prefill`` (laguna.py:1225-1240): writes
        deterministic KV bytes for the prompt, then sets kv_len/committed
        with the real +1 invariant (committed = prompt + [first_token])."""
        if self.slot_kv_len[slot] != 0:
            raise RuntimeError(f"slot {slot} is not fresh (kv_len={self.slot_kv_len[slot]})")
        prompt_len = len(prompt_ids)
        h = _digest("prefill", tuple(prompt_ids))
        self._write_positions(slot, 0, prompt_len, h)
        first_token = _int_from(h, 0, 32768)
        self.slot_kv_len[slot] = prompt_len
        self.slot_committed_tokens[slot] = list(prompt_ids) + [first_token]
        return first_token


class FakeDFlashEngine:
    """CPU-only stand-in for ``DFlashEngine``. Wraps a :class:`FakeBackend`
    exactly like the real ``DFlashEngine`` wraps a ``LagunaBackend`` (via
    ``self.backend``)."""

    def __init__(
        self,
        backend: FakeBackend,
        *,
        draft_window: int = 40,
        num_draft_layers: int = 2,
        num_kv_heads: int = 2,
        head_dim: int = 4,
        qo_max: int = DEFAULT_QO_MAX,
        num_query_per_req: int = 16,
    ) -> None:
        self.backend = backend
        self.block_size = backend.block_size
        self.num_slots = backend.num_slots
        self._num_query_per_req = num_query_per_req
        self._draft_layer_names = [f"draft.layer{i}" for i in range(num_draft_layers)]
        self._draft_blocks_per_slot = ring_blocks_for_window(
            draft_window, backend.block_size, qo_max
        )
        self._draft_ring_slots = self._draft_blocks_per_slot * backend.block_size

        num_phys = backend.num_slots + backend.RESERVED_PHYSICAL_SLOTS
        total_blocks = num_phys * self._draft_blocks_per_slot
        self._draft_kv_caches: dict[str, torch.Tensor] = {
            name: torch.zeros(
                2, total_blocks, backend.block_size, num_kv_heads, head_dim, dtype=torch.uint8
            )
            for name in self._draft_layer_names
        }

    # ── addressing, mirrors laguna_dflash.py's draft ring exactly ────────

    def _draft_address(self, slot: int, pos: int) -> tuple[int, int]:
        phys = self.backend._physical_slot(slot)
        base = phys * self._draft_blocks_per_slot
        bs = self.block_size
        ring_slot_idx = pos % self._draft_ring_slots
        return base + ring_slot_idx // bs, ring_slot_idx % bs

    def _write_draft_positions(self, slot: int, start_pos: int, count: int, h: bytes) -> None:
        for i in range(count):
            pos = start_pos + i
            for name in self._draft_layer_names:
                block, off = self._draft_address(slot, pos)
                self._draft_kv_caches[name][:, block, off, :, :] = _byte_at(h, name, pos)

    def _draft_digest(self, slot: int) -> bytes:
        phys = self.backend._physical_slot(slot)
        start = phys * self._draft_blocks_per_slot
        end = start + self._draft_blocks_per_slot
        parts = [_tensor_digest(t, start, end) for t in self._draft_kv_caches.values()]
        return _digest("draft_digest", tuple(parts))

    def _draft_forward(self, slot: int, anchor: int, kv_len: int) -> list[int]:
        """Mirrors ``DFlashEngine._draft_forward``
        (laguna_dflash.py:583-640): deterministic function of (anchor,
        kv_len, current draft-ring bytes), and -- like the real forward
        pass, which writes new KV for the mask-token positions via
        ``slot_mapping`` -- also writes new draft-ring bytes for the
        newly-scanned positions [kv_len, kv_len + num_query_per_req)."""
        before = self._draft_digest(slot)
        h = _digest("draft_forward", anchor, kv_len, before)
        num_draft = self._num_query_per_req - 1
        draft_tokens = [_int_from(h, i, 32768) for i in range(num_draft)]
        self._write_draft_positions(slot, kv_len, self._num_query_per_req, h)
        return draft_tokens

    def dflash_prefill_bootstrap(self, slot: int, prompt_ids: list[int]) -> dict:
        """Mirrors ``DFlashEngine.dflash_prefill_bootstrap``
        (laguna_dflash.py:1320-1352)."""
        backend = self.backend
        first_token = backend.prefill(slot, prompt_ids)
        prompt_len = len(prompt_ids)
        h = _digest("draft_bootstrap_precompute", tuple(prompt_ids))
        self._write_draft_positions(slot, 0, prompt_len, h)
        anchor = first_token
        kv_len = backend.slot_kv_len[slot]
        draft_tokens = self._draft_forward(slot, anchor, kv_len)
        return {"anchor": anchor, "draft_tokens": draft_tokens}

    def dflash_round(self, slot: int, anchor: int, draft_tokens: list[int]) -> dict:
        """Mirrors ``DFlashEngine.dflash_round``
        (laguna_dflash.py:1354-1462): output + newly-written KV bytes are a
        deterministic function of (anchor, draft_tokens, kv_len, and the
        CURRENT live bytes across all three KV regions for this slot)."""
        backend = self.backend
        kv_len = backend.slot_kv_len[slot]
        phys = backend._physical_slot(slot)

        full_start = phys * backend.blocks_per_slot
        full_used_end = full_start + (-(-kv_len // backend.block_size) if kv_len > 0 else 0)
        ring_start = phys * backend._ring_blocks_per_slot
        ring_end = ring_start + backend._ring_blocks_per_slot
        full_digests = tuple(
            _tensor_digest(backend.kv_caches[n], full_start, full_used_end)
            for n in backend._full_layer_names
        )
        swa_digests = tuple(
            _tensor_digest(backend.kv_caches[n], ring_start, ring_end)
            for n in backend._swa_layer_names
        )
        draft_digest = self._draft_digest(slot)

        h = _digest(
            "dflash_round",
            anchor,
            tuple(draft_tokens),
            kv_len,
            full_digests,
            swa_digests,
            draft_digest,
        )
        context_count = 1 + (h[0] % (len(draft_tokens) + 1))
        committed = [_int_from(h, i, 32768) for i in range(context_count)]

        backend._write_positions(slot, kv_len, context_count, h)
        backend.slot_kv_len[slot] += context_count
        backend.slot_committed_tokens[slot].extend(committed)

        new_bonus = committed[-1]
        new_kv_len = backend.slot_kv_len[slot]
        next_draft_tokens = self._draft_forward(slot, new_bonus, new_kv_len)

        return {
            "committed": committed,
            "next_anchor": new_bonus,
            "next_draft_tokens": next_draft_tokens,
            "context_count": context_count,
        }


def reset_all(engine: FakeDFlashEngine) -> None:
    """Test convenience mirroring
    ``bfdiag.daemon.session.reset_laguna_engine`` against the fakes: reset
    every slot's backend state, zero every draft KV tensor entirely."""
    backend = engine.backend
    for slot in range(backend.num_slots):
        backend.reset_slot(slot)
    for tensor in engine._draft_kv_caches.values():
        tensor.zero_()
