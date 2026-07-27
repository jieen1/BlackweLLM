"""Declarative manifest of everything one Laguna+DFlash logical slot's
persistent, checkpoint-worthy state consists of.

This is the single most important file in this package: get this list
wrong (miss an item, or misjudge its addressing) and a restored slot looks
fine but produces silently-wrong output -- worse than a crash, per the task
brief this package was built against.

Starting point: ``bfdiag/daemon/session.py::RESET_CHECKLIST`` -- another
agent's from-the-source checklist of "what must be cleared between two
experiments sharing a warm daemon". The overlap is deliberate: **checkpoint
must save exactly what reset would otherwise clear**, plus a handful of
things reset does not have to care about but restore does (see the "ring
write-phase" and "next-round anchor/draft-tokens" entries below -- both are
things a *clean reset* can ignore because a fresh prefill overwrites them
before anything reads them, but that a *restore* must get exactly right,
because after restore the very next read is live production code, not a
fresh prefill).

Every ``StateItem`` below was verified by reading the current on-disk
``runtime/backends/laguna.py``/``laguna_dflash.py`` source (via
``codegraph_explore`` + line-cited ``Read``, not guessed) -- see each
item's ``code_ref``. Nothing here imports ``runtime.*`` or ``torch`` at
module scope (this module has zero import-time GPU/vLLM risk); the actual
tensor I/O in ``store.py``/``restore.py`` is duck-typed against whatever
``backend``/``engine`` objects the caller passes in (real or
:mod:`bfdiag.checkpoint.testing`'s ``FakeBackend``/``FakeDFlashEngine``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ─────────────────────────────────────────────────────────────────────────
# 1. The declarative checklist itself
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StateItem:
    """One entry in the "what does a checkpointed slot consist of" list.

    ``category`` is one of:

    * ``"device_tensor"``   -- lives in a real (GPU, in production) tensor,
      one entry per attention layer name; saved/restored verbatim as raw
      bytes (safetensors).
    * ``"host_scalar"``     -- a plain Python int, one per slot.
    * ``"host_list"``       -- a plain Python list, one per slot.
    * ``"derived_no_store"``-- NOT separately persisted: provably a pure
      function of other items already in this list, given matching static
      geometry (see ``note`` for the derivation and its code citation).
      Listed here anyway because the task spec explicitly worried about it
      (ring write-pointer/phase, next-round anchor/draft-tokens) and "it
      turns out to need no storage, here is why" is itself a checkable
      claim, not an omission.
    * ``"not_applicable"``  -- named in the task spec by analogy with
      ``DirectModelRunner``'s state, but does not exist on the Laguna
      backend at all (verified against current code, not assumed from the
      spec's wording).
    """

    name: str
    category: str
    per_layer: bool
    source: str
    code_ref: str
    note: str


SLOT_STATE_ITEMS: tuple[StateItem, ...] = (
    StateItem(
        name="slot_kv_len",
        category="host_scalar",
        per_layer=False,
        source="backend.slot_kv_len[slot]",
        code_ref="runtime/backends/laguna.py:346 (declaration), :1639-1641 (reset_slot)",
        note=(
            "The single absolute-position source of truth. Every KV ring "
            "address anywhere in the codebase (full-attention block index, "
            "SWA ring slot, DFlash draft ring slot) is computed as a pure "
            "function of an absolute token position that is either this "
            "value or derived from it at the call site -- see the "
            "'ring write-phase' derived item below. Getting this exactly "
            "right is necessary AND (given matching geometry) sufficient "
            "to reconstruct every ring's addressing after restore."
        ),
    ),
    StateItem(
        name="slot_committed_tokens",
        category="host_list",
        per_layer=False,
        source="backend.slot_committed_tokens[slot]",
        code_ref=(
            "runtime/backends/laguna.py:347 (declaration), :1239 (prefill), "
            "laguna_dflash.py:1431-1433 (dflash_round append)"
        ),
        note=(
            "Full token history for this slot, length == kv_len + 1 "
            "(NOT kv_len) at every observable point after prefill -- "
            "verified by bfdiag.invariants.checks.check_committed_ahead_"
            "of_kv_by_one, which fires on EVERY dflash_round in production. "
            "The +1 is because the last entry is the just-sampled next "
            "token whose own KV row does not exist yet (see that check's "
            "docstring for the exact derivation). Restoring this list "
            "verbatim is also what makes DFlashEngine.generate_verify_"
            "only's find_prefix_match reuse (laguna.py:1800-1818) 'just "
            "work' post-restore for free -- see that item below."
        ),
    ),
    StateItem(
        name="full-attention KV cache blocks",
        category="device_tensor",
        per_layer=True,
        source=(
            "backend.kv_caches[name][:, full_start:full_end] for name in "
            "backend._full_layer_names, where full_start = physical_slot * "
            "blocks_per_slot and full_end = full_start + ceil(kv_len / "
            "block_size) -- NOTE the leading `:,`: this package deliberately "
            "does NOT replicate reset_slot's own missing-leading-colon bug, "
            "see the dedicated bug_found_not_fixed item below"
        ),
        code_ref=(
            "runtime/backends/laguna.py:1643-1647 (reset_slot's own "
            "addressing, which zeros the FULL blocks_per_slot allocation); "
            ":290-306 (allocation, shape (2, n_blocks, block_size, "
            "num_kv_heads, head_dim))"
        ),
        note=(
            "IMPORTANT SIZE OPTIMIZATION vs. reset_slot: reset_slot clears "
            "the entire static blocks_per_slot allocation (it must, since "
            "any future prefill up to that capacity could land there); a "
            "checkpoint only needs ceil(kv_len / block_size) blocks -- the "
            "ones that actually hold live data. Saving the full "
            "blocks_per_slot allocation regardless of actual context "
            "length would be up to ~8x oversized for a typical 64K-in-a-"
            "512K-capacity slot. This is exactly why the volume table in "
            "notes/2026-07-27-bfdiag-checkpoint-restore.md scales with "
            "context length, not with the configured capacity."
        ),
    ),
    StateItem(
        name="SWA ring KV cache blocks",
        category="device_tensor",
        per_layer=True,
        source=(
            "backend.kv_caches[name][:, ring_start:ring_end] for name in "
            "backend._swa_layer_names, where ring_start = physical_slot * "
            "_ring_blocks_per_slot and ring_end = ring_start + "
            "_ring_blocks_per_slot (the FULL ring capacity, always -- see "
            "note; leading `:,` again deliberate, see the bug_found_not_"
            "fixed item below)"
        ),
        code_ref=(
            "runtime/backends/laguna.py:1648-1653 (reset_slot's addressing, "
            "which this mirrors exactly, unlike the full-attention item "
            "above); :278-281 (_ring_blocks_per_slot / _ring_slots_per_slot "
            "derivation from _swa_window + block_size + SWA_QO_MAX)"
        ),
        note=(
            "Unlike full-attention KV, always save the WHOLE ring capacity "
            "regardless of kv_len: the ring is fixed-size and addressed "
            "mod capacity (see the ring write-phase item below), so any "
            "physical position in it can be 'live' (still inside the "
            "attention window) no matter how large kv_len has grown. This "
            "is a small, bounded cost (36 SWA layers x a few hundred KiB "
            "per slot, see the notes file's volume table) -- not worth "
            "computing a tighter live-subset."
        ),
    ),
    StateItem(
        name="DFlash draft KV cache ring blocks",
        category="device_tensor",
        per_layer=True,
        source=(
            "engine._draft_kv_caches[name][:, draft_start:draft_end] for "
            "name in engine._draft_layer_names, where draft_start = "
            "physical_slot * _draft_blocks_per_slot and draft_end = "
            "draft_start + _draft_blocks_per_slot (full ring capacity, same "
            "reasoning as the SWA ring above)"
        ),
        code_ref=(
            "runtime/backends/laguna_dflash.py:292-305 "
            "(_alloc_draft_kv_cache: allocation + _draft_blocks_per_slot = "
            "_ring_blocks_for_window(DRAFT_WINDOW, block_size, "
            "NUM_QUERY_PER_REQ)); :1419-1429 (dflash_round's ring-addressed "
            "write, same modulo-arithmetic pattern as the main SWA ring); "
            ":651-652 (_precompute_context_kv's identical addressing)"
        ),
        note=(
            "NOT part of bfdiag/daemon/session.py's RESET_CHECKLIST "
            "zeroing being 'load-bearing' the way the main KV caches are -- "
            "that note explicitly says omitting the zero-fill 'has not "
            "been shown to change results' for RESET (a subsequent "
            "prefill overwrites before anything reads). For CHECKPOINT "
            "RESTORE this reasoning does NOT carry over: after restore, "
            "the very next call is dflash_round on LIVE, already-committed "
            "history -- any ring position within the current draft window "
            "genuinely needs to hold the SAME bytes it held at save time, "
            "or the draft model's next forward pass (which reads this "
            "ring) silently produces different draft_tokens than "
            "production would have. This is the item most likely to be "
            "silently dropped by someone porting reset-style thinking to "
            "checkpoint/restore -- flagging it prominently here and in the "
            "'drop one item' parametrized test."
        ),
    ),
    StateItem(
        name="SWA / draft ring write-phase (write pointer, wraparound state)",
        category="derived_no_store",
        per_layer=False,
        source="n/a -- no such field exists anywhere in the codebase",
        code_ref=(
            "runtime/backends/laguna.py:1108,1143 (_copy_scratch_to_ring / "
            "_copy_ring_to_scratch: `ring_slot_idx = abs_pos % ring_slots`), "
            ":624,632,822,832,977,1023 (every SWA ring decode/prefill "
            "addressing site, all `pos % ring_slots`); "
            "laguna_dflash.py:1424 (`ring_blocks = (context_positions % "
            "ring_slots) // bs`), :655 (`ring_block = (position % "
            "ring_slots) // bs`)"
        ),
        note=(
            "This was the task brief's single biggest named risk ('the "
            "ring's write pointer/phase must be saved together, or "
            "wraparound will not line up after restore') -- and, after "
            "reading every ring-addressing call site in both files, the "
            "finding is: THERE IS NO SEPARATE CURSOR TO SAVE. Every single "
            "call site computes its ring slot as `absolute_position % "
            "ring_slots_per_slot`, where `absolute_position` is always "
            "either `slot_kv_len` itself or a value derived from it in "
            "that same call (e.g. `torch.arange(kv_len, kv_len + "
            "context_count)`). There is no independent 'current write "
            "head' variable anywhere -- the phase is *stateless*, "
            "recomputed fresh from slot_kv_len on every access. "
            "Consequently: restoring slot_kv_len exactly (already required, "
            "see that item) plus the ring's raw block bytes (already "
            "required, see the two ring items above), under a MATCHING "
            "static geometry (block_size, _ring_blocks_per_slot / "
            "_draft_blocks_per_slot -- see the fingerprint's hard-reject "
            "keys), is provably sufficient to put every ring back into "
            "the exact wraparound phase it was in at save time. No new "
            "field, no new bug surface -- but this claim is exactly what "
            "the wraparound-heavy parametrized restore test exists to "
            "keep honest, not merely assert in prose."
        ),
    ),
    StateItem(
        name="next-round (anchor, draft_tokens) pair",
        category="derived_no_store",
        per_layer=False,
        source="n/a -- lives only as a caller-local loop variable, never on backend/engine",
        code_ref=(
            "runtime/backends/laguna_dflash.py:1342-1347 "
            "(dflash_prefill_bootstrap: `bonus_token = first_token; "
            "draft_tokens = self._draft_cg.replay(...) or "
            "self._draft_forward(...)`); :1396-1398,1431-1433 "
            "(dflash_round: `new_bonus = decision['next_anchor']` == "
            "`decision['committed'][-1]`, and that same committed list is "
            "what gets appended to slot_committed_tokens)"
        ),
        note=(
            "Neither DFlashEngine nor LagunaBackend stores 'the anchor/"
            "draft_tokens for the next round' as an attribute anywhere -- "
            "production code threads it through the CALLER's own loop "
            "(ServerEngine._step_sync / generate_verify_only's while loop). "
            "Two facts make this safely re-derivable rather than a hidden "
            "'must be threaded in externally' requirement: (1) anchor is "
            "always exactly slot_committed_tokens[slot][-1] -- the last "
            "committed token IS the next round's bonus token, by "
            "construction, every single round including right after "
            "prefill; (2) draft_tokens is always a pure, deterministic "
            "function of (draft KV cache content + anchor + kv_len), "
            "computed via _draft_forward/_draft_cg.replay -- exactly the "
            "same call dflash_prefill_bootstrap itself makes. So "
            "restore_checkpoint recomputes both fresh, by calling the SAME "
            "production code path, rather than persisting them: if the "
            "restored draft KV ring (the item above) is byte-correct, this "
            "recomputation reproduces the exact save-time value; if it "
            "isn't, this is precisely the divergence verify.py's baseline "
            "replay is built to catch."
        ),
    ),
    StateItem(
        name="Laguna's lightweight per-slot prefix reuse (find_prefix_match)",
        category="derived_no_store",
        per_layer=False,
        source="n/a -- pure function of slot_committed_tokens + slot_kv_len, both already listed",
        code_ref="runtime/backends/laguna.py:1800-1818",
        note=(
            "find_prefix_match walks backend.slot_committed_tokens[slot] "
            "against a new prompt and returns how many tokens already "
            "match, block-aligned. It reads no other state. Once "
            "slot_committed_tokens/slot_kv_len are restored, this "
            "mechanism 'just works' against the restored slot for free -- "
            "no separate handling needed. (This is a DIFFERENT mechanism "
            "from this whole checkpoint package: find_prefix_match only "
            "helps within one already-warm process/slot; it cannot survive "
            "a daemon restart, which is exactly the gap this package "
            "exists to close -- see the notes file's 'why not just use "
            "find_prefix_match' aside.)"
        ),
    ),
    StateItem(
        name="CUDA Graph capture-time warmup residue",
        category="not_applicable",
        per_layer=False,
        source="n/a for checkpoint content -- but load-bearing for restore's write order",
        code_ref=(
            "bfdiag/daemon/session.py RESET_CHECKLIST's own entry, citing "
            "laguna_cuda_graph.py:294,702 and laguna_dflash_cudagraph.py:"
            "301,544 (all four CG classes warm up using slot 0 / tail-slot "
            "dummy tokens written directly into real logical slots)"
        ),
        note=(
            "Not separate checkpoint CONTENT (it's not something to save), "
            "but it dictates restore's write order: restore_checkpoint "
            "must call backend.reset_slot(slot) (plus zero the draft ring "
            "range for that slot) BEFORE writing the checkpointed tensors "
            "in, exactly the same discipline session.reset_laguna_engine "
            "uses for a cold load() -- otherwise warmup residue (or a "
            "PREVIOUS experiment's leftovers, in a hot daemon) could "
            "corrupt the tail of a ring whose live window doesn't happen "
            "to cover every position this checkpoint restores. Listed here "
            "as 'not_applicable' to the SAVE side but load-bearing on the "
            "RESTORE side."
        ),
    ),
    StateItem(
        name="BUG FOUND (not fixed): reset_slot's block-range slice hits the wrong tensor axis",
        category="bug_found_not_fixed",
        per_layer=False,
        source="n/a -- this is a finding about runtime/ code, not a checkpoint content item",
        code_ref="runtime/backends/laguna.py:1647,1653 (reset_slot)",
        note=(
            "Discovered while writing this package's own save/restore tensor "
            "slicing (a multi-slot smoke test -- num_slots=2 -- surfaced it "
            "immediately; NOT fixed, runtime/ is out of scope for this task, "
            "flagged here for whoever owns runtime/backends/laguna.py, same "
            "as bfdiag/daemon/session.py's RESET_CHECKLIST already flags an "
            "unrelated self._decode_cg.reset() bug). KV cache tensors are "
            "allocated as shape (2, num_blocks, block_size, num_kv_heads, "
            "head_dim) -- dim 0 is the K/V axis (size 2), dim 1 is the block "
            "index (laguna.py:301 comment: 'dim=0: 0=K, 1=V'; confirmed by "
            "every OTHER access site in the file, e.g. "
            "_copy_scratch_to_ring's `ring[:, db, do:do+n]`, which always "
            "slices dim 0 with a bare `:` and indexes dim 1 by block). "
            "reset_slot's own block-clearing lines are the ONE place in the "
            "file that omits the leading `:,`: `self.kv_caches[name]"
            "[full_start:full_end].zero_()` slices dim 0 (size 2), not dim 1 "
            "-- for slot 0 (full_start=0) this clips to the whole dim 0 and "
            "leaves dim 1 (blocks) UNRESTRICTED, zeroing every slot's "
            "blocks at once; for any slot > 0, full_start already exceeds "
            "dim 0's size and the slice is empty, so reset_slot(slot>0) "
            "zeros NOTHING. This is completely masked in current "
            "production use because DFlash requires capacity==1 "
            "(ServerEngine enforces num_slots==1 whenever DFlash speculative "
            "decoding is enabled, per laguna_dflash.py's docstring on "
            "mtp_verify_and_commit_batch) -- with exactly one slot, 'zero "
            "everything' and 'zero slot 0's blocks' are the same operation, "
            "so the bug has no observable effect today. It would matter the "
            "moment Laguna is ever run with num_slots > 1. This package's "
            "OWN code (state.py/store.py/restore.py/testing.py) does NOT "
            "replicate this bug -- every block-range slice here uses the "
            "correct `tensor[:, start:end]` form, verified by "
            "tests/test_bfdiag_checkpoint_state.py's dedicated regression "
            "test and by this package's own multi-slot (num_slots=2) test "
            "coverage, which would have caught exactly this class of error."
        ),
    ),
    StateItem(
        name="GDN (Gated DeltaNet) recurrent conv/ssm state",
        category="not_applicable",
        per_layer=False,
        source="n/a -- Laguna has no GDN layers",
        code_ref="runtime/backends/laguna.py:400 (gdn_layer_names=[]); runtime/gdn_state.py",
        note=(
            "Re-verified directly (not just trusted from RESET_CHECKLIST): "
            "LagunaBackend.__init__ passes gdn_layer_names=[] when building "
            "its ModelSpec (laguna.py:396-405, comment 'Laguna has no GDN/"
            "SSM recursive state'). runtime/gdn_state.py belongs entirely "
            "to the separate runtime/direct_model_runner.py (Qwen3.6) path, "
            "which bfdiag's LagunaEngineProvider never loads. Named in the "
            "task spec by analogy; does not apply here."
        ),
    ),
    StateItem(
        name="Content-addressed persistent prefix cache (BlockPool / prefix_cache.py)",
        category="not_applicable",
        per_layer=False,
        source="n/a -- LagunaBackend has no persistent prefix cache",
        code_ref="runtime/backends/laguna.py:1655-1658 (reconcile_prefix_hit stub returns 0)",
        note=(
            "Re-verified directly: LagunaBackend.reconcile_prefix_hit is an "
            "explicit stub ('E1: Laguna has no persistent content-"
            "addressed prefix cache yet (roadmap L2/L3 TODO) -- every "
            "admission is a cold miss'). runtime/block_pool.py and "
            "runtime/prefix_cache.py are exclusively used by "
            "runtime/direct_model_runner.py, never referenced from "
            "runtime/backends/laguna*.py. If/when Laguna grows a real "
            "persistent prefix cache (notes/2026-07-27-laguna-prefix-"
            "cache-scoping.md), this checklist needs a matching new item."
        ),
    ),
)


def describe_state_items() -> list[dict[str, Any]]:
    """JSON-safe dump of :data:`SLOT_STATE_ITEMS`, for ``bf checkpoint show
    --schema`` or ad-hoc inspection."""
    return [
        {
            "name": item.name,
            "category": item.category,
            "per_layer": item.per_layer,
            "source": item.source,
            "code_ref": item.code_ref,
            "note": item.note,
        }
        for item in SLOT_STATE_ITEMS
    ]


# ─────────────────────────────────────────────────────────────────────────
# 2. Concrete per-slot geometry: resolves the abstract items above into
#    actual block ranges for a specific (backend, engine, slot).
# ─────────────────────────────────────────────────────────────────────────


def ring_blocks_for_window(window: int, block_size: int, qo_max: int) -> int:
    """Local re-implementation of ``runtime/backends/laguna.py:50-51``'s
    ``_ring_blocks_for_window`` (``cdiv(window - 1 + qo_max, block_size) +
    1``). Duplicated rather than imported so this module has zero import-
    time dependency on ``runtime.*``/``torch``/vLLM (see module docstring);
    the two must be kept in sync by hand if the formula ever changes --
    there is no way around that without importing the real module.
    """
    return -(-(window - 1 + qo_max) // block_size) + 1


@dataclass(frozen=True)
class SlotGeometry:
    """Everything needed to compute block ranges for one (backend, engine,
    slot) triple, resolved once via :func:`slot_geometry` and then reused
    by :func:`full_block_range`/:func:`swa_ring_block_range`/
    :func:`draft_ring_block_range`.
    """

    slot: int
    physical_slot: int
    reserved_physical_slots: int
    num_slots: int
    block_size: int
    blocks_per_slot: int
    ring_blocks_per_slot: int
    draft_blocks_per_slot: int
    swa_window: int
    full_layer_names: tuple[str, ...]
    swa_layer_names: tuple[str, ...]
    draft_layer_names: tuple[str, ...]


def slot_geometry(backend: Any, engine: Any, slot: int) -> SlotGeometry:
    """Resolve a :class:`SlotGeometry` from a live (or fake) ``backend``/
    ``engine`` pair, duck-typed against the exact attribute names
    ``runtime/backends/laguna.py``/``laguna_dflash.py`` set on themselves
    (see each attribute's citation in :data:`SLOT_STATE_ITEMS`). No
    ``runtime.*`` import is needed: every attribute here is read off the
    already-constructed object the caller passes in.
    """
    # RESERVED_PHYSICAL_SLOTS is a MODULE-level constant in laguna.py (= 0),
    # never assigned onto backend instances -- so there is nothing to read
    # off `backend` by that exact name in production. Default to 0 (the
    # only value it has ever had), but allow an instance attribute to
    # override in case a future refactor exposes it.
    reserved = int(getattr(backend, "RESERVED_PHYSICAL_SLOTS", 0) or 0)
    return SlotGeometry(
        slot=slot,
        physical_slot=slot + reserved,
        reserved_physical_slots=reserved,
        num_slots=int(backend.num_slots),
        block_size=int(backend.block_size),
        blocks_per_slot=int(backend.blocks_per_slot),
        ring_blocks_per_slot=int(getattr(backend, "_ring_blocks_per_slot", 0)),
        draft_blocks_per_slot=int(getattr(engine, "_draft_blocks_per_slot", 0)),
        swa_window=int(getattr(backend, "_swa_window", 0)),
        full_layer_names=tuple(getattr(backend, "_full_layer_names", ())),
        swa_layer_names=tuple(getattr(backend, "_swa_layer_names", ())),
        draft_layer_names=tuple(getattr(engine, "_draft_layer_names", ())),
    )


def full_block_range(geom: SlotGeometry, kv_len: int) -> tuple[int, int]:
    """Block range to save/restore for full-attention layers: only the
    blocks that actually hold live data (``ceil(kv_len / block_size)``),
    NOT the full static ``blocks_per_slot`` allocation -- see the
    "full-attention KV cache blocks" item's note for why."""
    start = geom.physical_slot * geom.blocks_per_slot
    if kv_len <= 0:
        return start, start
    used_blocks = -(-kv_len // geom.block_size)  # ceil division
    used_blocks = min(used_blocks, geom.blocks_per_slot)
    return start, start + used_blocks


def swa_ring_block_range(geom: SlotGeometry) -> tuple[int, int]:
    """Block range for SWA-ring layers: always the FULL ring capacity,
    regardless of ``kv_len`` -- see that item's note."""
    start = geom.physical_slot * geom.ring_blocks_per_slot
    return start, start + geom.ring_blocks_per_slot


def draft_ring_block_range(geom: SlotGeometry) -> tuple[int, int]:
    """Block range for the DFlash draft KV ring: always the full ring
    capacity, same reasoning as :func:`swa_ring_block_range`."""
    start = geom.physical_slot * geom.draft_blocks_per_slot
    return start, start + geom.draft_blocks_per_slot


# ─────────────────────────────────────────────────────────────────────────
# 3. Volume estimation (static, config-only -- no live engine needed).
#    Constants below are cited findings, not assumptions: see
#    notes/2026-07-27-bfdiag-checkpoint-restore.md for the full derivation
#    and the resulting size table across context lengths / block sizes.
# ─────────────────────────────────────────────────────────────────────────

#: 12 full-attention layers, 36 SWA layers -- docstring at the top of
#: runtime/backends/laguna.py:7 ("48 layers (12 full attn 48-head + 36 SWA
#: 72-head window=512)"); num_qo_heads there refers to query heads, KV
#: heads are uniformly 8 for this TP=1 runtime (laguna.py:130 comment).
NUM_FULL_LAYERS = 12
NUM_SWA_LAYERS = 36
#: 6 draft layers, all SWA (dflash_constants.py: DRAFT_NUM_LAYERS = 6).
NUM_DRAFT_LAYERS = 6

NUM_KV_HEADS = 8
HEAD_DIM = 128
#: FP8 KV cache is stored as raw torch.uint8 (laguna.py:302-304: `kv_dtype =
#: torch.uint8 if "fp8" in cache_dtype_str else layer.kv_cache_torch_dtype`)
#: -- i.e. 1 byte/element in the (by far most common, NVFP4-quantized-model)
#: production configuration. A non-fp8 (e.g. bf16) cache_dtype would double
#: every figure below (dtype_size=2).
FP8_DTYPE_SIZE = 1
BF16_DTYPE_SIZE = 2

#: DFlash's SWA_QO_MAX / NUM_QUERY_PER_REQ = 16 (verify round = 1 bonus + 15
#: draft tokens); DRAFT_WINDOW == swa_window == 512 in production, so the
#: main SWA ring and the draft ring happen to use the identical formula.
QO_MAX = 16


def bytes_per_token_per_layer(
    num_kv_heads: int = NUM_KV_HEADS, head_dim: int = HEAD_DIM, dtype_size: int = FP8_DTYPE_SIZE
) -> int:
    """K + V, one layer, one token: ``2 * num_kv_heads * head_dim *
    dtype_size``. With the production defaults this is
    ``2 * 8 * 128 * 1 = 2048`` bytes = 2 KiB/token/layer."""
    return 2 * num_kv_heads * head_dim * dtype_size


@dataclass(frozen=True)
class CheckpointSizeEstimate:
    context_tokens: int
    block_size: int
    full_attn_bytes: int
    swa_ring_bytes: int
    draft_ring_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.full_attn_bytes + self.swa_ring_bytes + self.draft_ring_bytes


def estimate_checkpoint_bytes(
    context_tokens: int,
    block_size: int,
    *,
    swa_window: int = 512,
    draft_window: int = 512,
    dtype_size: int = FP8_DTYPE_SIZE,
) -> CheckpointSizeEstimate:
    """Static (no live engine needed) size estimate for one slot's
    checkpoint at a given context length and block size -- used to build
    the volume table in notes/2026-07-27-bfdiag-checkpoint-restore.md.
    """
    per_token_per_layer = bytes_per_token_per_layer(dtype_size=dtype_size)
    full_attn_bytes = context_tokens * NUM_FULL_LAYERS * per_token_per_layer

    ring_blocks = ring_blocks_for_window(swa_window, block_size, QO_MAX)
    ring_capacity_tokens = ring_blocks * block_size
    swa_ring_bytes = ring_capacity_tokens * NUM_SWA_LAYERS * per_token_per_layer

    draft_ring_blocks = ring_blocks_for_window(draft_window, block_size, QO_MAX)
    draft_ring_capacity_tokens = draft_ring_blocks * block_size
    draft_ring_bytes = draft_ring_capacity_tokens * NUM_DRAFT_LAYERS * per_token_per_layer

    return CheckpointSizeEstimate(
        context_tokens=context_tokens,
        block_size=block_size,
        full_attn_bytes=full_attn_bytes,
        swa_ring_bytes=swa_ring_bytes,
        draft_ring_bytes=draft_ring_bytes,
    )


def _fmt_mib(n: int) -> str:
    return f"{n / (1024 * 1024):.1f} MiB"


def _fmt_gib(n: int) -> str:
    return f"{n / (1024 * 1024 * 1024):.3f} GiB"


if __name__ == "__main__":
    print("checkpoint state item count:", len(SLOT_STATE_ITEMS))
    for item in SLOT_STATE_ITEMS:
        print(f"  [{item.category:18s}] {item.name}")

    print()
    print("volume table (bytes-per-token-per-layer = fp8/uint8, 1 byte):")
    print(f"  bytes/token/layer = {bytes_per_token_per_layer()} (= 2 KiB)")
    for bs in (64, 128):
        print(f"\n  -- block_size={bs} --")
        for ctx in (4096, 16384, 32768, 65536, 131072, 204800, 262144):
            est = estimate_checkpoint_bytes(ctx, bs)
            print(
                f"    ctx={ctx:>7d} tok  full={_fmt_gib(est.full_attn_bytes):>10s}"
                f"  swa_ring={_fmt_mib(est.swa_ring_bytes):>9s}"
                f"  draft_ring={_fmt_mib(est.draft_ring_bytes):>9s}"
                f"  total={_fmt_gib(est.total_bytes):>10s}"
            )
