"""Core acceptance tests for the checkpoint/restore feature.

1. State completeness: a "dirty" FakeBackend/FakeDFlashEngine (every
   checklist item non-default) -> save -> fresh engine -> restore ->
   every item matches byte-for-byte / value-for-value.
2. Parametrized "drop one item" test: corrupt or omit exactly one
   checkpoint item and prove restore/verify catches it.
3. Fingerprint rejection: a bs=64 checkpoint must not restore into a
   bs=128 engine, and the error must name block_size.
4. Verify safety valve: a tampered baseline must cause restore to raise
   and refuse to hand back a RestoreResult.

Everything here runs against :mod:`bfdiag.checkpoint.testing`'s pure-CPU
fakes -- no GPU, no real ``runtime.*`` import.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from bfdiag.checkpoint import restore, state, store, verify
from bfdiag.checkpoint.testing import FakeBackend, FakeDFlashEngine


def _fresh_engine(**kwargs) -> FakeDFlashEngine:
    defaults = dict(num_slots=2, block_size=16, blocks_per_slot=32, swa_window=40)
    defaults.update(kwargs)
    backend = FakeBackend(**defaults)
    return FakeDFlashEngine(backend, draft_window=40, num_draft_layers=2)


def _prime_slot(engine: FakeDFlashEngine, slot: int, prompt_len: int = 90, rounds: int = 8) -> None:
    """Runs prefill + several dflash_round calls so kv_len grows well past
    the (small, test-sized) SWA/draft ring capacity -- exercising real
    wraparound, not just a same-round-as-prefill trivial case."""
    prompt = list(range(1, prompt_len + 1))
    boot = engine.dflash_prefill_bootstrap(slot, prompt)
    anchor, draft_tokens = boot["anchor"], boot["draft_tokens"]
    for _ in range(rounds):
        dec = engine.dflash_round(slot, anchor, draft_tokens)
        anchor, draft_tokens = dec["next_anchor"], dec["next_draft_tokens"]


def _snapshot_slot_state(engine: FakeDFlashEngine, slot: int) -> dict:
    """Ground-truth snapshot of every SLOT_STATE_ITEMS "device_tensor"/
    "host_scalar"/"host_list" entry, taken directly (not via save_checkpoint,
    which would mutate the slot via its own baseline probe)."""
    backend = engine.backend
    geom = state.slot_geometry(backend, engine, slot)
    kv_len = backend.slot_kv_len[slot]
    full_start, full_end = state.full_block_range(geom, kv_len)
    swa_start, swa_end = state.swa_ring_block_range(geom)
    draft_start, draft_end = state.draft_ring_block_range(geom)
    return {
        "kv_len": kv_len,
        "committed_tokens": list(backend.slot_committed_tokens[slot]),
        "full": {
            n: backend.kv_caches[n][:, full_start:full_end].clone() for n in geom.full_layer_names
        },
        "swa": {
            n: backend.kv_caches[n][:, swa_start:swa_end].clone() for n in geom.swa_layer_names
        },
        "draft": {
            n: engine._draft_kv_caches[n][:, draft_start:draft_end].clone()
            for n in geom.draft_layer_names
        },
        "ranges": {
            "full": (full_start, full_end),
            "swa": (swa_start, swa_end),
            "draft": (draft_start, draft_end),
        },
    }


# --- 1. state completeness ------------------------------------------------


def test_full_state_integrity_after_restore(tmp_path: Path) -> None:
    engine = _fresh_engine()
    slot = 0
    _prime_slot(engine, slot)
    snapshot = _snapshot_slot_state(engine, slot)

    # Sanity: this is genuinely a "dirty" (non-default) state on every axis.
    assert snapshot["kv_len"] > 0
    assert len(snapshot["committed_tokens"]) == snapshot["kv_len"] + 1
    assert all(t.numel() > 0 and bool(t.any()) for t in snapshot["full"].values())
    assert all(t.numel() > 0 and bool(t.any()) for t in snapshot["swa"].values())
    assert all(t.numel() > 0 and bool(t.any()) for t in snapshot["draft"].values())
    # The rings must actually have wrapped at least once for this to be a
    # meaningful wraparound test (ring capacity is small on purpose).
    assert snapshot["kv_len"] > engine._draft_blocks_per_slot * engine.block_size

    store.save_checkpoint(engine, slot, "integrity", root=tmp_path, baseline_steps=2)

    fresh = _fresh_engine()
    result = restore.restore_checkpoint(
        fresh, slot, "integrity", root=tmp_path, verify_after=False, derive_next_round=False
    )
    assert result.verified is False  # a "pure" restore: no verify probe, no next-round derivation

    fb = fresh.backend
    assert fb.slot_kv_len[slot] == snapshot["kv_len"]
    assert fb.slot_committed_tokens[slot] == snapshot["committed_tokens"]
    full_start, full_end = snapshot["ranges"]["full"]
    for name, tensor in snapshot["full"].items():
        assert torch.equal(fb.kv_caches[name][:, full_start:full_end], tensor)
    swa_start, swa_end = snapshot["ranges"]["swa"]
    for name, tensor in snapshot["swa"].items():
        assert torch.equal(fb.kv_caches[name][:, swa_start:swa_end], tensor)
    draft_start, draft_end = snapshot["ranges"]["draft"]
    for name, tensor in snapshot["draft"].items():
        assert torch.equal(fresh._draft_kv_caches[name][:, draft_start:draft_end], tensor)

    # Target-slot-only restore: the other slot must remain completely untouched.
    other_slot = 1
    assert fb.slot_kv_len[other_slot] == 0
    assert fb.slot_committed_tokens[other_slot] == []


def test_restoring_into_nonzero_slot_does_not_disturb_other_live_slots(tmp_path: Path) -> None:
    engine = _fresh_engine()
    _prime_slot(engine, slot=0, prompt_len=50, rounds=3)
    store.save_checkpoint(engine, 0, "src", root=tmp_path, baseline_steps=1)

    # A SEPARATE, already-live engine with its own distinct state in slot 0.
    target = _fresh_engine()
    _prime_slot(target, slot=0, prompt_len=70, rounds=4)
    slot0_snapshot = _snapshot_slot_state(target, 0)

    result = restore.restore_checkpoint(target, 1, "src", root=tmp_path)
    assert result.verified

    # slot 0's tensors/bookkeeping must be bit-for-bit unchanged.
    after = _snapshot_slot_state(target, 0)
    assert after["kv_len"] == slot0_snapshot["kv_len"]
    assert after["committed_tokens"] == slot0_snapshot["committed_tokens"]
    for name, tensor in slot0_snapshot["full"].items():
        assert torch.equal(after["full"][name], tensor)


# --- 2. verified full round trip (via verify_after=True) ------------------


def test_restore_with_verify_after_true_succeeds_end_to_end(tmp_path: Path) -> None:
    engine = _fresh_engine()
    _prime_slot(engine, slot=0)
    store.save_checkpoint(engine, 0, "e2e", root=tmp_path, baseline_steps=3)

    fresh = _fresh_engine()
    result = restore.restore_checkpoint(fresh, 0, "e2e", root=tmp_path)
    assert result.verified
    assert isinstance(result.anchor, int)
    assert len(result.draft_tokens) > 0
    assert len(result.verified_tokens) > 0
    # The caller can keep generating immediately from the returned inputs.
    dec = fresh.dflash_round(0, result.anchor, result.draft_tokens)
    assert len(dec["committed"]) >= 1


# --- 3. fingerprint rejection ------------------------------------------------


def test_restore_rejects_block_size_mismatch(tmp_path: Path) -> None:
    """The task brief's flagship risk: bs=64 checkpoint -> bs=128 engine
    must be refused, and the error must name block_size."""
    engine_64 = _fresh_engine(block_size=16, blocks_per_slot=8, swa_window=40)
    _prime_slot(engine_64, slot=0, prompt_len=50, rounds=2)
    store.save_checkpoint(engine_64, 0, "bs64", root=tmp_path, baseline_steps=1)

    engine_32 = _fresh_engine(block_size=32, blocks_per_slot=8, swa_window=40)
    with pytest.raises(store.FingerprintMismatchError, match="block_size"):
        restore.restore_checkpoint(engine_32, 0, "bs64", root=tmp_path)


def test_restore_rejects_num_slots_mismatch(tmp_path: Path) -> None:
    engine_a = _fresh_engine(num_slots=2)
    _prime_slot(engine_a, slot=0, prompt_len=50, rounds=2)
    store.save_checkpoint(engine_a, 0, "slots2", root=tmp_path, baseline_steps=1)

    engine_b = _fresh_engine(num_slots=4)
    with pytest.raises(store.FingerprintMismatchError, match="num_slots"):
        restore.restore_checkpoint(engine_b, 0, "slots2", root=tmp_path)


def test_restore_does_not_block_on_soft_git_sha_drift_by_default(tmp_path: Path) -> None:
    """Repo commit velocity is very high (see notes) -- git-sha drift must
    not, by default, block restoring an otherwise-compatible checkpoint."""
    engine = _fresh_engine()
    _prime_slot(engine, slot=0, prompt_len=50, rounds=2)
    store.save_checkpoint(engine, 0, "softdrift", root=tmp_path, baseline_steps=1)

    manifest_path = store.checkpoint_dir("softdrift", tmp_path) / "manifest.json"
    data = json.loads(manifest_path.read_text())
    git_info = data["fingerprint"].setdefault("git", {})
    git_info.setdefault("qwen-sm120-runtime", {})["sha"] = "deadbeef"
    manifest_path.write_text(json.dumps(data))

    fresh = _fresh_engine()
    result = restore.restore_checkpoint(fresh, 0, "softdrift", root=tmp_path)
    assert result.verified
    assert any("qwen-sm120-runtime" in d for d in result.soft_fingerprint_diff)

    # But require_clean_fingerprint=True must upgrade it to a hard failure.
    fresh2 = _fresh_engine()
    with pytest.raises(store.FingerprintMismatchError):
        restore.restore_checkpoint(
            fresh2, 0, "softdrift", root=tmp_path, require_clean_fingerprint=True
        )


# --- 4. verify safety valve ---------------------------------------------------


def test_verify_safety_valve_rejects_tampered_baseline(tmp_path: Path) -> None:
    engine = _fresh_engine()
    _prime_slot(engine, slot=0)
    store.save_checkpoint(engine, 0, "tampered-baseline", root=tmp_path, baseline_steps=2)

    manifest_path = store.checkpoint_dir("tampered-baseline", tmp_path) / "manifest.json"
    data = json.loads(manifest_path.read_text())
    tokens = data["baseline"]["committed_tokens"]
    assert tokens, "baseline must be non-empty for this test to be meaningful"
    tokens[0] = (tokens[0] + 1) % 32768
    manifest_path.write_text(json.dumps(data))

    fresh = _fresh_engine()
    with pytest.raises(verify.CheckpointVerificationError):
        restore.restore_checkpoint(fresh, 0, "tampered-baseline", root=tmp_path)

    # And critically: no slot should be considered usable after a raised
    # verification error -- the caller never receives a RestoreResult.
    # (implicitly proven by the raise above; nothing further to assert)


# --- drop-one-item parametrized test ------------------------------------------


def _corrupt_tensor_byte(root: Path, name: str, category: str) -> None:
    path = store.checkpoint_dir(name, root) / "tensors.safetensors"
    tensors = load_file(str(path))
    key = next(k for k in tensors if k.startswith(f"{category}/"))
    flat = tensors[key].flatten().clone()
    flat[0] = flat[0] + 1  # uint8 wraparound (255 -> 0) is fine -- just needs to differ
    tensors[key] = flat.reshape(tensors[key].shape)
    save_file(tensors, str(path))


def _drop_tensor_category(root: Path, name: str, category: str) -> None:
    """Simulates 'this checklist item was never saved at all' by deleting
    every tensor entry of one category from BOTH the manifest index and
    the safetensors payload."""
    manifest_path = store.checkpoint_dir(name, root) / "manifest.json"
    data = json.loads(manifest_path.read_text())
    dropped_keys = [e["key"] for e in data["tensors"] if e["category"] == category]
    data["tensors"] = [e for e in data["tensors"] if e["category"] != category]
    manifest_path.write_text(json.dumps(data))

    tensors_path = store.checkpoint_dir(name, root) / "tensors.safetensors"
    tensors = load_file(str(tensors_path))
    for k in dropped_keys:
        del tensors[k]
    save_file(tensors, str(tensors_path))


def _corrupt_kv_len_off_by_one(root: Path, name: str) -> None:
    manifest_path = store.checkpoint_dir(name, root) / "manifest.json"
    data = json.loads(manifest_path.read_text())
    data["slot_kv_len"] = data["slot_kv_len"] - 1
    manifest_path.write_text(json.dumps(data))


def _truncate_committed_tokens(root: Path, name: str) -> None:
    manifest_path = store.checkpoint_dir(name, root) / "manifest.json"
    data = json.loads(manifest_path.read_text())
    data["slot_committed_tokens"] = data["slot_committed_tokens"][:-1]
    manifest_path.write_text(json.dumps(data))


CORRUPTIONS = {
    "full_kv_byte_flip": lambda root, name: _corrupt_tensor_byte(root, name, "full"),
    "swa_kv_byte_flip": lambda root, name: _corrupt_tensor_byte(root, name, "swa"),
    "draft_kv_byte_flip": lambda root, name: _corrupt_tensor_byte(root, name, "draft"),
    "draft_kv_entirely_missing": lambda root, name: _drop_tensor_category(root, name, "draft"),
    "slot_kv_len_off_by_one": lambda root, name: _corrupt_kv_len_off_by_one(root, name),
    "slot_committed_tokens_truncated": lambda root, name: _truncate_committed_tokens(root, name),
}


@pytest.mark.parametrize("case_name", sorted(CORRUPTIONS))
def test_dropping_or_corrupting_any_single_item_is_caught(tmp_path: Path, case_name: str) -> None:
    """Proves the acceptance criterion: omitting/corrupting ANY ONE
    checklist item causes a detectable failure -- either the verify safety
    valve (CheckpointVerificationError) or a lower-level consistency error
    (e.g. a shape mismatch when writing a now-missing tensor back), never a
    silent, plausible-looking-but-wrong success."""
    engine = _fresh_engine()
    _prime_slot(engine, slot=0)
    store.save_checkpoint(engine, 0, "corrupt-me", root=tmp_path, baseline_steps=2)

    CORRUPTIONS[case_name](tmp_path, "corrupt-me")

    fresh = _fresh_engine()
    with pytest.raises(Exception):  # noqa: B017 - deliberately broad, see docstring
        restore.restore_checkpoint(fresh, 0, "corrupt-me", root=tmp_path)
