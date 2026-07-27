"""Tests for ``bfdiag/checkpoint/store.py``: manifest round-tripping,
safetensors persistence, listing/loading/removal, and fingerprint
capture/compatibility checking.

Every test here uses :mod:`bfdiag.checkpoint.testing`'s ``FakeBackend``/
``FakeDFlashEngine`` (pure CPU tensors) and an explicit ``root=tmp_path``
-- no GPU, no real ``runtime.*`` import, no shared/mutated global state.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bfdiag.checkpoint import store
from bfdiag.checkpoint.testing import FakeBackend, FakeDFlashEngine


def _fresh_engine(**kwargs) -> FakeDFlashEngine:
    defaults = dict(num_slots=2, block_size=16, blocks_per_slot=32, swa_window=40)
    defaults.update(kwargs)
    backend = FakeBackend(**defaults)
    return FakeDFlashEngine(backend, draft_window=40, num_draft_layers=2)


def _prime_slot(engine: FakeDFlashEngine, slot: int, prompt_len: int = 60, rounds: int = 5) -> None:
    prompt = list(range(1, prompt_len + 1))
    boot = engine.dflash_prefill_bootstrap(slot, prompt)
    anchor, draft_tokens = boot["anchor"], boot["draft_tokens"]
    for _ in range(rounds):
        dec = engine.dflash_round(slot, anchor, draft_tokens)
        anchor, draft_tokens = dec["next_anchor"], dec["next_draft_tokens"]


# --- save_checkpoint ----------------------------------------------------------


def test_save_checkpoint_writes_manifest_and_tensors(tmp_path: Path) -> None:
    engine = _fresh_engine()
    _prime_slot(engine, slot=0)
    kv_len_before_baseline = engine.backend.slot_kv_len[0]

    manifest = store.save_checkpoint(engine, 0, "ckpt-a", root=tmp_path, baseline_steps=2)

    assert manifest.name == "ckpt-a"
    assert manifest.slot == 0
    assert manifest.slot_kv_len == kv_len_before_baseline
    assert manifest.slot_committed_tokens == engine.backend.slot_committed_tokens[0][
        : kv_len_before_baseline + 1
    ]
    assert (tmp_path / "ckpt-a" / "manifest.json").exists()
    assert (tmp_path / "ckpt-a" / "tensors.safetensors").exists()
    assert manifest.size_bytes["total"] > 0
    assert manifest.size_bytes["total"] == (
        manifest.size_bytes["full"] + manifest.size_bytes["swa"] + manifest.size_bytes["draft"]
    )
    assert len(manifest.tensors) == 6  # 2 full + 2 swa + 2 draft layers


def test_save_checkpoint_mutates_live_slot_past_the_snapshot_boundary(tmp_path: Path) -> None:
    """Documented side effect: the baseline probe advances the live slot."""
    engine = _fresh_engine()
    _prime_slot(engine, slot=0)
    kv_len_at_save_point = engine.backend.slot_kv_len[0]

    store.save_checkpoint(engine, 0, "ckpt-b", root=tmp_path, baseline_steps=3)

    assert engine.backend.slot_kv_len[0] > kv_len_at_save_point


def test_save_checkpoint_rejects_slot_with_no_committed_tokens(tmp_path: Path) -> None:
    engine = _fresh_engine()
    with pytest.raises(ValueError, match="nothing to checkpoint"):
        store.save_checkpoint(engine, 0, "empty-slot", root=tmp_path)


def test_save_checkpoint_overwrite_protection(tmp_path: Path) -> None:
    engine = _fresh_engine()
    _prime_slot(engine, slot=0)
    store.save_checkpoint(engine, 0, "dup", root=tmp_path)
    with pytest.raises(FileExistsError):
        store.save_checkpoint(engine, 0, "dup", root=tmp_path)
    # overwrite=True must succeed
    store.save_checkpoint(engine, 0, "dup", root=tmp_path, overwrite=True)


# --- manifest round trip -------------------------------------------------------


def test_manifest_to_dict_from_dict_round_trip(tmp_path: Path) -> None:
    engine = _fresh_engine()
    _prime_slot(engine, slot=0)
    manifest = store.save_checkpoint(engine, 0, "roundtrip", root=tmp_path)

    restored = store.CheckpointManifest.from_dict(manifest.to_dict())
    assert restored.to_dict() == manifest.to_dict()


# --- list / load / remove ------------------------------------------------------


def test_list_load_remove_checkpoints(tmp_path: Path) -> None:
    engine = _fresh_engine()
    _prime_slot(engine, slot=0)
    store.save_checkpoint(engine, 0, "one", root=tmp_path)
    _prime_slot(engine, slot=1)
    store.save_checkpoint(engine, 1, "two", root=tmp_path)

    names = {m.name for m in store.list_checkpoints(root=tmp_path)}
    assert names == {"one", "two"}

    loaded = store.load_manifest("two", root=tmp_path)
    assert loaded.slot == 1

    store.remove_checkpoint("one", root=tmp_path)
    assert {m.name for m in store.list_checkpoints(root=tmp_path)} == {"two"}

    with pytest.raises(KeyError):
        store.load_manifest("one", root=tmp_path)
    with pytest.raises(KeyError):
        store.remove_checkpoint("one", root=tmp_path)


def test_list_checkpoints_on_empty_root_returns_empty(tmp_path: Path) -> None:
    assert store.list_checkpoints(root=tmp_path / "does-not-exist-yet") == []


# --- fingerprint capture / compatibility --------------------------------------


def test_check_fingerprint_compatible_no_mismatch_for_identical_geometry() -> None:
    engine_a = _fresh_engine()
    engine_b = _fresh_engine()
    from bfdiag.checkpoint.state import slot_geometry

    geom_a = slot_geometry(engine_a.backend, engine_a, 0)
    geom_b = slot_geometry(engine_b.backend, engine_b, 0)
    fp_a = store.capture_fingerprint(engine_a.backend, engine_a, geom_a, model_revision="rev1")
    fp_b = store.capture_fingerprint(engine_b.backend, engine_b, geom_b, model_revision="rev1")
    assert store.check_fingerprint_compatible(fp_a, fp_b) == []


def test_check_fingerprint_compatible_names_block_size_mismatch() -> None:
    """The task brief's flagship example: a bs=64 checkpoint must not
    restore into a bs=128 engine, and the error must name block_size."""
    from bfdiag.checkpoint.state import slot_geometry

    engine_64 = _fresh_engine(block_size=16, blocks_per_slot=8, swa_window=40)
    engine_128 = _fresh_engine(block_size=32, blocks_per_slot=8, swa_window=40)

    geom_64 = slot_geometry(engine_64.backend, engine_64, 0)
    geom_128 = slot_geometry(engine_128.backend, engine_128, 0)
    fp_saved = store.capture_fingerprint(engine_64.backend, engine_64, geom_64)
    fp_current = store.capture_fingerprint(engine_128.backend, engine_128, geom_128)

    mismatches = store.check_fingerprint_compatible(fp_saved, fp_current)
    assert any(m.startswith("block_size:") for m in mismatches)


def test_check_fingerprint_compatible_names_model_revision_mismatch() -> None:
    from bfdiag.checkpoint.state import slot_geometry

    engine = _fresh_engine()
    geom = slot_geometry(engine.backend, engine, 0)
    fp_saved = store.capture_fingerprint(engine.backend, engine, geom, model_revision="rev-A")
    fp_current = store.capture_fingerprint(engine.backend, engine, geom, model_revision="rev-B")
    mismatches = store.check_fingerprint_compatible(fp_saved, fp_current)
    assert any(m.startswith("model_revision:") for m in mismatches)


def test_soft_fingerprint_diff_is_informational_only() -> None:
    """Git-sha drift is reported but is NOT in HARD_FINGERPRINT_KEYS --
    see store.py's module docstring on SOFT_FINGERPRINT_PATHS for why."""
    saved = {"git": {"qwen-sm120-runtime": {"sha": "aaa111", "dirty": False, "branch": "main"}}}
    current = {"git": {"qwen-sm120-runtime": {"sha": "bbb222", "dirty": False, "branch": "main"}}}
    diffs = store.soft_fingerprint_diff(saved, current)
    assert any("qwen-sm120-runtime" in d for d in diffs)
    # and none of HARD_FINGERPRINT_KEYS's fields are touched by this diff
    assert store.check_fingerprint_compatible(saved, current) == []


def test_prompt_hash_is_stable_and_content_sensitive() -> None:
    a = store.prompt_hash([1, 2, 3])
    b = store.prompt_hash([1, 2, 3])
    c = store.prompt_hash([1, 2, 4])
    assert a == b
    assert a != c
