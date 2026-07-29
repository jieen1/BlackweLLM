"""Persistence for one slot's checkpoint: safetensors tensor payload + a
JSON manifest carrying bookkeeping, a fingerprint, and a deterministic
verification baseline.

Layout on disk, under ``${QSR_BFDIAG_DIR:-<repo>/.bfdiag}/checkpoints/``::

    checkpoints/<name>/manifest.json          # everything except raw tensors
    checkpoints/<name>/tensors.safetensors    # raw KV bytes (see state.py)

No ``runtime.*`` import anywhere in this module -- ``backend``/``engine``
are duck-typed (mirrors ``bfdiag/daemon/session.py::reset_laguna_engine``'s
"engine: Any" convention), so this module is equally happy operating on the
real (never-executed, per this task's no-GPU constraint) Laguna objects or
on :mod:`bfdiag.checkpoint.testing`'s ``FakeBackend``/``FakeDFlashEngine``.
``torch``/``safetensors`` ARE imported at module scope -- both are pure
CPU-safe imports (neither touches CUDA merely by being imported), and every
tensor operation here (``.to("cpu")``, slicing, ``.numel()``) is safe
whether the source tensor is already CPU (tests) or a real GPU tensor
(production, never executed in this task).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from bfdiag.checkpoint import verify
from bfdiag.checkpoint.state import (
    SlotGeometry,
    draft_ring_block_range,
    full_block_range,
    slot_geometry,
    swa_ring_block_range,
)
from bfdiag.record import fingerprint as _fp
from bfdiag.record.store import bfdiag_dir

CHECKPOINT_SCHEMA_VERSION = 1

#: Number of dflash_round() calls to run (from the freshly-derived
#: anchor/draft_tokens, see verify.py) to build the deterministic
#: verification baseline recorded at save time and re-checked at restore
#: time. Small and cheap -- this is a correctness gate, not a benchmark.
DEFAULT_BASELINE_STEPS = 4

#: Fingerprint fields (read from the manifest's "laguna_geometry" section)
#: whose mismatch means restoring into this engine is STRUCTURALLY unsafe:
#: wrong tensor shapes, wrong ring addressing, or different model weights.
#: A mismatch here always raises -- there is no override, because the
#: failure mode is silent memory corruption or a shape error, not merely
#: "the numbers might have drifted" (that risk is what verify.py's
#: deterministic replay exists to catch instead).
HARD_FINGERPRINT_KEYS: tuple[str, ...] = (
    "block_size",
    "blocks_per_slot",
    "ring_blocks_per_slot",
    "swa_window",
    "draft_blocks_per_slot",
    "kv_dtype",
    "num_slots",
    "model_revision",
)

#: Fields that are recorded and SHOWN prominently on every restore, but do
#: NOT block it by default: this repo's commit velocity is very high
#: (see the user's own project memory: "提交节奏极快" -- near-100 commits/
#: day is normal), and the entire point of this feature is to keep re-
#: running the SAME prefilled 64K context while iterating on decode-path
#: code across many daemon sessions/days. Hard-blocking on every git-sha
#: drift would make checkpoints unusable for their actual purpose. Pass
#: ``require_clean_fingerprint=True`` to ``restore_checkpoint`` to upgrade
#: these to hard failures too (e.g. when trying to reproduce one exact
#: historical run bit-for-bit). See notes/2026-07-27-bfdiag-checkpoint-
#: restore.md's "指纹与拒绝策略" section for the full reasoning.
SOFT_FINGERPRINT_PATHS: tuple[str, ...] = (
    "git.qwen-sm120-runtime.sha",
    "git.sparkinfer.sha",
)


class FingerprintMismatchError(RuntimeError):
    """Raised by :func:`bfdiag.checkpoint.restore.restore_checkpoint` when a
    HARD_FINGERPRINT_KEYS field differs between the saved checkpoint and
    the live engine -- restoring would be structurally unsafe."""


def default_checkpoint_root() -> Path:
    """``${QSR_BFDIAG_DIR:-<repo>/.bfdiag}/checkpoints`` -- re-evaluated on
    every call (like ``bfdiag.record.store.bfdiag_dir``) so tests that
    monkeypatch ``QSR_BFDIAG_DIR`` see it take effect."""
    return bfdiag_dir() / "checkpoints"


def checkpoint_dir(name: str, root: Path | str | None = None) -> Path:
    base = Path(root) if root is not None else default_checkpoint_root()
    return base / name


def _manifest_path(name: str, root: Path | str | None) -> Path:
    return checkpoint_dir(name, root) / "manifest.json"


def _tensors_path(name: str, root: Path | str | None) -> Path:
    return checkpoint_dir(name, root) / "tensors.safetensors"


def _atomic_write_text(path: Path, text: str) -> None:
    """Same discipline as ``bfdiag/record/store.py::_atomic_write_text``:
    write-to-temp-then-rename, so a reader never observes a half-written
    manifest.json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp_path)
        raise


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def prompt_hash(token_ids: list[int]) -> str:
    """Short, stable hash of a token-id sequence -- purely informational
    (shown in ``bf checkpoint show``), not part of HARD_FINGERPRINT_KEYS."""
    return hashlib.sha256(repr(list(token_ids)).encode()).hexdigest()[:16]


@dataclass
class CheckpointManifest:
    """Everything about one checkpoint except the raw tensor bytes
    (those live in ``tensors.safetensors``, indexed by ``tensors`` below)."""

    name: str
    schema_version: int
    created_at: str
    slot: int
    geometry: dict[str, Any]
    slot_kv_len: int
    slot_committed_tokens: list[int]
    fingerprint: dict[str, Any]
    baseline: dict[str, Any]
    tensors: list[dict[str, Any]]
    size_bytes: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointManifest:
        return cls(
            name=data["name"],
            schema_version=int(data.get("schema_version", CHECKPOINT_SCHEMA_VERSION)),
            created_at=data.get("created_at", ""),
            slot=int(data["slot"]),
            geometry=dict(data.get("geometry") or {}),
            slot_kv_len=int(data["slot_kv_len"]),
            slot_committed_tokens=list(data.get("slot_committed_tokens") or []),
            fingerprint=dict(data.get("fingerprint") or {}),
            baseline=dict(data.get("baseline") or {}),
            tensors=list(data.get("tensors") or []),
            size_bytes=dict(data.get("size_bytes") or {}),
        )


def _get_path(data: dict[str, Any], dotted: str) -> Any:
    node: Any = data
    for part in dotted.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    return node


def capture_fingerprint(
    backend: Any,
    engine: Any,
    geom: SlotGeometry,
    *,
    prompt_ids: list[int] | None = None,
    model_revision: str | None = None,
    repo_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build the full checkpoint fingerprint: reuses
    ``bfdiag.record.fingerprint.capture()`` for the environment/git/GPU/
    Python portion (DRY with the rest of bfdiag -- same three repos, same
    ``QSR_REPO_*`` overrides), plus a ``laguna_geometry`` section holding
    exactly the fields ``HARD_FINGERPRINT_KEYS`` checks.

    ``model_revision`` is caller-supplied (e.g. from a provider's
    ``describe()["model_revision"]``) because neither ``LagunaBackend`` nor
    ``DFlashEngine`` expose it as an attribute themselves -- only the
    ``bfdiag.daemon.provider.LagunaEngineProvider`` layer above them
    computes it (see that module's ``_extract_revision``). ``None`` is a
    valid, honest value here if the caller has no provider layer to ask.
    """
    workload: dict[str, Any] = {"block_size": geom.block_size}
    if prompt_ids is not None:
        workload["prompt_hash"] = prompt_hash(prompt_ids)
        workload["prompt_len"] = len(prompt_ids)

    base = _fp.capture(
        model={"revision": model_revision},
        workload=workload,
        repo_paths=repo_paths,
    ).to_dict()

    kv_dtype = _kv_dtype_str(backend, geom)
    base["laguna_geometry"] = {
        "block_size": geom.block_size,
        "blocks_per_slot": geom.blocks_per_slot,
        "ring_blocks_per_slot": geom.ring_blocks_per_slot,
        "swa_window": geom.swa_window,
        "draft_blocks_per_slot": geom.draft_blocks_per_slot,
        "num_slots": geom.num_slots,
        "kv_dtype": kv_dtype,
        "model_revision": model_revision,
    }
    return base


def _kv_dtype_str(backend: Any, geom: SlotGeometry) -> str | None:
    """Read the dtype directly off a real full-attention KV tensor rather
    than threading ``cache_dtype_str`` through -- simpler, and exactly the
    dtype that will actually be (de)serialized."""
    if not geom.full_layer_names:
        return None
    tensor = backend.kv_caches[geom.full_layer_names[0]]
    return str(tensor.dtype)


def check_fingerprint_compatible(
    saved: dict[str, Any],
    current: dict[str, Any],
    *,
    hard_keys: tuple[str, ...] = HARD_FINGERPRINT_KEYS,
) -> list[str]:
    """Return a list of ``"key: saved=X current=Y"`` strings for every
    ``hard_keys`` field that differs between the saved and current
    fingerprint's ``laguna_geometry`` section. Empty list == safe to
    restore. Called by :func:`bfdiag.checkpoint.restore.restore_checkpoint`,
    which raises :class:`FingerprintMismatchError` naming every mismatch
    found (block_size in particular -- see this module's docstring on
    HARD_FINGERPRINT_KEYS for why that one is the flagship example, given
    the project's in-flight 64->128 block_size migration)."""
    saved_geom = saved.get("laguna_geometry") or {}
    current_geom = current.get("laguna_geometry") or {}
    mismatches = []
    for key in hard_keys:
        sv, cv = saved_geom.get(key), current_geom.get(key)
        if sv != cv:
            mismatches.append(f"{key}: saved={sv!r} current={cv!r}")
    return mismatches


def soft_fingerprint_diff(
    saved: dict[str, Any],
    current: dict[str, Any],
    *,
    soft_paths: tuple[str, ...] = SOFT_FINGERPRINT_PATHS,
) -> list[str]:
    """Informational-only diff (git shas by default) -- never blocks
    restore unless the caller passes ``require_clean_fingerprint=True``.
    Same ``"path: saved=X current=Y"`` format as
    :func:`check_fingerprint_compatible` for consistent display."""
    diffs = []
    for path in soft_paths:
        sv, cv = _get_path(saved, path), _get_path(current, path)
        if sv != cv:
            diffs.append(f"{path}: saved={sv!r} current={cv!r}")
    return diffs


def _gather_tensors(
    backend: Any, engine: Any, geom: SlotGeometry, kv_len: int
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    """Slice out exactly the block ranges ``state.py`` says matter, copy to
    CPU (a no-op for already-CPU tensors; the required step for real GPU
    tensors -- never executed against a real GPU in this task), and build
    the tensor index recorded in the manifest."""
    tensors: dict[str, torch.Tensor] = {}
    entries: list[dict[str, Any]] = []

    full_start, full_end = full_block_range(geom, kv_len)
    for name in geom.full_layer_names:
        chunk = backend.kv_caches[name][:, full_start:full_end].detach().to("cpu").contiguous()
        key = f"full/{name}"
        tensors[key] = chunk
        entries.append(
            {
                "key": key,
                "category": "full",
                "layer_name": name,
                "shape": list(chunk.shape),
                "dtype": str(chunk.dtype),
                "num_blocks": full_end - full_start,
            }
        )

    ring_start, ring_end = swa_ring_block_range(geom)
    for name in geom.swa_layer_names:
        chunk = backend.kv_caches[name][:, ring_start:ring_end].detach().to("cpu").contiguous()
        key = f"swa/{name}"
        tensors[key] = chunk
        entries.append(
            {
                "key": key,
                "category": "swa",
                "layer_name": name,
                "shape": list(chunk.shape),
                "dtype": str(chunk.dtype),
                "num_blocks": ring_end - ring_start,
            }
        )

    draft_start, draft_end = draft_ring_block_range(geom)
    for name in geom.draft_layer_names:
        raw = engine._draft_kv_caches[name][:, draft_start:draft_end]
        chunk = raw.detach().to("cpu").contiguous()
        key = f"draft/{name}"
        tensors[key] = chunk
        entries.append(
            {
                "key": key,
                "category": "draft",
                "layer_name": name,
                "shape": list(chunk.shape),
                "dtype": str(chunk.dtype),
                "num_blocks": draft_end - draft_start,
            }
        )

    return tensors, entries


def _size_bytes(tensors: dict[str, torch.Tensor]) -> dict[str, int]:
    totals = {"full": 0, "swa": 0, "draft": 0}
    for key, tensor in tensors.items():
        category = key.split("/", 1)[0]
        totals[category] = totals.get(category, 0) + tensor.numel() * tensor.element_size()
    totals["total"] = sum(totals.values())
    return totals


def save_checkpoint(
    engine: Any,
    slot: int,
    name: str,
    *,
    baseline_steps: int = DEFAULT_BASELINE_STEPS,
    model_revision: str | None = None,
    root: Path | str | None = None,
    repo_paths: dict[str, str] | None = None,
    overwrite: bool = False,
) -> CheckpointManifest:
    """Snapshot slot ``slot``'s complete state (see ``state.py``'s
    ``SLOT_STATE_ITEMS``) and persist it as ``<root>/<name>/``.

    Intended call site: immediately after
    ``engine.dflash_prefill_bootstrap(slot, prompt)`` (or after N
    ``dflash_round`` calls) -- i.e. exactly the "prefill 完成后主动存档"
    moment the task brief describes.

    Documented side effect: after taking the tensor snapshot, this
    function runs ``baseline_steps`` further ``dflash_round`` calls against
    the LIVE slot to build the deterministic verification baseline (see
    :mod:`bfdiag.checkpoint.verify`) -- this necessarily advances the live
    slot's kv_len/committed_tokens past the checkpointed point. Callers who
    need the live slot to remain exactly at the pre-save state afterward
    must snapshot before calling this, or call
    ``engine.backend.reset_slot(slot)`` themselves once ``save_checkpoint``
    returns.
    """
    backend = engine.backend
    geom = slot_geometry(backend, engine, slot)
    kv_len = backend.slot_kv_len[slot]
    committed = list(backend.slot_committed_tokens[slot])
    if not committed:
        raise ValueError(
            f"slot {slot} has no committed tokens -- nothing to checkpoint "
            "(did you prefill this slot first?)"
        )

    tensors, entries = _gather_tensors(backend, engine, geom, kv_len)
    fp = capture_fingerprint(
        backend,
        engine,
        geom,
        prompt_ids=committed,
        model_revision=model_revision,
        repo_paths=repo_paths,
    )
    baseline = verify.run_probe(engine, slot, steps=baseline_steps)

    manifest = CheckpointManifest(
        name=name,
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        created_at=_now_iso(),
        slot=slot,
        geometry=dataclasses.asdict(geom),
        slot_kv_len=kv_len,
        slot_committed_tokens=committed,
        fingerprint=fp,
        baseline=baseline.to_dict(),
        tensors=entries,
        size_bytes=_size_bytes(tensors),
    )

    ckpt_dir = checkpoint_dir(name, root)
    if ckpt_dir.exists() and not overwrite:
        raise FileExistsError(
            f"checkpoint {name!r} already exists at {ckpt_dir} (pass overwrite=True to replace)"
        )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(_tensors_path(name, root)),
        metadata={"bfdiag_checkpoint_schema": str(CHECKPOINT_SCHEMA_VERSION), "name": name},
    )
    _atomic_write_text(
        _manifest_path(name, root),
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False, sort_keys=True),
    )
    return manifest


def load_manifest(name: str, root: Path | str | None = None) -> CheckpointManifest:
    path = _manifest_path(name, root)
    if not path.exists():
        raise KeyError(f"no such checkpoint: {name!r} (looked for {path})")
    return CheckpointManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_tensors(name: str, root: Path | str | None = None) -> dict[str, torch.Tensor]:
    path = _tensors_path(name, root)
    if not path.exists():
        raise KeyError(f"no such checkpoint tensors file: {name!r} (looked for {path})")
    return load_file(str(path))


def list_checkpoints(root: Path | str | None = None) -> list[CheckpointManifest]:
    """Newest-first, by ``created_at``."""
    base = Path(root) if root is not None else default_checkpoint_root()
    if not base.exists():
        return []
    manifests = []
    for child in sorted(base.iterdir()):
        if (child / "manifest.json").exists():
            manifests.append(load_manifest(child.name, root=root))
    manifests.sort(key=lambda m: m.created_at, reverse=True)
    return manifests


def remove_checkpoint(name: str, root: Path | str | None = None) -> None:
    ckpt_dir = checkpoint_dir(name, root)
    if not ckpt_dir.exists():
        raise KeyError(f"no such checkpoint: {name!r} (looked for {ckpt_dir})")
    shutil.rmtree(ckpt_dir)
