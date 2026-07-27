"""T1 signature baselines: record a "known good" run's per-(site, layer)
signatures, then judge later runs against it.

This is a *self*-consistency check -- one engine implementation compared
against its own previously-recorded good behavior -- which is a different
job from ``bfdiag/divergence`` (an *oracle-vs-engine* comparison of two
different implementations of the same model). The two nonetheless share one
piece of hard-won knowledge, which this module deliberately keeps
consistent rather than re-deriving from scratch:

**Depth-relaxed thresholds.** A fixed bound across all 48 layers either
false-positives on deep layers (rounding-order drift compounds with depth)
or false-negatives on shallow ones. ``bfdiag/divergence/thresholds.py``
(a parallel agent's module, evidence-grounded against real oracle-diff
measurements -- see notes/2026-07-27-bfdiag-oracle-divergence.md) argues for
widening the allowed error budget with depth via a capped
``sqrt(layer_idx)`` growth model: independent per-layer rounding errors
accumulate like a random walk, so the standard deviation of the accumulated
error grows with the square root of the number of layers. This module
reuses *the same growth model and the same constants*
(``_DEPTH_GROWTH_COEFFICIENT = 0.3``, capped at ``_MAX_GROWTH = 3.0``) for
its own, differently-shaped thresholds (absmax ratio / L2 relative
deviation, vs. that module's cosine/top-k-agreement bars) -- see
notes/2026-07-27-bfprobe-t1-signatures.md's "阈值策略" section for the worked
comparison. This module does not import ``bfdiag`` (that package is owned by
a different agent and is not guaranteed to be present in this package's
dependency graph); the growth model is reimplemented here, deliberately
byte-for-byte identical in shape, not linked at the code level.

Unlike ``bfdiag/divergence/thresholds.py``'s floors (each backed by a real
oracle-diff measurement), this module's *base* ratios
(``_BASE_ABSMAX_HEADROOM``, ``_BASE_L2_REL_DEV``) are provisional -- argued
from first principles (see notes), not yet calibrated against a real
baseline-vs-repeat-run measurement, because no GPU access is available in
this repo's current environment. See
notes/2026-07-27-bfprobe-t1-signatures.md's GPU-verification checklist.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

from bfprobe.reduce import Signature

#: Same growth-model constants as bfdiag/divergence/thresholds.py (see this
#: module's docstring): the per-layer error budget can at most triple by the
#: deepest layers of the 48-layer Laguna-S-2.1 stack.
_MAX_GROWTH = 3.0
_DEPTH_GROWTH_COEFFICIENT = 0.3

#: Layer-0 floors for this module's two upper-bound-style checks (not
#: GPU-calibrated yet -- see module docstring):
#: - absmax may exceed the baseline by up to 50% at layer 0 before being
#!   flagged; deeper layers get up to 50% * growth (150% at the depth cap).
#: - L2 may deviate from the baseline by up to 50% (relative) at layer 0,
#!   growing the same way.
_BASE_ABSMAX_HEADROOM = 1.5
_BASE_L2_REL_DEV = 0.5


def _depth_growth(layer_idx: int) -> float:
    """Identical shape to bfdiag/divergence/thresholds.py's
    ``_depth_growth``: capped ``1 + coefficient * sqrt(layer_idx)``."""
    return min(_MAX_GROWTH, 1.0 + _DEPTH_GROWTH_COEFFICIENT * sqrt(max(layer_idx, 0)))


def absmax_ratio_bound(layer_idx: int) -> float:
    """How far above ``baseline.absmax`` the current absmax may go before
    being flagged, at this layer depth. ``1.0`` would mean "no headroom at
    all"; this grows the *headroom* (not the ratio itself) by depth, e.g.
    layer 0 allows 1.5x, the depth cap allows ``1 + 0.5*3.0 = 2.5x``."""
    growth = _depth_growth(layer_idx)
    return 1.0 + (_BASE_ABSMAX_HEADROOM - 1.0) * growth


def l2_rel_dev_bound(layer_idx: int) -> float:
    """Maximum allowed ``|current.l2 - baseline.l2| / baseline.l2`` at this
    layer depth."""
    growth = _depth_growth(layer_idx)
    return _BASE_L2_REL_DEV * growth


@dataclass(frozen=True)
class BaselineFingerprint:
    """Identifies *what* a baseline was recorded against, so a stale
    baseline (wrong model revision, wrong code) is never silently trusted."""

    model_revision: str
    git_sha: str


@dataclass(frozen=True)
class BaselineEntry:
    """The recorded-good signature for one ``(site_id, layer)`` pair."""

    site_id: int
    layer: int
    absmax: float
    l2: float
    mean: float
    numel: int


@dataclass(frozen=True)
class Baseline:
    """A full baseline: one ``BaselineEntry`` per ``(site_id, layer)`` seen
    in the recording run, plus the fingerprint of what produced it."""

    fingerprint: BaselineFingerprint
    entries: dict[tuple[int, int], BaselineEntry]

    def entry_for(self, site_id: int, layer: int) -> BaselineEntry | None:
        return self.entries.get((site_id, layer))


def current_git_sha(repo_root: Path | None = None) -> str:
    """Best-effort current commit hash, for stamping a recorded baseline.
    Falls back to ``"unknown"`` rather than raising -- a baseline missing a
    git sha is still useful; a baseline recording that crashes because git
    isn't available is not."""
    root = repo_root if repo_root is not None else Path(__file__).resolve().parent.parent
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def record_baseline(
    signatures: list[tuple[int, int, Signature]],
    *,
    model_revision: str,
    git_sha: str,
) -> Baseline:
    """Build a ``Baseline`` from a "known good" run's signatures.

    ``signatures`` is a list of ``(site_id, layer, Signature)`` -- typically
    every ``SignatureRecord`` from several rounds of
    ``SignatureRing.read_all()`` (via
    ``SignatureRecord.as_signature()``), so a given ``(site_id, layer)``
    pair usually appears once per round. Multiple observations of the same
    pair are folded together conservatively:

    - ``absmax``: the max observed (a baseline for "how big did it ever
      legitimately get" should not be an average that a normal peak round
      would already exceed).
    - ``l2``/``mean``: the mean observed (these are used for *relative
      deviation* checks, where a typical/central value is the right
      reference).
    - ``numel``: the last observed value (expected constant per site/layer
      for a fixed model+batch shape; not asserted here since a probe site
      could legitimately see varying batch sizes across rounds).

    Pure function: no file I/O. See ``save_baseline``/``load_baseline`` for
    persistence.
    """
    if not signatures:
        raise ValueError("cannot record a baseline from an empty signature list")

    grouped: dict[tuple[int, int], list[Signature]] = {}
    for site_id, layer, signature in signatures:
        grouped.setdefault((site_id, layer), []).append(signature)

    entries: dict[tuple[int, int], BaselineEntry] = {}
    for (site_id, layer), sigs in grouped.items():
        absmax = max(sig.absmax for sig in sigs)
        l2 = sum(sig.l2 for sig in sigs) / len(sigs)
        mean = sum(sig.mean for sig in sigs) / len(sigs)
        numel = sigs[-1].numel
        entries[(site_id, layer)] = BaselineEntry(
            site_id=site_id, layer=layer, absmax=absmax, l2=l2, mean=mean, numel=numel
        )

    fingerprint = BaselineFingerprint(model_revision=model_revision, git_sha=git_sha)
    return Baseline(fingerprint=fingerprint, entries=entries)


def save_baseline(baseline: Baseline, path: Path) -> None:
    """Persist a baseline as JSON: fingerprint + a flat list of entries."""
    payload = {
        "fingerprint": {
            "model_revision": baseline.fingerprint.model_revision,
            "git_sha": baseline.fingerprint.git_sha,
        },
        "entries": [
            {
                "site_id": entry.site_id,
                "layer": entry.layer,
                "absmax": entry.absmax,
                "l2": entry.l2,
                "mean": entry.mean,
                "numel": entry.numel,
            }
            for entry in baseline.entries.values()
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_baseline(path: Path) -> Baseline:
    """Read back a baseline written by ``save_baseline``."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    fingerprint = BaselineFingerprint(**payload["fingerprint"])
    entries: dict[tuple[int, int], BaselineEntry] = {}
    for raw_entry in payload["entries"]:
        entry = BaselineEntry(**raw_entry)
        entries[(entry.site_id, entry.layer)] = entry
    return Baseline(fingerprint=fingerprint, entries=entries)


@dataclass(frozen=True)
class OutOfBandVerdict:
    """Pure judgment result for one signature against its baseline entry."""

    out_of_band: bool
    reasons: tuple[str, ...]


def judge(baseline_entry: BaselineEntry, current: Signature) -> OutOfBandVerdict:
    """Pure function: is ``current`` out of band relative to
    ``baseline_entry``, at ``baseline_entry.layer``'s depth-relaxed
    thresholds?

    Order of checks matters for the reported reason, not for the verdict
    (``out_of_band`` is true iff *any* reason fires): NaN/Inf are checked
    first because they are unconditionally fatal regardless of what the
    other statistics say (see ``bfprobe.reduce``'s module docstring on why
    absmax/l2/mean are left unmasked and can themselves be NaN/Inf-poisoned
    -- this check does not need them to be finite to make its decision).
    """
    reasons: list[str] = []
    if current.nan_count > 0:
        reasons.append(f"nan_count={current.nan_count}>0")
    if current.inf_count > 0:
        reasons.append(f"inf_count={current.inf_count}>0")
    if reasons:
        return OutOfBandVerdict(out_of_band=True, reasons=tuple(reasons))

    layer = baseline_entry.layer
    if baseline_entry.absmax > 0:
        bound = baseline_entry.absmax * absmax_ratio_bound(layer)
        if current.absmax > bound:
            reasons.append(f"absmax={current.absmax:.6g}>bound={bound:.6g}")
    elif current.absmax > 0:
        # Baseline saw an all-zero (or absent-signal) tensor; any nonzero
        # magnitude now is a real change, not floating-point noise.
        reasons.append(f"absmax={current.absmax:.6g}>baseline_absmax=0")

    if baseline_entry.l2 > 0:
        rel_dev = abs(current.l2 - baseline_entry.l2) / baseline_entry.l2
        bound = l2_rel_dev_bound(layer)
        if rel_dev > bound:
            reasons.append(f"l2_rel_dev={rel_dev:.6g}>bound={bound:.6g}")
    elif current.l2 > 0:
        reasons.append(f"l2={current.l2:.6g}>baseline_l2=0")

    return OutOfBandVerdict(out_of_band=bool(reasons), reasons=tuple(reasons))


if __name__ == "__main__":
    demo_signatures = [
        (200, layer, Signature(absmax=1.0, l2=10.0, mean=0.1, nan_count=0, inf_count=0, numel=100))
        for layer in range(48)
    ]
    demo_baseline = record_baseline(demo_signatures, model_revision="demo", git_sha="deadbeef")
    demo_current = Signature(
        absmax=1.0, l2=10.0, mean=0.1, nan_count=0, inf_count=0, numel=100
    )
    print(judge(demo_baseline.entry_for(200, 31), demo_current))
