"""Offline T1 scan: turn a sequence of signatures + a baseline into "here is
the first (round, layer, tap) that went out of band".

Pure logic only -- no file I/O (that is ``bfprobe.scan_cli``'s job) and no
model/tensor access (this operates purely on already-reduced
``bfprobe.signature.SignatureRecord`` rows, exactly the shape
``SignatureRing.read_all()`` produces or ``bfprobe.signature.load_json``
reads back). This mirrors ``bfdiag/divergence/scan.py``'s split (pure scan
vs. CLI plumbing) for the same reason: the interesting algorithm --
"iterate in time order, stop at the first bad one" -- should be testable
with plain synthetic data, with no dependency on how the data got there.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from bfprobe.baseline import Baseline, OutOfBandVerdict, judge
from bfprobe.signature import SignatureRecord

#: This package's T1 tap-kind ids (see notes/2026-07-27-bfprobe-t1-
#: signatures.md's "site_id 分配规则"). ``site_id`` identifies which kind of
#: tensor was tapped -- *not* which layer, since the layer index is already
#: its own column (``SignatureRecord.layer``) -- so 48 layers x 4 taps needs
#: only 4 distinct ids here, leaving 200-299's remaining 96 ids free for
#: future tap kinds. Names chosen to match
#: ``bfdiag/divergence/thresholds.py``'s kind constants
#: (``INPUT_LAYERNORM``/``ATTN_OUT``/``POST_ATTENTION_LAYERNORM``/
#: ``MOE_OUT``) so cross-referencing a T1 site with that module's
#: oracle-diff thresholds is a name lookup, not a mapping table.
SITE_INPUT_LAYERNORM = 200
SITE_ATTN_OUT = 201
SITE_POST_ATTENTION_LAYERNORM = 202
SITE_MOE_OUT = 203

SITE_NAMES: dict[int, str] = {
    SITE_INPUT_LAYERNORM: "input_layernorm",
    SITE_ATTN_OUT: "self_attn",
    SITE_POST_ATTENTION_LAYERNORM: "post_attention_layernorm",
    SITE_MOE_OUT: "mlp",
}


def site_name(site_id: int) -> str:
    return SITE_NAMES.get(site_id, f"site_{site_id}")


@dataclass(frozen=True)
class SignatureVerdict:
    """One judged signature: the record plus its pass/fail verdict."""

    record: SignatureRecord
    verdict: OutOfBandVerdict


@dataclass(frozen=True)
class FirstOutOfBand:
    """The answer this whole package exists to produce: the first point,
    in time order, where something looked wrong."""

    round_idx: int
    layer: int
    site_id: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ScanReport:
    """Every judged signature (in ``(round_idx, layer, site_id)`` order) plus
    the first out-of-band one, if any."""

    verdicts: tuple[SignatureVerdict, ...]
    skipped_no_baseline: tuple[SignatureRecord, ...]
    first_out_of_band: FirstOutOfBand | None

    @property
    def has_out_of_band(self) -> bool:
        return self.first_out_of_band is not None

    def timeline_for(self, site_id: int, layer: int) -> tuple[SignatureVerdict, ...]:
        """All judged signatures for one ``(site_id, layer)``, in round
        order -- "the time series for this tap" a human would want to plot."""
        return tuple(
            item
            for item in self.verdicts
            if item.record.site_id == site_id and item.record.layer == layer
        )


def scan(records: Iterable[SignatureRecord], baseline: Baseline) -> ScanReport:
    """Judge every record against ``baseline``, in ``(round_idx, layer,
    site_id)`` order, and report the first out-of-band one.

    Records are sorted defensively (not just trusted to already be in time
    order) so the "first" in ``first_out_of_band`` is unambiguous regardless
    of what order the caller's iterable happens to yield them in. A record
    whose ``(site_id, layer)`` has no baseline entry is skipped -- not
    treated as a failure -- matching
    ``bfdiag/divergence/scan.py``'s "missing on one side -> silently skip"
    convention; those are collected in ``skipped_no_baseline`` so a caller
    can still notice ("scanned 180/192 sites, 12 had no baseline") without
    it silently looking like a clean pass.
    """
    ordered = sorted(records, key=lambda record: (record.round_idx, record.layer, record.site_id))

    verdicts: list[SignatureVerdict] = []
    skipped: list[SignatureRecord] = []
    first: FirstOutOfBand | None = None
    for record in ordered:
        entry = baseline.entry_for(record.site_id, record.layer)
        if entry is None:
            skipped.append(record)
            continue
        verdict = judge(entry, record.as_signature())
        verdicts.append(SignatureVerdict(record=record, verdict=verdict))
        if verdict.out_of_band and first is None:
            first = FirstOutOfBand(
                round_idx=record.round_idx,
                layer=record.layer,
                site_id=record.site_id,
                reasons=verdict.reasons,
            )

    return ScanReport(
        verdicts=tuple(verdicts),
        skipped_no_baseline=tuple(skipped),
        first_out_of_band=first,
    )


# ---------------------------------------------------------------------------
# Report rendering. Kept in this module (not a separate ``report.py``) since
# this package does not own ``bfprobe/report.py`` -- see
# notes/2026-07-27-bfprobe-t1-signatures.md.
# ---------------------------------------------------------------------------


def to_json_dict(report: ScanReport) -> dict[str, Any]:
    """Machine-readable rendering, used by ``bf probe scan --json``."""
    return {
        "has_out_of_band": report.has_out_of_band,
        "first_out_of_band": (
            {
                "round_idx": report.first_out_of_band.round_idx,
                "layer": report.first_out_of_band.layer,
                "site_id": report.first_out_of_band.site_id,
                "site_name": site_name(report.first_out_of_band.site_id),
                "reasons": list(report.first_out_of_band.reasons),
            }
            if report.first_out_of_band is not None
            else None
        ),
        "num_judged": len(report.verdicts),
        "num_skipped_no_baseline": len(report.skipped_no_baseline),
        "num_out_of_band": sum(1 for item in report.verdicts if item.verdict.out_of_band),
    }


def format_text_report(report: ScanReport) -> str:
    """Render a short, human-readable conclusion: pass/fail counts plus the
    first-out-of-band drill-down, if any."""
    lines: list[str] = []
    num_out_of_band = sum(1 for item in report.verdicts if item.verdict.out_of_band)
    lines.append(
        f"judged {len(report.verdicts)} signatures "
        f"({num_out_of_band} out of band, "
        f"{len(report.skipped_no_baseline)} skipped -- no baseline)"
    )
    lines.append("-" * 64)
    if report.has_out_of_band:
        first = report.first_out_of_band
        assert first is not None
        lines.append(
            f"第一个越界点: round {first.round_idx}, layer {first.layer}, "
            f"{site_name(first.site_id)} (site_id={first.site_id})"
        )
        lines.append(f"  原因: {', '.join(first.reasons)}")
    else:
        lines.append("未发现越界: 全部签名均在阈值内。")
    return "\n".join(lines)


if __name__ == "__main__":
    from bfprobe.baseline import record_baseline
    from bfprobe.reduce import Signature
    from bfprobe.signature import SignatureRecord

    good = Signature(absmax=1.0, l2=10.0, mean=0.1, nan_count=0, inf_count=0, numel=100)
    demo_baseline = record_baseline(
        [(SITE_MOE_OUT, layer, good) for layer in range(48)],
        model_revision="demo",
        git_sha="deadbeef",
    )
    demo_records = [
        SignatureRecord(
            seq=layer,
            site_id=SITE_MOE_OUT,
            round_idx=0,
            layer=layer,
            absmax=(50.0 if layer == 31 else 1.0),
            l2=10.0,
            mean=0.1,
            nan_count=0,
            inf_count=0,
            numel=100,
        )
        for layer in range(48)
    ]
    demo_report = scan(demo_records, demo_baseline)
    print(format_text_report(demo_report))
