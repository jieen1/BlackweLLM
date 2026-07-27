"""Dump-time trace I/O: ring snapshot -> ``trace.jsonl``, and reading it back
for ``bf trace show``/``bf trace diff``.

Everything here runs at run-end (or on-demand for the CLI) -- never from the
hot path. Resolving timing (``RoundRing.snapshot``) may synchronize CUDA;
that cost is paid exactly once here, not per round.
"""

from __future__ import annotations

from pathlib import Path

from bfdiag.trace import events
from bfdiag.trace.ring import RoundRing


def write_trace(ring: RoundRing, path: Path) -> Path:
    """Resolve ``ring`` and write it to ``path`` as newline-delimited JSON,
    oldest round first. Creates parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = ring.snapshot()
    with path.open("w") as f:
        for row in rows:
            f.write(row.to_json())
            f.write("\n")
    return path


def read_trace(path: Path) -> list[events.RoundEvent]:
    """Read a ``trace.jsonl`` file back into a list of ``RoundEvent``,
    in file order (== chronological, since ``write_trace`` writes oldest
    first)."""
    path = Path(path)
    out: list[events.RoundEvent] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(events.RoundEvent.from_json(line))
    return out


def resolve_run_dir(bfdiag_dir: Path, run_id: str) -> Path:
    return Path(bfdiag_dir) / "runs" / run_id


def trace_path_for_run(bfdiag_dir: Path, run_id: str) -> Path:
    return resolve_run_dir(bfdiag_dir, run_id) / "trace.jsonl"


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        ring = RoundRing(4, use_cuda=False)
        row = ring.begin_round(0, 10)
        ring.mark(row, events.PHASE_VERIFY)
        ring.finish_round(
            row,
            events.PHASE_DRAFT,
            path=events.Path.CG_REPLAY,
            cg_miss_reason=events.CgMissReason.NONE,
            draft_tokens_n=15,
            accepted_n=15,
            reject_position=-1,
            bonus_token=7,
        )
        out_path = Path(tmp) / "trace.jsonl"
        write_trace(ring, out_path)
        loaded = read_trace(out_path)
        assert len(loaded) == 1
        assert loaded[0].accepted_n == 15
        print("dump.py self-test OK:", loaded[0])
