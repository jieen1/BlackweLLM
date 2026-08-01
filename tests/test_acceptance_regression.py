"""CPU contracts for the acceptance-regression fixture metadata."""

from __future__ import annotations

import json
import runpy
from pathlib import Path
from types import SimpleNamespace


def test_fixture_provenance_keeps_paired_source_identity() -> None:
    module = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "benchmarks" / "acceptance_regression.py"),
        run_name="acceptance_regression_test",
    )
    record = SimpleNamespace(
        run_id="run-123",
        fingerprint=SimpleNamespace(
            git={
                "qwen-sm120-runtime": SimpleNamespace(
                    sha="runtime-sha", branch="main", dirty=False
                ),
                "sparkinfer": SimpleNamespace(sha="sparkinfer-sha", branch="master", dirty=True),
            }
        ),
    )

    assert module["artifact_provenance"](record) == {
        "bfdiag_run_id": "run-123",
        "runtime": {"sha": "runtime-sha", "branch": "main", "dirty": False},
        "sparkinfer": {"sha": "sparkinfer-sha", "branch": "master", "dirty": True},
    }


def test_save_fixture_attaches_complete_rounds_to_the_run(tmp_path: Path) -> None:
    module = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "benchmarks" / "acceptance_regression.py"),
        run_name="acceptance_regression_test",
    )
    record = SimpleNamespace(
        run_id="run-123",
        fingerprint=SimpleNamespace(git={}),
    )
    attached: list[tuple[str, Path]] = []
    rec = SimpleNamespace(
        record=record,
        artifact=lambda name, path: attached.append((name, Path(path))),
    )
    results = [{"label": "fox-64K", "rounds": [{"tok_s": 300.0}]}]

    out = module["save_fixture"](
        rec, {"all": {"mean": 1.0}}, results, output_path=tmp_path / "run.json"
    )

    assert attached == [("acceptance_fixture", out)]
    assert json.loads(out.read_text())["results"] == results
