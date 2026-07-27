"""Core bfdiag.record behavior: schema round-trip, the sqlite store, the
``run_record`` context manager, the zero-invasion ``auto_record()`` entry
point, and the ``bf`` CLI dispatcher.

Nothing in this file touches the GPU: fingerprints captured here always run
with ``QSR_BFDIAG_DIR`` pointed at a tmp directory and (where fingerprint
capture happens implicitly, e.g. inside ``run_record``) rely on
``fingerprint.capture()``'s own no-GPU-machine tolerance rather than any
GPU state.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from bfdiag import cli as bfdiag_cli
from bfdiag.record import RunHandle, run_record
from bfdiag.record.schema import Fingerprint, GitRepoInfo, RunRecord, new_run_id, utc_now_iso
from bfdiag.record.store import RunStore

_REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def store(tmp_path: Path) -> RunStore:
    return RunStore(tmp_path / ".bfdiag")


def _make_record(run_id: str | None = None, **overrides) -> RunRecord:
    defaults = dict(
        run_id=run_id or new_run_id(),
        started_at=utc_now_iso(),
        script="benchmarks/example.py",
        argv=["--flag"],
        status="ok",
    )
    defaults.update(overrides)
    return RunRecord(**defaults)


# --- schema -----------------------------------------------------------------


def test_run_record_to_dict_from_dict_round_trip() -> None:
    record = _make_record(
        finished_at=utc_now_iso(),
        fingerprint=Fingerprint(
            git={"vllm": GitRepoInfo(sha="abc123", dirty=False, branch="main")}
        ),
        metrics={"acceptance_rate": 0.687},
        artifacts={"profile": "runs/x/artifacts/profile.json"},
    )
    restored = RunRecord.from_dict(json.loads(json.dumps(record.to_dict())))
    assert restored == record


def test_run_record_get_path_dotted_lookup() -> None:
    record = _make_record(
        fingerprint=Fingerprint(git={"vllm": GitRepoInfo(sha="deadbeef")}),
    )
    assert record.get_path("fingerprint.git.vllm.sha") == "deadbeef"
    assert record.get_path("fingerprint.git.vllm.branch") is None
    assert record.get_path("fingerprint.does.not.exist") is None


# --- store --------------------------------------------------------------


def test_store_save_and_load_round_trip(store: RunStore) -> None:
    record = _make_record(metrics={"acceptance_rate": 0.687})
    store.save(record)
    loaded = store.load(record.run_id)
    assert loaded == record


def test_store_load_missing_run_raises_keyerror(store: RunStore) -> None:
    with pytest.raises(KeyError):
        store.load("no-such-run")


def test_store_writes_record_json_file(store: RunStore) -> None:
    record = _make_record()
    store.save(record)
    path = store.record_path(record.run_id)
    assert path.is_file()
    assert json.loads(path.read_text())["run_id"] == record.run_id


def test_store_list_runs_newest_first(store: RunStore) -> None:
    first = _make_record(started_at="2026-07-27T10:00:00+00:00")
    second = _make_record(started_at="2026-07-27T11:00:00+00:00")
    store.save(first)
    store.save(second)
    runs = store.list_runs()
    assert [r.run_id for r in runs] == [second.run_id, first.run_id]


def test_store_list_runs_respects_limit(store: RunStore) -> None:
    for _ in range(5):
        store.save(_make_record())
    assert len(store.list_runs(limit=2)) == 2


def test_store_resolve_run_id_exact_and_prefix(store: RunStore) -> None:
    record = _make_record(run_id="abcdef012345")
    store.save(record)
    assert store.resolve_run_id("abcdef012345") == "abcdef012345"
    assert store.resolve_run_id("abcdef") == "abcdef012345"


def test_store_resolve_run_id_missing_raises_keyerror(store: RunStore) -> None:
    with pytest.raises(KeyError):
        store.resolve_run_id("nope")


def test_store_resolve_run_id_ambiguous_prefix_raises_valueerror(store: RunStore) -> None:
    store.save(_make_record(run_id="abc111111111"))
    store.save(_make_record(run_id="abc222222222"))
    with pytest.raises(ValueError):
        store.resolve_run_id("abc")


def test_store_query_metric(store: RunStore) -> None:
    a = _make_record(metrics={"acceptance_rate": 1.0})
    b = _make_record(metrics={"acceptance_rate": 0.687})
    store.save(a)
    store.save(b)
    rows = dict(store.query_metric("acceptance_rate"))
    assert rows[a.run_id] == 1.0
    assert rows[b.run_id] == 0.687


def test_store_save_is_atomic_transaction(store: RunStore) -> None:
    """A run's metrics rows always match what's in its record.json -- an
    update replaces both together, never leaves stale metric rows behind.
    """
    record = _make_record(metrics={"a": 1.0, "b": 2.0})
    store.save(record)
    record.metrics = {"a": 1.0}  # "b" dropped between rounds
    store.save(record)
    rows = store.query_metric("b")
    assert rows == []
    rows_a = store.query_metric("a")
    assert dict(rows_a)[record.run_id] == 1.0


# --- run_record() context manager -------------------------------------------


def test_run_record_success_path_persists_ok_status(store: RunStore) -> None:
    with run_record(script="demo.py", workload={"k": 15}, store=store) as rec:
        assert isinstance(rec, RunHandle)
        rec.metric("acceptance_rate", 0.687)

    loaded = store.load(rec.run_id)
    assert loaded.status == "ok"
    assert loaded.finished_at is not None
    assert loaded.metrics["acceptance_rate"] == 0.687
    assert loaded.fingerprint.workload.k == 15


def test_run_record_exports_and_restores_env_vars(store: RunStore, monkeypatch) -> None:
    monkeypatch.delenv("QSR_BFDIAG_RUN_ID", raising=False)
    monkeypatch.delenv("QSR_RUN_RECORD", raising=False)
    captured_run_id = None
    with run_record(script="demo.py", store=store) as rec:
        captured_run_id = rec.run_id
        assert os.environ["QSR_BFDIAG_RUN_ID"] == rec.run_id
        assert os.environ["QSR_RUN_RECORD"] == str(store.record_path(rec.run_id))
        assert os.environ["QSR_BFDIAG_DIR"] == str(store.root)
    assert "QSR_BFDIAG_RUN_ID" not in os.environ
    assert "QSR_RUN_RECORD" not in os.environ
    assert captured_run_id is not None


def test_run_record_persists_failed_status_with_traceback_on_exception(store: RunStore) -> None:
    run_id = None
    with pytest.raises(RuntimeError, match="boom"):
        with run_record(script="demo.py", store=store) as rec:
            run_id = rec.run_id
            rec.metric("acceptance_rate", 0.3)
            raise RuntimeError("boom")

    loaded = store.load(run_id)
    assert loaded.status == "failed"
    assert loaded.error is not None
    assert "RuntimeError: boom" in loaded.error
    assert loaded.finished_at is not None
    # metrics recorded before the crash are not lost -- a crashed experiment
    # is still data.
    assert loaded.metrics["acceptance_rate"] == 0.3


def test_run_record_artifact_registers_relpath(store: RunStore, tmp_path: Path) -> None:
    src = tmp_path / "profile.json"
    src.write_text("{}", encoding="utf-8")
    with run_record(script="demo.py", store=store) as rec:
        rec.artifact("profile", src)
        run_id = rec.run_id

    loaded = store.load(run_id)
    relpath = loaded.artifacts["profile"]
    assert (store.root / relpath).is_file()


# --- auto_record() zero-invasion path (subprocess, so atexit/excepthook are
# exercised for real without affecting this test process) -------------------


def _run_adopt_subprocess(tmp_path: Path, body: str) -> subprocess.CompletedProcess:
    script = f"import sys\nfrom bfdiag.record import auto_record\n{body}\n"
    env = dict(os.environ)
    env["QSR_BFDIAG_DIR"] = str(tmp_path / ".bfdiag")
    env["PYTHONPATH"] = str(_REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_auto_record_persists_ok_on_clean_exit(tmp_path: Path) -> None:
    result = _run_adopt_subprocess(
        tmp_path, "handle = auto_record(script='adopt_demo.py')\nprint(handle.run_id)"
    )
    assert result.returncode == 0, result.stderr
    run_id = result.stdout.strip().splitlines()[-1]

    loaded_store = RunStore(tmp_path / ".bfdiag")
    record = loaded_store.load(run_id)
    assert record.status == "ok"
    assert record.finished_at is not None
    assert record.script == "adopt_demo.py"


def test_auto_record_persists_failed_on_uncaught_exception(tmp_path: Path) -> None:
    result = _run_adopt_subprocess(
        tmp_path,
        "handle = auto_record(script='adopt_crash.py')\n"
        "print(handle.run_id)\n"
        "raise RuntimeError('kaboom')\n",
    )
    assert result.returncode != 0
    run_id = result.stdout.strip().splitlines()[-1]

    loaded_store = RunStore(tmp_path / ".bfdiag")
    record = loaded_store.load(run_id)
    assert record.status == "failed"
    assert "RuntimeError: kaboom" in record.error
    assert record.finished_at is not None


# --- bf CLI dispatcher ----------------------------------------------------


def test_cli_discovers_record_subcommands(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("QSR_BFDIAG_DIR", str(tmp_path / ".bfdiag"))
    parser = bfdiag_cli.build_parser()
    args = parser.parse_args(["ls"])
    assert args.func is not None


def test_cli_ls_show_diff_end_to_end(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setenv("QSR_BFDIAG_DIR", str(tmp_path / ".bfdiag"))
    cli_store = RunStore(tmp_path / ".bfdiag")
    a = _make_record(metrics={"acceptance_rate": 1.0})
    b = _make_record(metrics={"acceptance_rate": 0.687})
    cli_store.save(a)
    cli_store.save(b)

    assert bfdiag_cli.main(["ls", "-n", "5"]) == 0
    out = capsys.readouterr().out
    assert a.run_id in out and b.run_id in out

    assert bfdiag_cli.main(["show", a.run_id[:8]]) == 0
    out = capsys.readouterr().out
    assert "acceptance_rate: 1.0" in out

    exit_code = bfdiag_cli.main(["diff", a.run_id, b.run_id, "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_a"] == a.run_id
    assert payload["run_b"] == b.run_id
    assert exit_code in (0, 2)


def test_cli_no_subcommand_prints_help_and_returns_nonzero(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    monkeypatch.setenv("QSR_BFDIAG_DIR", str(tmp_path / ".bfdiag"))
    exit_code = bfdiag_cli.main([])
    assert exit_code == 1
    assert "usage" in capsys.readouterr().out.lower()
