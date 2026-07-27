"""CPU-only tests for ``bfdiag.determinism`` (the QSR_FORCE_SYNC /
QSR_DETERMINISTIC single source of truth), its wiring into
``bfdiag.trace.ring``'s hot path, ``bfdiag.record.fingerprint``'s run
fingerprint, and ``bfdiag.record.differ``'s comparability gate.

Hard rule for every test in this file (see the task brief this implements):
no test ever lets a real ``torch.cuda.*`` call reach actual hardware. Every
test that needs to observe "did synchronize get called" replaces
``determinism.torch`` with a small fake object (``_FakeTorch`` below) rather
than monkeypatching individual ``torch.cuda.*`` attributes on the real
module -- this sandbox has no GPU and must never probe for one, even
indirectly via ``torch.cuda.is_available()``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import timeit
from pathlib import Path

import pytest

from bfdiag import det_cli, determinism
from bfdiag.record import fingerprint
from bfdiag.record.differ import check_comparability
from bfdiag.record.schema import Fingerprint, RunRecord
from bfdiag.trace import events
from bfdiag.trace.ring import RoundRing

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Every env var any bundle item (or `bf determinism env`) might mutate.
# Restored after every test in this file regardless of pass/fail, since the
# production code mutates os.environ directly (not through monkeypatch), so
# monkeypatch's own auto-revert doesn't see those writes.
_MUTATED_ENV_VARS = (
    "CUBLAS_WORKSPACE_CONFIG",
    determinism.SEED_ENV,
    determinism.AUTOTUNE_CACHE_ENV,
    determinism.SPARKINFER_MOE_DETERMINISTIC_ENV,
    *determinism.CUDA_GRAPH_ENV_VARS,
)


@pytest.fixture(autouse=True)
def _restore_mutated_env():
    saved = {name: os.environ.get(name) for name in _MUTATED_ENV_VARS}
    yield
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


class _FakeCuda:
    def __init__(self, available: bool) -> None:
        self._available = available
        self.sync_calls = 0

    def is_available(self) -> bool:
        return self._available

    def synchronize(self) -> None:
        self.sync_calls += 1


class _FakeTorch:
    """Stands in for the real ``torch`` module inside ``determinism.py`` so
    tests can drive every branch (CUDA available/unavailable, deterministic
    flag on/off) without ever touching real hardware."""

    def __init__(self, *, cuda_available: bool = True) -> None:
        self.cuda = _FakeCuda(cuda_available)
        self._deterministic_enabled = False
        self.manual_seed_calls: list[int] = []
        self.use_deterministic_algorithms_calls: list[bool] = []

    def use_deterministic_algorithms(self, flag: bool) -> None:
        self.use_deterministic_algorithms_calls.append(bool(flag))
        self._deterministic_enabled = bool(flag)

    def are_deterministic_algorithms_enabled(self) -> bool:
        return self._deterministic_enabled

    def manual_seed(self, seed: int) -> None:
        self.manual_seed_calls.append(seed)


class _FakeNumpyRandom:
    def __init__(self) -> None:
        self.seed_calls: list[int] = []

    def seed(self, value: int) -> None:
        self.seed_calls.append(value)


class _FakeNumpy:
    def __init__(self) -> None:
        self.random = _FakeNumpyRandom()


# --------------------------------------------------------------------------
# 1. Sync-point gating: QSR_FORCE_SYNC=1 syncs at every mark; =0 never does;
#    CUDA-unavailable auto-degrades to a no-op either way.
# --------------------------------------------------------------------------


class TestForceSyncGating:
    def test_maybe_sync_off_by_default_never_calls_synchronize(self, monkeypatch):
        fake = _FakeTorch(cuda_available=True)
        monkeypatch.setattr(determinism, "torch", fake)
        monkeypatch.setattr(determinism, "FORCE_SYNC", False)

        assert determinism.maybe_sync() is False
        assert fake.cuda.sync_calls == 0

    def test_maybe_sync_on_calls_synchronize_when_cuda_available(self, monkeypatch):
        fake = _FakeTorch(cuda_available=True)
        monkeypatch.setattr(determinism, "torch", fake)
        monkeypatch.setattr(determinism, "FORCE_SYNC", True)

        assert determinism.maybe_sync() is True
        assert fake.cuda.sync_calls == 1
        # idempotent-in-spirit: calling again syncs again (it's per-call, not
        # a one-shot latch)
        determinism.maybe_sync()
        assert fake.cuda.sync_calls == 2

    def test_maybe_sync_on_but_no_cuda_is_a_no_op(self, monkeypatch):
        fake = _FakeTorch(cuda_available=False)
        monkeypatch.setattr(determinism, "torch", fake)
        monkeypatch.setattr(determinism, "FORCE_SYNC", True)

        assert determinism.maybe_sync() is False
        assert fake.cuda.sync_calls == 0

    def test_maybe_sync_on_but_torch_not_installed_is_a_no_op(self, monkeypatch):
        monkeypatch.setattr(determinism, "torch", None)
        monkeypatch.setattr(determinism, "FORCE_SYNC", True)

        assert determinism.maybe_sync() is False  # must not raise (no .cuda to call)

    def test_ring_mark_and_begin_round_sync_when_force_sync_on(self, monkeypatch):
        from bfdiag.trace import ring as ring_module

        fake = _FakeTorch(cuda_available=True)
        monkeypatch.setattr(determinism, "torch", fake)
        monkeypatch.setattr(determinism, "FORCE_SYNC", True)

        ring = RoundRing(4, use_cuda=False)
        row = ring.begin_round(slot=0, kv_len_before=10)
        assert fake.cuda.sync_calls == 1  # begin_round's own sync

        ring.mark(row, events.PHASE_VERIFY)
        assert fake.cuda.sync_calls == 2
        ring.mark(row, events.PHASE_COMMIT)
        assert fake.cuda.sync_calls == 3

        # finish_round calls self.mark() as its first action -> inherits the
        # sync without a separate explicit call.
        ring.finish_round(
            row,
            events.PHASE_DRAFT,
            path=events.Path.CG_REPLAY,
            cg_miss_reason=events.CgMissReason.NONE,
            draft_tokens_n=15,
            accepted_n=15,
            reject_position=-1,
            bonus_token=1,
        )
        assert fake.cuda.sync_calls == 4

        # ring_module's module-level `determinism` binding is the same
        # object we patched (import bfdiag.determinism as determinism in
        # ring.py) -- confirm the free-function API sees it too.
        assert ring_module.determinism is determinism

    def test_ring_never_syncs_when_force_sync_off(self, monkeypatch):
        fake = _FakeTorch(cuda_available=True)
        monkeypatch.setattr(determinism, "torch", fake)
        monkeypatch.setattr(determinism, "FORCE_SYNC", False)

        ring = RoundRing(4, use_cuda=False)
        row = ring.begin_round(slot=0, kv_len_before=10)
        ring.mark(row, events.PHASE_VERIFY)
        ring.finish_round(
            row,
            events.PHASE_DRAFT,
            path=events.Path.CG_REPLAY,
            cg_miss_reason=events.CgMissReason.NONE,
            draft_tokens_n=15,
            accepted_n=15,
            reject_position=-1,
            bonus_token=1,
        )
        assert fake.cuda.sync_calls == 0

    def test_ring_force_sync_on_but_no_cuda_does_not_crash(self, monkeypatch):
        fake = _FakeTorch(cuda_available=False)
        monkeypatch.setattr(determinism, "torch", fake)
        monkeypatch.setattr(determinism, "FORCE_SYNC", True)

        ring = RoundRing(4, use_cuda=False)
        row = ring.begin_round(slot=0, kv_len_before=10)
        ring.mark(row, events.PHASE_VERIFY)
        ring.finish_round(
            row,
            events.PHASE_DRAFT,
            path=events.Path.CG_REPLAY,
            cg_miss_reason=events.CgMissReason.NONE,
            draft_tokens_n=15,
            accepted_n=15,
            reject_position=-1,
            bonus_token=1,
        )
        assert fake.cuda.sync_calls == 0  # degraded to no-op, no crash


# --------------------------------------------------------------------------
# 2. Zero-overhead when off (complements test_bfdiag_ring.py's own
#    TestDisabledPathOverhead, which is unaffected since ring.mark()/
#    begin_round() are never even called when QSR_TRACE=0 -- the caller
#    guards with `if TRACE_ENABLED:` before entering this module at all).
# --------------------------------------------------------------------------


class TestZeroOverheadWhenOff:
    def test_maybe_sync_disabled_short_circuit_is_cheap(self):
        assert determinism.FORCE_SYNC is False, (
            "QSR_FORCE_SYNC was set to 1 in this test environment; this "
            "microbenchmark requires the default (off) state"
        )
        number = 200_000
        best = min(timeit.repeat(determinism.maybe_sync, repeat=5, number=number))
        per_call_ns = (best / number) * 1e9
        assert per_call_ns < 100.0, (
            f"maybe_sync() disabled-path cost {per_call_ns:.1f}ns exceeds the 100ns budget"
        )

    def test_ring_mark_with_force_sync_off_stays_cheap(self, monkeypatch):
        monkeypatch.setattr(determinism, "FORCE_SYNC", False)
        ring = RoundRing(8, use_cuda=False)
        row = ring.begin_round(0, 0)

        def do_mark() -> None:
            ring.mark(row, events.PHASE_VERIFY)

        number = 50_000
        best = min(timeit.repeat(do_mark, repeat=5, number=number))
        per_call_ns = (best / number) * 1e9
        # Generous budget: this measures the *whole* mark() call (Timeline
        # record + array writes), not just the new FORCE_SYNC guard -- the
        # point is proving the guard didn't introduce anything expensive.
        assert per_call_ns < 2000.0, (
            f"RoundRing.mark() with FORCE_SYNC off costs {per_call_ns:.1f}ns/call"
        )


# --------------------------------------------------------------------------
# 3. apply() idempotency + honest skip reporting (CUDA unavailable).
# --------------------------------------------------------------------------


class TestApplyBundle:
    def test_apply_off_reports_not_enabled_for_every_mutable_item(self, monkeypatch):
        fake_torch = _FakeTorch(cuda_available=True)
        monkeypatch.setattr(determinism, "torch", fake_torch)
        monkeypatch.setattr(determinism, "numpy", _FakeNumpy())

        report = determinism.apply(deterministic=False, force_sync=False, mutate=True)
        assert report.deterministic is False
        assert report.force_sync is False
        by_name = {item.name: item for item in report.bundle}
        assert by_name["torch_deterministic_algorithms"].status == "not_enabled"
        assert by_name["seed_all"].status == "not_enabled"
        assert by_name["autotune_cache"].status == "not_enabled"
        assert by_name["cuda_graph_disable"].status == "not_enabled"
        # observational item is unconditional, regardless of `deterministic`
        assert by_name["sparkinfer_moe_deterministic_output"].status == "observed_only"
        # nothing was actually mutated
        assert fake_torch.use_deterministic_algorithms_calls == []
        assert fake_torch.manual_seed_calls == []

    def test_apply_on_mutates_and_reports_applied(self, monkeypatch, tmp_path):
        fake_torch = _FakeTorch(cuda_available=True)
        fake_numpy = _FakeNumpy()
        monkeypatch.setattr(determinism, "torch", fake_torch)
        monkeypatch.setattr(determinism, "numpy", fake_numpy)
        monkeypatch.setattr(determinism, "default_autotune_cache_dir", lambda: tmp_path / "cache")

        report = determinism.apply(deterministic=True, seed=42, mutate=True)

        by_name = {item.name: item for item in report.bundle}
        assert by_name["torch_deterministic_algorithms"].status == "applied"
        assert fake_torch.use_deterministic_algorithms_calls == [True]
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == determinism.CUBLAS_WORKSPACE_CONFIG_VALUE

        assert by_name["seed_all"].status == "applied"
        assert fake_torch.manual_seed_calls == [42]
        assert fake_numpy.random.seed_calls == [42]

        assert by_name["autotune_cache"].status == "applied"
        assert (tmp_path / "cache").is_dir()
        assert os.environ[determinism.AUTOTUNE_CACHE_ENV] == str(tmp_path / "cache")

        # cuda_graph_disable stays off: disable_cuda_graph defaults to False
        assert by_name["cuda_graph_disable"].status == "not_enabled"

    def test_apply_is_idempotent(self, monkeypatch, tmp_path):
        fake_torch = _FakeTorch(cuda_available=True)
        monkeypatch.setattr(determinism, "torch", fake_torch)
        monkeypatch.setattr(determinism, "numpy", _FakeNumpy())
        monkeypatch.setattr(determinism, "default_autotune_cache_dir", lambda: tmp_path / "cache")

        first = determinism.apply(deterministic=True, seed=7, mutate=True)
        second = determinism.apply(deterministic=True, seed=7, mutate=True)

        assert first.to_dict() == second.to_dict()
        # the underlying mutation itself doesn't error or diverge on replay
        assert fake_torch.use_deterministic_algorithms_calls == [True, True]

    def test_apply_disable_cuda_graph_skips_honestly_when_cuda_unavailable(self, monkeypatch):
        fake_torch = _FakeTorch(cuda_available=False)
        monkeypatch.setattr(determinism, "torch", fake_torch)
        monkeypatch.setattr(determinism, "numpy", _FakeNumpy())

        report = determinism.apply(deterministic=True, disable_cuda_graph=True, mutate=True)
        item = {i.name: i for i in report.bundle}["cuda_graph_disable"]

        assert item.status == "skipped_no_cuda"  # honest skip, not a fake "applied"
        for name in determinism.CUDA_GRAPH_ENV_VARS:
            assert os.environ.get(name) != "0"  # nothing was actually set

    def test_apply_disable_cuda_graph_applies_when_cuda_available(self, monkeypatch):
        fake_torch = _FakeTorch(cuda_available=True)
        monkeypatch.setattr(determinism, "torch", fake_torch)
        monkeypatch.setattr(determinism, "numpy", _FakeNumpy())

        report = determinism.apply(deterministic=True, disable_cuda_graph=True, mutate=True)
        item = {i.name: i for i in report.bundle}["cuda_graph_disable"]

        assert item.status == "applied"
        for name in determinism.CUDA_GRAPH_ENV_VARS:
            assert os.environ[name] == "0"

    def test_apply_reports_skipped_when_torch_not_installed(self, monkeypatch):
        monkeypatch.setattr(determinism, "torch", None)
        report = determinism.apply(deterministic=True, mutate=True)
        item = {i.name: i for i in report.bundle}["torch_deterministic_algorithms"]
        assert item.status == "skipped_no_torch"

    def test_mutate_false_never_mutates(self, monkeypatch):
        fake_torch = _FakeTorch(cuda_available=True)
        monkeypatch.setattr(determinism, "torch", fake_torch)
        monkeypatch.setattr(determinism, "numpy", _FakeNumpy())

        determinism.apply(deterministic=True, mutate=False)

        assert fake_torch.use_deterministic_algorithms_calls == []
        assert fake_torch.manual_seed_calls == []

    def test_sparkinfer_item_observes_current_env_without_setting_it(self, monkeypatch):
        monkeypatch.delenv(determinism.SPARKINFER_MOE_DETERMINISTIC_ENV, raising=False)
        report = determinism.apply(deterministic=True, mutate=True)
        item = {i.name: i for i in report.bundle}["sparkinfer_moe_deterministic_output"]
        assert item.status == "observed_only"
        assert "unset" in item.detail
        # bfdiag must never set this itself
        assert determinism.SPARKINFER_MOE_DETERMINISTIC_ENV not in os.environ

        monkeypatch.setenv(determinism.SPARKINFER_MOE_DETERMINISTIC_ENV, "1")
        report2 = determinism.apply(deterministic=True, mutate=True)
        item2 = {i.name: i for i in report2.bundle}["sparkinfer_moe_deterministic_output"]
        assert "1" in item2.detail


# --------------------------------------------------------------------------
# Warning on import when QSR_FORCE_SYNC=1 (subprocess -- avoids in-process
# module-reload trickery and any risk of leaking global state into other
# tests in this session).
# --------------------------------------------------------------------------


class TestForceSyncWarning:
    def test_warns_on_import_when_enabled(self):
        env = dict(os.environ)
        env["QSR_FORCE_SYNC"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", "import bfdiag.determinism"],
            cwd=str(_REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "QSR_FORCE_SYNC=1" in result.stderr
        assert "NOT valid performance data" in result.stderr

    def test_does_not_warn_by_default(self):
        env = dict(os.environ)
        env.pop("QSR_FORCE_SYNC", None)
        result = subprocess.run(
            [sys.executable, "-c", "import bfdiag.determinism"],
            cwd=str(_REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "QSR_FORCE_SYNC" not in result.stderr


# --------------------------------------------------------------------------
# 4/5. fingerprint.py wiring: never mutates, always observes; differ.py
#      flags force_sync drift as NOT COMPARABLE.
# --------------------------------------------------------------------------


class TestFingerprintWiring:
    def test_capture_determinism_never_mutates(self, monkeypatch):
        calls: list[dict] = []
        real_apply = determinism.apply

        def spy_apply(**kwargs):
            calls.append(kwargs)
            return real_apply(**kwargs)

        monkeypatch.setattr(fingerprint._determinism, "apply", spy_apply)
        monkeypatch.setattr(determinism, "torch", _FakeTorch(cuda_available=True))

        result = fingerprint.capture_determinism()
        assert calls == [{"mutate": False}]
        assert result["force_sync"] is False
        assert result["deterministic"] is False
        assert "bundle" in result

    def test_capture_determinism_never_raises(self, monkeypatch):
        def boom(**kwargs):
            raise RuntimeError("simulated failure")

        monkeypatch.setattr(fingerprint._determinism, "apply", boom)
        assert fingerprint.capture_determinism() == {}

    def test_capture_folds_determinism_into_extra(self, monkeypatch):
        monkeypatch.setattr(fingerprint.shutil, "which", lambda _name: None)
        result = fingerprint.capture()
        assert "determinism" in result.extra
        assert result.extra["determinism"]["force_sync"] is False


class TestDifferComparability:
    def _record(self, run_id: str, *, force_sync: bool) -> RunRecord:
        return RunRecord(
            run_id=run_id,
            started_at="2026-07-27T10:00:00+00:00",
            fingerprint=Fingerprint(extra={"determinism": {"force_sync": force_sync}}),
        )

    def test_force_sync_drift_is_flagged_not_comparable(self):
        run_a = self._record("syncA0000001", force_sync=False)
        run_b = self._record("syncB0000001", force_sync=True)

        breaks = check_comparability(run_a, run_b)
        assert [b.path for b in breaks] == ["fingerprint.extra.determinism.force_sync"]

    def test_same_force_sync_is_comparable(self):
        run_a = self._record("syncC0000001", force_sync=True)
        run_b = self._record("syncD0000001", force_sync=True)

        breaks = check_comparability(run_a, run_b)
        assert breaks == []


# --------------------------------------------------------------------------
# bf determinism show / env (det_cli.py's register(), mounted standalone --
# see this module's docstring on why `bf`'s own auto-discovery doesn't wire
# it up yet).
# --------------------------------------------------------------------------


class TestDetCli:
    def _run(self, argv: list[str]) -> int:
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command", required=True)
        det_cli.register(subparsers)
        args = parser.parse_args(argv)
        return args.func(args)

    def test_show_renders_text(self, capsys, monkeypatch):
        monkeypatch.setattr(determinism, "torch", _FakeTorch(cuda_available=True))
        code = self._run(["determinism", "show"])
        assert code == 0
        out = capsys.readouterr().out
        assert "QSR_FORCE_SYNC:" in out
        assert "QSR_DETERMINISTIC:" in out
        assert "seed_all" in out

    def test_show_json_is_valid_json(self, capsys, monkeypatch):
        import json

        monkeypatch.setattr(determinism, "torch", _FakeTorch(cuda_available=True))
        code = self._run(["determinism", "show", "--json"])
        assert code == 0
        parsed = json.loads(capsys.readouterr().out)
        assert "bundle" in parsed
        assert "seed_all" in parsed["bundle"]

    def test_env_nothing_requested_is_an_error(self, capsys):
        code = self._run(["determinism", "env"])
        assert code == 1
        assert "nothing requested" in capsys.readouterr().err

    def test_env_deterministic_and_force_sync_prints_exports(self, capsys):
        code = self._run(["determinism", "env", "--deterministic", "--force-sync"])
        assert code == 0
        out = capsys.readouterr().out
        assert "export QSR_DETERMINISTIC=1" in out
        assert "export QSR_FORCE_SYNC=1" in out
        assert "export CUBLAS_WORKSPACE_CONFIG=" in out
        assert "export QSR_SEED=0" in out
        assert f"export {determinism.AUTOTUNE_CACHE_ENV}=" in out

    def test_env_disable_cuda_graph_flag(self, capsys):
        code = self._run(["determinism", "env", "--disable-cuda-graph"])
        assert code == 0
        out = capsys.readouterr().out
        for name in determinism.CUDA_GRAPH_ENV_VARS:
            assert f"export {name}=0" in out
