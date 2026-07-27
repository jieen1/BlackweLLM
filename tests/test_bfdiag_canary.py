"""Unit + integration tests for bfdiag/daemon/canary.py -- CPU-only, uses
FakeEngineProvider throughout (no GPU/torch anywhere in this file).
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest

from bfdiag.daemon.canary import CanaryChecker
from bfdiag.daemon.client import Client
from bfdiag.daemon.provider import FakeEngineProvider
from bfdiag.daemon.server import Daemon


@pytest.fixture
def state_dir(tmp_path: Path) -> Path:
    return tmp_path / "state"


class TestCanaryChecker:
    def test_first_check_records_baseline(self, state_dir: Path):
        checker = CanaryChecker(state_dir, git_sha="abc123")
        fake = FakeEngineProvider()
        fake.load()

        result = checker.check(fake)

        assert result.ok is True
        assert "baseline recorded" in result.detail
        assert checker.baseline_path().exists()

    def test_repeat_check_on_clean_engine_matches(self, state_dir: Path):
        checker = CanaryChecker(state_dir, git_sha="abc123")
        fake = FakeEngineProvider()
        fake.load()

        first = checker.check(fake)
        second = checker.check(fake)

        assert first.ok and second.ok
        assert first.observed == second.observed

    def test_pollution_causes_mismatch(self, state_dir: Path):
        checker = CanaryChecker(state_dir, git_sha="abc123")
        fake = FakeEngineProvider()
        fake.load()

        checker.check(fake)  # records baseline
        fake.pollute()
        result = checker.check(fake)

        assert result.ok is False
        assert result.mismatch_at is not None
        assert "mismatch" in result.detail

    def test_reset_restores_clean_canary(self, state_dir: Path):
        checker = CanaryChecker(state_dir, git_sha="abc123")
        fake = FakeEngineProvider()
        fake.load()

        checker.check(fake)
        fake.pollute()
        assert checker.check(fake).ok is False

        fake.reset()
        assert checker.check(fake).ok is True

    def test_fingerprint_change_re_records_instead_of_flagging(self, state_dir: Path):
        checker_v1 = CanaryChecker(state_dir, git_sha="sha-v1")
        fake = FakeEngineProvider()
        fake.load()
        baseline = checker_v1.check(fake)
        assert baseline.ok is True

        # A genuine code/model change (different git sha) must not be
        # misreported as corruption -- it should just re-record.
        checker_v2 = CanaryChecker(state_dir, git_sha="sha-v2")
        result = checker_v2.check(fake)
        assert result.ok is True
        assert "baseline recorded" in result.detail

    def test_disabled_canary_always_ok_and_does_not_call_generate(self, state_dir: Path):
        checker = CanaryChecker(state_dir, enabled=False, git_sha="abc123")

        class ExplodingProvider:
            def generate(self, *args, **kwargs):
                raise AssertionError("generate() must not be called when canary is disabled")

            def describe(self):
                return {"model_revision": "n/a"}

        result = checker.check(ExplodingProvider())
        assert result.ok is True
        assert result.detail == "canary disabled"

    def test_reset_baseline_forces_re_record(self, state_dir: Path):
        checker = CanaryChecker(state_dir, git_sha="abc123")
        fake = FakeEngineProvider()
        fake.load()
        checker.check(fake)
        assert checker.baseline_path().exists()

        checker.reset_baseline()
        assert not checker.baseline_path().exists()

        result = checker.check(fake)
        assert "baseline recorded" in result.detail

    def test_length_mismatch_is_flagged(self, state_dir: Path):
        checker = CanaryChecker(state_dir, git_sha="abc123", steps=4)
        fake = FakeEngineProvider()
        fake.load()
        baseline_result = checker.check(fake)
        assert baseline_result.ok

        # Simulate a shorter observed sequence by hand-editing the stored
        # baseline to be longer than what generate() will now produce.
        import json

        stored = json.loads(checker.baseline_path().read_text())
        stored["tokens"] = [*stored["tokens"], 999]
        checker.baseline_path().write_text(json.dumps(stored))

        result = checker.check(fake)
        assert result.ok is False
        assert "length mismatch" in result.detail

    def test_git_sha_env_override(self, state_dir: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("QSR_BFD_GIT_SHA", "env-sha")
        checker = CanaryChecker(state_dir)
        fake = FakeEngineProvider()
        fake.load()
        checker.check(fake)
        import json

        stored = json.loads(checker.baseline_path().read_text())
        assert stored["fingerprint"] == "fake-v1:env-sha"


class TestCanaryFailureRestartIntegration:
    """The core safety path end to end: canary mismatch -> refuse exec ->
    mark TAINTED -> auto-restart -> daemon recovers -- entirely against
    FakeEngineProvider, per this task's hard no-GPU constraint."""

    def test_pollution_refuses_exec_then_restart_recovers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # Route bfdiag_dir() (and therefore the canary baseline file) into
        # an isolated per-test directory via the real env-var contract,
        # instead of the actual repo's .bfdiag/.
        monkeypatch.setenv("QSR_BFDIAG_DIR", str(tmp_path))
        socket_dir = Path(tempfile.mkdtemp())
        socket_path = socket_dir / "d.sock"
        load_counts: list[FakeEngineProvider] = []

        def factory() -> FakeEngineProvider:
            provider = FakeEngineProvider(model_revision="fake-canary-v1")
            load_counts.append(provider)
            return provider

        daemon = Daemon(
            provider_factory=factory,
            socket_path=socket_path,
            canary_enabled=True,
            restart_on_taint=True,
        )
        daemon.serve_in_background()
        client = Client(socket_path=socket_path, timeout_s=5.0)
        try:
            # First exec: canary has no baseline yet, records one, allowed.
            first = client.exec_code("result = 1")
            assert first.ok is True
            assert daemon.status()["state"] == "READY"

            # Pollute via the exec namespace, exactly as a careless
            # experiment would leave residual engine state behind.
            pollute_response = client.exec_code("provider.pollute()")
            assert pollute_response.ok is True  # the exec itself succeeds...

            # ...but the NEXT exec's canary pre-check must catch the
            # divergence and refuse to run the requested code at all.
            second = client.exec_code("result = 2")
            assert second.ok is False
            assert "canary mismatch" in (second.error or "")

            # Restart-on-taint means the daemon recovers automatically: a
            # fresh provider was constructed and loaded (dirty=0 again),
            # and it must match the ORIGINAL baseline (same model_revision
            # fingerprint) so normal service resumes without a new
            # baseline needing to be recorded.
            assert len(load_counts) == 2  # original + restarted instance
            status_after = client.status().result
            assert status_after["state"] == "READY"
            assert status_after["restart_count"] == 1

            third = client.exec_code("result = 3")
            assert third.ok is True
            assert third.result == 3
        finally:
            client.shutdown()
            time.sleep(0.3)

    def test_crash_on_reset_taints_and_restarts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("QSR_BFDIAG_DIR", str(tmp_path))
        socket_dir = Path(tempfile.mkdtemp())
        socket_path = socket_dir / "d.sock"

        def factory() -> FakeEngineProvider:
            return FakeEngineProvider(crash_on_reset=True)

        daemon = Daemon(
            provider_factory=factory,
            socket_path=socket_path,
            canary_enabled=False,  # isolate the reset-crash path specifically
            restart_on_taint=True,
        )
        daemon.serve_in_background()
        client = Client(socket_path=socket_path, timeout_s=5.0)
        try:
            response = client.reset()
            assert response.ok is False
            assert "reset() failed" in (response.error or "")

            status = client.status().result
            assert status["state"] == "READY"  # restarted with a fresh, non-crashing...
            assert status["restart_count"] == 1
        finally:
            client.shutdown()
            time.sleep(0.3)
