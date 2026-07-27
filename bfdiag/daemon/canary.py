"""Canary self-check: the core dirty-state safety net for the bfdiag warm
daemon.

Before every ``exec``, the daemon replays a fixed prompt through
``EngineProvider.generate()`` with greedy sampling for a fixed number of
steps and compares the resulting token sequence, POSITION BY POSITION,
against a recorded baseline. Any mismatch means some earlier experiment
left residual state behind (a KV cache row, a slot counter, a CUDA Graph
replay buffer, a prefix-cache hit that shouldn't have hit, ...) that
changes this "clean" run's output -- i.e. the daemon can no longer be
trusted to produce independent results, so the next experiment must not
run against it.

The baseline is recorded on first use (the first canary check after
``EngineProvider.load()``) and persisted to disk keyed by a fingerprint
(model revision + git SHA) so that a genuine model/code change is not
misreported as corruption -- see ``CanaryChecker._fingerprint``.

This module has no dependency on ``provider.py`` (it duck-types
``provider.generate()``/``provider.describe()``) and no dependency on
``server.py`` -- callers pass in the state directory explicitly, keeping
this fully unit-testable in isolation (see
``tests/test_bfdiag_canary.py``).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

#: Small, fixed "prompt" -- for the fake provider these are just ints; for
#: the real engine these must be valid token ids for whatever tokenizer is
#: loaded. Kept deliberately short so the canary is cheap to run before
#: every single exec.
DEFAULT_CANARY_PROMPT_IDS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
DEFAULT_CANARY_STEPS = 8

_BASELINE_FILENAME = "canary_baseline.json"


class _GeneratesGreedily(Protocol):
    def generate(
        self, prompt_ids: list[int], max_tokens: int, *, temperature: float = 0.0
    ) -> list[int]: ...

    def describe(self) -> dict[str, Any]: ...


@dataclass
class CanaryResult:
    """Outcome of one canary check."""

    ok: bool
    baseline: list[int] | None
    observed: list[int]
    mismatch_at: int | None
    detail: str


class CanaryChecker:
    """Fixed-prompt, fixed-step, greedy self-check with an on-disk,
    fingerprinted baseline.

    ``state_dir`` is the directory the baseline JSON file lives in (the
    caller -- ``server.py`` -- resolves this to
    ``${QSR_BFDIAG_DIR:-<repo>/.bfdiag}``; this class does not know or care
    about that convention, it just needs *a* directory).
    """

    def __init__(
        self,
        state_dir: str | Path,
        *,
        prompt_ids: tuple[int, ...] | list[int] = DEFAULT_CANARY_PROMPT_IDS,
        steps: int = DEFAULT_CANARY_STEPS,
        enabled: bool = True,
        git_sha: str | None = None,
    ) -> None:
        self._state_dir = Path(state_dir)
        self._prompt_ids = list(prompt_ids)
        self._steps = steps
        self._enabled = enabled
        self._git_sha_override = git_sha

    @property
    def enabled(self) -> bool:
        return self._enabled

    def baseline_path(self) -> Path:
        return self._state_dir / _BASELINE_FILENAME

    def check(self, provider: _GeneratesGreedily) -> CanaryResult:
        """Run the canary against ``provider``. Records a fresh baseline
        (and returns ok=True) the first time it sees a given fingerprint;
        otherwise compares against the stored one, position by position."""
        if not self._enabled:
            return CanaryResult(
                ok=True, baseline=None, observed=[], mismatch_at=None, detail="canary disabled"
            )

        fingerprint = self._fingerprint(provider)
        observed = provider.generate(list(self._prompt_ids), self._steps, temperature=0.0)

        stored = self._load_baseline()
        if stored is None or stored.get("fingerprint") != fingerprint:
            self._save_baseline(fingerprint, observed)
            return CanaryResult(
                ok=True,
                baseline=observed,
                observed=observed,
                mismatch_at=None,
                detail="baseline recorded (first run for this fingerprint)",
            )

        baseline_tokens: list[int] = stored["tokens"]
        for i, (expected, actual) in enumerate(zip(baseline_tokens, observed)):
            if expected != actual:
                return CanaryResult(
                    ok=False,
                    baseline=baseline_tokens,
                    observed=observed,
                    mismatch_at=i,
                    detail=f"token mismatch at step {i}: expected {expected}, got {actual}",
                )
        if len(baseline_tokens) != len(observed):
            return CanaryResult(
                ok=False,
                baseline=baseline_tokens,
                observed=observed,
                mismatch_at=min(len(baseline_tokens), len(observed)),
                detail=(
                    f"length mismatch: baseline has {len(baseline_tokens)} tokens, "
                    f"observed {len(observed)}"
                ),
            )
        return CanaryResult(
            ok=True, baseline=baseline_tokens, observed=observed, mismatch_at=None, detail="match"
        )

    def reset_baseline(self) -> None:
        """Delete the stored baseline, forcing the next ``check()`` to
        re-record it. Useful after an intentional model/code change that
        doesn't happen to change the fingerprint, or from a REPL/exec
        session while debugging the canary itself."""
        self.baseline_path().unlink(missing_ok=True)

    def _fingerprint(self, provider: _GeneratesGreedily) -> str:
        desc = provider.describe()
        model_revision = desc.get("model_revision", "unknown")
        return f"{model_revision}:{self._git_sha()}"

    def _git_sha(self) -> str:
        if self._git_sha_override is not None:
            return self._git_sha_override
        env_override = os.environ.get("QSR_BFD_GIT_SHA")
        if env_override:
            return env_override
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=2,
                cwd=Path(__file__).resolve().parents[2],
            )
        except (OSError, subprocess.SubprocessError):
            return "unknown"
        if result.returncode != 0:
            return "unknown"
        return result.stdout.strip() or "unknown"

    def _load_baseline(self) -> dict[str, Any] | None:
        path = self.baseline_path()
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def _save_baseline(self, fingerprint: str, tokens: list[int]) -> None:
        path = self.baseline_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "fingerprint": fingerprint,
            "tokens": tokens,
            "recorded_at": time.time(),
        }
        path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    import tempfile

    from bfdiag.daemon.provider import FakeEngineProvider

    with tempfile.TemporaryDirectory() as tmp:
        checker = CanaryChecker(tmp)
        fake = FakeEngineProvider()
        fake.load()

        first = checker.check(fake)
        print("first check (records baseline):", first)

        second = checker.check(fake)
        print("second check (clean, should match):", second.ok, second.detail)

        fake.pollute()
        third = checker.check(fake)
        print("third check (polluted, should mismatch):", third.ok, third.detail)
