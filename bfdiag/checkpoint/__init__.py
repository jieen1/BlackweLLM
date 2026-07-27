"""Active checkpoint / restore for a Laguna+DFlash logical slot -- skip the
repeated 64K prefill when re-investigating the same bug.

This is NOT the same thing as the (separately planned) P3 "pre-trigger
freeze" from ``notes/2026-07-27-probe-system-design-and-plan.md`` sec 5:
P3 is *passive* (a background trigger function decides, post hoc, to freeze
a rolling GPU ring buffer of the last ~50 decode rounds when an anomaly
fires) and covers *decode-phase telemetry*. This package is *active* (a
human explicitly picks a moment -- almost always "right after this slot's
64K prefill just finished" -- and says "save this exact spot") and covers
*the whole reproducible slot state*, so a brand new daemon process, days
later, can restore it and resume generation without re-running the prefill
at all. See ``notes/2026-07-27-bfdiag-checkpoint-restore.md`` sec "与 P3
的区别" for the full comparison.

Submodules:

* :mod:`bfdiag.checkpoint.state` -- the declarative manifest of exactly what
  "one slot's complete state" consists of (names, source objects, shapes,
  dtypes, code citations). Read this file first; it is the spec this whole
  package implements.
* :mod:`bfdiag.checkpoint.store` -- safetensors + JSON-manifest persistence,
  fingerprinting, and fingerprint-compatibility checking.
* :mod:`bfdiag.checkpoint.restore` -- writes a saved checkpoint back into a
  live engine's target slot and hands off to :mod:`verify`.
* :mod:`bfdiag.checkpoint.verify` -- the safety valve: a deterministic
  greedy replay whose token-for-token output must match the baseline
  recorded at save time, or restore refuses to hand the slot back.
* :mod:`bfdiag.checkpoint.cli` -- ``bf checkpoint save|list|show|restore|rm``.
* :mod:`bfdiag.checkpoint.testing` -- ``FakeBackend``/``FakeDFlashEngine``,
  pure-Python/CPU-tensor stand-ins used by every test in this package (this
  whole package was written and reviewed against the current on-disk
  ``runtime/`` source, per this task's hard no-GPU constraint, but the real
  ``LagunaBackend``/``DFlashEngine`` code path has never been executed --
  see the notes file's GPU-validation TODO list).
"""

from __future__ import annotations
