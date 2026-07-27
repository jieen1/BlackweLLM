"""Oracle-vs-engine per-layer activation divergence scanning (bfdiag plan 5).

Ties together ``oracle/capture_hooks.py`` (activation capture),
``oracle/comparator.py`` (numeric comparison primitives), a tiered on-disk
oracle activation cache (``cache.py``), a pure divergence scanner
(``scan.py``), depth-aware composite thresholds (``thresholds.py``), and a
human/machine-readable report (``report.py``) exposed as ``bf divergence``
via ``cli.py``.

See notes/2026-07-27-bfdiag-oracle-divergence.md for the full design
rationale, cache size/tier trade-off analysis, and the real-module-name
evidence this package's defaults are grounded in.
"""

from __future__ import annotations
