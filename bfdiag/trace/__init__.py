"""Flight recorder: always-on, ring-buffered per-round trace events for the
DFlash/Laguna decode loop.

Submodules are intentionally import-light at the package level (no torch/
numpy import here) so ``import bfdiag.trace`` never pays for a heavy
dependency the caller may not need yet -- ``ring``/``timing`` import numpy-
free (stdlib ``array``) and lazily probe for torch, so the whole subpackage
stays importable on a CPU-only, torch-less box (see notes/2026-07-27-
bfdiag-flight-recorder.md).
"""

from __future__ import annotations
