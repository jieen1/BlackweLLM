"""Host-side invariant assertions, gated by ``QSR_ASSERT_LEVEL`` (0 default
off / 1 cheap / 2 also-expensive). See ``bfdiag.invariants.registry`` for the
``check()`` API and ``bfdiag.invariants.checks`` for the concrete invariants
wired into the DFlash/Laguna decode loop.
"""

from __future__ import annotations
