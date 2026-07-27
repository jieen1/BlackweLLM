"""Minimal, local stand-in for the not-yet-landed ``bfprobe.bus`` module.

``bfprobe/bus.py`` is owned by the P1 agent (probe bus: global enable flag +
tiered ``emit_*`` write API) and may not exist yet in any given worktree. This
stub reproduces just enough of its public surface --
``TIER_EVENT``/``TIER_SIGNATURE``/``TIER_TENSOR``, ``PROBE_ENABLED``, and
``emit_signature`` -- so that ``bfprobe/signature.py`` (and its tests) can be
imported and exercised standalone via::

    try:
        from bfprobe.bus import TIER_SIGNATURE, PROBE_ENABLED
    except ImportError:
        from bfprobe._bus_stub import TIER_SIGNATURE, PROBE_ENABLED

Once the real ``bus.py`` lands, the ``try`` branch wins automatically and this
module is only ever reached by direct unit tests of the stub itself. No
behavior here should be load-bearing for production; when the two packages
are merged, whoever owns ``bus.py`` reconciles both.

``PROBE_ENABLED`` defaults to ``False`` so merely importing this stub (e.g.
transitively, before ``bus.py`` exists) never turns probing on by accident.
"""

from __future__ import annotations

#: Tier identifiers, mirrored from the bus contract in
#: notes/2026-07-27-probe-system-design-and-plan.md section 3: T0 host
#: events, T1 GPU signatures (this package's job), T2 full GPU tensors.
TIER_EVENT, TIER_SIGNATURE, TIER_TENSOR = 0, 1, 2

#: Module-level flag every call site is expected to check with a single
#: ``if`` before doing any probe work. Kept as a plain module attribute (not
#: a function) so the disabled-path cost is one attribute load + branch.
PROBE_ENABLED: bool = False


def emit_signature(site_id: int, tensor: object) -> None:
    """No-op placeholder matching the real bus's T1 write API.

    The real ``bus.emit_signature`` is expected to reduce ``tensor`` (via
    ``bfprobe.reduce``) and forward the result into a live
    ``bfprobe.signature.SignatureRing``. This stub does neither -- it exists
    only so ``signature.py``'s ``try/except ImportError`` against
    ``bfprobe.bus`` has something importable to fall back to. Production
    code should never end up calling this stub with ``PROBE_ENABLED=True``;
    tests that want ring-recording behavior should drive
    ``bfprobe.signature.SignatureRing`` / ``bfprobe.signature.emit`` directly
    instead of going through this function.
    """
    del site_id, tensor
