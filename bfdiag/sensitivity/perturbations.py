"""Named allocator perturbations applied immediately before a measurement.

On 2026-07-27 DFlash acceptance at ``block_size=128`` was found to take
three different values (0.452525 / 0.602564 / 0.675362) on the same
prompt, same config, same commit -- decided purely by caching-allocator
layout. ``block_size=64`` was bit-identical under the same perturbations.
These are the knobs that sweep produced, kept as a named, reusable set so
the finding stays reproducible instead of living in a scratch script.

Perturbation names are part of the record: a measurement without one is
not comparable to a measurement with one.
"""

from __future__ import annotations

import gc
import re
from collections.abc import Callable

# ``torch.cuda.empty_cache()`` is deliberately NOT offered. Once the
# DFlash/decode CUDA Graphs are captured it releases blocks the graphs
# still reference, and the next replay dies with
# ``CUDA error: an illegal memory access was encountered``. Observed
# 2026-07-27; see notes/2026-07-27-allocator-sensitivity.md.
FORBIDDEN = {"empty_cache"}

_PAD_RE = re.compile(r"^(pad|holdpad)(\d+)$")


def known_names() -> tuple[str, ...]:
    """The fixed names; ``pad<N>``/``holdpad<N>`` take any MiB size."""
    return ("none", "gc", "reset", "gc+reset", "pad<N>", "holdpad<N>")


def parse(name: str) -> tuple[str, int | None]:
    """Split a perturbation name into (kind, mib). Raises on unknown names."""
    if name in FORBIDDEN:
        raise ValueError(
            f"{name!r} is unsafe after CUDA Graph capture: it frees blocks the "
            "captured graphs still point at, and the next replay hits an "
            "illegal memory access"
        )
    if name in ("none", "gc", "reset", "gc+reset"):
        return name, None
    m = _PAD_RE.match(name)
    if m:
        return m.group(1), int(m.group(2))
    raise ValueError(f"unknown perturbation {name!r}; known: {known_names()}")


def build(name: str, *, backend=None, slot: int = 0, holder: list | None = None) -> Callable:
    """Return a zero-arg callable applying ``name``.

    ``backend`` is only needed for the reset variants; ``holder`` receives
    the tensor for ``holdpad<N>`` so it stays alive past the call.
    """
    kind, mib = parse(name)

    if kind == "none":
        return lambda: None
    if kind == "gc":
        return gc.collect
    if kind in ("reset", "gc+reset"):
        if backend is None:
            raise ValueError(f"perturbation {name!r} needs a backend")

        def _reset() -> None:
            if kind == "gc+reset":
                gc.collect()
            backend.reset_slot(slot)

        return _reset

    def _pad() -> None:
        import torch

        t = torch.empty(int(mib) * 2**20, dtype=torch.uint8, device="cuda")
        if kind == "holdpad":
            if holder is None:
                raise ValueError("holdpad needs a holder list to keep the tensor alive")
            holder.append(t)
        else:
            del t

    return _pad
