"""Find CUDA tensors that only a Python reference cycle keeps alive.

Such a tensor is dead weight the cyclic collector will reclaim at an
unpredictable moment -- and on this engine that moment changes numerical
results, because freeing it changes where later allocations land.

2026-07-27: after engine construction this found a single 588.00 MiB
all-zero ``[100352, 3072]`` bf16 Parameter (exactly ``vocab_size x
hidden_size``, i.e. an unused lm_head) held by a cycle, reachable only
through a module ``_parameters`` dict. ``gc.collect()`` reclaiming it
moved DFlash acceptance at block_size=128 from 0.452525 to 0.675362.

GPU-only; torch is imported lazily so the module stays CPU-importable.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass


@dataclass(frozen=True)
class CycleTensor:
    nbytes: int
    shape: tuple[int, ...]
    dtype: str
    all_zero: bool | None
    referrer_types: tuple[str, ...]

    @property
    def mib(self) -> float:
        return self.nbytes / 2**20


def find_cycle_held_cuda_tensors(
    *, top: int = 10, check_zero_up_to_mib: float = 4096.0
) -> list[CycleTensor]:
    """Collect cycles with SAVEALL and report the CUDA tensors inside.

    Call this with automatic GC disabled from process start
    (``gc.disable()`` before importing torch), otherwise the collector
    may already have reclaimed the cycle and this returns nothing --
    which is itself the observation that the timing is nondeterministic.
    """
    import torch

    gc.set_debug(gc.DEBUG_SAVEALL)
    try:
        gc.collect()
        garbage = list(gc.garbage)
    finally:
        gc.garbage.clear()
        gc.set_debug(0)

    found: list[CycleTensor] = []
    for obj in garbage:
        try:
            if not (isinstance(obj, torch.Tensor) and obj.is_cuda):
                continue
            nbytes = obj.untyped_storage().nbytes()
        except Exception:  # noqa: BLE001 - odd objects break on attribute access
            continue
        all_zero = None
        if nbytes / 2**20 <= check_zero_up_to_mib:
            try:
                all_zero = bool(not obj.any().item())
            except Exception:  # noqa: BLE001
                all_zero = None
        refs = tuple(
            sorted({f"{type(r).__module__}.{type(r).__qualname__}" for r in gc.get_referrers(obj)})
        )[:6]
        found.append(
            CycleTensor(
                nbytes=nbytes,
                shape=tuple(obj.shape),
                dtype=str(obj.dtype),
                all_zero=all_zero,
                referrer_types=refs,
            )
        )
    found.sort(key=lambda t: -t.nbytes)
    return found[:top]


def format_report(items: list[CycleTensor]) -> str:
    if not items:
        return (
            "no cycle-held CUDA tensors found.\n"
            "If automatic GC was left enabled, it may simply have collected them "
            "already -- rerun with gc.disable() before importing torch."
        )
    lines = [f"{len(items)} cycle-held CUDA tensor(s), largest first:", ""]
    total = 0.0
    for t in items:
        total += t.mib
        zero = {True: "all-zero", False: "has data", None: "?"}[t.all_zero]
        lines.append(
            f"  {t.mib:10.2f} MiB  {str(t.shape):<24} {t.dtype:<16} {zero:<9} "
            f"refs={','.join(x.split('.')[-1] for x in t.referrer_types)}"
        )
    lines += ["", f"  total held by cycles: {total:.2f} MiB"]
    return "\n".join(lines)
