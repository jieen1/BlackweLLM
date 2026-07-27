"""Is a measurement a property of the code, or of the allocator layout?

See notes/2026-07-27-allocator-sensitivity.md. At block_size=128 DFlash
acceptance took three values (0.452525 / 0.602564 / 0.675362) on one
prompt and one commit, decided by caching-allocator state alone;
block_size=64 was bit-identical throughout.
"""

from bfdiag.sensitivity.perturbations import build, known_names, parse
from bfdiag.sensitivity.verdict import Measurement, Verdict, format_table, judge

__all__ = [
    "Measurement",
    "Verdict",
    "build",
    "format_table",
    "judge",
    "known_names",
    "parse",
]
