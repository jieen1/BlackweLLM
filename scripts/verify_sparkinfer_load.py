"""Ground-truth check: which SparkInfer checkout is actually loaded in THIS
process right now, and is the Laguna analytic-decode gate open for it.

Why this exists (see runtime/backends/_sparkinfer_import.py's docstring and
notes/2026-08-01-sparkinfer-patch-recovery-and-repro.md §7.2 for the full
history): `BF_SPARKINFER_PATH` used to be honored ONLY by
`laguna_sparkinfer_attn.py`/`laguna_sparkinfer_moe.py`, each inserting it
into `sys.path` right before their own `from sparkinfer...` import -- but
`laguna.py`'s `_patch_moe_sparkinfer` did an earlier, uncontrolled direct
`from sparkinfer.moe.fused_moe._impl import ...`, which was the real FIRST
touch of the `sparkinfer` name on the actual Laguna startup path. That made
the env var silently do nothing, no error, no warning. All three call
sites (and this script) now route through the single controlled resolver
`runtime.backends._sparkinfer_import.ensure_sparkinfer_path()`, which is
idempotent and raises loudly instead of silently no-opping if something
*else* imported `sparkinfer` first. This script calls it explicitly, in the
same "resolve, then import" order `laguna.py` uses, so a pass here is
evidence about the real startup path, not just about this script's own
import order.

Never assume the env var took effect just because this script didn't
error -- it also asserts the loaded path matches what was requested, and
fails loudly (non-zero exit via the raised exception) if it doesn't.

Two ways to run it:

  Cold (no model, no daemon, ~2s -- just needs a CUDA-capable process). Run
  from the repo root with PYTHONPATH=. -- this is a plain `python
  scripts/foo.py` invocation, so `sys.path[0]` is `scripts/`, not the repo
  root; without PYTHONPATH this hits the *other* worktree trap (see "bf and
  worktrees" in docs/diagnostics-guide.md) and imports `runtime` from
  whichever checkout the venv's pip-editable install happens to be pinned
  to, not this one:
    cd <this worktree> && BF_SPARKINFER_PATH=/path/to/checkout PYTHONPATH=. \\
        ~/.venvs/vllm/bin/python scripts/verify_sparkinfer_load.py

  Inside the warm bfdiag daemon (confirms what the *running* daemon loaded,
  which is what actually matters for a benchmark run -- the env var must be
  set before `bf daemon start`, since switching SparkInfer is a load-time
  change and does not take effect via `bf exec` alone). `bf` resolves its
  own worktree correctly on its own (see docs/diagnostics-guide.md), so no
  PYTHONPATH juggling is needed here:
    bf exec scripts/verify_sparkinfer_load.py

The three shapes checked mirror sparkinfer commit 7a1d69d's gating change
(`_is_laguna_fp8_gqa6_analytic_decode_graph`, generalized from the exact
TP=2 shape 24q/4kv/page128 to any `gqa_group_size == 6`,
`page_size in (64, 128)`): Laguna's real TP=1 FULL-attention shape at both
page sizes should gate OPEN once patched; the SWA shape (gqa=9) is designed
to stay on the generic path regardless.
"""

# NOTE: deliberately no `from __future__ import annotations` -- `bf exec
# <file>` (bfdiag/daemon/client.py Client.exec_file) prepends a
# `__file__ = '...'` assignment line before the script's own source, which
# pushes any `from __future__ import ...` off line 1 and trips Python's
# "must occur at the beginning of the file" rule. Found the hard way running
# this exact script through `bf exec`.

import pathlib
import subprocess

from runtime.backends._sparkinfer_import import ensure_sparkinfer_path

_requested_sparkinfer_path = ensure_sparkinfer_path()

# Wrapped in try/except (a `try:` block is a recognized conditional-import
# idiom, not flagged by E402) purely so `ensure_sparkinfer_path()` above can
# run as an ordinary statement between two import groups without tripping
# "module level import not at top of file" -- the whole point of this
# script is that path resolution MUST happen before this import, so the two
# genuinely cannot be reordered.
try:
    import sparkinfer
    import torch
    from sparkinfer.attention.paged.planner import _is_laguna_fp8_gqa6_analytic_decode_graph
except ImportError as exc:
    raise ImportError(
        f"sparkinfer not importable from {_requested_sparkinfer_path!r} "
        "(BF_SPARKINFER_PATH or the default) -- see "
        "runtime/backends/_sparkinfer_import.py"
    ) from exc

_CASES = [
    # (label, num_q_heads, num_kv_heads, page_size, window_left)
    ("Laguna FULL  48q/8kv page64 ", 48, 8, 64, -1),
    ("Laguna FULL  48q/8kv page128", 48, 8, 128, -1),
    ("Laguna SWA   72q/8kv page64 ", 72, 8, 64, 511),
]


def _git(repo_dir: pathlib.Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_dir), *args], capture_output=True, text=True, check=False
    )
    return result.stdout.strip()


def main() -> dict:
    pkg_dir = pathlib.Path(sparkinfer.__file__).resolve().parent
    repo_dir = pkg_dir.parent
    sha = _git(repo_dir, "rev-parse", "--short", "HEAD")
    dirty = bool(_git(repo_dir, "status", "--porcelain"))

    print(f"BF_SPARKINFER_PATH requested = {_requested_sparkinfer_path}")
    print(f"sparkinfer.__file__ = {sparkinfer.__file__}")
    print(f"repo dir            = {repo_dir}")
    print(f"git HEAD            = {sha}{'  (DIRTY)' if dirty else ''}")
    if str(repo_dir) != _requested_sparkinfer_path:
        raise RuntimeError(
            f"BF_SPARKINFER_PATH={_requested_sparkinfer_path!r} was requested but "
            f"sparkinfer actually loaded from {repo_dir} -- the override did NOT "
            "take effect. This should be impossible after the "
            "_sparkinfer_import.ensure_sparkinfer_path() fix; see "
            "runtime/backends/_sparkinfer_import.py."
        )
    print("BF_SPARKINFER_PATH honored: loaded repo matches the requested path.")

    device = torch.device("cuda")
    gates = {}
    for label, num_q_heads, num_kv_heads, page_size, window_left in _CASES:
        gate_open = _is_laguna_fp8_gqa6_analytic_decode_graph(
            device=device,
            q_dtype=torch.bfloat16,
            kv_dtype=torch.float8_e4m3fn,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=128,
            head_dim_vo=128,
            page_size=page_size,
            batch=1,
            window_left=window_left,
        )
        gates[label.strip()] = gate_open
        print(f"  {label} -> analytic gate {'OPEN' if gate_open else 'closed'}")

    result = {
        "sparkinfer_file": str(sparkinfer.__file__),
        "sparkinfer_repo": str(repo_dir),
        "sparkinfer_head": sha,
        "sparkinfer_dirty": dirty,
        "gates": gates,
    }
    return result


# Run unconditionally at module scope (not gated on `__name__ == "__main__"`):
# `bf exec` compiles this file's source and execs it with __name__ forced to
# "__main__" in a throwaway namespace, then reports back whatever ends up
# bound to a top-level `result` name in that namespace. Plain
# `python scripts/verify_sparkinfer_load.py` also has __name__ == "__main__",
# so gating on it would just be dead weight here -- either way we want main()
# to run and `result` to end up bound.
result = main()
