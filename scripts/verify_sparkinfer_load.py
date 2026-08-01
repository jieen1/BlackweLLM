"""Ground-truth check: which SparkInfer checkout is actually loaded in THIS
process right now, and is the Laguna analytic-decode gate open for it.

Why this exists: `runtime/backends/laguna_sparkinfer_attn.py` and
`laguna_sparkinfer_moe.py` resolve `sparkinfer` via
`sys.path.insert(0, os.environ.get("BF_SPARKINFER_PATH", "/home/bot/project/sparkinfer"))`
followed by a lazy, function-scoped `import sparkinfer...`. Whether that env
var actually wins depends on import order and on nothing having imported
`sparkinfer` earlier in the process (import results are cached in
sys.modules; a later sys.path change cannot retroactively redirect an
already-imported module). Never assume the env var took effect -- check.

Two ways to run it:

  Cold (no model, no daemon, ~2s -- just needs a CUDA-capable process):
    BF_SPARKINFER_PATH=/path/to/checkout ~/.venvs/vllm/bin/python \\
        scripts/verify_sparkinfer_load.py

  Inside the warm bfdiag daemon (confirms what the *running* daemon loaded,
  which is what actually matters for a benchmark run -- the env var must be
  set before `bf daemon start`, since switching SparkInfer is a load-time
  change and does not take effect via `bf exec` alone):
    scripts/bf-t0.sh exec scripts/verify_sparkinfer_load.py

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

import sparkinfer
import torch
from sparkinfer.attention.paged.planner import _is_laguna_fp8_gqa6_analytic_decode_graph

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

    print(f"sparkinfer.__file__ = {sparkinfer.__file__}")
    print(f"repo dir            = {repo_dir}")
    print(f"git HEAD            = {sha}{'  (DIRTY)' if dirty else ''}")

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
