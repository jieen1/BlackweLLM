#!/usr/bin/env bash
# Wrapper for `bf` (bfdiag CLI) scoped to THIS worktree.
#
# Why this exists: the venv's `bf` console-script lives in
# ~/.venvs/vllm/bin/bf. When Python resolves `import bfdiag` (or `runtime`,
# `server`, ...) for that script, sys.path[0] is the *script's own directory*
# (the venv bin/), not the caller's cwd. That directory contains no project
# code, so resolution falls through to the venv's pip-editable finder for the
# "blackwellm" package -- and that finder has the main worktree
# (/home/bot/project/qwen-sm120-runtime) HARD-CODED as the source location,
# regardless of which worktree you `cd`ed into or ran `bf` from.
#
# Net effect: plain `cd <this-worktree> && bf daemon start` silently loads
# bfdiag/runtime/server/benchmarks from the OTHER worktree, with no error and
# no warning. Run records, .bfdiag state, and the daemon's own code all come
# from the wrong checkout. Verified empirically 2026-08-01 (see
# notes/2026-08-01-sparkinfer-patch-recovery-and-repro.md).
#
# Fix: PYTHONPATH entries are searched by the stdlib PathFinder, which sits
# *before* the pip-editable MetaPathFinder in sys.meta_path. Pointing
# PYTHONPATH at this worktree's root makes `import bfdiag` resolve here
# first, before the editable finder is ever consulted.
#
# Usage: scripts/bf-t0.sh <same args as `bf`>
#   scripts/bf-t0.sh daemon start --provider laguna
#   scripts/bf-t0.sh exec benchmarks/acceptance_regression.py
#   scripts/bf-t0.sh ls
#
# To load a non-default SparkInfer checkout (e.g. for an A/B control group),
# export BF_SPARKINFER_PATH (and, for accurate bfdiag provenance, the
# matching QSR_REPO_SPARKINFER) before calling this wrapper -- see
# runtime/backends/laguna_sparkinfer_attn.py / laguna_sparkinfer_moe.py for
# how BF_SPARKINFER_PATH is consumed, and always confirm what actually loaded
# with `scripts/verify_sparkinfer_load.py` run through `bf exec`, not by
# assuming the env var took effect.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BF_BIN="${BF_BIN:-$HOME/.venvs/vllm/bin/bf}"

if [[ ! -x "$BF_BIN" ]]; then
    echo "bf-t0: no bf executable at $BF_BIN (set BF_BIN=...)" >&2
    exit 1
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "$BF_BIN" "$@"
