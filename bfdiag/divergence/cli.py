"""``bf divergence`` -- oracle-vs-engine per-layer activation divergence scan.

Auto-discovered by the (not-yet-built, owned by another agent) top-level
``bfdiag/cli.py`` dispatcher via ``register(subparsers) -> None``. Until
that dispatcher exists, this subcommand is fully self-testable standalone:

    python -m bfdiag.divergence.cli --prompt <path-to-token-ids.json>

The real end-to-end path (reading/populating the oracle cache, capturing our
own engine's activations off a live model) requires a GPU and a loaded
Laguna backend; it is written but intentionally never exercised by this
package's test suite (see notes/2026-07-27-bfdiag-oracle-divergence.md's
GPU-verification checklist). Everything up to and including the scan/report
step is exercised on CPU via ``bfdiag.divergence.capture.FakeCaptureSource``
in tests/test_bfdiag_divergence.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bfdiag.divergence.cache import (
    CacheKey,
    CaptureConfig,
    compute_prompt_hash,
    read_oracle_cache,
    to_activation_trace,
)
from bfdiag.divergence.capture import CaptureSource, EngineCaptureSource, default_module_names
from bfdiag.divergence.report import format_text_report, to_json_dict
from bfdiag.divergence.scan import DivergenceReport, scan_layers

#: Laguna-S-2.1's real total decoder-layer count (16 full-attention groups
#: are folded into the same 0..47 index space as the 47 MoE layers -- see
#: ``runtime/backends/laguna_sparkinfer_moe.py``'s ``MOE_LAYER_IDS =
#: list(range(1, 48))``, i.e. layers 0-47, 48 layers total).
DEFAULT_NUM_LAYERS = 48


def register(subparsers: argparse._SubParsersAction) -> None:
    """Wire ``bf divergence`` into the shared ``bfdiag`` CLI dispatcher."""
    parser = subparsers.add_parser(
        "divergence",
        help="oracle-vs-engine per-layer activation divergence scan",
        description=__doc__,
    )
    _add_arguments(parser)
    parser.set_defaults(func=_run)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bf divergence", description="oracle-vs-engine per-layer divergence scan"
    )
    _add_arguments(parser)
    return parser


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--prompt", required=True, help="path to a JSON file containing a list of token ids"
    )
    parser.add_argument(
        "--layers", default="all", help="'all' or a comma list, e.g. '0-15,17,20-23'"
    )
    parser.add_argument("--json", action="store_true", help="emit a machine-readable JSON report")
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="ignore any existing oracle cache entry and require a fresh capture",
    )
    parser.add_argument(
        "--model-revision", default="unknown", help="oracle model revision tag (cache key)"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=DEFAULT_NUM_LAYERS,
        help=f"total decoder layer count (default {DEFAULT_NUM_LAYERS}, Laguna-S-2.1)",
    )


def _parse_layers(spec: str, num_layers: int) -> tuple[int, ...]:
    if spec == "all":
        return tuple(range(num_layers))
    layers: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", 1)
            layers.update(range(int(start), int(end) + 1))
        else:
            layers.add(int(chunk))
    return tuple(sorted(layers))


def _load_prompt_token_ids(prompt: str) -> list[int]:
    path = Path(prompt)
    if not path.exists():
        raise FileNotFoundError(
            f"no token-ids fixture at {path!s} -- pass a path to a JSON file containing a "
            "list of int token ids (e.g. produced from oracle.fixtures golden cases)"
        )
    token_ids = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(token_ids, list) or not all(isinstance(item, int) for item in token_ids):
        raise ValueError(f"{path!s} must contain a JSON list of integers")
    return token_ids


def scan_prompt(
    oracle_source: CaptureSource,
    engine_source: CaptureSource,
    prompt_token_ids: list[int],
    *,
    top_k: int = 10,
) -> DivergenceReport:
    """Capture both sides for one prompt and scan for divergence.

    Both sources implement ``bfdiag.divergence.capture.CaptureSource`` --
    this function doesn't care whether they're ``FakeCaptureSource`` (used
    by tests) or cache-/model-backed sources (real CLI usage), which is what
    keeps the orchestration itself unit-testable.
    """
    oracle_trace = oracle_source.capture(prompt_token_ids)
    engine_trace = engine_source.capture(prompt_token_ids)
    return scan_layers(oracle_trace, engine_trace, top_k=top_k)


class _OracleCacheSource:
    """``CaptureSource`` backed by a populated on-disk oracle cache entry."""

    def __init__(self, entries: dict) -> None:
        self._trace = to_activation_trace(entries)

    def capture(self, prompt_token_ids: list[int]) -> dict:
        del prompt_token_ids  # the cache is already keyed by prompt hash
        return self._trace


def _run(args: argparse.Namespace) -> int:
    prompt_token_ids = _load_prompt_token_ids(args.prompt)
    layer_indices = _parse_layers(args.layers, args.num_layers)
    module_names = default_module_names(args.num_layers)

    key = CacheKey(
        model_revision=args.model_revision,
        prompt_hash=compute_prompt_hash(prompt_token_ids),
        layer_set=layer_indices if args.layers != "all" else "all",
        capture_config=CaptureConfig(),
    )

    oracle_source: CaptureSource | None = None
    if not args.refresh_cache:
        cached = read_oracle_cache(key)
        if cached is not None:
            entries, lookup = cached
            print(lookup.message, file=sys.stderr)
            oracle_source = _OracleCacheSource(entries)

    if oracle_source is None:
        print(
            "oracle cache miss (or --refresh-cache passed) -- this bfdiag build has no "
            "in-repo vLLM oracle runner (see oracle/vllm_reference.py's docstring: hooks run "
            "in a separate vLLM checkout, artifacts are consumed here, never produced here). "
            "Populate the cache via bfdiag.divergence.cache.write_oracle_cache with a trace "
            "captured out-of-process, then re-run without --refresh-cache. See "
            "notes/2026-07-27-bfdiag-oracle-divergence.md's GPU-verification checklist.",
            file=sys.stderr,
        )
        return 2

    # GPU-only from here: requires a live, already-constructed engine backend.
    # Never exercised by tests -- see ``_construct_live_engine_backend``.
    try:
        backend = _construct_live_engine_backend()
    except NotImplementedError as error:
        print(f"engine-side capture unavailable: {error}", file=sys.stderr)
        return 2

    engine_source = EngineCaptureSource(backend, module_names)
    report = scan_prompt(oracle_source, engine_source, prompt_token_ids)
    if args.json:
        print(json.dumps(to_json_dict(report), indent=2))
    else:
        print(format_text_report(report))
    return 1 if report.has_divergence else 0


def _construct_live_engine_backend() -> object:
    """Construct a live, GPU-backed engine backend for real CLI usage.

    Deliberately unimplemented: constructing a ``LagunaBackend`` requires a
    ``VllmConfig`` and a loaded checkpoint that only make sense inside the
    server's own bootstrap path (``server/engine.py``), which is out of
    scope for a diagnostics tool to own. Real GPU usage should call
    ``scan_prompt`` directly from a script that already has a backend
    (wrapped in ``bfdiag.divergence.capture.EngineCaptureSource``) instead
    of going through this bare CLI entry point. See notes/2026-07-27-bfdiag-
    oracle-divergence.md's GPU-verification checklist.
    """
    raise NotImplementedError(
        "bf divergence has no built-in backend bootstrap -- see this function's docstring"
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
