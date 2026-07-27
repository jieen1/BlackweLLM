"""``bf submit``: a minimal FIFO experiment queue with Cartesian-product
env-var sweeps, run through the warm daemon so N variants share ONE
loaded engine.

    bf submit --sweep 'QSR_DFLASH_CUDA_GRAPH=0,1' --sweep 'QSR_ASSERT_LEVEL=0,2' script.py

expands to 4 variants (the Cartesian product of the two sweeps), each run
as one ``exec`` request against the daemon in turn -- turning "run this
script 4 times, babysitting each one" into "submit once, go do something
else." Ordering across variants is exactly submission order: the daemon's
own single FIFO worker thread (see ``server.py``) is what guarantees no
two variants ever touch the engine concurrently, so this module does not
need any locking of its own -- it just calls the client once per variant
and waits for each response before sending the next.

Each variant gets its own ``run_id`` (``.bfdiag/runs/<run_id>/``) so other
bfdiag subsystems' run recorders can key off ``QSR_BFDIAG_RUN_ID`` the same
way a single ``bf exec`` call would. To avoid clobbering another agent's
run-record schema in the same directory, this module only ever writes
files prefixed ``queue_*`` there.

**Hot/cold boundary (hard requirement)**: some env vars are read exactly
once when the Laguna backend/DFlash engine are constructed (see
``provider.LOAD_TIME_ENV_VARS``) -- sweeping one of those through an
already-loaded hot daemon changes nothing in the running engine and would
silently produce N identical, meaningless "variants". ``submit()`` refuses
outright (raises ``ValueError``) rather than doing that; use
``bf run --cold --sweep ...`` (``cli.py``/``_cmd_run``) instead, which
starts one independent process per variant.
"""

from __future__ import annotations

import itertools
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from bfdiag.daemon.client import Client
from bfdiag.daemon.provider import LOAD_TIME_ENV_VARS
from bfdiag.daemon.server import bfdiag_dir


def parse_sweep(spec: str) -> tuple[str, list[str]]:
    """Parse ``'NAME=v1,v2,v3'`` into ``("NAME", ["v1", "v2", "v3"])``."""
    if "=" not in spec:
        raise ValueError(f"--sweep must look like NAME=v1,v2,...: got {spec!r}")
    name, values = spec.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"--sweep is missing a variable name: {spec!r}")
    values_list = [v.strip() for v in values.split(",") if v.strip() != ""]
    if not values_list:
        raise ValueError(f"--sweep has no values: {spec!r}")
    return name, values_list


def expand_sweeps(specs: list[str]) -> list[dict[str, str]]:
    """Cartesian product of all ``--sweep`` specs -> one env-var overlay
    dict per variant. No sweeps at all -> a single empty-overlay variant
    (plain ``bf submit script.py`` with no ``--sweep``)."""
    if not specs:
        return [{}]
    names: list[str] = []
    value_lists: list[list[str]] = []
    for spec in specs:
        name, values = parse_sweep(spec)
        names.append(name)
        value_lists.append(values)
    return [dict(zip(names, combo, strict=True)) for combo in itertools.product(*value_lists)]


def check_sweep_is_hot_safe(specs: list[str]) -> None:
    """Refuse (``ValueError``) any ``--sweep`` spec whose variable name is
    one of ``provider.LOAD_TIME_ENV_VARS`` -- see module docstring. This
    is a hard requirement: producing N identical "variants" because the
    swept variable was actually locked in at engine-construction time is
    strictly worse than erroring, because it looks like real data.
    """
    swept_names = (parse_sweep(spec)[0] for spec in specs)
    locked = sorted(name for name in swept_names if name in LOAD_TIME_ENV_VARS)
    if locked:
        raise ValueError(
            f"--sweep touches load-time-locked variable(s) {locked}: these are read once "
            "when the engine is constructed, so sweeping them through an already-loaded "
            "hot daemon would silently produce identical, meaningless variants. Use "
            "`bf run --cold --sweep ...` instead, which starts one independent process "
            "per variant."
        )


def make_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"


def _run_dir(run_id: str) -> Path:
    directory = bfdiag_dir() / "runs" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _wrap_script_code(script_path: Path, env_overlay: dict[str, str]) -> str:
    """Build the code string sent to the daemon: apply the env overlay,
    then exec the target script's own source (the daemon's exec namespace
    always sets ``__name__ == "__main__"``, so scripts using the standard
    ``if __name__ == "__main__":`` idiom run exactly as they would
    stand-alone)."""
    overlay_lines = "\n".join(
        f"os.environ[{name!r}] = {value!r}" for name, value in env_overlay.items()
    )
    return (
        "import os\n"
        f"{overlay_lines}\n"
        f"exec(compile(open({str(script_path)!r}).read(), {str(script_path)!r}, 'exec'))\n"
    )


@dataclass
class SubmitResult:
    run_id: str
    env: dict[str, str]
    ok: bool
    elapsed_s: float
    run_dir: Path
    error: str | None = None


def submit(
    script_path: str | Path,
    sweeps: list[str] | None = None,
    *,
    client: Client | None = None,
    timeout_s: float | None = None,
) -> list[SubmitResult]:
    """Run ``script_path`` once per Cartesian-product combination of
    ``--sweep`` specs (or once, with no overlay, if ``sweeps`` is falsy),
    FIFO through the warm daemon. Each variant's request/response is
    recorded under ``.bfdiag/runs/<run_id>/queue_{request,response}.json``.

    Raises ``ValueError`` immediately (before running anything) if any
    ``--sweep`` targets a load-time-locked env var -- see
    ``check_sweep_is_hot_safe``.
    """
    check_sweep_is_hot_safe(sweeps or [])
    path = Path(script_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(script_path)
    overlays = expand_sweeps(sweeps or [])
    active_client = client or Client()

    results: list[SubmitResult] = []
    for overlay in overlays:
        run_id = make_run_id(prefix=path.stem)
        run_dir = _run_dir(run_id)
        code = _wrap_script_code(path, overlay)

        (run_dir / "queue_request.json").write_text(
            json.dumps({"script": str(path), "env": overlay, "run_id": run_id}, indent=2)
        )

        t0 = time.perf_counter()
        response = active_client.exec_code(code, run_id=run_id, timeout_s=timeout_s)
        elapsed = time.perf_counter() - t0

        (run_dir / "queue_response.json").write_text(
            json.dumps(response.to_dict(), indent=2, default=str)
        )

        results.append(
            SubmitResult(
                run_id=run_id,
                env=overlay,
                ok=response.ok,
                elapsed_s=elapsed,
                run_dir=run_dir,
                error=response.error,
            )
        )
    return results


def format_results(results: list[SubmitResult]) -> str:
    """Small human-readable summary table for the CLI."""
    lines = []
    for result in results:
        env_str = ",".join(f"{k}={v}" for k, v in result.env.items()) or "(no sweep)"
        status = "OK" if result.ok else f"FAILED: {result.error}"
        lines.append(f"  [{result.run_id}] {env_str} -- {status} ({result.elapsed_s:.2f}s)")
    return "\n".join(lines)


def _demo_args(argv: list[str]) -> tuple[str, list[str]]:
    if not argv:
        raise SystemExit("usage: python -m bfdiag.daemon.queue SCRIPT [--sweep NAME=v1,v2 ...]")
    script = argv[0]
    sweeps: list[str] = []
    i = 1
    while i < len(argv):
        if argv[i] == "--sweep" and i + 1 < len(argv):
            sweeps.append(argv[i + 1])
            i += 2
        else:
            i += 1
    return script, sweeps


if __name__ == "__main__":
    import sys

    script_arg, sweep_args = _demo_args(sys.argv[1:])
    print(f"expanded sweeps for {script_arg}:")
    for combo in expand_sweeps(sweep_args):
        print(" ", combo or "(no sweep)")
