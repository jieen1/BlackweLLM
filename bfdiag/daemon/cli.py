"""``bf daemon``/``bf exec``/``bf repl``/``bf submit`` command wiring.

Auto-discovered by ``bfdiag/cli.py`` (owned by another agent) via the
``register(subparsers) -> None`` convention described in the task spec:
each subparser this module adds calls ``.set_defaults(func=...)`` with a
``callable(args: argparse.Namespace) -> int`` (the standard argparse
dispatch idiom -- ``bfdiag/cli.py`` is expected to do
``args.func(args)`` after parsing). This file has its own
``if __name__ == "__main__":`` entry point so it is self-testable without
depending on that dispatcher at all.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from bfdiag.daemon.client import Client, DaemonNotRunning
from bfdiag.daemon.provider import LOAD_TIME_CONFIG_KEYS, requires_cold_restart
from bfdiag.daemon.queue import expand_sweeps, format_results, submit
from bfdiag.daemon.server import default_socket_path


def register(subparsers: argparse._SubParsersAction) -> None:
    """Mount ``daemon``, ``exec``, ``repl``, ``submit``, ``run`` onto the
    shared top-level argparse subparsers object."""
    _register_daemon(subparsers)
    _register_exec(subparsers)
    _register_repl(subparsers)
    _register_submit(subparsers)
    _register_run(subparsers)


def _register_daemon(subparsers: argparse._SubParsersAction) -> None:
    daemon_parser = subparsers.add_parser("daemon", help="manage the bfdiag warm daemon")
    daemon_sub = daemon_parser.add_subparsers(dest="daemon_command", required=True)

    start_parser = daemon_sub.add_parser(
        "start", help="start the daemon (reuses an already-running instance)"
    )
    start_parser.add_argument("--provider", choices=["fake", "laguna"], default="laguna")
    start_parser.add_argument("--socket", default=None)
    start_parser.add_argument("--model-path", default=None)
    start_parser.add_argument("--num-slots", type=int, default=1)
    start_parser.add_argument("--blocks-per-slot", type=int, default=4096)
    start_parser.add_argument("--dtype", default="bfloat16")
    start_parser.add_argument("--max-model-len", type=int, default=131072)
    start_parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    start_parser.add_argument("--no-canary", action="store_true")
    start_parser.add_argument(
        "--idle-ttl-s",
        type=float,
        default=None,
        help="auto-shutdown after this many idle seconds (default 900; 0 disables)",
    )
    start_parser.add_argument(
        "--wait-s", type=float, default=10.0, help="how long to wait for readiness before returning"
    )
    start_parser.set_defaults(func=_cmd_daemon_start)

    status_parser = daemon_sub.add_parser("status", help="show daemon status")
    status_parser.add_argument("--socket", default=None)
    status_parser.set_defaults(func=_cmd_daemon_status)

    stop_parser = daemon_sub.add_parser("stop", help="stop the daemon")
    stop_parser.add_argument("--socket", default=None)
    stop_parser.set_defaults(func=_cmd_daemon_stop)


def _register_exec(subparsers: argparse._SubParsersAction) -> None:
    exec_parser = subparsers.add_parser("exec", help="run code in the warm daemon")
    exec_parser.add_argument("file", nargs="?", help="path to a Python file to execute")
    exec_parser.add_argument("-c", "--code", default=None, help="inline code to execute")
    exec_parser.add_argument("--socket", default=None)
    exec_parser.add_argument("--timeout-s", type=float, default=None)
    exec_parser.add_argument("--run-id", default=None)
    exec_parser.set_defaults(func=_cmd_exec)


def _register_repl(subparsers: argparse._SubParsersAction) -> None:
    repl_parser = subparsers.add_parser("repl", help="interactive REPL against the warm daemon")
    repl_parser.add_argument("--socket", default=None)
    repl_parser.add_argument("--timeout-s", type=float, default=None)
    repl_parser.set_defaults(func=_cmd_repl)


def _register_submit(subparsers: argparse._SubParsersAction) -> None:
    submit_parser = subparsers.add_parser(
        "submit", help="FIFO-submit a script through the daemon, optionally as a sweep"
    )
    submit_parser.add_argument("script", help="path to a Python script")
    submit_parser.add_argument(
        "--sweep",
        action="append",
        default=[],
        metavar="NAME=v1,v2,...",
        help="Cartesian-product env-var sweep; repeatable",
    )
    submit_parser.add_argument("--socket", default=None)
    submit_parser.add_argument("--timeout-s", type=float, default=None)
    submit_parser.set_defaults(func=_cmd_submit)


def _register_run(subparsers: argparse._SubParsersAction) -> None:
    run_parser = subparsers.add_parser(
        "run",
        help="run a script as an independent cold-start process (no daemon involved)",
    )
    run_parser.add_argument("script", help="path to a Python script")
    run_parser.add_argument(
        "--cold",
        action="store_true",
        required=True,
        help="explicit: this is the only mode today (a real, from-scratch process per run)",
    )
    run_parser.add_argument(
        "--sweep",
        action="append",
        default=[],
        metavar="NAME=v1,v2,...",
        help="Cartesian-product env-var sweep; repeatable. Unlike `bf submit`, load-time "
        "config IS safe to sweep here -- each variant gets its own fresh process",
    )
    run_parser.set_defaults(func=_cmd_run)


# -- daemon start/status/stop ---------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _client_for(args: argparse.Namespace) -> Client:
    return Client(socket_path=args.socket, timeout_s=getattr(args, "timeout_s", None))


def _print_status(client: Client) -> None:
    response = client.status()
    print(json.dumps(response.result, indent=2, default=str))


def _requested_load_config(args: argparse.Namespace) -> dict[str, object]:
    """The load-time config this CLI invocation is asking for, in the same
    key-space as ``LagunaEngineProvider.describe()["load_config"]`` (see
    ``provider.LOAD_TIME_CONFIG_KEYS``) -- used to decide whether reusing
    an already-running daemon is actually safe."""
    return {
        "model_path": args.model_path,
        "num_slots": args.num_slots,
        "blocks_per_slot": args.blocks_per_slot,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }


def _cmd_daemon_start(args: argparse.Namespace) -> int:
    socket_path = Path(args.socket) if args.socket else default_socket_path()
    client = Client(socket_path=socket_path)
    try:
        if client.ping().ok:
            running_status = client.status().result
            running_cfg = running_status.get("provider", {}).get("load_config", {})
            requested_cfg = _requested_load_config(args)
            # Only compare keys the running provider actually tracks (a
            # FakeEngineProvider's load_config is deliberately sparse --
            # missing-there means "not meaningful for this provider", not
            # "a requested change"), and treat an unset --model-path
            # (None) as "keep whatever's already loaded", not a deliberate
            # override attempt.
            comparable_requested_cfg = {
                key: value for key, value in requested_cfg.items() if key in running_cfg
            }
            if requested_cfg.get("model_path") is None:
                comparable_requested_cfg.pop("model_path", None)
            mismatched = requires_cold_restart(
                running_cfg, comparable_requested_cfg, locked_keys=LOAD_TIME_CONFIG_KEYS
            )
            if mismatched:
                print(
                    f"bf daemon start: a daemon is already running at {socket_path} but with "
                    f"a DIFFERENT load-time config for {mismatched} -- reusing it would silently "
                    "keep serving the OLD config, not the one just requested. Run "
                    "`bf daemon stop` first, then start again with the new config.",
                    file=sys.stderr,
                )
                _print_status(client)
                return 1
            print(f"bfdiag daemon already running at {socket_path} (reusing).")
            _print_status(client)
            return 0
    except DaemonNotRunning:
        pass

    socket_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = socket_path.parent / "daemon.log"
    cmd = [
        sys.executable,
        "-m",
        "bfdiag.daemon.server",
        "--provider",
        args.provider,
        "--socket",
        str(socket_path),
        "--blocks-per-slot",
        str(args.blocks_per_slot),
        "--dtype",
        args.dtype,
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    if args.model_path:
        cmd += ["--model-path", args.model_path]
    if args.num_slots:
        cmd += ["--num-slots", str(args.num_slots)]
    if args.no_canary:
        cmd.append("--no-canary")
    if args.idle_ttl_s is not None:
        cmd += ["--idle-ttl-s", str(args.idle_ttl_s)]

    with open(log_path, "ab") as log_file:
        subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            cwd=str(_repo_root()),
        )
    print(f"bfdiag daemon starting (provider={args.provider}); log: {log_path}")

    deadline = time.monotonic() + max(args.wait_s, 0.0)
    while time.monotonic() < deadline:
        try:
            if client.ping().ok:
                print("bfdiag daemon is ready.")
                _print_status(client)
                return 0
        except DaemonNotRunning:
            pass
        time.sleep(0.2)

    print(
        f"bfdiag daemon still starting after {args.wait_s:.0f}s "
        f"(model load can take minutes for --provider laguna); check {log_path} "
        "or run `bf daemon status` again shortly."
    )
    return 0


def _cmd_daemon_status(args: argparse.Namespace) -> int:
    socket_path = Path(args.socket) if args.socket else default_socket_path()
    client = Client(socket_path=socket_path)
    try:
        _print_status(client)
        return 0
    except DaemonNotRunning:
        print(f"bfdiag daemon: not running (no socket at {socket_path})")
        return 1


def _cmd_daemon_stop(args: argparse.Namespace) -> int:
    socket_path = Path(args.socket) if args.socket else default_socket_path()
    client = Client(socket_path=socket_path)
    try:
        response = client.shutdown()
    except DaemonNotRunning:
        print(f"bfdiag daemon: not running (no socket at {socket_path})")
        return 0
    print(f"bfdiag daemon: {response.result}")
    return 0 if response.ok else 1


# -- exec -------------------------------------------------------------------


def _cmd_exec(args: argparse.Namespace) -> int:
    if bool(args.file) == bool(args.code):
        print("bf exec: pass exactly one of FILE or -c/--code", file=sys.stderr)
        return 2
    client = _client_for(args)
    try:
        if args.file:
            response = client.exec_file(args.file, run_id=args.run_id, timeout_s=args.timeout_s)
        else:
            response = client.exec_code(args.code, run_id=args.run_id, timeout_s=args.timeout_s)
    except DaemonNotRunning as exc:
        print(f"bf exec: {exc}", file=sys.stderr)
        return 1

    if response.stdout:
        sys.stdout.write(response.stdout)
    if response.stderr:
        sys.stderr.write(response.stderr)
    if not response.ok:
        print(f"bf exec: FAILED ({response.error or 'see traceback'})", file=sys.stderr)
        if response.traceback:
            sys.stderr.write(response.traceback)
        return 1
    if response.result is not None:
        print(f"result: {response.result!r}")
    return 0


# -- repl ---------------------------------------------------------------


def _cmd_repl(args: argparse.Namespace) -> int:
    import readline  # noqa: F401  (enables line editing/history for input())

    client = _client_for(args)
    try:
        client.ping()
    except DaemonNotRunning as exc:
        print(f"bf repl: {exc}", file=sys.stderr)
        return 1

    print("bfdiag repl -- blank line submits the buffered block; 'exit'/'quit' to leave.")
    buffer: list[str] = []
    while True:
        try:
            line = input("bf... " if buffer else "bf> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        stripped = line.strip()
        if not buffer and stripped in ("exit", "quit"):
            break
        if stripped == "":
            if not buffer:
                continue
            code = "\n".join(buffer)
            buffer = []
            response = client.exec_code(code)
            if response.stdout:
                sys.stdout.write(response.stdout)
            if response.stderr:
                sys.stderr.write(response.stderr)
            if not response.ok:
                print(f"error: {response.error or response.traceback}")
            elif response.result is not None:
                print(repr(response.result))
            continue
        buffer.append(line)
    return 0


# -- submit ---------------------------------------------------------------


def _cmd_submit(args: argparse.Namespace) -> int:
    client = _client_for(args)
    try:
        results = submit(args.script, args.sweep, client=client, timeout_s=args.timeout_s)
    except (FileNotFoundError, ValueError) as exc:
        print(f"bf submit: {exc}", file=sys.stderr)
        return 2
    print(f"bf submit: {len(results)} variant(s) run:")
    print(format_results(results))
    return 0 if all(r.ok for r in results) else 1


# -- run --cold -------------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> int:
    """Cold path for load-time config sweeps: no daemon, each variant is a
    genuinely independent ``python script.py`` process. This is the
    counterpart ``bf submit`` points to when a sweep touches a
    load-time-locked variable (see ``queue.check_sweep_is_hot_safe``)."""
    script_path = Path(args.script).resolve()
    if not script_path.is_file():
        print(f"bf run: no such file: {script_path}", file=sys.stderr)
        return 2
    try:
        overlays = expand_sweeps(args.sweep)
    except ValueError as exc:
        print(f"bf run: {exc}", file=sys.stderr)
        return 2

    exit_codes: list[int] = []
    for overlay in overlays:
        env = dict(os.environ)
        env.update(overlay)
        label = ",".join(f"{k}={v}" for k, v in overlay.items()) or "(no sweep)"
        print(f"bf run --cold: {label}")
        result = subprocess.run([sys.executable, str(script_path)], env=env, check=False)
        exit_codes.append(result.returncode)
    return 0 if all(code == 0 for code in exit_codes) else 1


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(prog="bf")
    _subparsers = _parser.add_subparsers(dest="command", required=True)
    register(_subparsers)
    _args = _parser.parse_args()
    raise SystemExit(_args.func(_args))
