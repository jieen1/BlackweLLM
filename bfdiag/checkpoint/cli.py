"""``bf checkpoint save|list|show|restore|rm`` -- CLI surface for this
package.

Follows the dispatcher contract from ``bfdiag/cli.py``: :func:`register`
adds subparsers and attaches a ``func(args) -> int`` handler to each via
``set_defaults(func=...)``; the top-level ``bf`` dispatcher discovers this
module automatically (any ``bfdiag.<subpackage>.cli`` with a
``register(subparsers)``) and calls ``args.func(args)``.

``list``/``show``/``rm`` are pure filesystem operations against the
checkpoint store and need no live engine -- they run directly in this
process. ``save``/``restore`` need a LIVE ``engine``/``backend`` object,
which only exists inside the warm daemon process (``bfdiag/daemon/``); this
module reaches it the same way any other diagnostic script would --
``bfdiag.daemon.client.Client.exec_code()`` -- rather than adding a new
daemon RPC opcode (``bfdiag/daemon/*`` is out of this package's file scope,
see notes/2026-07-27-bfdiag-checkpoint-restore.md's integration-TODO
section for what a real ``bf exec --from-checkpoint`` flag on the
EXISTING ``bf exec`` command would need).
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

from bfdiag.checkpoint import store
from bfdiag.checkpoint.state import describe_state_items

_DEFAULT_SLOT = 0


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024
    return f"{n:.1f} GiB"


def _get_client(timeout_s: float | None):
    from bfdiag.daemon.client import Client

    return Client(timeout_s=timeout_s)


def _exec_or_report(client, code: str, timeout_s: float | None, *, label: str):
    """Run ``code`` against the warm daemon, turning ``DaemonNotRunning``
    into the same clear, non-traceback error message every other ``bf``
    daemon-backed subcommand shows -- ``save``/``restore`` need a LIVE
    engine, which only exists inside a running ``bf daemon start`` process.
    """
    from bfdiag.daemon.client import DaemonNotRunning

    try:
        return client.exec_code(code, timeout_s=timeout_s), None
    except DaemonNotRunning as exc:
        return None, f"{label}: no warm daemon running ({exc}). Start one with `bf daemon start`."


def _cmd_list(args: argparse.Namespace) -> int:
    manifests = store.list_checkpoints(root=args.root)
    if args.json:
        print(json.dumps([m.to_dict() for m in manifests], indent=2, ensure_ascii=False))
        return 0
    if not manifests:
        print("(no checkpoints saved yet)")
        return 0
    for m in manifests:
        total = m.size_bytes.get("total", 0)
        print(
            f"{m.name:20s}  {m.created_at}  slot={m.slot}  "
            f"kv_len={m.slot_kv_len:>8d}  size={_fmt_bytes(total):>10s}"
        )
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    try:
        manifest = store.load_manifest(args.name, root=args.root)
    except KeyError as exc:
        print(f"bf checkpoint show: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False))
        return 0
    print(f"name:          {manifest.name}")
    print(f"created_at:    {manifest.created_at}")
    print(f"slot:          {manifest.slot}")
    print(f"slot_kv_len:   {manifest.slot_kv_len}")
    print(f"committed_len: {len(manifest.slot_committed_tokens)}")
    print("size_bytes:")
    for category, n in sorted(manifest.size_bytes.items()):
        print(f"  {category:8s}: {_fmt_bytes(n)}")
    print("fingerprint (laguna_geometry):")
    geom = manifest.fingerprint.get("laguna_geometry", {})
    for key, value in sorted(geom.items()):
        print(f"  {key}: {value}")
    print("fingerprint (git shas, informational -- see soft_fingerprint_diff on restore):")
    for repo, info in sorted((manifest.fingerprint.get("git") or {}).items()):
        print(f"  {repo}: {info.get('sha')}")
    print(f"baseline: steps={manifest.baseline.get('steps')} "
          f"committed_tokens={manifest.baseline.get('committed_tokens')}")
    return 0


def _cmd_rm(args: argparse.Namespace) -> int:
    try:
        store.remove_checkpoint(args.name, root=args.root)
    except KeyError as exc:
        print(f"bf checkpoint rm: {exc}", file=sys.stderr)
        return 1
    print(f"removed checkpoint {args.name!r}")
    return 0


def _cmd_schema(args: argparse.Namespace) -> int:
    items = describe_state_items()
    if args.json:
        print(json.dumps(items, indent=2, ensure_ascii=False))
        return 0
    for item in items:
        print(f"[{item['category']:20s}] {item['name']}")
        print(f"    source:   {item['source']}")
        print(f"    code_ref: {item['code_ref']}")
    return 0


def _cmd_save(args: argparse.Namespace) -> int:
    client = _get_client(args.timeout)
    root_repr = repr(str(args.root)) if args.root else "None"
    code = textwrap.dedent(
        f"""
        from bfdiag.checkpoint.store import save_checkpoint
        _bf_ckpt_manifest = save_checkpoint(
            engine,
            {args.slot!r},
            {args.name!r},
            baseline_steps={args.baseline_steps!r},
            model_revision={args.model_revision!r},
            root={root_repr},
            overwrite={args.overwrite!r},
        )
        result = _bf_ckpt_manifest.to_dict()
        """
    )
    response, error = _exec_or_report(client, code, args.timeout, label="bf checkpoint save")
    if error:
        print(error, file=sys.stderr)
        return 1
    if not response.ok:
        print(f"bf checkpoint save: daemon exec failed: {response.error}", file=sys.stderr)
        if response.traceback:
            print(response.traceback, file=sys.stderr)
        return 1
    manifest_dict = response.result or {}
    total = (manifest_dict.get("size_bytes") or {}).get("total", 0)
    print(f"saved checkpoint {args.name!r}: slot={args.slot} size={_fmt_bytes(total)}")
    return 0


def _cmd_restore(args: argparse.Namespace) -> int:
    client = _get_client(args.timeout)
    root_repr = repr(str(args.root)) if args.root else "None"
    code = textwrap.dedent(
        f"""
        from bfdiag.checkpoint.restore import restore_checkpoint
        _bf_restore_result = restore_checkpoint(
            engine,
            {args.slot!r},
            {args.name!r},
            root={root_repr},
            verify_after={(not args.no_verify)!r},
            require_clean_fingerprint={args.require_clean_fingerprint!r},
        )
        anchor = _bf_restore_result.anchor
        draft_tokens = _bf_restore_result.draft_tokens
        result = {{
            "name": _bf_restore_result.name,
            "slot": _bf_restore_result.slot,
            "verified": _bf_restore_result.verified,
            "anchor": _bf_restore_result.anchor,
            "verified_tokens": _bf_restore_result.verified_tokens,
            "soft_fingerprint_diff": _bf_restore_result.soft_fingerprint_diff,
        }}
        """
    )
    if args.exec_file:
        code += "\n" + Path(args.exec_file).read_text()
    response, error = _exec_or_report(client, code, args.timeout, label="bf checkpoint restore")
    if error:
        print(error, file=sys.stderr)
        return 1
    if not response.ok:
        print(f"bf checkpoint restore: daemon exec failed: {response.error}", file=sys.stderr)
        if response.traceback:
            print(response.traceback, file=sys.stderr)
        return 1
    result = response.result or {}
    print(
        f"restored checkpoint {args.name!r}: slot={result.get('slot')} "
        f"verified={result.get('verified')}"
    )
    diffs = result.get("soft_fingerprint_diff") or []
    if diffs:
        print("soft fingerprint diffs (informational, did not block restore):")
        for d in diffs:
            print(f"  {d}")
    return 0


def register(subparsers) -> None:
    checkpoint_parser = subparsers.add_parser(
        "checkpoint", help="active checkpoint/restore for a Laguna+DFlash slot"
    )
    checkpoint_subparsers = checkpoint_parser.add_subparsers(dest="checkpoint_command")

    list_parser = checkpoint_subparsers.add_parser("list", help="list saved checkpoints")
    list_parser.add_argument("--root", type=Path, default=None)
    list_parser.add_argument("--json", action="store_true")
    list_parser.set_defaults(func=_cmd_list)

    show_parser = checkpoint_subparsers.add_parser("show", help="show one checkpoint's manifest")
    show_parser.add_argument("name")
    show_parser.add_argument("--root", type=Path, default=None)
    show_parser.add_argument("--json", action="store_true")
    show_parser.set_defaults(func=_cmd_show)

    rm_parser = checkpoint_subparsers.add_parser("rm", help="delete a saved checkpoint")
    rm_parser.add_argument("name")
    rm_parser.add_argument("--root", type=Path, default=None)
    rm_parser.set_defaults(func=_cmd_rm)

    schema_parser = checkpoint_subparsers.add_parser(
        "schema", help="print the declarative state-item checklist (state.py)"
    )
    schema_parser.add_argument("--json", action="store_true")
    schema_parser.set_defaults(func=_cmd_schema)

    save_parser = checkpoint_subparsers.add_parser(
        "save", help="save the CURRENT warm daemon slot's state (prefill 完成后主动存档)"
    )
    save_parser.add_argument("name")
    save_parser.add_argument("--slot", type=int, default=_DEFAULT_SLOT)
    save_parser.add_argument("--baseline-steps", type=int, default=store.DEFAULT_BASELINE_STEPS)
    save_parser.add_argument("--model-revision", type=str, default=None)
    save_parser.add_argument("--root", type=Path, default=None)
    save_parser.add_argument("--overwrite", action="store_true")
    save_parser.add_argument("--timeout", type=float, default=None)
    save_parser.set_defaults(func=_cmd_save)

    restore_parser = checkpoint_subparsers.add_parser(
        "restore", help="restore a checkpoint into the warm daemon, skipping prefill"
    )
    restore_parser.add_argument("name")
    restore_parser.add_argument("--slot", type=int, default=_DEFAULT_SLOT)
    restore_parser.add_argument("--root", type=Path, default=None)
    restore_parser.add_argument("--no-verify", action="store_true")
    restore_parser.add_argument("--require-clean-fingerprint", action="store_true")
    restore_parser.add_argument("--timeout", type=float, default=None)
    restore_parser.add_argument(
        "--exec-file",
        type=str,
        default=None,
        help=(
            "run this script's code in the SAME daemon exec call right after restoring "
            "(sees the restored 'anchor'/'draft_tokens' locals) -- the closest approximation "
            "to 'bf exec --from-checkpoint <name> script.py' this package can offer without "
            "modifying bfdiag/daemon/cli.py (out of this package's file scope; see the notes "
            "file for the one-line change that would wire this up as a real --from-checkpoint "
            "flag on bf exec itself)"
        ),
    )
    restore_parser.set_defaults(func=_cmd_restore)

    def _no_subcommand(args: argparse.Namespace) -> int:
        checkpoint_parser.print_help()
        return 1

    checkpoint_parser.set_defaults(func=_no_subcommand)
