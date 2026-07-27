"""``bf shapes`` -- print/diff kernel-isolation-test shapes derived from the
real model config, with ``block_size`` (KV page_size) explicit.

``register(subparsers)`` is the contract ``bfdiag/cli.py``'s auto-discovery
dispatcher calls (see that module's docstring -- not owned by this
module/agent). Uses the standard ``argparse`` ``set_defaults(func=...)``
pattern so the dispatcher can just do ``args.func(args)`` after parsing.

    bf shapes                                    # both supported block sizes (64, 128)
    bf shapes --block-size 128
    bf shapes --block-size 64 --block-size 128 --diff   # only what changed
    bf shapes --json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from bfdiag.shapes import LagunaConfigError, ModelShapes, model_shapes
from bfdiag.shapes.model import cdiv

DEFAULT_BLOCK_SIZES = (64, 128)
"""The two page_size values ``runtime/backends/laguna.py``'s
``LagunaBackend.__init__`` currently accepts (``block_size not in (64,
128): raise``). Used only as ``bf shapes``' no-argument default -- every
shape is still computed fresh from the real config for each one, nothing
here is a fallback for a missing value."""

DEFAULT_KV_LEN = 65536
"""Matches the default CTX in benchmarks/ab_dflash_block_size_64_vs_128.py
(the A/B script this feature exists to support) -- not a model constant."""

DEFAULT_CHUNK_TOKENS = 8192
"""Matches QSR_PREFILL_CHUNK's default in runtime/backends/laguna.py."""


def _flatten(
    S: ModelShapes,
    *,
    kv_len: int,
    chunk_tokens: int,
    num_tokens: int,
    num_slots: int,
    blocks_per_slot: int,
) -> dict[str, Any]:
    """All interesting scalars/shapes for one ``ModelShapes``, as a flat
    ``{key: value}`` map -- the representation ``bf shapes --diff`` compares
    key-by-key between two block_size configurations."""
    out: dict[str, Any] = {}

    for group in ("full", "sliding"):
        if group not in S.config.groups:
            continue

        d = S.decode_attention(group=group, kv_len=kv_len)
        out[f"decode/{group}.q"] = d.shapes()["q"]
        out[f"decode/{group}.k_cache"] = d.shapes()["k_cache"]
        out[f"decode/{group}.page_table"] = d.shapes()["page_table"]
        out[f"decode/{group}.cache_seqlens"] = d.shapes()["cache_seqlens"]
        out[f"decode/{group}.max_pages"] = d.max_pages
        out[f"decode/{group}.cache_seqlen"] = d.cache_seqlen
        if d.swa is not None:
            out[f"decode/{group}.window_start"] = d.swa.window_start
            out[f"decode/{group}.aligned_start"] = d.swa.aligned_start
            out[f"decode/{group}.aligned_len"] = d.swa.aligned_len
            out[f"decode/{group}.n_ring"] = d.swa.n_ring

        v = S.verify_attention(group=group, kv_len=kv_len)
        out[f"verify/{group}.q"] = v.shapes()["q"]
        out[f"verify/{group}.k_cache"] = v.shapes()["k_cache"]
        out[f"verify/{group}.page_table"] = v.shapes()["page_table"]
        out[f"verify/{group}.max_pages"] = v.max_pages
        out[f"verify/{group}.cache_seqlen"] = v.cache_seqlen
        if v.swa is not None:
            out[f"verify/{group}.aligned_len"] = v.swa.aligned_len
            out[f"verify/{group}.n_ring"] = v.swa.n_ring

        if group == "sliding":
            out["ring_capacity/sliding"] = S.ring_capacity(group)
            out["kv_cache/sliding"] = S.kv_cache_shape(group=group, num_slots=num_slots)
        else:
            out["kv_cache/full"] = S.kv_cache_shape(
                group=group, num_slots=num_slots, blocks_per_slot=blocks_per_slot
            )

        if group == "full":
            p = S.prefill_attention(
                group=group, kv_len_before=kv_len, chunk_tokens=chunk_tokens
            )
        else:
            p = None
        if p is not None:
            out["prefill/full.q"] = p.shapes()["q"]
            out["prefill/full.max_pages"] = p.max_pages

    scratch = S.prefill_swa_scratch(chunk_tokens=chunk_tokens)
    out["prefill_swa_scratch.shape"] = scratch.shape()
    out["prefill_swa_scratch.scratch_blocks"] = scratch.scratch_blocks

    dd = S.draft_decode_attention(kv_len=kv_len)
    out["draft_decode.q"] = dd.shapes()["q"]
    out["draft_decode.k_cache"] = dd.shapes()["k_cache"]
    out["draft_decode.aligned_len"] = dd.swa.aligned_len
    out["draft_decode.n_ring"] = dd.swa.n_ring
    dv = S.draft_verify_attention(kv_len=kv_len)
    out["draft_verify.q"] = dv.shapes()["q"]
    out["draft_verify.k_cache"] = dv.shapes()["k_cache"]
    out["draft_verify.aligned_len"] = dv.swa.aligned_len
    out["draft_verify.n_ring"] = dv.swa.n_ring
    out["draft_ring_capacity"] = S.draft_ring_capacity()

    for g in S.dense_gemms(num_tokens=num_tokens):
        out[f"gemm/{g.name}"] = (g.m, g.n, g.k)
    for g in S.draft_gemms(num_tokens=num_tokens):
        out[f"gemm/{g.name}"] = (g.m, g.n, g.k)

    for name, shape in S.moe_stacked_expert_shapes().items():
        out[f"moe/stacked.{name}"] = shape
    for name, shape in S.moe_sparkinfer_shapes().items():
        out[f"moe/sparkinfer.{name}"] = shape
    for name, shape in S.moe_router_shapes(num_tokens=num_tokens).items():
        out[f"moe/{name}"] = shape

    return out


def diff_flat(
    a: dict[str, Any], b: dict[str, Any]
) -> tuple[list[tuple[str, Any, Any]], list[str]]:
    """Key-by-key diff of two flattened shape maps. Returns (changed, unchanged)."""
    changed: list[tuple[str, Any, Any]] = []
    unchanged: list[str] = []
    for key in sorted(set(a) | set(b)):
        va, vb = a.get(key, "<missing>"), b.get(key, "<missing>")
        if va != vb:
            changed.append((key, va, vb))
        else:
            unchanged.append(key)
    return changed, unchanged


def render_table(block_size: int, flat: dict[str, Any]) -> str:
    lines = [f"=== bf shapes: block_size={block_size} ==="]
    prev_prefix = None
    for key in sorted(flat):
        prefix = key.split(".", 1)[0].split("/", 1)[0]
        if prefix != prev_prefix:
            lines.append(f"-- {prefix} --")
            prev_prefix = prefix
        lines.append(f"  {key}: {flat[key]}")
    return "\n".join(lines)


def render_diff(
    bs_a: int, bs_b: int, changed: list[tuple[str, Any, Any]], unchanged: list[str]
) -> str:
    lines = [f"=== bf shapes --diff {bs_a} {bs_b} ==="]
    if changed:
        lines.append(f"⚠ {len(changed)} shape(s) changed:")
        for key, va, vb in changed:
            lines.append(f"  {key}: {va} -> {vb}")
    else:
        lines.append("(no shapes changed)")
    lines.append("")
    lines.append(f"-- unchanged ({len(unchanged)}) --")
    for key in unchanged:
        lines.append(f"  {key}")
    return "\n".join(lines)


def _cmd_shapes(args: argparse.Namespace) -> int:
    block_sizes: list[int] = args.block_size or list(DEFAULT_BLOCK_SIZES)
    if args.diff and len(block_sizes) != 2:
        print(
            "bf shapes --diff needs exactly two --block-size values "
            f"(got {block_sizes}; default is {list(DEFAULT_BLOCK_SIZES)})",
            file=sys.stderr,
        )
        return 1

    try:
        shapes_by_bs: dict[int, ModelShapes] = {
            bs: model_shapes(bs, model_path=args.model_path, draft_model_path=args.draft_model_path)
            for bs in block_sizes
        }
        flat_by_bs: dict[int, dict[str, Any]] = {}
        for bs, S in shapes_by_bs.items():
            bps = args.blocks_per_slot
            if bps is None:
                # Illustrative default: size full-attn KV cache to exactly
                # hold the kv_len being used for decode/verify shapes above
                # -- not an arbitrary constant, see DEFAULT_KV_LEN docstring.
                bps = cdiv(args.kv_len + 1, bs)
            flat_by_bs[bs] = _flatten(
                S,
                kv_len=args.kv_len,
                chunk_tokens=args.chunk_tokens,
                num_tokens=args.num_tokens,
                num_slots=args.num_slots,
                blocks_per_slot=bps,
            )
    except LagunaConfigError as exc:
        print(f"bf shapes: {exc}", file=sys.stderr)
        return 1

    if args.diff:
        bs_a, bs_b = block_sizes
        changed, unchanged = diff_flat(flat_by_bs[bs_a], flat_by_bs[bs_b])
        if args.json:
            print(
                json.dumps(
                    {
                        "block_size_a": bs_a,
                        "block_size_b": bs_b,
                        "changed": [
                            {"key": k, "a": va, "b": vb} for k, va, vb in changed
                        ],
                        "unchanged": unchanged,
                    },
                    indent=2,
                )
            )
        else:
            print(render_diff(bs_a, bs_b, changed, unchanged))
        return 0 if not changed else 2

    for bs in block_sizes:
        if args.json:
            print(json.dumps({"block_size": bs, "shapes": flat_by_bs[bs]}, indent=2))
        else:
            print(render_table(bs, flat_by_bs[bs]))
            print()
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    """Mount ``bf shapes`` onto the dispatcher's subparsers."""
    p = subparsers.add_parser(
        "shapes",
        help="derive kernel-isolation-test shapes from the real model config "
        "(block_size explicit; --diff shows what changes across page_size)",
    )
    p.add_argument(
        "--block-size",
        action="append",
        type=int,
        default=None,
        help="KV page_size(s) to derive shapes for; repeatable "
        f"(default: {list(DEFAULT_BLOCK_SIZES)}, the values "
        "runtime/backends/laguna.py currently accepts)",
    )
    p.add_argument(
        "--diff",
        action="store_true",
        help="show only what changed between exactly two --block-size values "
        "(exit code 2 if anything changed, matching bf diff's convention)",
    )
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument(
        "--kv-len",
        type=int,
        default=DEFAULT_KV_LEN,
        help=f"context length for decode/verify shapes (default {DEFAULT_KV_LEN}, "
        "matching benchmarks/ab_dflash_block_size_64_vs_128.py's default CTX)",
    )
    p.add_argument(
        "--chunk-tokens",
        type=int,
        default=DEFAULT_CHUNK_TOKENS,
        help=f"prefill chunk size (default {DEFAULT_CHUNK_TOKENS}, matching "
        "QSR_PREFILL_CHUNK's default)",
    )
    p.add_argument(
        "--num-tokens", type=int, default=1, help="token count for GEMM/MoE M dimension"
    )
    p.add_argument("--num-slots", type=int, default=1, help="slot count for KV cache shapes")
    p.add_argument(
        "--blocks-per-slot",
        type=int,
        default=None,
        help="full-attention KV cache capacity in blocks/slot (default: exactly enough "
        "to hold --kv-len)",
    )
    p.add_argument("--model-path", default=None, help="override the target model's config dir")
    p.add_argument(
        "--draft-model-path", default=None, help="override the draft model's config dir"
    )
    p.set_defaults(func=_cmd_shapes)


def _build_standalone_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bfdiag.shapes.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    register(subparsers)
    return parser


if __name__ == "__main__":
    _parser = _build_standalone_parser()
    _args = _parser.parse_args()
    raise SystemExit(_args.func(_args))
