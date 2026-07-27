"""Tests for bfdiag.shapes.cli: ``bf shapes`` / ``bf shapes --diff``.

Acceptance criterion #3: ``--diff 64 128`` must accurately list which
shapes changed (n_ring / aligned_len / max_pages among them) and which
didn't (GEMM/MoE shapes, which don't depend on block_size at all).
"""

from __future__ import annotations

import argparse
import json

import pytest

from bfdiag.shapes.cli import diff_flat, register
from bfdiag.shapes.model import DEFAULT_DRAFT_MODEL_ID, DEFAULT_MODEL_ID


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bf")
    subparsers = parser.add_subparsers(dest="command", required=True)
    register(subparsers)
    return parser


def _run(argv: list[str]) -> tuple[int, argparse.Namespace]:
    parser = _build_parser()
    args = parser.parse_args(argv)
    rc = args.func(args)
    return rc, args


def test_diff_flat_reports_changed_and_unchanged():
    a = {"x": (1, 2), "y": 5, "z": "same"}
    b = {"x": (1, 3), "y": 5, "z": "same"}
    changed, unchanged = diff_flat(a, b)
    assert changed == [("x", (1, 2), (1, 3))]
    assert unchanged == ["y", "z"]


def test_diff_flat_handles_missing_keys():
    a = {"only_a": 1, "shared": 2}
    b = {"only_b": 1, "shared": 2}
    changed, unchanged = diff_flat(a, b)
    changed_keys = {k for k, _, _ in changed}
    assert "only_a" in changed_keys
    assert "only_b" in changed_keys
    assert unchanged == ["shared"]


@pytest.mark.requires_hf_snapshot(DEFAULT_MODEL_ID)
@pytest.mark.requires_hf_snapshot(DEFAULT_DRAFT_MODEL_ID)
def test_cli_diff_64_vs_128_flags_ring_and_pages(capsys):
    rc, _ = _run(
        ["shapes", "--block-size", "64", "--block-size", "128", "--diff", "--kv-len", "65600"]
    )
    out = capsys.readouterr().out
    assert rc == 2  # something changed -> nonzero exit, matches `bf diff`'s convention
    assert "decode/sliding.n_ring" in out
    assert "decode/sliding.aligned_len" in out
    assert "decode/full.max_pages" in out
    assert "ring_capacity/sliding" in out
    assert "-- unchanged" in out
    # GEMM/MoE shapes must not depend on block_size at all
    assert "gemm/full.q_proj" not in out.split("-- unchanged")[0]


@pytest.mark.requires_hf_snapshot(DEFAULT_MODEL_ID)
@pytest.mark.requires_hf_snapshot(DEFAULT_DRAFT_MODEL_ID)
def test_cli_diff_lists_gemm_and_moe_as_unchanged(capsys):
    rc, _ = _run(
        ["shapes", "--block-size", "64", "--block-size", "128", "--diff", "--kv-len", "65600"]
    )
    out = capsys.readouterr().out
    unchanged_section = out.split("-- unchanged")[1]
    assert "gemm/full.q_proj" in unchanged_section
    assert "gemm/lm_head" in unchanged_section
    assert "moe/sparkinfer.w13_fp4" in unchanged_section


def test_cli_diff_requires_exactly_two_block_sizes(capsys):
    rc, _ = _run(["shapes", "--block-size", "64", "--diff"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "exactly two" in err


@pytest.mark.requires_hf_snapshot(DEFAULT_MODEL_ID)
@pytest.mark.requires_hf_snapshot(DEFAULT_DRAFT_MODEL_ID)
def test_cli_diff_no_change_when_same_block_size_twice(capsys):
    rc, _ = _run(["shapes", "--block-size", "64", "--block-size", "64", "--diff"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "(no shapes changed)" in out


@pytest.mark.requires_hf_snapshot(DEFAULT_MODEL_ID)
@pytest.mark.requires_hf_snapshot(DEFAULT_DRAFT_MODEL_ID)
def test_cli_default_block_sizes_are_64_and_128(capsys):
    rc, _ = _run(["shapes", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    docs = [json.loads(line) for line in _split_json_objects(out)]
    block_sizes = {d["block_size"] for d in docs}
    assert block_sizes == {64, 128}


def _split_json_objects(text: str) -> list[str]:
    """``bf shapes --json`` prints one JSON object per block_size,
    back-to-back; split them by tracking brace depth."""
    objs = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                objs.append(text[start : i + 1])
                start = None
    return objs


@pytest.mark.requires_hf_snapshot(DEFAULT_MODEL_ID)
@pytest.mark.requires_hf_snapshot(DEFAULT_DRAFT_MODEL_ID)
def test_cli_json_diff_output_is_valid_json(capsys):
    rc, _ = _run(
        [
            "shapes",
            "--block-size",
            "64",
            "--block-size",
            "128",
            "--diff",
            "--json",
            "--kv-len",
            "65600",
        ]
    )
    out = capsys.readouterr().out
    doc = json.loads(out)
    assert doc["block_size_a"] == 64
    assert doc["block_size_b"] == 128
    assert any(item["key"] == "decode/sliding.n_ring" for item in doc["changed"])
    assert "gemm/lm_head" in doc["unchanged"]


def test_cli_missing_config_reports_error_not_traceback(tmp_path, capsys):
    rc, _ = _run(["shapes", "--model-path", str(tmp_path / "nowhere")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "bf shapes:" in err
    assert "no config.json" in err
