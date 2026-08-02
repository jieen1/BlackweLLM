"""Test fakes must accept every keyword ``ServerEngine`` actually passes.

`runtime/backends/protocol.py::check_conformance` verifies that a backend has
the members its capabilities claim. It says nothing about whether those members
can be *called* the way the engine calls them, and that gap has already cost a
day.

2026-08-02: E2-b added a keyword-only ``params_per_slot`` to
``prefill_chunked_begin`` and updated every fake on its own branch. Step 7-b,
developed in parallel, created ``tests/test_engine_prefix_cache_admission.py``
with two fakes carrying the older signature. Each branch was green. Merged, the
engine's call raised ``TypeError`` inside the admission ``try/except``, which
fails the futures and drops the requests -- leaving nothing active and nothing
waiting, so ``_step_sync`` fell into its idle blocking read and waited on a pipe
no one would write to. The suite went from 63 seconds to never finishing, and
pytest printed nothing at all: a signature mismatch surfaced as a hang.

Static analysis, deliberately. Importing the test modules to introspect them
would drag torch into the torch-free CI job, and the question -- "does this
`def` accept that keyword" -- is answerable from the syntax tree alone.

Known limits, stated rather than hidden: only ``self.runner.<method>(...)`` call
sites are scanned, so a call through a local alias is missed, and a fake using
``**kwargs`` is accepted without inspecting what it does with them.
"""

from __future__ import annotations

import ast
import pathlib

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_ENGINE = _REPO_ROOT / "server" / "engine.py"


def _runner_call_keywords() -> dict[str, set[str]]:
    """method name -> every keyword ``ServerEngine`` passes to it."""
    tree = ast.parse(_ENGINE.read_text(encoding="utf-8"))
    found: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # self.runner.<method>(...)
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "runner"
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "self"
        ):
            names = {kw.arg for kw in node.keywords if kw.arg is not None}
            if names:
                found.setdefault(func.attr, set()).update(names)
    return found


def _fake_definitions(method: str) -> list[tuple[pathlib.Path, ast.FunctionDef]]:
    """Every ``def <method>`` defined inside tests/."""
    out = []
    for path in sorted((_REPO_ROOT / "tests").glob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a broken test file is its own failure
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method:
                out.append((path, node))
    return out


def _accepted_keywords(fn: ast.FunctionDef) -> set[str] | None:
    """Keywords this def accepts, or None if it takes ``**kwargs`` (accepts all)."""
    if fn.args.kwarg is not None:
        return None
    return {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}


def test_test_fakes_accept_every_keyword_the_engine_passes():
    call_keywords = _runner_call_keywords()
    assert call_keywords, (
        "no self.runner.<method>(keyword=...) call sites found in server/engine.py -- "
        "the scanner stopped matching rather than the engine losing its keywords"
    )

    problems: list[str] = []
    for method, keywords in sorted(call_keywords.items()):
        for path, fn in _fake_definitions(method):
            accepted = _accepted_keywords(fn)
            if accepted is None:
                continue
            missing = keywords - accepted
            if missing:
                problems.append(
                    f"{path.relative_to(_REPO_ROOT)}:{fn.lineno} "
                    f"{method}() rejects {sorted(missing)} which server/engine.py passes"
                )

    assert not problems, (
        "a test fake cannot be called the way the engine calls it. This does not "
        "fail loudly at runtime -- ServerEngine's admission try/except swallows the "
        "TypeError, drops the requests, and the engine blocks on its idle read, so "
        "the suite hangs with no output:\n  " + "\n  ".join(problems)
    )
