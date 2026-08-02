"""A script must derive its repo root from ``__file__``, never hardcode one.

Scripts under ``scripts/`` prepend the repo root to ``sys.path`` before
importing ``runtime``, because the editable ``blackwellm`` install otherwise
resolves ``runtime`` to whichever checkout it was installed from regardless
of cwd -- so a bare ``python scripts/foo.py`` from a worktree silently grades
main's code. The established form is::

    _ROOT = str(Path(__file__).resolve().parent.parent)
    sys.path.insert(0, _ROOT)
    import runtime
    assert runtime.__file__.startswith(_ROOT), ...

Eight scripts instead wrote the literal path of the throwaway worktree they
happened to be authored in -- ``_ROOT = "/home/bot/project/qsr-w-b3a"`` and
friends. That is wrong the moment anyone runs them from anywhere else, and it
becomes *fatal* once the worktree is cleaned up: the assert fires and the
script cannot run at all. Which is exactly what happened on 2026-08-03 --
retiring 39 stale worktrees turned eight measurement scripts into
hard-failures, discovered only when one of them was needed to re-measure MTP
acceptance on the standard checkpoint.

Two properties make this worth a gate rather than a one-time cleanup:

- The failure is delayed and its cause is offscreen. The script breaks
  because of an unrelated housekeeping action taken days later, and the
  traceback names an editable install, not a deleted directory.
- It regenerates naturally. Every one of these was written inside a worktree,
  where hardcoding the path one is standing in looks correct and works
  perfectly until it doesn't.

The same shape as ``tests/test_scripts_checkpoint_resolution_gate.py``: the
literal is banned so the derivation has to be used. This is the repo-root
half; that one is the checkpoint-path half.

Pure AST/source inspection -- no imports, no torch, no GPU, no checkpoints.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"

#: Any absolute path under the directory that holds this repo's checkouts.
#: Deliberately broad -- the point is that NO sibling-checkout path belongs in
#: a script, not merely the eight that were wrong on 2026-08-03.
_HARDCODED_CHECKOUT_PATH = re.compile(r'["\']/home/bot/project/[^"\']*["\']')

#: Assignments the ban does not apply to. `scripts/` legitimately refers to
#: sibling *repositories* (sparkinfer) via env-var-backed resolution; what is
#: banned is a literal standing in for a path that should be derived.
_ALLOWED_PREFIXES = ("_DEFAULT_",)


def _script_files() -> list[Path]:
    return sorted(p for p in _SCRIPTS_DIR.glob("*.py") if p.name != "__init__.py")


def test_there_are_scripts_to_check():
    """Guard against the glob silently matching nothing and the gate passing vacuously."""
    assert len(_script_files()) > 20


@pytest.mark.parametrize("script", _script_files(), ids=lambda p: p.name)
def test_script_does_not_hardcode_a_checkout_path(script: Path):
    source = script.read_text()
    offenders: list[str] = []
    for lineno, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#") or not _HARDCODED_CHECKOUT_PATH.search(line):
            continue
        if any(stripped.startswith(prefix) for prefix in _ALLOWED_PREFIXES):
            continue
        offenders.append(f"  {script.name}:{lineno}: {stripped}")

    assert not offenders, (
        "hardcoded checkout path(s) found:\n"
        + "\n".join(offenders)
        + "\n\nDerive it instead: _ROOT = str(Path(__file__).resolve().parent.parent). "
        "A literal path to a sibling checkout breaks for anyone running from a "
        "different worktree, and stops working entirely once that worktree is "
        "removed -- see this file's docstring."
    )


def test_scripts_that_insert_a_root_derive_it_from_file():
    """The positive form: if a script builds ``sys.path`` at all, it must derive the root.

    Complements the ban above. A script could avoid the banned literal by
    building the same path through string concatenation or ``os.environ`` with
    a hardcoded default; requiring the ``Path(__file__)`` derivation makes the
    intended form the only passing one.
    """
    bad: list[str] = []
    for script in _script_files():
        source = script.read_text()
        if "sys.path.insert" not in source:
            continue
        if "Path(__file__)" not in source:
            bad.append(script.name)

    assert not bad, (
        f"{bad} modify sys.path without deriving the path from __file__ -- "
        "they will import from whichever checkout the editable install points "
        "at, not the one they live in"
    )
