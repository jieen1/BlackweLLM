"""Regression coverage for the worktree-safe checkout guard in ``bfdiag.cli``.

Background (see docs/diagnostics-guide.md, "bf and worktrees", and
notes/2026-08-01-sparkinfer-patch-recovery-and-repro.md §7.1): the installed
``bf`` console script resolves ``import bfdiag`` through the venv's
pip-editable finder, which has ONE checkout's path hard-coded regardless of
which worktree you actually ran ``bf`` from. ``_ensure_correct_checkout``
detects that mismatch and either relaunches against the right checkout or,
if a relaunch already happened once and the mismatch persists, fails loudly
instead of silently running the wrong code.

These tests exercise the real ``_ensure_correct_checkout`` / ``_find_repo_root``
functions (not a synthetic replica of the bug pattern) against throwaway
directories on disk, with ``os.execve`` monkeypatched to a recorder instead
of actually replacing the test process -- this is a real regression gate,
not a pattern demo. A live end-to-end demonstration using two genuine git
worktrees and a genuine `pip install -e` venv is recorded in
notes/2026-08-01-sparkinfer-patch-recovery-and-repro.md.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from bfdiag import cli as bfdiag_cli


def _make_checkout(root: Path) -> Path:
    """A minimal on-disk stand-in for a ``blackwellm`` checkout: just enough
    for ``_find_repo_root`` to recognize it (``bfdiag/__init__.py`` +
    a ``pyproject.toml`` naming this package)."""
    (root / "bfdiag").mkdir(parents=True)
    (root / "bfdiag" / "__init__.py").write_text("")
    (root / "pyproject.toml").write_text('[project]\nname = "blackwellm"\nversion = "0.0.0"\n')
    return root


# -- _find_repo_root ---------------------------------------------------------


def test_find_repo_root_finds_checkout_at_start(tmp_path: Path) -> None:
    checkout = _make_checkout(tmp_path / "repo")
    assert bfdiag_cli._find_repo_root(checkout) == checkout


def test_find_repo_root_walks_up_from_a_subdirectory(tmp_path: Path) -> None:
    checkout = _make_checkout(tmp_path / "repo")
    nested = checkout / "tests" / "test_bfdiag_something"
    nested.mkdir(parents=True)
    assert bfdiag_cli._find_repo_root(nested) == checkout


def test_find_repo_root_returns_none_outside_any_checkout(tmp_path: Path) -> None:
    outside = tmp_path / "not-a-checkout"
    outside.mkdir()
    assert bfdiag_cli._find_repo_root(outside) is None


def test_find_repo_root_ignores_a_bfdiag_dir_with_unrelated_pyproject(tmp_path: Path) -> None:
    # A directory that happens to have a `bfdiag/__init__.py` and *some*
    # pyproject.toml, but not one naming this package, must not be
    # mistaken for a blackwellm checkout.
    decoy = tmp_path / "unrelated"
    (decoy / "bfdiag").mkdir(parents=True)
    (decoy / "bfdiag" / "__init__.py").write_text("")
    (decoy / "pyproject.toml").write_text('[project]\nname = "some-other-package"\n')
    assert bfdiag_cli._find_repo_root(decoy) is None


# -- _ensure_correct_checkout -------------------------------------------------


def test_ensure_correct_checkout_is_a_noop_when_cwd_matches_loaded_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkout = _make_checkout(tmp_path / "repo")
    monkeypatch.setattr(bfdiag_cli, "_loaded_repo_root", lambda: checkout)
    relaunched = []
    monkeypatch.setattr(bfdiag_cli, "_relaunch", lambda argv, env: relaunched.append((argv, env)))

    bfdiag_cli._ensure_correct_checkout(cwd=checkout)

    assert relaunched == []


def test_ensure_correct_checkout_is_a_noop_outside_any_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loaded = _make_checkout(tmp_path / "loaded")
    outside = tmp_path / "somewhere-else"
    outside.mkdir()
    monkeypatch.setattr(bfdiag_cli, "_loaded_repo_root", lambda: loaded)
    relaunched = []
    monkeypatch.setattr(bfdiag_cli, "_relaunch", lambda argv, env: relaunched.append((argv, env)))

    bfdiag_cli._ensure_correct_checkout(cwd=outside)

    assert relaunched == []


def test_ensure_correct_checkout_relaunches_against_the_cwd_checkout_on_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    loaded = _make_checkout(tmp_path / "loaded-elsewhere")
    wanted = _make_checkout(tmp_path / "actual-worktree")
    monkeypatch.setattr(bfdiag_cli, "_loaded_repo_root", lambda: loaded)
    monkeypatch.delenv(bfdiag_cli._RELAUNCH_GUARD_ENV, raising=False)
    monkeypatch.setattr("sys.argv", ["bf", "daemon", "status"])
    relaunched = []
    monkeypatch.setattr(bfdiag_cli, "_relaunch", lambda argv, env: relaunched.append((argv, env)))

    bfdiag_cli._ensure_correct_checkout(cwd=wanted)

    assert len(relaunched) == 1
    argv, env = relaunched[0]
    # Re-launches through the worktree-aware `-m bfdiag.cli` entry point,
    # not by re-executing the (possibly-relocated) console script path.
    assert argv[1:3] == ["-m", "bfdiag.cli"]
    assert argv[3:] == ["daemon", "status"]
    assert env[bfdiag_cli._RELAUNCH_GUARD_ENV] == "1"
    # PYTHONPATH is pointed at the *wanted* checkout so the stdlib
    # PathFinder resolves it before the pip-editable finder is consulted.
    assert str(wanted) in env["PYTHONPATH"].split(os.pathsep)

    notice = capsys.readouterr().err
    assert str(loaded) in notice
    assert str(wanted) in notice


def test_ensure_correct_checkout_preserves_existing_pythonpath(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loaded = _make_checkout(tmp_path / "loaded-elsewhere")
    wanted = _make_checkout(tmp_path / "actual-worktree")
    monkeypatch.setattr(bfdiag_cli, "_loaded_repo_root", lambda: loaded)
    monkeypatch.delenv(bfdiag_cli._RELAUNCH_GUARD_ENV, raising=False)
    monkeypatch.setenv("PYTHONPATH", "/some/preexisting/entry")
    relaunched = []
    monkeypatch.setattr(bfdiag_cli, "_relaunch", lambda argv, env: relaunched.append((argv, env)))

    bfdiag_cli._ensure_correct_checkout(cwd=wanted)

    _, env = relaunched[0]
    entries = env["PYTHONPATH"].split(os.pathsep)
    assert entries[0] == str(wanted)
    assert "/some/preexisting/entry" in entries


def test_ensure_correct_checkout_fails_loudly_if_a_relaunch_already_happened(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If the guard env var is already set and the mismatch is STILL there,
    this must not try to relaunch again (no infinite loop) and must not
    silently proceed on the wrong checkout either."""
    loaded = _make_checkout(tmp_path / "loaded-elsewhere")
    wanted = _make_checkout(tmp_path / "actual-worktree")
    monkeypatch.setattr(bfdiag_cli, "_loaded_repo_root", lambda: loaded)
    monkeypatch.setenv(bfdiag_cli._RELAUNCH_GUARD_ENV, "1")
    relaunched = []
    monkeypatch.setattr(bfdiag_cli, "_relaunch", lambda argv, env: relaunched.append((argv, env)))

    with pytest.raises(SystemExit):
        bfdiag_cli._ensure_correct_checkout(cwd=wanted)

    assert relaunched == []


def test_main_calls_ensure_correct_checkout_before_dispatching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`main()` must check the checkout before touching argparse/subcommand
    discovery -- otherwise a mismatched checkout could already have done
    something (e.g. import a broken sibling subpackage) before the guard
    gets a chance to run."""
    calls: list[str] = []
    monkeypatch.setattr(bfdiag_cli, "_ensure_correct_checkout", lambda: calls.append("checked"))
    monkeypatch.setattr(
        bfdiag_cli,
        "build_parser",
        lambda debug=False: calls.append("built") or _DummyParser(),
    )

    bfdiag_cli.main(["--debug"])

    assert calls == ["checked", "built"]


class _DummyParser:
    def parse_args(self, argv):
        class _Args:
            func = None

        return _Args()

    def print_help(self):
        pass
