"""Unit coverage for ``runtime.backends._sparkinfer_import``.

Background (see the module's own docstring and
notes/2026-08-01-sparkinfer-patch-recovery-and-repro.md §7.2):
``BF_SPARKINFER_PATH`` used to be honored by ``laguna_sparkinfer_attn.py``
and ``laguna_sparkinfer_moe.py`` inserting it into ``sys.path`` right before
their own ``from sparkinfer...`` import -- but ``laguna.py``'s
``_patch_moe_sparkinfer`` did an EARLIER, uncontrolled direct
``from sparkinfer.moe.fused_moe._impl import ...``, which was the real
first touch of the ``sparkinfer`` name on the actual Laguna startup path.
Once that happens, no later ``sys.path`` edit can redirect the already-
cached module, so the env var silently did nothing.

This module has zero torch/sparkinfer dependency by design (pure
``os``/``pathlib``/``sys``), so these tests run in the no-torch dev venv
too and don't need a real sparkinfer checkout on disk -- they fabricate a
fake ``sparkinfer`` entry in ``sys.modules`` to exercise the "already
imported" branch, which is the exact scenario the bug lived in.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from runtime.backends import _sparkinfer_import as sut


@pytest.fixture(autouse=True)
def _clean_sparkinfer_state(monkeypatch: pytest.MonkeyPatch):
    """Every test gets a pristine ``sys.modules``/``sys.path`` w.r.t.
    ``sparkinfer`` -- this module must never actually be importable as part
    of these tests (that's the whole point: these test the *resolver*,
    not sparkinfer itself)."""
    monkeypatch.delenv("BF_SPARKINFER_PATH", raising=False)
    monkeypatch.delitem(sys.modules, "b12x", raising=False)
    original_path = list(sys.path)
    yield
    sys.path[:] = original_path


def _fake_module_at(root: Path) -> types.ModuleType:
    fake = types.ModuleType("b12x")
    fake.__file__ = str(root / "b12x" / "__init__.py")
    return fake


def test_requested_path_defaults_when_env_unset() -> None:
    assert sut.requested_sparkinfer_path() == sut.DEFAULT_SPARKINFER_PATH


def test_requested_path_honors_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BF_SPARKINFER_PATH", "/some/other/checkout")
    assert sut.requested_sparkinfer_path() == "/some/other/checkout"


def test_ensure_sparkinfer_path_inserts_requested_path_when_not_yet_imported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BF_SPARKINFER_PATH", "/some/other/checkout")
    assert "b12x" not in sys.modules

    result = sut.ensure_sparkinfer_path()

    assert result == "/some/other/checkout"
    assert sys.path[0] == "/some/other/checkout"


def test_ensure_sparkinfer_path_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BF_SPARKINFER_PATH", "/some/other/checkout")

    first = sut.ensure_sparkinfer_path()
    path_after_first = list(sys.path)
    second = sut.ensure_sparkinfer_path()

    assert first == second == "/some/other/checkout"
    assert sys.path == path_after_first  # no duplicate insertion


def test_ensure_sparkinfer_path_noop_when_already_loaded_from_the_requested_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact non-bug case: something already imported sparkinfer, but
    from precisely the checkout that was requested anyway -- this must be a
    silent, error-free no-op."""
    monkeypatch.setenv("BF_SPARKINFER_PATH", "/requested/checkout")
    sys.modules["b12x"] = _fake_module_at(Path("/requested/checkout"))

    result = sut.ensure_sparkinfer_path()

    assert result == "/requested/checkout"


def test_ensure_sparkinfer_path_raises_when_already_loaded_from_elsewhere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The actual bug scenario, caught instead of silently ignored: some
    uncontrolled import beat every known call site to `sparkinfer`, so the
    requested override can no longer take effect. This must fail loudly,
    not silently keep serving the wrong checkout."""
    monkeypatch.setenv("BF_SPARKINFER_PATH", "/requested/checkout")
    sys.modules["b12x"] = _fake_module_at(Path("/some/uncontrolled/checkout"))

    with pytest.raises(RuntimeError, match="ALREADY imported"):
        sut.ensure_sparkinfer_path()


def test_ensure_sparkinfer_path_does_not_duplicate_an_existing_sys_path_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BF_SPARKINFER_PATH", "/some/other/checkout")
    sys.path.insert(0, "/some/other/checkout")
    path_before = list(sys.path)

    sut.ensure_sparkinfer_path()

    assert sys.path == path_before
