"""Every ``scripts/`` file must resolve the Qwen3.6 checkpoint through
``runtime.checkpoints``, never a hardcoded ``models--...`` path literal.

**The failure mode this guards.** Before ``runtime/checkpoints.py`` existed,
22 separate scripts each declared their own ``MODEL_PATH = (...)`` string
constant, hardcoding a full local HF hub cache path -- 20 of them pinned to
``nvidia/Qwen3.6-27B-NVFP4`` (the modelopt checkpoint every script happened
to be written against first), 2 pinned to ``unsloth/Qwen3.6-27B-NVFP4``.
Nothing forced any of them to notice when the user declared
``unsloth/Qwen3.6-27B-NVFP4`` the *standard* checkpoint that actually ships
(``3c2d0a8``) -- the entire B1/B2/B3 measurement corpus quietly kept
grading a checkpoint nobody serves anymore, and nobody could tell by
looking at any individual script (each one looked complete and correct in
isolation; only a cross-script grep revealed the drift).

Migrating the 22 (``checkpoint-unify-20260803``) fixes today's drift, but a
resolution point only stays authoritative if nothing can quietly bypass it.
The next person adding a probe script under time pressure will reach for
"just copy the ``MODEL_PATH`` block from a script that already works" --
which reintroduces exactly this bug, silently, one script at a time, the
same way it happened the first time. This test is what makes that copy-paste
fail loudly at CI time instead of being discovered the next time someone
audits ``scripts/`` by hand.

**What "hardcoded" means here, precisely.** Any string literal anywhere in a
``scripts/*.py`` file that contains the HF hub cache directory name for
either checkpoint (``models--nvidia--Qwen3.6-27B-NVFP4`` or
``models--unsloth--Qwen3.6-27B-NVFP4``) -- whether assigned to a variable
named ``MODEL_PATH``, ``DEFAULT_MODEL_PATH``, ``CKPT``, or anything else,
and whether written as one literal or split across implicit string
concatenation (``"a" "b"``, what every migrated script used for its
multi-line path). Detected via the AST, not a plain grep, specifically so
implicit-concatenation splits and f-strings can't dodge it (a plain
line-based grep only sees each physical line, and would miss a literal
deliberately split so no single line contains the full substring).

**What this deliberately does NOT flag**, and why:

- ``runtime/checkpoints.py`` itself -- the one place allowed to know these
  strings, by construction (it holds the HF *repo ids*, e.g.
  ``"unsloth/Qwen3.6-27B-NVFP4"``, not a local cache path -- see below for
  why even that doesn't trip the same substring check).
- Anything outside ``scripts/`` -- ``runtime/``, ``server/``, ``tests/``,
  ``notes/``, ``docs/`` are out of this test's scope; this guards the
  specific "22 scripts, 22 copies" failure mode, not a repo-wide ban on the
  string "Qwen3.6-27B-NVFP4" (which appears legitimately all over the
  docs/notes as prose).
- Other checkpoints entirely -- e.g. ``scripts/laguna_verify_prefill_jit_
  and_greedy.py`` hardcodes a real local path to
  ``poolside/Laguna-S-2.1-NVFP4``, a different model with no standard/
  modelopt duality and no ``runtime.checkpoints`` entry; that is legitimate
  and this test does not touch it. The check below matches the two
  checkpoint directory names specifically, not a generic ``models--``
  prefix, so it cannot false-positive on Laguna or on a future unrelated
  checkpoint someone hardcodes deliberately for a good reason.

Pure filesystem + AST parsing, no imports of ``scripts/`` code and no
checkpoint access -- passes identically whether or not either checkpoint is
actually present on the machine (including the torch-free CI job's empty
HF cache).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"

#: The HF hub cache directory name for each checkpoint -- exactly the
#: substring every one of the original 22 hardcoded copies embedded
#: (``models--<org>--<repo>``, the literal `~/.cache/huggingface/hub`
#: subdirectory naming). Deliberately NOT the bare repo id
#: (``"unsloth/Qwen3.6-27B-NVFP4"``) -- that string legitimately appears in
#: ``runtime/checkpoints.py`` itself (``STANDARD_CHECKPOINT_REPO`` /
#: ``MODELOPT_CHECKPOINT_REPO``) and in module docstrings/comments across
#: the migrated scripts explaining *why* they resolve the way they do;
#: flagging it would make this test fail on the very comments that document
#: the fix.
_FORBIDDEN_SUBSTRINGS = (
    "models--nvidia--Qwen3.6-27B-NVFP4",
    "models--unsloth--Qwen3.6-27B-NVFP4",
)


def _script_files() -> list[Path]:
    assert _SCRIPTS_DIR.is_dir(), f"expected {_SCRIPTS_DIR} to exist"
    return sorted(_SCRIPTS_DIR.glob("*.py"))


def _hardcoded_literals(path: Path) -> list[tuple[int, str]]:
    """``(line, matched substring)`` for every string constant in ``path``
    that embeds a forbidden checkpoint cache-directory name.

    Walks the AST rather than grepping lines so an implicit-concatenation
    split (``"models--nvidia--Qwen3.6-27B-NVFP4/" "snapshots/" + h``) is
    still caught as one already-joined ``ast.Constant`` -- the exact shape
    every migrated script's old ``MODEL_PATH = (...)`` block used.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for forbidden in _FORBIDDEN_SUBSTRINGS:
                if forbidden in node.value:
                    hits.append((node.lineno, forbidden))
    return hits


class TestNoHardcodedCheckpointPathInScripts:
    @pytest.mark.parametrize("path", _script_files(), ids=lambda p: p.name)
    def test_script_has_no_hardcoded_checkpoint_literal(self, path: Path) -> None:
        hits = _hardcoded_literals(path)
        assert not hits, (
            f"{path.name} hardcodes a checkpoint path literal: "
            f"{[(f'line {ln}', s) for ln, s in hits]}. Resolve it through "
            "runtime.checkpoints.standard_checkpoint_path() (the model this "
            "runtime ships) or .modelopt_checkpoint_path() (only if this "
            "script specifically exercises the modelopt adapter or "
            "reproduces a modelopt-specific historical measurement -- say "
            "which, in a comment, same as the other scripts pinned to "
            "modelopt) instead of a literal path."
        )

    def test_the_gate_actually_finds_something_on_a_known_bad_file(self, tmp_path) -> None:
        """The gate must not be vacuous -- prove it fires on exactly the
        pattern it exists to catch (the old, pre-migration style), so a
        change to ``_hardcoded_literals`` that accidentally stops matching
        anything is itself caught here rather than by every parametrized
        case above going quiet at once.
        """
        bad_script = tmp_path / "would_be_scripts" / "b9_hypothetical_probe.py"
        bad_script.parent.mkdir()
        bad_script.write_text(
            "MODEL_PATH = (\n"
            '    "/home/bot/.cache/huggingface/hub/models--nvidia--Qwen3.6-27B-NVFP4/"\n'
            '    "snapshots/0893e1606ff3d5f97a441f405d5fc541a6bdf404"\n'
            ")\n"
        )
        hits = _hardcoded_literals(bad_script)
        assert hits == [(2, "models--nvidia--Qwen3.6-27B-NVFP4")]

    def test_the_gate_ignores_the_bare_repo_id(self, tmp_path) -> None:
        """A script that only names the HF repo id (as
        ``runtime/checkpoints.py`` and every migrated script's explanatory
        comments do) must not trip this gate -- only a real local cache
        path does.
        """
        fine_script = tmp_path / "fine.py"
        fine_script.write_text(
            "# See unsloth/Qwen3.6-27B-NVFP4 and nvidia/Qwen3.6-27B-NVFP4.\n"
            "from runtime.checkpoints import standard_checkpoint_path\n"
            "MODEL_PATH = standard_checkpoint_path()\n"
        )
        assert _hardcoded_literals(fine_script) == []

    def test_scripts_dir_is_not_accidentally_empty(self) -> None:
        """Guards against the parametrization silently collecting zero
        files (e.g. a path typo in ``_SCRIPTS_DIR``) and the class above
        passing vacuously with nothing to check."""
        files = _script_files()
        assert len(files) > 20, (
            f"only found {len(files)} files under {_SCRIPTS_DIR} -- expected "
            "scripts/ to hold many more; is _SCRIPTS_DIR pointed at the "
            "right place?"
        )
