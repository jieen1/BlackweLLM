"""``runtime.checkpoints`` resolution logic -- the module every migrated
``scripts/`` file now trusts instead of hardcoding its own path.

**Why this needs its own tests, not just the migration.** A resolution
point that is wrong once is worse than 22 independent hardcoded paths that
are each wrong independently: every caller now shares the same bug, and
"but it worked when I ran it" stops being useful evidence once one machine's
lucky cache layout (exactly one snapshot dir) is doing the proving. These
tests exercise the module against synthetic HF-cache-shaped directories
under ``tmp_path`` rather than the real ``~/.cache/huggingface/hub`` --
same reasoning as ``tests/test_registry_quant_format_gate.py``'s synthetic
configs -- so they hold on a machine with an empty HuggingFace cache (the
torch-free CI job) and actually exercise the failure paths (missing
checkpoint, ambiguous multiple snapshots) that a real, currently-healthy
local cache never hits.

Three behaviors matter enough to pin down explicitly:

1. Exactly one snapshot resolves without an env var -- the common case,
   and the reason dynamic resolution (not a hardcoded hash) was chosen at
   all: a real snapshot hash rotates on re-download, and every migrated
   script needs to keep working when it does.
2. Zero or 2+ snapshots refuse to guess, loudly, naming the env var that
   fixes it -- silently picking one (``sorted()[0]``, "newest mtime", or
   similar) would reintroduce exactly the "quietly grading the wrong
   checkpoint" bug this whole round exists to close, just one layer lower.
3. The env var override is honored, and validated (a typo'd override must
   fail loudly, not silently fall through to the default checkpoint --
   that fallback would defeat the point of an explicit override).
"""

from __future__ import annotations

import pytest

from runtime.checkpoints import (
    MODELOPT_CHECKPOINT_REPO,
    QSR_QWEN36_MODELOPT_CHECKPOINT,
    QSR_QWEN36_STANDARD_CHECKPOINT,
    STANDARD_CHECKPOINT_REPO,
    CheckpointNotFoundError,
    _repo_cache_dirname,
    _resolve_snapshot_dir,
    modelopt_checkpoint_path,
    standard_checkpoint_path,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Neither override env var may leak in from the real shell -- a
    developer's exported ``QSR_QWEN36_STANDARD_CHECKPOINT`` must not change
    what these tests exercise."""
    monkeypatch.delenv(QSR_QWEN36_STANDARD_CHECKPOINT, raising=False)
    monkeypatch.delenv(QSR_QWEN36_MODELOPT_CHECKPOINT, raising=False)


def _fake_hub_cache(tmp_path, repo_id: str, *snapshot_names: str):
    """Build ``<tmp_path>/hub/models--<org>--<repo>/snapshots/<name>/`` for
    each name in ``snapshot_names`` -- the directory shape a real HF hub
    cache download produces, minus the actual checkpoint files (nothing
    here needs them)."""
    cache_dir = tmp_path / "hub" / _repo_cache_dirname(repo_id)
    snapshots_dir = cache_dir / "snapshots"
    for name in snapshot_names:
        (snapshots_dir / name).mkdir(parents=True)
    return cache_dir


class TestSingleSnapshotResolves:
    def test_exactly_one_snapshot_is_picked_without_a_hardcoded_hash(self, tmp_path, monkeypatch):
        cache_dir = _fake_hub_cache(tmp_path, STANDARD_CHECKPOINT_REPO, "abc123")
        monkeypatch.setattr("runtime.checkpoints._DEFAULT_HF_HUB_CACHE", tmp_path / "hub")
        resolved = _resolve_snapshot_dir(
            STANDARD_CHECKPOINT_REPO, env_var=QSR_QWEN36_STANDARD_CHECKPOINT
        )
        assert resolved == cache_dir / "snapshots" / "abc123"

    def test_a_rotated_hash_is_picked_up_without_editing_anything(self, tmp_path, monkeypatch):
        """The whole point of resolving dynamically instead of pinning a
        hash: re-downloading a checkpoint under a new snapshot hash must
        not require touching this module (or, pre-migration, 22 scripts)."""
        monkeypatch.setattr("runtime.checkpoints._DEFAULT_HF_HUB_CACHE", tmp_path / "hub")
        _fake_hub_cache(tmp_path, STANDARD_CHECKPOINT_REPO, "old-hash-111")
        first = _resolve_snapshot_dir(
            STANDARD_CHECKPOINT_REPO, env_var=QSR_QWEN36_STANDARD_CHECKPOINT
        )
        assert first.name == "old-hash-111"

        # Simulate a fresh download replacing the snapshot (old one gone,
        # new hash present) -- the realistic re-download shape, not an
        # addition.
        import shutil

        shutil.rmtree(tmp_path / "hub" / _repo_cache_dirname(STANDARD_CHECKPOINT_REPO))
        _fake_hub_cache(tmp_path, STANDARD_CHECKPOINT_REPO, "new-hash-222")
        second = _resolve_snapshot_dir(
            STANDARD_CHECKPOINT_REPO, env_var=QSR_QWEN36_STANDARD_CHECKPOINT
        )
        assert second.name == "new-hash-222"


class TestAmbiguityAndAbsenceRefuseToGuess:
    def test_missing_cache_directory_raises_naming_the_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setattr("runtime.checkpoints._DEFAULT_HF_HUB_CACHE", tmp_path / "hub")
        with pytest.raises(CheckpointNotFoundError) as excinfo:
            _resolve_snapshot_dir(STANDARD_CHECKPOINT_REPO, env_var=QSR_QWEN36_STANDARD_CHECKPOINT)
        assert QSR_QWEN36_STANDARD_CHECKPOINT in str(excinfo.value)
        assert STANDARD_CHECKPOINT_REPO in str(excinfo.value)

    def test_empty_snapshots_dir_raises_rather_than_returning_nothing(self, tmp_path, monkeypatch):
        """An interrupted/corrupted download can leave ``snapshots/`` empty
        -- this must not be confused with "exactly one, use it" or silently
        return a directory that turns out to hold no checkpoint files."""
        monkeypatch.setattr("runtime.checkpoints._DEFAULT_HF_HUB_CACHE", tmp_path / "hub")
        (tmp_path / "hub" / _repo_cache_dirname(STANDARD_CHECKPOINT_REPO) / "snapshots").mkdir(
            parents=True
        )
        with pytest.raises(CheckpointNotFoundError):
            _resolve_snapshot_dir(STANDARD_CHECKPOINT_REPO, env_var=QSR_QWEN36_STANDARD_CHECKPOINT)

    def test_two_snapshots_refuses_to_pick_one(self, tmp_path, monkeypatch):
        """This is the case a naive ``sorted(...)[0]`` would get away with
        silently -- and silently is exactly the failure mode this whole
        round of work exists to close one layer up (scripts quietly
        grading the wrong checkpoint). Both candidate names must appear in
        the error so a human can tell which to pick."""
        monkeypatch.setattr("runtime.checkpoints._DEFAULT_HF_HUB_CACHE", tmp_path / "hub")
        _fake_hub_cache(tmp_path, STANDARD_CHECKPOINT_REPO, "hash-aaa", "hash-bbb")
        with pytest.raises(CheckpointNotFoundError) as excinfo:
            _resolve_snapshot_dir(STANDARD_CHECKPOINT_REPO, env_var=QSR_QWEN36_STANDARD_CHECKPOINT)
        message = str(excinfo.value)
        assert "hash-aaa" in message
        assert "hash-bbb" in message
        assert QSR_QWEN36_STANDARD_CHECKPOINT in message


class TestEnvVarOverride:
    def test_override_is_used_verbatim(self, tmp_path, monkeypatch):
        override_dir = tmp_path / "somewhere" / "else"
        override_dir.mkdir(parents=True)
        monkeypatch.setenv(QSR_QWEN36_STANDARD_CHECKPOINT, str(override_dir))
        resolved = _resolve_snapshot_dir(
            STANDARD_CHECKPOINT_REPO, env_var=QSR_QWEN36_STANDARD_CHECKPOINT
        )
        assert resolved == override_dir

    def test_override_does_not_require_the_real_cache_to_exist(self, tmp_path, monkeypatch):
        """No ``_DEFAULT_HF_HUB_CACHE`` patch here -- the override must work
        even when the default HF hub cache location is untouched/absent,
        which is exactly the "empty HuggingFace cache" case this test suite
        must pass under."""
        override_dir = tmp_path / "my_checkpoint"
        override_dir.mkdir()
        monkeypatch.setenv(QSR_QWEN36_MODELOPT_CHECKPOINT, str(override_dir))
        resolved = _resolve_snapshot_dir(
            MODELOPT_CHECKPOINT_REPO, env_var=QSR_QWEN36_MODELOPT_CHECKPOINT
        )
        assert resolved == override_dir

    def test_a_nonexistent_override_fails_loudly_not_silently(self, monkeypatch):
        """A typo'd override must not silently fall back to the default HF
        hub cache lookup -- that would defeat the entire point of an
        explicit override (the user asked for THIS path, not "try this,
        else guess")."""
        monkeypatch.setenv(QSR_QWEN36_STANDARD_CHECKPOINT, "/definitely/not/a/real/path")
        with pytest.raises(CheckpointNotFoundError) as excinfo:
            _resolve_snapshot_dir(STANDARD_CHECKPOINT_REPO, env_var=QSR_QWEN36_STANDARD_CHECKPOINT)
        assert QSR_QWEN36_STANDARD_CHECKPOINT in str(excinfo.value)


class TestTheTwoPublicFunctionsAreIndependent:
    """``standard_checkpoint_path()`` and ``modelopt_checkpoint_path()``
    must never conflate the two checkpoints -- neither one falling back to
    the other on failure is the load-bearing property (see the module
    docstring's "no silent fallback" paragraph): a missing standard
    checkpoint must be reported as missing, not quietly answered with
    modelopt's path.
    """

    def test_each_function_reads_only_its_own_env_var(self, tmp_path, monkeypatch):
        std_dir = tmp_path / "std"
        std_dir.mkdir()
        monkeypatch.setenv(QSR_QWEN36_STANDARD_CHECKPOINT, str(std_dir))
        # Modelopt's cache is absent and its env var is unset -- must fail
        # independently of the standard checkpoint resolving fine.
        monkeypatch.setattr("runtime.checkpoints._DEFAULT_HF_HUB_CACHE", tmp_path / "hub")

        assert standard_checkpoint_path() == str(std_dir)
        with pytest.raises(CheckpointNotFoundError):
            modelopt_checkpoint_path()

    def test_module_import_alone_never_touches_the_filesystem(self):
        """``import runtime.checkpoints`` must always succeed -- including
        with an empty HF cache -- because resolution is lazy. Re-importing
        here (already imported at module load above) is itself the
        assertion: if resolution were eager, collecting this test file at
        all would have already raised on a machine with no checkpoints."""
        import runtime.checkpoints  # noqa: PLC0415

        assert callable(runtime.checkpoints.standard_checkpoint_path)
        assert callable(runtime.checkpoints.modelopt_checkpoint_path)
