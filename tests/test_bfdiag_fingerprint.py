"""fingerprint.capture() must never raise, even with no GPU or sparkinfer
checkout -- that's the whole point of this module (see
``notes/2026-07-27-bfdiag-run-records.md``).

No test in this file ever invokes the real ``nvidia-smi`` binary: GPU
queries are exercised entirely through monkeypatched ``fingerprint._run``
and ``fingerprint.shutil.which`` so these tests are safe to run alongside
another process actively using the one GPU in the box.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from bfdiag.record import fingerprint
from bfdiag.record.schema import GpuInfo, ModelInfo, WorkloadInfo

# A real "nvidia-smi --query-gpu=... --format=csv,noheader,nounits" line,
# captured once from this machine before GPU-adjacent commands were banned
# for this task -- reused as a static fixture so parsing logic is tested
# against real-world output without ever shelling out again.
_REAL_CORE_CSV_LINE = (
    "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, "
    "610.47, 2272, 13365, 300.00, 97887, Enabled"
)


def _run_git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _init_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _run_git(["init", "-q"], cwd=path)
    _run_git(["config", "user.email", "bfdiag-test@example.com"], cwd=path)
    _run_git(["config", "user.name", "bfdiag-test"], cwd=path)
    (path / "README").write_text("hello\n", encoding="utf-8")
    _run_git(["add", "README"], cwd=path)
    _run_git(["commit", "-q", "-m", "initial"], cwd=path)


# --- git ---------------------------------------------------------------


def test_capture_git_repo_missing_path_returns_all_none() -> None:
    info = fingerprint.capture_git_repo("/no/such/path/at/all")
    assert info.sha is None
    assert info.dirty is None
    assert info.branch is None


def test_capture_git_repo_none_path_returns_all_none() -> None:
    info = fingerprint.capture_git_repo(None)
    assert info == fingerprint.GitRepoInfo()


def test_capture_git_repo_clean_checkout(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    info = fingerprint.capture_git_repo(str(tmp_path))
    assert info.sha is not None and len(info.sha) == 40
    assert info.dirty is False
    assert info.branch is not None


def test_capture_git_repo_dirty_checkout(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "README").write_text("changed\n", encoding="utf-8")
    info = fingerprint.capture_git_repo(str(tmp_path))
    assert info.dirty is True


def test_capture_git_never_raises_when_all_repos_missing(tmp_path: Path) -> None:
    missing = str(tmp_path / "does-not-exist")
    result = fingerprint.capture_git({"qwen-sm120-runtime": missing, "sparkinfer": missing})
    assert set(result) == {"qwen-sm120-runtime", "sparkinfer"}
    for info in result.values():
        assert info.sha is None
        assert info.dirty is None


def test_capture_git_honors_env_overrides(tmp_path: Path, monkeypatch) -> None:
    _init_repo(tmp_path)
    monkeypatch.setenv("QSR_REPO_SPARKINFER", str(tmp_path))
    result = fingerprint.capture_git({"qwen-sm120-runtime": "/no/such/path"})
    assert result["sparkinfer"].sha is not None


def test_default_runtime_repo_path_is_the_executing_worktree() -> None:
    assert fingerprint._repo_paths()["qwen-sm120-runtime"] == str(
        Path(fingerprint.__file__).resolve().parents[2]
    )


# --- env -----------------------------------------------------------------


def test_capture_env_filters_by_prefix(monkeypatch) -> None:
    monkeypatch.setenv("QSR_TEST_MARKER", "1")
    monkeypatch.setenv("CUDA_TEST_MARKER", "3")
    monkeypatch.setenv("TORCH_TEST_MARKER", "4")
    monkeypatch.setenv("NVIDIA_TEST_MARKER", "5")
    monkeypatch.setenv("SOME_UNRELATED_VAR", "should-not-appear")
    monkeypatch.setenv("MY_QSR_FOO", "not a real prefix match, does not start with QSR_")

    env = fingerprint.capture_env()

    assert env["QSR_TEST_MARKER"] == "1"
    assert env["CUDA_TEST_MARKER"] == "3"
    assert env["TORCH_TEST_MARKER"] == "4"
    assert env["NVIDIA_TEST_MARKER"] == "5"
    assert "SOME_UNRELATED_VAR" not in env
    assert "MY_QSR_FOO" not in env


# --- gpu (mocked -- never touches the real nvidia-smi) --------------------


def test_capture_gpu_without_nvidia_smi_binary(monkeypatch) -> None:
    monkeypatch.setattr(fingerprint.shutil, "which", lambda _name: None)
    info = fingerprint.capture_gpu()
    assert info == GpuInfo()


def test_capture_gpu_parses_real_csv_sample(monkeypatch) -> None:
    monkeypatch.setattr(fingerprint.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")

    def fake_run(cmd, cwd=None, timeout=5.0):
        if "cuda_version" in " ".join(cmd):
            return None  # not every driver supports this query field
        return _REAL_CORE_CSV_LINE

    monkeypatch.setattr(fingerprint, "_run", fake_run)

    info = fingerprint.capture_gpu()
    assert info.name == "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"
    assert info.driver == "610.47"
    assert info.sm_clock_mhz == 2272
    assert info.mem_clock_mhz == 13365
    assert info.power_limit_w == 300.0
    assert info.total_mem_mib == 97887
    assert info.persistence_mode == "Enabled"
    assert info.cuda is None  # gracefully degraded, not raised


def test_capture_gpu_degrades_gracefully_when_query_fails(monkeypatch) -> None:
    monkeypatch.setattr(fingerprint.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(fingerprint, "_run", lambda *a, **k: None)
    info = fingerprint.capture_gpu()
    assert info == GpuInfo()


def test_capture_gpu_degrades_on_malformed_output(monkeypatch) -> None:
    monkeypatch.setattr(fingerprint.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(fingerprint, "_run", lambda *a, **k: "not,enough,fields")
    info = fingerprint.capture_gpu()
    assert info == GpuInfo()


# --- python ---------------------------------------------------------------


def test_capture_python_version_matches_runtime() -> None:
    import sys

    info = fingerprint.capture_python()
    assert info.version == sys.version.split()[0]
    assert info.torch is None or isinstance(info.torch, str)
    assert info.transformers is None or isinstance(info.transformers, str)


# --- capture() end-to-end ---------------------------------------------------


def test_capture_never_raises_with_everything_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(fingerprint.shutil, "which", lambda _name: None)
    missing = str(tmp_path / "nope")
    result = fingerprint.capture(
        model={"path": "/models/laguna"},
        workload={"k": 15, "greedy": True},
        repo_paths={"qwen-sm120-runtime": missing, "sparkinfer": missing},
    )
    assert result.gpu == GpuInfo()
    for info in result.git.values():
        assert info.sha is None
    assert result.model.path == "/models/laguna"
    assert result.workload.k == 15


def test_capture_filters_unknown_model_and_workload_keys_into_extra(monkeypatch) -> None:
    monkeypatch.setattr(fingerprint.shutil, "which", lambda _name: None)
    result = fingerprint.capture(
        model={"path": "/models/laguna", "bogus_model_field": 1},
        workload={"k": 15, "bogus_workload_field": "z"},
    )
    assert result.model == ModelInfo(path="/models/laguna")
    assert result.workload.k == 15
    assert result.extra["model_extra"] == {"bogus_model_field": 1}
    assert result.extra["workload_extra"] == {"bogus_workload_field": "z"}


def test_capture_defaults_to_empty_workload_and_model_info(monkeypatch) -> None:
    monkeypatch.setattr(fingerprint.shutil, "which", lambda _name: None)
    result = fingerprint.capture()
    assert result.model == ModelInfo()
    assert result.workload == WorkloadInfo()
