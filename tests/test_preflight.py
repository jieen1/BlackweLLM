"""Unit tests for runtime/preflight.py.

CPU-only by construction: every check function accepts an injected probe
(GpuProbe / SparkInferProbe), so none of these tests need torch, CUDA, or
SparkInfer to be installed. The module itself must also be importable
without torch — test_module_has_no_third_party_import_time_dependency below
guards that contract directly.
"""

from __future__ import annotations

import json

from runtime.preflight import (
    REQUIRED_COMPUTE_CAPABILITY,
    REQUIRED_TORCH_VERSION,
    CheckResult,
    GpuProbe,
    PreflightReport,
    SparkInferProbe,
    _parse_version_prefix,
    check_checkpoint_config,
    check_checkpoint_path,
    check_compute_capability,
    check_cuda_driver_version,
    check_cuda_runtime_version,
    check_gpu_present,
    check_sparkinfer_analytic_decode_gate,
    check_sparkinfer_contract,
    check_torch_version,
    run_preflight,
)

# --- helpers -----------------------------------------------------------------


def _healthy_gpu_probe(**overrides) -> GpuProbe:
    base = dict(
        torch_importable=True,
        torch_version="2.13.0a0+gitcf30153",
        cuda_available=True,
        device_name="NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
        compute_capability=(12, 0),
        cuda_runtime_version="13.3",
        driver_version="610.47",
        error=None,
    )
    base.update(overrides)
    return GpuProbe(**base)


def _healthy_sparkinfer_probe(**overrides) -> SparkInferProbe:
    base = dict(
        importable=True,
        version="1.0.1",
        module_path="/home/bot/project/sparkinfer/sparkinfer/__init__.py",
        gate_found=True,
        gate_accepts_production_shape=True,
        error=None,
    )
    base.update(overrides)
    return SparkInferProbe(**base)


# --- module import discipline ------------------------------------------------


def test_module_has_no_third_party_import_time_dependency() -> None:
    """Mirrors bfdiag/'s import discipline: importing runtime.preflight must
    not require torch or sparkinfer, since it is meant to be imported before
    any GPU work happens (and unit-tested on machines without a GPU).
    """
    import ast

    import runtime.preflight as module

    with open(module.__file__, encoding="utf-8") as handle:
        source = ast.parse(handle.read())
    top_level_imports = [
        node
        for node in ast.walk(source)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and node.col_offset == 0  # only module-level statements
    ]
    third_party_roots = {"torch", "sparkinfer", "numpy", "transformers"}
    for node in top_level_imports:
        if isinstance(node, ast.Import):
            names = [alias.name.split(".")[0] for alias in node.names]
        else:
            names = [node.module.split(".")[0]] if node.module else []
        for name in names:
            assert name not in third_party_roots, (
                f"runtime.preflight imports {name!r} at module scope; it must "
                "be imported lazily inside a function so this module stays "
                "importable without a GPU or without torch installed."
            )


# --- version parsing helpers --------------------------------------------------


def test_parse_version_prefix_handles_nightly_and_local_suffix() -> None:
    assert _parse_version_prefix("2.13.0a0+gitcf30153") == (2, 13, 0)
    assert _parse_version_prefix("0.7.0") == (0, 7, 0)
    assert _parse_version_prefix("610.47") == (610, 47)
    assert _parse_version_prefix("not-a-version") is None


# --- GPU checks ---------------------------------------------------------------


def test_gpu_present_passes_when_cuda_available() -> None:
    result = check_gpu_present(_healthy_gpu_probe())
    assert result.passed
    assert result.severity == "fatal"


def test_gpu_present_fails_with_remediation_when_no_cuda() -> None:
    probe = GpuProbe(
        torch_importable=True,
        torch_version="2.13.0a0",
        cuda_available=False,
        device_name=None,
        compute_capability=None,
        cuda_runtime_version=None,
        driver_version=None,
        error="torch.cuda.is_available() is False",
    )
    result = check_gpu_present(probe)
    assert not result.passed
    assert result.severity == "fatal"
    assert result.remediation


def test_compute_capability_rejects_non_sm120() -> None:
    probe = _healthy_gpu_probe(compute_capability=(9, 0))
    result = check_compute_capability(probe)
    assert not result.passed
    assert result.severity == "fatal"
    assert "SM120" in result.remediation or "12.0" in result.expected


def test_compute_capability_accepts_sm120() -> None:
    probe = _healthy_gpu_probe(compute_capability=REQUIRED_COMPUTE_CAPABILITY)
    result = check_compute_capability(probe)
    assert result.passed


def test_compute_capability_handles_missing_device() -> None:
    probe = _healthy_gpu_probe(cuda_available=False, compute_capability=None)
    result = check_compute_capability(probe)
    assert not result.passed
    assert "unknown" in result.actual


def test_torch_version_accepts_exact_pypi_release() -> None:
    probe = _healthy_gpu_probe(torch_version=".".join(map(str, REQUIRED_TORCH_VERSION)))
    assert check_torch_version(probe).passed


def test_torch_version_accepts_local_prerelease_build_of_same_release() -> None:
    """The core scenario this check exists for: the reference environment
    reports `2.13.0a0+gitcf30153` — a self-compiled pre-release of the pinned
    `2.13.0`, not a stock PyPI wheel. It must pass, not fail just because the
    version string carries an `a0` pre-release marker and a `+git...` local
    build segment.
    """
    probe = _healthy_gpu_probe(torch_version="2.13.0a0+gitcf30153")
    result = check_torch_version(probe)
    assert result.passed
    assert result.severity == "fatal"


def test_torch_version_accepts_other_prerelease_and_local_suffix_spellings() -> None:
    for spelling in ("2.13.0rc1", "2.13.0.dev20260731", "2.13.0+cu126"):
        probe = _healthy_gpu_probe(torch_version=spelling)
        assert check_torch_version(probe).passed, spelling


def test_torch_version_rejects_too_old() -> None:
    probe = _healthy_gpu_probe(torch_version="2.11.0")
    result = check_torch_version(probe)
    assert not result.passed
    assert result.severity == "fatal"


def test_torch_version_rejects_prerelease_of_a_different_release() -> None:
    """A pre-release only satisfies the contract if its release segment
    (major.minor.patch) matches exactly -- `2.14.0a0` is not `2.13.0` just
    because both are pre-release-flavored version strings.
    """
    probe = _healthy_gpu_probe(torch_version="2.14.0a0+gitdeadbeef")
    result = check_torch_version(probe)
    assert not result.passed


def test_torch_version_fails_when_not_importable() -> None:
    probe = _healthy_gpu_probe(torch_importable=False, torch_version=None)
    result = check_torch_version(probe)
    assert not result.passed
    assert result.severity == "fatal"


def test_cuda_driver_version_is_warning_severity_and_advisory_when_unknown() -> None:
    probe = _healthy_gpu_probe(driver_version=None)
    result = check_cuda_driver_version(probe)
    assert not result.passed
    assert result.severity == "warning"


def test_cuda_driver_version_passes_on_verified_value() -> None:
    result = check_cuda_driver_version(_healthy_gpu_probe())
    assert result.passed
    assert result.severity == "warning"


def test_cuda_runtime_version_passes_on_verified_value() -> None:
    result = check_cuda_runtime_version(_healthy_gpu_probe())
    assert result.passed


def test_cuda_runtime_version_warns_when_absent() -> None:
    result = check_cuda_runtime_version(_healthy_gpu_probe(cuda_runtime_version=None))
    assert not result.passed
    assert result.severity == "warning"


# --- SparkInfer checks ---------------------------------------------------------


def test_sparkinfer_contract_fails_fatally_when_not_importable() -> None:
    probe = SparkInferProbe(
        importable=False,
        version=None,
        module_path=None,
        gate_found=False,
        gate_accepts_production_shape=None,
        error="sparkinfer is not importable: No module named 'sparkinfer'",
    )
    result = check_sparkinfer_contract(probe)
    assert not result.passed
    assert result.severity == "fatal"
    assert "pip install" in result.remediation


def test_sparkinfer_contract_passes_when_importable_and_versioned() -> None:
    result = check_sparkinfer_contract(_healthy_sparkinfer_probe())
    assert result.passed


def test_sparkinfer_gate_warns_not_fails_when_unpatched() -> None:
    """The core T0-5 scenario: SparkInfer is importable and fine, but the
    analytic-decode gate rejects our production (48/8) shape because the
    local gating patch is absent. This must be a warning, not fatal — the
    generic kernel is still correct, just slower.
    """
    probe = _healthy_sparkinfer_probe(gate_accepts_production_shape=False)
    result = check_sparkinfer_analytic_decode_gate(probe)
    assert not result.passed
    assert result.severity == "warning"
    assert "docs/sparkinfer-upstream-handoff.md" in result.remediation


def test_sparkinfer_gate_passes_when_patched() -> None:
    result = check_sparkinfer_analytic_decode_gate(_healthy_sparkinfer_probe())
    assert result.passed


def test_sparkinfer_gate_warns_when_not_evaluable() -> None:
    probe = _healthy_sparkinfer_probe(
        gate_accepts_production_shape=None, error="no CUDA device available"
    )
    result = check_sparkinfer_analytic_decode_gate(probe)
    assert not result.passed
    assert result.severity == "warning"


def test_sparkinfer_gate_warns_when_sparkinfer_missing() -> None:
    probe = SparkInferProbe(
        importable=False,
        version=None,
        module_path=None,
        gate_found=False,
        gate_accepts_production_shape=None,
        error="not importable",
    )
    result = check_sparkinfer_analytic_decode_gate(probe)
    assert not result.passed
    assert result.severity == "warning"


# --- checkpoint checks ---------------------------------------------------------


def test_checkpoint_path_passes_for_existing_directory(tmp_path) -> None:
    result = check_checkpoint_path(tmp_path)
    assert result.passed


def test_checkpoint_path_fails_for_missing_directory(tmp_path) -> None:
    result = check_checkpoint_path(tmp_path / "does-not-exist")
    assert not result.passed
    assert result.severity == "fatal"


def test_checkpoint_config_passes_for_valid_json(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "laguna", "architectures": ["LagunaForCausalLM"]}),
        encoding="utf-8",
    )
    result = check_checkpoint_config(tmp_path)
    assert result.passed
    assert "laguna" in result.actual


def test_checkpoint_config_fails_when_missing(tmp_path) -> None:
    result = check_checkpoint_config(tmp_path)
    assert not result.passed
    assert result.severity == "fatal"


def test_checkpoint_config_fails_when_malformed(tmp_path) -> None:
    (tmp_path / "config.json").write_text("{not valid json", encoding="utf-8")
    result = check_checkpoint_config(tmp_path)
    assert not result.passed


def test_checkpoint_config_fails_when_not_a_json_object(tmp_path) -> None:
    (tmp_path / "config.json").write_text("[1, 2, 3]", encoding="utf-8")
    result = check_checkpoint_config(tmp_path)
    assert not result.passed


# --- CheckResult / PreflightReport -------------------------------------------


def test_check_result_str_includes_remediation_only_on_failure() -> None:
    passing = CheckResult(name="x", passed=True, severity="fatal", actual="ok", expected="ok")
    failing = CheckResult(
        name="x",
        passed=False,
        severity="fatal",
        actual="bad",
        expected="ok",
        remediation="fix it",
    )
    assert "fix it" not in str(passing)
    assert "fix it" in str(failing)


def test_preflight_report_ok_ignores_warnings() -> None:
    report = PreflightReport(
        checks=(
            CheckResult(name="a", passed=True, severity="fatal", actual="x", expected="x"),
            CheckResult(name="b", passed=False, severity="warning", actual="x", expected="y"),
        )
    )
    assert report.ok
    assert len(report.warnings) == 1
    assert len(report.fatal_failures) == 0


def test_preflight_report_not_ok_when_fatal_check_fails() -> None:
    report = PreflightReport(
        checks=(CheckResult(name="a", passed=False, severity="fatal", actual="x", expected="y"),)
    )
    assert not report.ok
    assert len(report.fatal_failures) == 1


# --- end-to-end run_preflight with fully injected probes ---------------------


def test_run_preflight_all_healthy_and_checkpoint_valid(tmp_path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "laguna"}), encoding="utf-8")
    report = run_preflight(
        tmp_path, gpu=_healthy_gpu_probe(), sparkinfer=_healthy_sparkinfer_probe()
    )
    assert report.ok
    assert not report.fatal_failures
    names = {c.name for c in report.checks}
    assert {
        "gpu_present",
        "compute_capability",
        "cuda_driver_version",
        "cuda_runtime_version",
        "torch_version",
        "sparkinfer_contract",
        "sparkinfer_analytic_decode_gate",
        "checkpoint_path",
        "checkpoint_config",
    } <= names


def test_run_preflight_blocks_on_wrong_gpu_but_not_on_unpatched_sparkinfer(tmp_path) -> None:
    """A single scenario exercising the exact split this module is designed
    around: a hardware-contract violation is fatal, but an unpatched
    SparkInfer (T0-5) is a warning that does not, by itself, block startup.
    """
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "laguna"}), encoding="utf-8")
    gpu = _healthy_gpu_probe(compute_capability=(9, 0))
    sparkinfer = _healthy_sparkinfer_probe(gate_accepts_production_shape=False)
    report = run_preflight(tmp_path, gpu=gpu, sparkinfer=sparkinfer)

    assert not report.ok
    fatal_names = {c.name for c in report.fatal_failures}
    assert "compute_capability" in fatal_names

    warning_names = {c.name for c in report.warnings}
    assert "sparkinfer_analytic_decode_gate" in warning_names
    assert "sparkinfer_analytic_decode_gate" not in fatal_names


def test_run_preflight_missing_checkpoint_is_fatal(tmp_path) -> None:
    report = run_preflight(
        tmp_path / "nope", gpu=_healthy_gpu_probe(), sparkinfer=_healthy_sparkinfer_probe()
    )
    assert not report.ok
    fatal_names = {c.name for c in report.fatal_failures}
    assert "checkpoint_path" in fatal_names
