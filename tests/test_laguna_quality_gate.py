"""Contracts for the native side of the Laguna quality oracle."""

import ast
from pathlib import Path

from benchmarks import laguna_quality_gate


def test_native_quality_gate_template_uses_owned_runtime_config() -> None:
    """The compared Laguna subprocess must not reconstruct a vLLM config."""
    script = laguna_quality_gate.BACKEND_SCRIPT.format(
        repo_root="/repo",
        model="model",
        prompts=[],
        eos=(),
        max_tok=1,
    )

    tree = ast.parse(script)
    assert "from runtime.laguna_config import build_laguna_config" in script
    assert "runtime_config = build_laguna_config(" in script
    backend_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "LagunaBackend"
    ]
    assert len(backend_calls) == 1
    assert isinstance(backend_calls[0].args[0], ast.Name)
    assert backend_calls[0].args[0].id == "runtime_config"
    assert "EngineArgs" not in script
    assert "vllm" not in script.lower()


def test_vllm_oracle_template_is_spawn_safe() -> None:
    """vLLM's WSL worker must not re-run LLM construction on spawn."""
    script = laguna_quality_gate.VLLM_SCRIPT.format(
        model="model",
        prompts=[],
        max_tok=1,
    )

    tree = ast.parse(script)
    main_functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    ]
    assert len(main_functions) == 1
    assert 'if __name__ == "__main__":' in script


def test_quality_gate_exposes_venv_tools_to_oracle_children(monkeypatch) -> None:
    """The isolated vLLM process must find its FlashInfer JIT `ninja` tool."""
    seen = {}

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["env"] = kwargs["env"]
        Path(command[-1]).write_text("[]")
        return Completed()

    monkeypatch.setattr(laguna_quality_gate, "PYTHON", "/tmp/venv/bin/python")
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.delenv("MAX_JOBS", raising=False)
    monkeypatch.setattr(laguna_quality_gate.subprocess, "run", fake_run)

    assert laguna_quality_gate.run_in_subprocess("pass", "oracle") == []
    assert seen["command"][0] == "/tmp/venv/bin/python"
    assert seen["env"]["PATH"] == "/tmp/venv/bin:/usr/bin"
    assert seen["env"]["MAX_JOBS"] == "2"
    assert laguna_quality_gate.ORACLE_TIMEOUT_SECONDS == 1800


def test_quality_gate_preserves_explicit_oracle_build_parallelism(monkeypatch) -> None:
    """A caller can choose a stricter build limit for a constrained host."""
    seen = {}

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, **kwargs):
        seen["env"] = kwargs["env"]
        Path(command[-1]).write_text("[]")
        return Completed()

    monkeypatch.setenv("MAX_JOBS", "2")
    monkeypatch.setattr(laguna_quality_gate.subprocess, "run", fake_run)

    assert laguna_quality_gate.run_in_subprocess("pass", "oracle") == []
    assert seen["env"]["MAX_JOBS"] == "2"
