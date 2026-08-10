"""RunRecord schema v1: the structured metadata every diagnostic run produces.

This is the shared contract for bfdiag: a run's environment fingerprint,
metrics, and artifacts, serializable to/from JSON and stable enough for the
sqlite store and the differ to depend on. See
``notes/2026-07-27-bfdiag-run-records.md`` for the design rationale.
"""

from __future__ import annotations

import dataclasses
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = 1


def new_run_id() -> str:
    """A short, sortable-enough identifier; uniqueness is all that matters."""
    return uuid.uuid4().hex[:12]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class GitRepoInfo:
    sha: str | None = None
    dirty: bool | None = None
    branch: str | None = None


@dataclass
class GpuInfo:
    name: str | None = None
    driver: str | None = None
    cuda: str | None = None
    sm_clock_mhz: int | None = None
    mem_clock_mhz: int | None = None
    power_limit_w: float | None = None
    total_mem_mib: int | None = None
    persistence_mode: str | None = None


@dataclass
class PythonInfo:
    version: str | None = None
    torch: str | None = None
    transformers: str | None = None


@dataclass
class ModelInfo:
    path: str | None = None
    revision: str | None = None
    dtype: str | None = None
    max_model_len: int | None = None
    quantization: str | None = None


@dataclass
class WorkloadInfo:
    # Stable workload identity.  These fields are first-class because a
    # benchmark contract change must make ``bf diff`` refuse comparison,
    # rather than hiding under ``fingerprint.extra.workload_extra``.
    contract: str | None = None
    contract_version: int | None = None
    workload_name: str | None = None
    prompt_hash: str | None = None
    prompt_len: int | None = None
    generated_tokens: int | None = None
    batch: int | None = None
    k: int | None = None
    seed: int | None = None
    greedy: bool | None = None
    block_size: int | None = None
    capacity: int | None = None
    max_model_len: int | None = None
    max_q_rows: int | None = None
    cuda_graph_status: str | None = None
    warm_only: bool | None = None
    # Blocks reserved per slot. NOT a capacity knob: it sets the
    # sparkinfer decode workspace's ``max_pages`` for full-attention
    # groups (``runtime/backends/laguna_cuda_graph.py``), which can change
    # kernel tiling and hence float reduction order. On 2026-07-27 a warm
    # daemon defaulting to 4096 was compared against a cold-start script
    # deriving 130, and the acceptance rates (0.6754 vs 0.452525) were
    # treated as comparable. Recorded separately from ``capacity``
    # (concurrent slots) because they mean different things.
    blocks_per_slot: int | None = None


def _dataclass_field_names(dc_type: type) -> set[str]:
    return {f.name for f in dataclasses.fields(dc_type)}


@dataclass
class Fingerprint:
    git: dict[str, GitRepoInfo] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    gpu: GpuInfo = field(default_factory=GpuInfo)
    python: PythonInfo = field(default_factory=PythonInfo)
    model: ModelInfo = field(default_factory=ModelInfo)
    workload: WorkloadInfo = field(default_factory=WorkloadInfo)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "git": {name: dataclasses.asdict(info) for name, info in self.git.items()},
            "env": dict(self.env),
            "gpu": dataclasses.asdict(self.gpu),
            "python": dataclasses.asdict(self.python),
            "model": dataclasses.asdict(self.model),
            "workload": dataclasses.asdict(self.workload),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Fingerprint:
        data = data or {}
        git = {name: GitRepoInfo(**info) for name, info in (data.get("git") or {}).items()}
        python_data = data.get("python") or {}
        python_fields = _dataclass_field_names(PythonInfo)
        return cls(
            git=git,
            env=dict(data.get("env") or {}),
            gpu=GpuInfo(**(data.get("gpu") or {})),
            # v1 records may contain the retired ``vllm`` package field.
            # Ignore it while reading so diagnostic history stays usable.
            python=PythonInfo(**{k: v for k, v in python_data.items() if k in python_fields}),
            model=ModelInfo(**(data.get("model") or {})),
            workload=WorkloadInfo(**(data.get("workload") or {})),
            extra=dict(data.get("extra") or {}),
        )


@dataclass
class RunRecord:
    run_id: str
    schema_version: int = SCHEMA_VERSION
    started_at: str = ""
    finished_at: str | None = None
    script: str = ""
    argv: list[str] = field(default_factory=list)
    status: str = "ok"  # "ok" | "failed"
    error: str | None = None
    fingerprint: Fingerprint = field(default_factory=Fingerprint)
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    trace_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "script": self.script,
            "argv": list(self.argv),
            "status": self.status,
            "error": self.error,
            "fingerprint": self.fingerprint.to_dict(),
            "metrics": dict(self.metrics),
            "artifacts": dict(self.artifacts),
            "trace_path": self.trace_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunRecord:
        return cls(
            run_id=data["run_id"],
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
            started_at=data.get("started_at", ""),
            finished_at=data.get("finished_at"),
            script=data.get("script", ""),
            argv=list(data.get("argv") or []),
            status=data.get("status", "ok"),
            error=data.get("error"),
            fingerprint=Fingerprint.from_dict(data.get("fingerprint")),
            metrics=dict(data.get("metrics") or {}),
            artifacts=dict(data.get("artifacts") or {}),
            trace_path=data.get("trace_path"),
        )

    def get_path(self, dotted_key: str) -> Any:
        """Look up a dotted key, e.g. ``fingerprint.workload.prompt_hash`` or
        ``fingerprint.git.sparkinfer.sha``, against this record's full dict form.
        Returns None for any missing intermediate key instead of raising.
        """
        node: Any = self.to_dict()
        for part in dotted_key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return None
        return node
