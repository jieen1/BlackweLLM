"""Cold-start Laguna SKU capacity measurement.

This is deliberately a ``bfdiag`` utility rather than a benchmark script:
capacity is a load-time property, so each invocation must run in a fresh
process via ``bf run --cold``.  It records the runtime and SparkInfer commits
with allocator and driver memory at load, after the requested simultaneous
prefills, and after the requested batched decode suffix.

Example::

    python -m bfdiag.cli run bfdiag/cold_capacity.py --cold \
      --sweep QSR_COLD_SKU=4x250k,QSR_COLD_NUM_SLOTS=4,\
QSR_COLD_PROMPT_TOKENS=250000,QSR_COLD_OUTPUT_TOKENS=1000,\
QSR_COLD_MAX_MODEL_LEN=251024,QSR_COLD_BLOCKS_PER_SLOT=3923
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bfdiag.record.store import _atomic_write_text, bfdiag_dir

_ROOT = Path(__file__).resolve().parents[1]
_SPARKINFER_ROOT = Path("/home/bot/project/sparkinfer")


class StaticHeadroomError(RuntimeError):
    """Load-time GPU headroom fell below the SKU safety floor."""


@dataclass(frozen=True)
class ColdCapacitySpec:
    sku: str
    num_slots: int
    prompt_tokens: int
    output_tokens: int
    max_model_len: int
    blocks_per_slot: int
    block_size: int = 64
    gpu_memory_utilization: float = 0.88

    def validate(self) -> None:
        for name in (
            "num_slots",
            "prompt_tokens",
            "output_tokens",
            "max_model_len",
            "blocks_per_slot",
            "block_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        required_tokens = self.prompt_tokens + self.output_tokens
        if self.max_model_len < required_tokens:
            raise ValueError(
                f"max_model_len={self.max_model_len} is below prompt + output "
                f"({required_tokens})"
            )
        required_blocks = blocks_for_tokens(required_tokens, self.block_size)
        if self.blocks_per_slot < required_blocks:
            raise ValueError(
                f"blocks_per_slot={self.blocks_per_slot} is below the required "
                f"{required_blocks} blocks for {required_tokens} tokens"
            )


def blocks_for_tokens(tokens: int, block_size: int) -> int:
    """Return the exact number of fixed-size KV blocks needed for ``tokens``."""
    if tokens <= 0 or block_size <= 0:
        raise ValueError("tokens and block_size must be positive")
    return (tokens + block_size - 1) // block_size


def spec_from_env(env: dict[str, str] | None = None) -> ColdCapacitySpec:
    env = os.environ if env is None else env
    spec = ColdCapacitySpec(
        sku=env.get("QSR_COLD_SKU", "unspecified"),
        num_slots=int(env.get("QSR_COLD_NUM_SLOTS", "4")),
        prompt_tokens=int(env.get("QSR_COLD_PROMPT_TOKENS", "250000")),
        output_tokens=int(env.get("QSR_COLD_OUTPUT_TOKENS", "1000")),
        max_model_len=int(env.get("QSR_COLD_MAX_MODEL_LEN", "251024")),
        blocks_per_slot=int(env.get("QSR_COLD_BLOCKS_PER_SLOT", "3923")),
        block_size=int(env.get("QSR_COLD_BLOCK_SIZE", "64")),
        gpu_memory_utilization=float(env.get("QSR_COLD_GPU_MEMORY_UTILIZATION", "0.88")),
    )
    spec.validate()
    return spec


def _git_sha(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _driver_memory_from_fields(fields: str) -> dict[str, float | int | None]:
    """Normalize an ``nvidia-smi`` used/total/free row without unit ambiguity."""
    values = [int(value.strip()) for value in fields.split(",")]
    if len(values) != 3:
        raise ValueError(f"expected used,total,free MiB, got {fields!r}")
    used_mib, total_mib, free_mib = values
    return {
        "driver_used_mib": used_mib,
        "driver_total_mib": total_mib,
        "driver_free_mib": free_mib,
        "driver_used_gib": round(used_mib / 1024, 3),
        "driver_total_gib": round(total_mib / 1024, 3),
        "driver_free_gib": round(free_mib / 1024, 3),
    }


def _driver_memory() -> dict[str, float | int | None]:
    unavailable = {
        "driver_used_mib": None,
        "driver_total_mib": None,
        "driver_free_mib": None,
        "driver_used_gib": None,
        "driver_total_gib": None,
        "driver_free_gib": None,
    }
    try:
        fields = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return _driver_memory_from_fields(fields.stdout.splitlines()[0])
    except (OSError, subprocess.CalledProcessError, IndexError, ValueError):
        return unavailable


def _memory_snapshot(torch: Any) -> dict[str, Any]:
    stats = torch.cuda.memory_stats()
    allocated = stats.get("allocated_bytes.all.current", 0)
    reserved = stats.get("reserved_bytes.all.current", 0)
    peak_allocated = stats.get("allocated_bytes.all.peak", 0)
    peak_reserved = stats.get("reserved_bytes.all.peak", 0)
    snapshot = {
        "allocated_mib": round(allocated / 2**20, 1),
        "reserved_mib": round(reserved / 2**20, 1),
        "peak_allocated_mib": round(peak_allocated / 2**20, 1),
        "peak_reserved_mib": round(peak_reserved / 2**20, 1),
        "allocation_retries": stats.get("num_alloc_retries", 0),
    }
    snapshot.update(_driver_memory())
    return snapshot


def _static_headroom_error(snapshot: dict[str, Any], minimum_free_mib: int) -> str | None:
    """Return a capacity-gate error without conflating MiB and GiB."""
    if minimum_free_mib < 0:
        raise ValueError("minimum_free_mib must not be negative")
    free_mib = snapshot.get("driver_free_mib")
    if free_mib is None or free_mib >= minimum_free_mib:
        return None
    return (
        "StaticHeadroomError: after_load driver_free_mib="
        f"{free_mib} is below the required {minimum_free_mib} MiB"
    )


def _prompt_ids(tokenizer: Any, prompt_tokens: int) -> list[int]:
    seed = tokenizer.encode(
        "The quick brown fox jumps over the lazy dog.", add_special_tokens=False
    )
    if not seed:
        raise RuntimeError("tokenizer produced no usable prompt tokens")
    return (seed * ((prompt_tokens + len(seed) - 1) // len(seed)))[:prompt_tokens]


def _prefill_slot_limit_from_env(num_slots: int, env: dict[str, str] | None = None) -> int:
    """Read the bounded cold-prefill stage without changing the SKU allocation."""
    env = os.environ if env is None else env
    limit = int(env.get("QSR_COLD_PREFILL_SLOT_LIMIT", str(num_slots)))
    if not 1 <= limit <= num_slots:
        raise ValueError(
            "QSR_COLD_PREFILL_SLOT_LIMIT must be between 1 and "
            f"QSR_COLD_NUM_SLOTS ({num_slots}), got {limit}"
        )
    return limit


def _route_tile_trace_from_host(
    *,
    token_map: list[int],
    expert_row_counts: list[int],
    expert_tile_base: list[int],
    physical_tiles_capacity: int,
    num_topk: int,
) -> dict[str, int]:
    """Decode the existing dynamic workspace into exact-order tile evidence."""
    from bfdiag.workloads import summarize_dynamic_route_tile_trace

    return summarize_dynamic_route_tile_trace(
        token_map=token_map,
        expert_row_counts=expert_row_counts,
        expert_tile_base=expert_tile_base,
        physical_tiles_capacity=physical_tiles_capacity,
        num_topk=num_topk,
    )


def _capture_route_tile_trace(backend: Any) -> dict[str, int]:
    """Copy only post-prefill routing metadata when the cold diagnostic asks."""
    layers = getattr(backend, "_moe_sparkinfer_layers", ())
    if not layers:
        raise RuntimeError("Laguna backend exposes no SparkInfer MoE layers")
    pool = layers[0].workspace
    candidates = [
        workspace
        for workspace in getattr(pool, "workspaces", {}).values()
        if all(
            hasattr(workspace, name)
            for name in (
                "token_map",
                "row_counts",
                "expert_tile_base",
                "physical_tiles_capacity",
                "num_topk",
            )
        )
    ]
    if not candidates:
        raise RuntimeError("no dynamic SparkInfer workspace was materialized")
    snapshots = [
        (sum(workspace.row_counts.cpu().tolist()), workspace)
        for workspace in candidates
    ]
    routed_rows, workspace = max(snapshots, key=lambda item: item[0])
    if routed_rows <= 0:
        capacities = [int(workspace.routed_rows_capacity) for workspace in candidates]
        raise RuntimeError(f"no dynamic workspace has routed rows; capacities={capacities}")
    return _route_tile_trace_from_host(
        token_map=workspace.token_map.cpu().tolist(),
        expert_row_counts=workspace.row_counts.cpu().tolist(),
        expert_tile_base=workspace.expert_tile_base.cpu().tolist(),
        physical_tiles_capacity=int(workspace.physical_tiles_capacity),
        num_topk=int(workspace.num_topk),
    )


def _safe_filename_part(value: str) -> str:
    """Keep a human-readable SKU in a single artifact filename."""
    return "".join(char if char.isalnum() or char in "._-" else "-" for char in value)


def _cold_artifact_path(sku: str, artifact_dir: Path | None = None) -> Path:
    directory = artifact_dir or bfdiag_dir() / "cold_capacity"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return directory / f"{timestamp}-{os.getpid()}-{_safe_filename_part(sku)}.json"


def persist_record(
    record: dict[str, Any],
    artifact_dir: Path | None = None,
    *,
    path: Path | None = None,
) -> Path:
    """Atomically retain a cold-run result even when its parent loses stdout."""
    path = path or _cold_artifact_path(
        str(record.get("spec", {}).get("sku", "invalid")), artifact_dir
    )
    record["artifact_path"] = str(path)
    _atomic_write_text(path, json.dumps(record, indent=2, sort_keys=True) + "\n")
    return path


def run(
    spec: ColdCapacitySpec,
    *,
    on_checkpoint: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Execute one fresh-process SKU measurement and return a JSON-safe record."""
    record: dict[str, Any] = {
        "ok": False,
        "spec": asdict(spec),
        "minimum_static_free_mib": int(os.environ.get("QSR_COLD_MIN_FREE_MIB", "2048")),
        "runtime_sha": _git_sha(_ROOT),
        "sparkinfer_sha": _git_sha(_SPARKINFER_ROOT),
        "memory": {},
    }
    provider: Any | None = None
    torch: Any | None = None

    def checkpoint() -> None:
        if on_checkpoint is None:
            return
        try:
            on_checkpoint(record)
        except Exception as exc:  # diagnostics must not disrupt capacity measurement
            record["checkpoint_write_error"] = f"{type(exc).__name__}: {exc}"

    try:
        import torch as torch_module

        from bfdiag.daemon.provider import LagunaEngineProvider

        torch = torch_module
        record["memory"]["before_load"] = _memory_snapshot(torch)
        checkpoint()
        provider = LagunaEngineProvider(
            num_slots=spec.num_slots,
            blocks_per_slot=spec.blocks_per_slot,
            block_size=spec.block_size,
            max_model_len=spec.max_model_len,
            gpu_memory_utilization=spec.gpu_memory_utilization,
        )
        def on_load_stage(stage: str) -> None:
            torch.cuda.synchronize()
            record["memory"][stage] = _memory_snapshot(torch)
            checkpoint()
            headroom_error = _static_headroom_error(
                record["memory"][stage], record["minimum_static_free_mib"]
            )
            if headroom_error is not None:
                raise StaticHeadroomError(headroom_error.removeprefix("StaticHeadroomError: "))

        provider.load(on_stage=on_load_stage)
        torch.cuda.synchronize()
        record["memory"]["after_load"] = _memory_snapshot(torch)
        checkpoint()
        headroom_error = _static_headroom_error(
            record["memory"]["after_load"], record["minimum_static_free_mib"]
        )
        if headroom_error is not None:
            record["error"] = headroom_error
            checkpoint()
            return record
        if os.environ.get("QSR_COLD_LOAD_ONLY") == "1":
            record["load_only"] = True
            record["ok"] = True
            checkpoint()
            return record

        prompt_ids = _prompt_ids(provider._tokenizer, spec.prompt_tokens)
        backend = provider._backend
        prefill_slot_limit = _prefill_slot_limit_from_env(spec.num_slots)
        record["prefill_slot_limit"] = prefill_slot_limit
        first_tokens: list[int] = []
        prefill_seconds: list[float] = []
        for slot in range(prefill_slot_limit):
            started = time.perf_counter()
            first_tokens.append(backend.prefill(slot, prompt_ids))
            torch.cuda.synchronize()
            prefill_seconds.append(round(time.perf_counter() - started, 3))
            record["prefill_seconds_by_slot"] = prefill_seconds
            record["memory"][f"after_prefill_slot_{slot + 1}"] = _memory_snapshot(torch)
            checkpoint()
        record["prefill_seconds_by_slot"] = prefill_seconds
        if os.environ.get("QSR_COLD_TRACE_ROUTE_TILES") == "1":
            try:
                record["route_tile_trace"] = _capture_route_tile_trace(backend)
            except Exception as trace_exc:
                record["route_tile_trace_error"] = (
                    f"{type(trace_exc).__name__}: {trace_exc}"
                )
        record["memory"]["after_all_prefills"] = _memory_snapshot(torch)
        checkpoint()

        next_tokens = first_tokens
        started = time.perf_counter()
        for _ in range(spec.output_tokens):
            next_tokens = backend.decode_batch(list(range(prefill_slot_limit)), next_tokens)
        torch.cuda.synchronize()
        record["decode_suffix_seconds"] = round(time.perf_counter() - started, 3)
        record["final_tokens"] = next_tokens
        record["memory"]["after_decode_suffix"] = _memory_snapshot(torch)
        record["ok"] = True
        checkpoint()
        return record
    except Exception as exc:  # capacity qualification must preserve OOM evidence
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
        if torch is not None:
            try:
                record["memory"]["at_failure"] = _memory_snapshot(torch)
            except Exception as snapshot_exc:  # a broken CUDA context is still evidence
                record["memory"]["failure_snapshot_error"] = (
                    f"{type(snapshot_exc).__name__}: {snapshot_exc}"
                )
        checkpoint()
        return record
    finally:
        if provider is not None:
            provider.unload()


def main() -> int:
    try:
        spec = spec_from_env()
    except Exception as exc:
        record = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "runtime_sha": _git_sha(_ROOT),
            "sparkinfer_sha": _git_sha(_SPARKINFER_ROOT),
        }
    else:
        artifact_path = _cold_artifact_path(spec.sku)
        record = run(spec, on_checkpoint=lambda item: persist_record(item, path=artifact_path))
    if "artifact_path" not in locals():
        artifact_path = _cold_artifact_path("invalid")
    try:
        persist_record(record, path=artifact_path)
    except Exception as exc:  # stdout remains a fallback when artifact storage is unavailable
        record["artifact_write_error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record.get("ok") and "artifact_write_error" not in record else 1


if __name__ == "__main__":
    sys.exit(main())
