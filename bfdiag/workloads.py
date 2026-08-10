"""Frozen, reusable performance workloads for the bfdiag warm engine.

These are diagnostic contracts, not ad-hoc benchmark scripts.  A workload
keeps its exact prompt construction, load-time geometry, reset sequence, and
metric names in one reviewable place so ``bf diff`` can reject accidental
apples-to-oranges comparisons.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from collections.abc import Callable
from typing import Any

from bfdiag.record import run_record
from bfdiag.record.store import default_store


def summarize_sparkinfer_workspace_pools(
    pools: list[Any],
    *,
    view_mapper: Any,
) -> dict[str, Any]:
    """Make a JSON-safe, view-level report for shared SparkInfer core arenas."""
    unique_pools = {id(pool): pool for pool in pools}
    arenas: list[dict[str, Any]] = []
    for pool in unique_pools.values():
        for key, arena in pool.core_arenas.items():
            plan = arena.plan
            views = [dict(view) for view in view_mapper(plan)]
            arena_nbytes = int(arena.shared_arena.numel() * arena.shared_arena.element_size())
            arenas.append(
                {
                    "workspace_key": repr(key),
                    "arena_nbytes": arena_nbytes,
                    "implementation": plan.implementation,
                    "deterministic_output": bool(plan.deterministic_output),
                    "routed_rows": int(plan.routed_rows),
                    "num_topk": int(plan.num_topk),
                    "views": views,
                }
            )
    return {"pool_count": len(unique_pools), "core_arenas": arenas}


def summarize_dynamic_route_tile_trace(
    *,
    token_map: list[int],
    expert_row_counts: list[int],
    expert_tile_base: list[int],
    physical_tiles_capacity: int,
    num_topk: int,
) -> dict[str, int]:
    """Decode actual dynamic route placement into exact-order tile evidence.

    The trace is descriptive, not a proposed launch order. Dependency cycles
    identify when a simple tile-reordered streaming reduction cannot preserve
    the existing fixed top-k accumulation order.
    """
    if physical_tiles_capacity <= 0 or num_topk <= 0:
        raise ValueError("physical_tiles_capacity and num_topk must be positive")
    if len(expert_tile_base) != len(expert_row_counts) + 1:
        raise ValueError("expert tile bases do not match expert row counts")
    if len(token_map) % physical_tiles_capacity:
        raise ValueError("token_map length is not divisible by physical tile capacity")

    from sparkinfer.moe.fused_moe._impl import _deterministic_route_tile_dependencies

    tile_m = len(token_map) // physical_tiles_capacity
    active_tiles = expert_tile_base[-1]
    if active_tiles < 0 or active_tiles > physical_tiles_capacity:
        raise ValueError("active physical tile count is outside workspace capacity")
    physical_to_pair = [-1] * (active_tiles * tile_m)
    for expert_index, rows in enumerate(expert_row_counts):
        tile_span = expert_tile_base[expert_index + 1] - expert_tile_base[expert_index]
        if rows < 0 or rows > tile_span * tile_m:
            raise ValueError(f"expert {expert_index} has invalid routed row count")
        start = expert_tile_base[expert_index] * tile_m
        physical_to_pair[start : start + rows] = token_map[start : start + rows]
    routed_rows = sum(expert_row_counts)
    if routed_rows % num_topk:
        raise ValueError("routed rows are not divisible by num_topk")
    dependencies = _deterministic_route_tile_dependencies(
        physical_to_pair=physical_to_pair,
        num_tokens=routed_rows // num_topk,
        num_topk=num_topk,
        tile_m=tile_m,
    )
    return {
        "routed_rows": routed_rows,
        "tile_m": tile_m,
        "active_tiles": dependencies.active_tiles,
        "dependency_edges": dependencies.dependency_edges,
        "cyclic_components": dependencies.cyclic_components,
        "largest_cyclic_component_tiles": dependencies.largest_cyclic_component_tiles,
        "largest_cyclic_component_route_rows": dependencies.largest_cyclic_component_route_rows,
    }


def capture_dynamic_route_tile_trace(backend: Any) -> dict[str, int]:
    """Copy the current dynamic workspace's small routing metadata to host."""
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
    snapshots = [(sum(workspace.row_counts.cpu().tolist()), workspace) for workspace in candidates]
    routed_rows, workspace = max(snapshots, key=lambda item: item[0])
    if routed_rows <= 0:
        capacities = [int(workspace.routed_rows_capacity) for workspace in candidates]
        raise RuntimeError(f"no dynamic workspace has routed rows; capacities={capacities}")
    return summarize_dynamic_route_tile_trace(
        token_map=workspace.token_map.cpu().tolist(),
        expert_row_counts=workspace.row_counts.cpu().tolist(),
        expert_tile_base=workspace.expert_tile_base.cpu().tolist(),
        physical_tiles_capacity=int(workspace.physical_tiles_capacity),
        num_topk=int(workspace.num_topk),
    )


def audit_sparkinfer_workspace(backend: Any) -> dict[str, Any]:
    """Persist the live SparkInfer core-arena view map for one warm backend."""
    from sparkinfer.moe.fused_moe._impl import (
        _core_workspace_view_map,
        _dynamic_core_workspace_liveness_map,
    )

    layers = getattr(backend, "_moe_sparkinfer_layers", ())
    if not layers:
        raise RuntimeError("Laguna backend has no patched SparkInfer MoE layers")
    report = summarize_sparkinfer_workspace_pools(
        [layer.workspace for layer in layers],
        view_mapper=lambda plan: (
            _dynamic_core_workspace_liveness_map(plan)
            if plan.implementation == "dynamic"
            else _core_workspace_view_map(plan)
        ),
    )
    report["route_tile_trace"] = capture_dynamic_route_tile_trace(backend)
    with run_record(
        script=__file__,
        workload={"kind": "sparkinfer-workspace-audit"},
        extra={"workload_extra": {"deterministic_moe": True}},
    ) as rec:
        artifacts = default_store().artifacts_dir(rec.run_id)
        artifacts.mkdir(parents=True, exist_ok=True)
        path = artifacts / "workspace_views.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        rec.artifact("workspace_views", path)
        rec.metric("workspace_pool_count", report["pool_count"])
        rec.metric("core_arena_count", len(report["core_arenas"]))
        rec.metric(
            "core_arena_bytes", sum(arena["arena_nbytes"] for arena in report["core_arenas"])
        )
        for name, value in report["route_tile_trace"].items():
            rec.metric(f"route_tile_{name}", value)
        run_id = rec.run_id
    return {"run_id": run_id, **report}


HISTORICAL_M1_CONTEXT_TOKENS = 65_536
HISTORICAL_M1_SUFFIX_TOKENS = 10_240
HISTORICAL_M1_NEW_TOKENS = 128
HISTORICAL_M1_BLOCK_SIZE = 64
HISTORICAL_M1_MAX_MODEL_LEN = 77_824
HISTORICAL_M1_BLOCKS_PER_SLOT = 1_248
HISTORICAL_M1_CONTRACT = "66d5913-repro_80tok_m1_decode_cg"

# This is the fixed 64K DFlash M=16 contract used by the historical
# verify-CUDA-graph experiments.  It deliberately retains the tokenizer-based
# prompt rather than substituting synthetic token ids: DFlash acceptance is a
# property of the exact draft/target token sequence, not just its length.
HISTORICAL_DFLASH_M16_CONTEXT_TOKENS = 65_536
HISTORICAL_DFLASH_M16_NEW_TOKENS = 256
HISTORICAL_DFLASH_M16_TEXT = "The quick brown fox jumps over the lazy dog. "
HISTORICAL_DFLASH_M16_CONTRACT = "dflash-m16-64k-quick-brown-fox"

# Exact workload from fd33368:benchmarks/full_comparison_ours.py.  Keep this
# separate from the small quick-brown-fox probe above: the historical
# throughput baseline uses a 55,536-token retained prefix plus a distinct
# 10,000-token suffix, and prefix-cache acceptance depends on that sequence.
HISTORICAL_DFLASH_PREFIX_CONTEXT_TOKENS = 65_536
HISTORICAL_DFLASH_PREFIX_BASE_TOKENS = 55_536
HISTORICAL_DFLASH_PREFIX_SUFFIX_TOKENS = 10_000
HISTORICAL_DFLASH_PREFIX_MAX_TOKENS = 256
HISTORICAL_DFLASH_PREFIX_BLOCK_SIZE = 64
HISTORICAL_DFLASH_PREFIX_BLOCKS_PER_SLOT = 1_536
HISTORICAL_DFLASH_PREFIX_MAX_MODEL_LEN = 262_144
HISTORICAL_DFLASH_PREFIX_CONTRACT = "fd33368-full-comparison-ours-64k"
HISTORICAL_DFLASH_PREFIX_BASE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "In a world of artificial intelligence and machine learning, "
    "the importance of efficient inference cannot be overstated. "
    "Modern language models require careful optimization of memory "
    "bandwidth and compute utilization to achieve real-time performance. "
    "Speculative decoding offers a promising approach by using a smaller "
    "draft model to propose multiple tokens in parallel, which are then "
    "verified by the larger target model in a single forward pass. "
)
HISTORICAL_DFLASH_PREFIX_SUFFIX_TEXT = (
    "Deep learning architectures have revolutionized natural language processing. "
    "Transformer models use self-attention mechanisms to capture long-range dependencies "
    "in sequential data. The key innovation is the multi-head attention mechanism which "
    "allows the model to jointly attend to information from different representation "
    "subspaces at different positions. Mixture of experts models further improve "
    "efficiency by routing tokens to specialized expert networks. Quantization "
    "techniques reduce memory footprint while maintaining model quality. "
)


def historical_m1_prompt_ids(length: int) -> list[int]:
    """The exact synthetic base prompt used by the frozen 80 tok/s run."""
    if length < 0:
        raise ValueError("length must be non-negative")
    chunk = list(range(1000, 1100))
    return (chunk * ((length + len(chunk) - 1) // len(chunk)))[:length]


def historical_dflash_m16_prompt_ids(
    tokenizer: Any, length: int = HISTORICAL_DFLASH_M16_CONTEXT_TOKENS
) -> list[int]:
    """Build the frozen tokenizer-derived DFlash prompt at an exact length.

    Encode the full repeated string in ONE call, then truncate.  Do NOT
    repeat a tokenized chunk -- tokenizer boundary merging makes
    N * len(encode(phrase)) != len(encode(phrase * N)), producing
    different boundary tokens and drastically different acceptance rates
    (0.69 vs 0.96 at 64K).  Matches acceptance_regression.py _repeat_text.
    """
    if length < 0:
        raise ValueError("length must be non-negative")
    chars_needed = length * 8
    text = HISTORICAL_DFLASH_M16_TEXT
    big = text * max(chars_needed // max(len(text), 1) + 1, 10)
    ids = tokenizer.encode(big, add_special_tokens=False)
    if len(ids) < length:
        raise ValueError(f"tokenizer produced only {len(ids)} tokens, need {length}")
    return ids[:length]


def capture_gpu_telemetry() -> dict[str, float | int | None]:
    """Snapshot runtime GPU conditions outside a completed timed sample."""
    fields = "temperature.gpu,power.draw,clocks.sm,memory.used"
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
        values = [item.strip() for item in result.stdout.splitlines()[0].split(",")]
        if result.returncode or len(values) != 4:
            raise ValueError("unexpected nvidia-smi output")
        return {
            "temperature_c": int(values[0]),
            "power_w": float(values[1]),
            "sm_clock_mhz": int(values[2]),
            "driver_used_mib": int(values[3]),
        }
    except (IndexError, OSError, subprocess.TimeoutExpired, ValueError):
        return {
            "temperature_c": None,
            "power_w": None,
            "sm_clock_mhz": None,
            "driver_used_mib": None,
        }


def _run_dflash_e2e_sample(
    backend: Any,
    engine: Any,
    prompt_ids: list[int],
    *,
    max_tokens: int,
    synchronize: Callable[[], None],
    telemetry: Callable[[], dict[str, float | int | None]] = capture_gpu_telemetry,
) -> dict[str, Any]:
    """Run the whole-generation production path once from an empty slot."""
    backend.reset_slot(0)
    synchronize()
    start = time.perf_counter()
    tokens, stats = engine.generate_verify_only(
        prompt_ids=prompt_ids,
        max_tokens=max_tokens,
        enable_prefix_cache=False,
        slot=0,
    )
    synchronize()
    return {
        "tok_s": float(stats["tok_per_s"]),
        "acceptance_rate": float(stats["acceptance_rate"]),
        "tokens_per_step": float(stats["tokens_per_step"]),
        "steps": int(stats["num_steps"]),
        "wall_s": time.perf_counter() - start,
        "output_hash": _token_ids_hash(tokens),
        "telemetry": telemetry(),
    }


def run_historical_dflash_m16_e2e(
    engine: Any,
    tokenizer: Any,
    *,
    warmup_rounds: int = 1,
    measure_rounds: int = 6,
    max_tokens: int = HISTORICAL_DFLASH_M16_NEW_TOKENS,
) -> dict[str, Any]:
    """Replay the dedicated historical fox-64K whole-generation contract."""
    if warmup_rounds < 0 or measure_rounds <= 0:
        raise ValueError("warmup_rounds must be non-negative and measure_rounds positive")

    import torch

    backend = engine.backend
    prompt_ids = historical_dflash_m16_prompt_ids(tokenizer)
    if backend.block_size != HISTORICAL_M1_BLOCK_SIZE:
        raise ValueError(
            f"{HISTORICAL_DFLASH_M16_CONTRACT} requires block_size="
            f"{HISTORICAL_M1_BLOCK_SIZE}, got {backend.block_size}"
        )
    workload = {
        "contract": HISTORICAL_DFLASH_M16_CONTRACT,
        "prompt_hash": _token_ids_hash(prompt_ids),
        "prompt_len": len(prompt_ids),
        "max_tokens": max_tokens,
        "k": 15,
        "greedy": True,
        "block_size": backend.block_size,
        "blocks_per_slot": backend.blocks_per_slot,
        "max_model_len": backend.runtime_config.model_config.max_model_len,
        "capacity": backend.num_slots,
        "dflash": True,
        "prefix_cache": False,
        "whole_generation": True,
    }
    with run_record(
        script=__file__,
        workload=workload,
        extra={
            "workload_extra": {
                "kind": "laguna-dflash-fox-64k-dedicated",
                "warmup_rounds": warmup_rounds,
                "measure_rounds": measure_rounds,
                "discard_first": warmup_rounds > 0,
            }
        },
    ) as rec:
        warmup = [
            _run_dflash_e2e_sample(
                backend,
                engine,
                prompt_ids,
                max_tokens=max_tokens,
                synchronize=torch.cuda.synchronize,
            )
            for _ in range(warmup_rounds)
        ]
        rounds = [
            _run_dflash_e2e_sample(
                backend,
                engine,
                prompt_ids,
                max_tokens=max_tokens,
                synchronize=torch.cuda.synchronize,
            )
            for _ in range(measure_rounds)
        ]
        for index, row in enumerate(rounds):
            rec.metric(f"sample_{index}_tok_s", row["tok_s"])
            rec.metric(f"sample_{index}_acceptance", row["acceptance_rate"])
            rec.metric(f"sample_{index}_tps", row["tokens_per_step"])
        median_tok_s = sorted(row["tok_s"] for row in rounds)[len(rounds) // 2]
        rec.metric("stable_median_tok_s", median_tok_s)
        rec.metric("stable_mean_tok_s", sum(row["tok_s"] for row in rounds) / len(rounds))
        rec.metric(
            "stable_mean_acceptance",
            sum(row["acceptance_rate"] for row in rounds) / len(rounds),
        )
        artifact = {"warmup": warmup, "rounds": rounds}
        artifacts = default_store().artifacts_dir(rec.run_id)
        artifacts.mkdir(parents=True, exist_ok=True)
        artifact_path = artifacts / "historical_dflash_m16_e2e.json"
        artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        rec.artifact("historical_dflash_m16_e2e", artifact_path)
        run_id = rec.run_id
    return {"run_id": run_id, "warmup": warmup, "rounds": rounds}


def _repeat_tokenized_text(tokenizer: Any, text: str, length: int) -> list[int]:
    """Encode repeated text to exactly length tokens.

    Encode the full repeated string in ONE call, then truncate.
    Tokenizer boundary merging makes chunk-repeat != whole-encode.
    """
    chars_needed = length * 8
    big = text * max(chars_needed // max(len(text), 1) + 1, 10)
    ids = tokenizer.encode(big, add_special_tokens=False)
    if len(ids) < length:
        raise ValueError(f"tokenizer produced only {len(ids)} tokens, need {length}")
    return ids[:length]


def historical_dflash_prefix_prompt_ids(tokenizer: Any) -> tuple[list[int], list[int]]:
    """Return fd33368's exact retained base and full 64K prompt token ids."""
    base_ids = _repeat_tokenized_text(
        tokenizer,
        HISTORICAL_DFLASH_PREFIX_BASE_TEXT,
        HISTORICAL_DFLASH_PREFIX_BASE_TOKENS,
    )
    suffix_ids = _repeat_tokenized_text(
        tokenizer,
        HISTORICAL_DFLASH_PREFIX_SUFFIX_TEXT,
        HISTORICAL_DFLASH_PREFIX_SUFFIX_TOKENS,
    )
    full_ids = base_ids + suffix_ids
    assert len(full_ids) == HISTORICAL_DFLASH_PREFIX_CONTEXT_TOKENS
    return base_ids, full_ids


def _token_ids_hash(token_ids: list[int]) -> str:
    return hashlib.sha256(b"".join(token.to_bytes(4, "little") for token in token_ids)).hexdigest()


DSV4_WARM_BASELINE_CONTRACT = "dsv4-warm-baseline"
DSV4_WARM_BASELINE_VERSION = 1
DSV4_WARM_BASELINE_GENERATED_TOKENS = 64
DSV4_WARM_BASELINE_SPECS = (
    {
        "name": "chat-short-96",
        "kind": "chat",
        "prompt_len": 96,
        "text": (
            "<|system|>\nYou are concise and exact.\n<|user|>\n"
            "Summarize why compressed attention helps long-context inference "
            "in two short sentences.\n<|assistant|>\n"
        ),
    },
    {
        "name": "chat-cross-128",
        "kind": "chat",
        "prompt_len": 129,
        "text": (
            "<|system|>\nYou are a runtime debugging assistant.\n<|user|>\n"
            "List three likely causes of a decode-only throughput regression "
            "after a graph-capture change, and keep each cause to one line.\n"
            "<|assistant|>\n"
        ),
    },
    {
        "name": "code-instruction-cross-256",
        "kind": "code_instruction",
        "prompt_len": 257,
        "text": (
            "You are reviewing a Python inference runtime.\n"
            "Task: explain the control flow, then rewrite the function so it "
            "returns both the running mean and the final token budget.\n\n"
            "```python\n"
            "def summarize(values):\n"
            "    total = 0\n"
            "    count = 0\n"
            "    for value in values:\n"
            "        total += value\n"
            "        count += 1\n"
            "    return total / max(count, 1)\n"
            "```\n\n"
            "Constraints:\n"
            "- preserve deterministic behavior\n"
            "- keep intermediate names explicit\n"
            "- mention any state that must be reset between runs\n"
        ),
    },
)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if not 0.0 <= q <= 1.0:
        raise ValueError("percentile q must be within [0, 1]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = q * (len(ordered) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def _nested_token_ids_hash(token_groups: list[list[int]]) -> str:
    payload = json.dumps(token_groups, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _cuda_synchronize() -> None:
    import torch

    torch.cuda.synchronize()


def _cuda_reset_peak_memory_stats() -> None:
    import torch

    torch.cuda.reset_peak_memory_stats()


def _cuda_max_memory_allocated() -> int:
    import torch

    return int(torch.cuda.max_memory_allocated())


def dsv4_warm_baseline_workloads(tokenizer: Any) -> list[dict[str, Any]]:
    """Build the fixed tokenizer-derived DSV4 warm-baseline prompt set."""
    workloads = []
    for spec in DSV4_WARM_BASELINE_SPECS:
        prompt_ids = _repeat_tokenized_text(tokenizer, spec["text"], spec["prompt_len"])
        workloads.append(
            {
                "name": spec["name"],
                "kind": spec["kind"],
                "prompt_ids": prompt_ids,
                "prompt_len": len(prompt_ids),
                "prompt_hash": _token_ids_hash(prompt_ids),
            }
        )
    return workloads


def _dsv4_decode_graph_status(backend: Any) -> str:
    snapshot = getattr(backend, "snapshot", None)
    if not callable(snapshot):
        return "unknown"
    try:
        status = dict(snapshot().dflash_cg_status)
    except Exception:
        return "unknown"
    return str(status.get("decode", "unknown"))


def _dsv4_model_identity(backend: Any) -> dict[str, str]:
    identity = getattr(backend, "bfdiag_model_identity", None)
    if not isinstance(identity, dict) or not identity.get("path") or not identity.get("revision"):
        raise RuntimeError(
            "DSV4 warm baseline requires backend.bfdiag_model_identity with "
            "non-empty path and revision; load it through DeepseekV4EngineProvider"
        )
    return {"path": str(identity["path"]), "revision": str(identity["revision"])}


def _run_dsv4_warm_baseline_sample(
    backend: Any,
    prompt_ids: list[int],
    *,
    batch: int,
    generated_tokens: int,
    synchronize: Callable[[], None] = _cuda_synchronize,
    reset_peak_memory: Callable[[], None] = _cuda_reset_peak_memory_stats,
    peak_memory_allocated: Callable[[], int] = _cuda_max_memory_allocated,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    if batch < 1:
        raise ValueError("batch must be positive")
    if generated_tokens < 2:
        raise ValueError("generated_tokens must be at least two")

    from runtime.sampling import SamplingParams

    slots = list(range(batch))
    params = [SamplingParams(temperature=0.0) for _ in slots]
    slot_outputs = [[] for _ in slots]
    decode_itl_ms: list[float] = []

    try:
        reset_peak_memory()
        synchronize()
        prefill_start = clock()
        current_tokens = []
        for slot_index, slot in enumerate(slots):
            token = int(backend.prefill(slot, prompt_ids))
            current_tokens.append(token)
            slot_outputs[slot_index].append(token)
        synchronize()
        prefill_s = clock() - prefill_start

        for _ in range(generated_tokens - 1):
            kv_lengths = [int(backend.slot_state(slot).kv_len) for slot in slots]
            synchronize()
            step_start = clock()
            next_tokens = backend.decode_batch_sampled(slots, current_tokens, kv_lengths, params)
            synchronize()
            decode_itl_ms.append((clock() - step_start) * 1000.0)
            current_tokens = [int(token) for token in next_tokens]
            for slot_index, token in enumerate(current_tokens):
                slot_outputs[slot_index].append(token)
        peak_memory_allocated_bytes = peak_memory_allocated()
    finally:
        for slot in slots:
            backend.reset_slot(slot)

    prefill_tokens = len(prompt_ids) * batch
    decode_steps = generated_tokens - 1
    decode_s = sum(decode_itl_ms) / 1000.0
    return {
        "batch": batch,
        "prompt_len": len(prompt_ids),
        "generated_tokens": generated_tokens,
        "prefill_s": prefill_s,
        "prefill_tok_s": prefill_tokens / prefill_s if prefill_s > 0.0 else 0.0,
        "decode_s": decode_s,
        "decode_steps": decode_steps,
        "itl_p50_ms": _percentile(decode_itl_ms, 0.50),
        "itl_p95_ms": _percentile(decode_itl_ms, 0.95),
        "tok_s": decode_steps / decode_s if decode_s > 0.0 else 0.0,
        "aggregate_tok_s": (decode_steps * batch) / decode_s if decode_s > 0.0 else 0.0,
        "output_hash": _nested_token_ids_hash(slot_outputs),
        "slot_output_hashes": [_token_ids_hash(tokens) for tokens in slot_outputs],
        "slot_outputs": slot_outputs,
        "decode_itl_ms": decode_itl_ms,
        "peak_memory_allocated_bytes": peak_memory_allocated_bytes,
    }


def run_dsv4_warm_baseline_case(
    backend: Any,
    tokenizer: Any,
    *,
    workload_name: str,
    batch: int = 1,
    generated_tokens: int = DSV4_WARM_BASELINE_GENERATED_TOKENS,
) -> dict[str, Any]:
    """Run one warm DSV4 baseline case.

    This measures an already-loaded backend only. It intentionally excludes
    model load, first-process initialization, and cold-TTFT claims; use a
    cold-process workflow for those questions.
    """
    if generated_tokens < 2:
        raise ValueError("generated_tokens must be at least two")
    if batch not in (1, 2, 4):
        raise ValueError("batch must be 1, 2, or 4")
    if batch > int(backend.num_slots):
        raise ValueError(f"batch={batch} requires backend.num_slots>={batch}")

    workload = next(
        (item for item in dsv4_warm_baseline_workloads(tokenizer) if item["name"] == workload_name),
        None,
    )
    if workload is None:
        known = ", ".join(spec["name"] for spec in DSV4_WARM_BASELINE_SPECS)
        raise ValueError(f"unknown workload_name {workload_name!r}; expected one of: {known}")

    model_identity = _dsv4_model_identity(backend)
    # Prime the exact path once before the measured sample so the result is a
    # warm-engine decode baseline rather than a first-touch artifact.
    warmup = _run_dsv4_warm_baseline_sample(
        backend,
        workload["prompt_ids"],
        batch=batch,
        generated_tokens=generated_tokens,
    )
    graph_capture_status = _dsv4_decode_graph_status(backend)
    with run_record(
        script=__file__,
        model=model_identity,
        workload={
            "contract": DSV4_WARM_BASELINE_CONTRACT,
            "contract_version": DSV4_WARM_BASELINE_VERSION,
            "workload_name": workload["name"],
            "prompt_hash": workload["prompt_hash"],
            "prompt_len": workload["prompt_len"],
            "generated_tokens": generated_tokens,
            "batch": batch,
            "max_model_len": int(backend.max_seq_len),
            "max_q_rows": int(backend.max_q_rows),
            "capacity": int(backend.num_slots),
            "cuda_graph_status": graph_capture_status,
            "warm_only": True,
        },
        extra={
            "workload_extra": {
                "kind": workload["kind"],
                "graph_capture_status": graph_capture_status,
                "warm_only": True,
            }
        },
    ) as rec:
        result = _run_dsv4_warm_baseline_sample(
            backend,
            workload["prompt_ids"],
            batch=batch,
            generated_tokens=generated_tokens,
        )
        if result["output_hash"] != warmup["output_hash"]:
            raise AssertionError(
                "DSV4 warm baseline output changed between identical reset runs: "
                f"warmup={warmup['output_hash']} measured={result['output_hash']}"
            )
        rec.metric("prefill_tok_s", result["prefill_tok_s"])
        rec.metric("decode_tok_s", result["tok_s"])
        rec.metric("aggregate_tok_s", result["aggregate_tok_s"])
        rec.metric("itl_p50_ms", result["itl_p50_ms"])
        rec.metric("itl_p95_ms", result["itl_p95_ms"])
        rec.metric("prefill_s", result["prefill_s"])
        rec.metric("decode_s", result["decode_s"])
        rec.metric("peak_memory_allocated_bytes", result["peak_memory_allocated_bytes"])
        rec.record.fingerprint.extra["output_hash"] = result["output_hash"]
        rec.record.fingerprint.extra["slot_output_hashes"] = result["slot_output_hashes"]
        rec.record.fingerprint.extra["decode_itl_ms"] = result["decode_itl_ms"]
        run_id = rec.run_id

    return {
        "run_id": run_id,
        "contract": DSV4_WARM_BASELINE_CONTRACT,
        "contract_version": DSV4_WARM_BASELINE_VERSION,
        "workload_name": workload["name"],
        "workload_kind": workload["kind"],
        "prompt_hash": workload["prompt_hash"],
        "prompt_len": workload["prompt_len"],
        "generated_tokens": generated_tokens,
        "batch": batch,
        "max_model_len": int(backend.max_seq_len),
        "max_q_rows": int(backend.max_q_rows),
        "num_slots": int(backend.num_slots),
        "graph_capture_status": graph_capture_status,
        **result,
    }


def run_dsv4_warm_baseline(
    backend: Any,
    tokenizer: Any,
    *,
    generated_tokens: int = DSV4_WARM_BASELINE_GENERATED_TOKENS,
) -> dict[str, Any]:
    """Run the fixed DSV4 warm-baseline suite for native B=1/2/4 buckets.

    This suite is only for a warm, already-loaded backend. It is not valid
    evidence for cold load cost, cold TTFT, or first-touch initialization.
    """
    if generated_tokens < 2:
        raise ValueError("generated_tokens must be at least two")

    results = []
    for workload in dsv4_warm_baseline_workloads(tokenizer):
        results.append(
            run_dsv4_warm_baseline_case(
                backend,
                tokenizer,
                workload_name=workload["name"],
                batch=1,
                generated_tokens=generated_tokens,
            )
        )
        for batch in (2, 4):
            if int(backend.num_slots) < batch:
                continue
            results.append(
                run_dsv4_warm_baseline_case(
                    backend,
                    tokenizer,
                    workload_name=workload["name"],
                    batch=batch,
                    generated_tokens=generated_tokens,
                )
            )
    return {
        "contract": DSV4_WARM_BASELINE_CONTRACT,
        "contract_version": DSV4_WARM_BASELINE_VERSION,
        "generated_tokens": generated_tokens,
        "results": results,
    }


def _tensor_content_hash(tensor: Any) -> str:
    """Hash a CPU materialization without weakening a numeric diagnostic.

    Divergence reports already materialize logits and aux tensors on CPU.  A
    content hash gives graph candidates a compact, exact comparison target
    without writing multi-megabyte tensor dumps for every probe.
    """
    contiguous = tensor.detach().contiguous()
    if str(contiguous.dtype) == "torch.bfloat16":
        import torch

        contiguous = contiguous.view(torch.uint8)
    contiguous = contiguous.cpu()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def capture_laguna_target_prefill_state(
    backend: Any,
    slot: int,
    prefix_len: int,
    first_token: int,
    aux_hidden_states: list[Any] | None,
) -> dict[str, Any]:
    """Exact, opt-in host hashes of one target slot immediately after prefill.

    The caller owns the diagnostic-only synchronization/copies.  Production
    generation never calls this helper, so its hashes cannot perturb target or
    draft execution order.
    """
    # Laguna allocates its own attention caches and deliberately uses a
    # different physical-slot convention from BlockPool.  Hash against the
    # allocator that owns these tensors; BlockPool reserves slot zero, while
    # Laguna's self-allocated cache starts logical slot zero at block zero.
    from runtime.backends.laguna import _physical_slot

    physical_slot = _physical_slot(slot)
    block_size = backend.block_size
    kv_len = int(backend.slot_kv_len[slot])
    full_blocks = (kv_len + block_size - 1) // block_size
    full_start = physical_slot * backend.blocks_per_slot
    ring_start = physical_slot * backend._ring_blocks_per_slot
    full_names = set(backend._full_layer_names)
    layer_hashes = []
    for name in backend.kv_caches:
        cache = backend.kv_caches[name]
        if name in full_names:
            view = cache[:, full_start : full_start + full_blocks]
            kind = "full"
        else:
            view = cache[:, ring_start : ring_start + backend._ring_blocks_per_slot]
            kind = "swa"
        layer_hashes.append(
            {
                "name": name,
                "kind": kind,
                "shape": list(view.shape),
                "sha256": _tensor_content_hash(view),
            }
        )
    aux = []
    for index, tensor in enumerate(aux_hidden_states or []):
        aux.append(
            {
                "index": index,
                "shape": list(tensor.shape),
                "sha256": _tensor_content_hash(tensor),
            }
        )
    return {
        "slot": slot,
        "physical_slot": physical_slot,
        "kv_len": kv_len,
        "prefix_len": prefix_len,
        "first_token": first_token,
        "layer_hashes": layer_hashes,
        "aux_hashes": aux,
    }


def compare_laguna_target_prefill_states(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Summarize exact target-state differences from two captured prefill boundaries."""
    scalar_fields = ("kv_len", "first_token")
    scalar_differences = {
        name: {"reference": reference[name], "candidate": candidate[name]}
        for name in scalar_fields
        if reference[name] != candidate[name]
    }
    layer_differences = [
        {
            "name": expected["name"],
            "kind": expected["kind"],
            "reference": expected["sha256"],
            "candidate": actual["sha256"],
        }
        for expected, actual in zip(reference["layer_hashes"], candidate["layer_hashes"])
        if expected != actual
    ]
    aux_differences = [
        {
            "index": expected["index"],
            "reference": expected["sha256"],
            "candidate": actual["sha256"],
        }
        for expected, actual in zip(reference["aux_hashes"], candidate["aux_hashes"])
        if expected != actual
    ]
    return {
        "scalar_differences": scalar_differences,
        "layer_difference_count": len(layer_differences),
        "first_layer_difference": layer_differences[0] if layer_differences else None,
        "aux_difference_count": len(aux_differences),
        "first_aux_difference": aux_differences[0] if aux_differences else None,
        "exact": not scalar_differences and not layer_differences and not aux_differences,
    }


def reset_dflash_workload_state(engine: Any) -> None:
    """Reset target/draft KV and undo any preceding M=1 graph patch."""
    from bfdiag.daemon.session import reset_laguna_engine

    # A previous M=1 workload may have left the backend temporarily patched
    # with its decode-CUDA-graph attention implementations.  Those are
    # invalid for DFlash's q=16 prefill and verify shapes.
    engine.backend._unpatch_impls_for_prefill()
    reset_laguna_engine(engine)


def restore_dflash_daemon_state(engine: Any) -> None:
    """Restore clean KV state and the normal decode-CUDA-graph bindings."""
    reset_dflash_workload_state(engine)
    engine.backend._repatch_impls_for_cg()


def check_dflash_server_step_parity(
    engine: Any,
    tokenizer: Any,
    *,
    max_tokens: int = 64,
) -> dict[str, Any]:
    """Prove the server's step-wise M=16 loop matches whole-generation DFlash."""
    if max_tokens < 2:
        raise ValueError("max_tokens must be at least two")

    prompt_ids = historical_dflash_m16_prompt_ids(tokenizer)
    reset_dflash_workload_state(engine)
    try:
        whole_tokens, _stats = engine.generate_verify_only(
            prompt_ids,
            max_tokens=max_tokens,
            temperature=0.0,
            slot=0,
            enable_prefix_cache=False,
        )

        reset_dflash_workload_state(engine)
        state = engine.dflash_prefill_bootstrap(0, prompt_ids)
        stepped_tokens = [state["anchor"]]
        while len(stepped_tokens) < max_tokens:
            decision = engine.dflash_round(0, state["anchor"], state["draft_tokens"])
            stepped_tokens.extend(decision["committed"])
            state = {
                "anchor": decision["next_anchor"],
                "draft_tokens": decision["next_draft_tokens"],
            }
        stepped_tokens = stepped_tokens[:max_tokens]
        if whole_tokens != stepped_tokens:
            mismatch = next(
                (
                    index
                    for index, pair in enumerate(zip(whole_tokens, stepped_tokens))
                    if pair[0] != pair[1]
                ),
                min(len(whole_tokens), len(stepped_tokens)),
            )
            whole_token = whole_tokens[mismatch] if mismatch < len(whole_tokens) else None
            stepped_token = stepped_tokens[mismatch] if mismatch < len(stepped_tokens) else None
            raise AssertionError(
                "DFlash server-step parity diverged at token "
                f"{mismatch}: whole={whole_token} step={stepped_token}"
            )
        return {"tokens": len(stepped_tokens), "parity": True}
    finally:
        reset_dflash_workload_state(engine)


def check_dflash_verify_cg_parity(
    engine: Any,
    tokenizer: Any,
    *,
    steps: int = 32,
) -> dict[str, Any]:
    """Compare captured and eager M=16 verify over the same DFlash state path.

    The draft CUDA graph remains enabled in both passes.  This isolates the
    target verify graph's attention metadata and aux-hidden-state capture from
    the draft model and from the ordinary server-step coordinator.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    captured_verify = engine._verify_cg
    if captured_verify is None:
        raise RuntimeError("DFlash verify CUDA Graph must be captured")

    prompt_ids = historical_dflash_m16_prompt_ids(tokenizer)
    prompt_hash = _token_ids_hash(prompt_ids)

    def run(
        verify_cg: Any | None,
    ) -> tuple[list[int], list[tuple[int, list[int], list[int]]]]:
        engine._verify_cg = verify_cg
        reset_dflash_workload_state(engine)
        state = engine.dflash_prefill_bootstrap(0, prompt_ids)
        output = [state["anchor"]]
        decisions = []
        for _ in range(steps):
            decision = engine.dflash_round(0, state["anchor"], state["draft_tokens"])
            output.extend(decision["committed"])
            decisions.append(
                (
                    decision["num_accepted"],
                    list(decision["committed"]),
                    list(decision["next_draft_tokens"]),
                )
            )
            state = {
                "anchor": decision["next_anchor"],
                "draft_tokens": decision["next_draft_tokens"],
            }
        return output, decisions

    try:
        with run_record(
            script=__file__,
            workload={
                "contract": HISTORICAL_DFLASH_M16_CONTRACT,
                "prompt_hash": prompt_hash,
                "prompt_len": len(prompt_ids),
                "k": 15,
                "greedy": True,
                "block_size": engine.backend.block_size,
                "blocks_per_slot": engine.backend.blocks_per_slot,
                "max_model_len": engine.backend.runtime_config.model_config.max_model_len,
                "capacity": engine.backend.num_slots,
                "dflash": True,
                "cuda_graph": True,
                "verify_cg_parity": True,
            },
        ) as rec:
            captured_output, captured_decisions = run(captured_verify)
            eager_output, eager_decisions = run(None)
            captured_hash = _token_ids_hash(captured_output)
            eager_hash = _token_ids_hash(eager_output)
            rec.metric("verify_parity_steps", steps)
            rec.metric("captured_output_tokens", len(captured_output))
            rec.metric("eager_output_tokens", len(eager_output))
            rec.record.fingerprint.extra["captured_output_hash"] = captured_hash
            rec.record.fingerprint.extra["eager_output_hash"] = eager_hash
            rec.record.fingerprint.extra["captured_decisions_hash"] = hashlib.sha256(
                json.dumps(captured_decisions, separators=(",", ":")).encode()
            ).hexdigest()
            rec.record.fingerprint.extra["eager_decisions_hash"] = hashlib.sha256(
                json.dumps(eager_decisions, separators=(",", ":")).encode()
            ).hexdigest()

            if captured_output != eager_output or captured_decisions != eager_decisions:
                mismatch_step = next(
                    (
                        index
                        for index, pair in enumerate(zip(captured_decisions, eager_decisions))
                        if pair[0] != pair[1]
                    ),
                    min(len(captured_decisions), len(eager_decisions)),
                )
                captured_decision = (
                    captured_decisions[mismatch_step]
                    if mismatch_step < len(captured_decisions)
                    else None
                )
                eager_decision = (
                    eager_decisions[mismatch_step] if mismatch_step < len(eager_decisions) else None
                )
                raise AssertionError(
                    "DFlash captured/eager verify parity diverged at step "
                    f"{mismatch_step}: captured={captured_decision} eager={eager_decision}"
                )
            run_id = rec.run_id
    finally:
        engine._verify_cg = captured_verify
        reset_dflash_workload_state(engine)

    return {
        "run_id": run_id,
        "steps": steps,
        "parity": True,
        "output_tokens": len(captured_output),
        "output_hash": captured_hash,
    }


def diagnose_dflash_verify_cg_divergence(
    engine: Any,
    tokenizer: Any,
    *,
    prefix_steps: int = 3,
) -> dict[str, Any]:
    """Record graph/eager logits and aux-layer error at one shared M=16 state."""
    if prefix_steps < 0:
        raise ValueError("prefix_steps must be non-negative")

    import torch

    captured_verify = engine._verify_cg
    if captured_verify is None:
        raise RuntimeError("DFlash verify CUDA Graph must be captured")
    prompt_ids = historical_dflash_m16_prompt_ids(tokenizer)

    def capture(verify_cg: Any | None) -> dict[str, Any]:
        engine._verify_cg = verify_cg
        reset_dflash_workload_state(engine)
        state = engine.dflash_prefill_bootstrap(0, prompt_ids)
        history: list[tuple[int, list[int], list[int]]] = []
        for _ in range(prefix_steps):
            decision = engine.dflash_round(0, state["anchor"], state["draft_tokens"])
            history.append(
                (
                    decision["num_accepted"],
                    list(decision["committed"]),
                    list(decision["next_draft_tokens"]),
                )
            )
            state = {
                "anchor": decision["next_anchor"],
                "draft_tokens": decision["next_draft_tokens"],
            }
        kv_len = engine.backend.slot_kv_len[0]
        verify_tokens = [state["anchor"]] + state["draft_tokens"]
        if verify_cg is not None:
            logits, aux = verify_cg.replay_with_aux(0, verify_tokens, kv_len)
        else:
            logits, aux = engine._forward_verify_with_aux(
                0, verify_tokens, kv_len, len(verify_tokens)
            )
        torch.cuda.synchronize()
        return {
            "history": history,
            "kv_len": kv_len,
            "anchor": state["anchor"],
            "draft_tokens": list(state["draft_tokens"]),
            "logits": logits.detach().float().cpu(),
            "aux": [item.detach().float().cpu() for item in aux] if aux is not None else None,
        }

    try:
        captured = capture(captured_verify)
        eager = capture(None)
    finally:
        engine._verify_cg = captured_verify
        reset_dflash_workload_state(engine)

    shared_state = all(
        captured[name] == eager[name] for name in ("history", "kv_len", "anchor", "draft_tokens")
    )
    if not shared_state:
        raise AssertionError("DFlash paths diverged before the requested diagnostic step")

    logits_delta = captured["logits"] - eager["logits"]
    captured_top1 = captured["logits"].argmax(dim=-1)
    eager_top1 = eager["logits"].argmax(dim=-1)
    top1_mismatch = (captured_top1 != eager_top1).nonzero().flatten().tolist()
    aux_report = []
    for index, (captured_aux, eager_aux) in enumerate(
        zip(captured["aux"] or [], eager["aux"] or [])
    ):
        delta = captured_aux - eager_aux
        aux_report.append(
            {
                "layer": index,
                "max_abs": float(delta.abs().max()),
                "rmse": float(delta.square().mean().sqrt()),
            }
        )

    report = {
        "prefix_steps": prefix_steps,
        "kv_len": captured["kv_len"],
        "anchor": captured["anchor"],
        "top1_mismatch_positions": top1_mismatch,
        "captured_top1": captured_top1.tolist(),
        "eager_top1": eager_top1.tolist(),
        "logits_max_abs": float(logits_delta.abs().max()),
        "logits_rmse": float(logits_delta.square().mean().sqrt()),
        "captured_logits_hash": _tensor_content_hash(captured["logits"]),
        "eager_logits_hash": _tensor_content_hash(eager["logits"]),
        "captured_aux_hashes": [_tensor_content_hash(item) for item in captured["aux"] or []],
        "eager_aux_hashes": [_tensor_content_hash(item) for item in eager["aux"] or []],
        "aux": aux_report,
    }
    with run_record(
        script=__file__,
        workload={
            "contract": HISTORICAL_DFLASH_M16_CONTRACT,
            "prompt_hash": _token_ids_hash(prompt_ids),
            "prompt_len": len(prompt_ids),
            "k": 15,
            "greedy": True,
            "block_size": engine.backend.block_size,
            "blocks_per_slot": engine.backend.blocks_per_slot,
            "max_model_len": engine.backend.runtime_config.model_config.max_model_len,
            "capacity": engine.backend.num_slots,
            "dflash": True,
            "cuda_graph": True,
            "verify_cg_divergence": True,
        },
    ) as rec:
        artifacts = default_store().artifacts_dir(rec.run_id)
        artifacts.mkdir(parents=True, exist_ok=True)
        report_path = artifacts / "verify_cg_divergence.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        rec.artifact("verify_cg_divergence", report_path)
        rec.metric("prefix_steps", prefix_steps)
        rec.metric("logits_max_abs", report["logits_max_abs"])
        rec.metric("logits_rmse", report["logits_rmse"])
        rec.metric("top1_mismatch_count", len(top1_mismatch))
        run_id = rec.run_id
    return {"run_id": run_id, **report}


def diagnose_dflash_verify_split_kv_chunking(
    engine: Any,
    *,
    probe_kv_lens: tuple[int, ...] = (64, 400, 500, 510, 511, 512, 513, 520, 600, 1000, 2016),
) -> dict[str, Any]:
    """Compare sparkinfer's verify-mode split-KV chunk sizing across the CG
    and eager verify paths, without running the model at all.

    Root-cause probe for notes/2026-08-02-eager-verify-cg-verify-divergence.md.
    Both ``LagunaCudaGraphVerify`` (CG verify, laguna_cuda_graph.py) and
    ``SparkinferPrefillWorkspace.forward`` (DFlash's eager verify fallback
    reached from ``_forward_verify_with_aux``, laguna_sparkinfer_attn.py)
    call sparkinfer's ``planner.create_paged_plan(mode="verify", ...)`` --
    but at different times, with different ``enable_cuda_graph``/
    ``cache_seqlens`` inputs:

    - CG (``LagunaCudaGraphVerify._init_workspaces``) calls it exactly ONCE,
      at construction, with ``enable_cuda_graph=True`` and
      ``cache_seqlens=[max_kv]`` -- the group's entire workspace capacity
      (``(blocks_per_slot + 16) * block_size - 1``), the worst case, not the
      real replay length. The resulting ``plan.kv_chunk_size`` is written
      into a device buffer (``kv_chunk_size_ptr``) once and, per
      ``PagedAttentionWorkspace.update_prefill_graph_replay_metadata``'s
      Triton kernel (``update_prefill_graph_work_metadata_triton``, the
      non-``ADAPTIVE_CHUNKING`` branch -- see below for why that branch
      never engages here), is READ BACK UNCHANGED on every replay: the
      chunk *size* in tokens never adapts to the real replay ``kv_len``,
      only the chunk *count* (``num_chunks_kv = ceil(pages_needed /
      chunk_pages)``) does, using that frozen size.
    - Eager (``SparkinferPrefillWorkspace.forward``) calls it FRESH on every
      real verify call, with ``enable_cuda_graph=False`` and the ACTUAL
      current ``kv_len``, under an effectively unconstrained
      ``max_batch_size_if_split`` budget -- so it is free to pick a much
      finer chunk size than CG's capacity-constrained worst-case one.

    ``ADAPTIVE_CHUNKING`` is sparkinfer's only mechanism for re-deriving
    chunk size at CG replay time (see
    ``PagedAttentionWorkspace._uses_laguna_verify_analytic_schedule`` and
    ``update_prefill_graph_work_metadata_triton``'s ``ADAPTIVE_CHUNKING``
    branch), and it is hard-gated to one Laguna-specific fast shape:
    ``cta_tile_q == 64`` (which itself requires ``packed_qo_len == 48``,
    i.e. a verify query window of exactly 8 tokens under GQA6) and
    ``page_size == 128``. This deployment's real DFlash verify window is
    ``NUM_QUERY_PER_REQ=16`` tokens (``dflash_constants.py``), not 8, and
    the production block size is 64, not 128 (``server/app.py``'s
    ``SERVER_BLOCK_SIZE`` default) -- so ``cta_tile_q`` resolves to 16 for
    both paths (confirmed below: both report the same ``cta_tile_q``) and
    the adaptive path never engages. CG's chunk size is frozen for the
    life of the process.

    Calls ``create_paged_plan`` directly against synthetic CUDA tensors
    (correct shapes/dtypes/devices only -- no real weights, no forward
    pass, no KV cache reads) so this needs no model load beyond whatever
    the caller already loaded to get ``engine``.
    """
    import torch

    from runtime.backends.dflash_constants import NUM_QUERY_PER_REQ

    backend = engine.backend
    block_size = int(backend.block_size)
    blocks_per_slot = int(backend.blocks_per_slot)
    full_groups = [key for key in backend._layer_groups if key[0] == -1]
    if not full_groups:
        raise RuntimeError(
            "diagnose_dflash_verify_split_kv_chunking requires a full-attention "
            "(window_left=-1) layer group -- none found on this backend"
        )
    window_left, num_q_heads, num_kv_heads = full_groups[0]
    head_dim = 128
    device = torch.device("cuda")

    def make_plan(*, cache_seqlen: int, enable_cuda_graph: bool, num_pages: int):
        from sparkinfer.attention.paged.planner import create_paged_plan

        q = torch.zeros(
            NUM_QUERY_PER_REQ, num_q_heads, head_dim, dtype=torch.bfloat16, device=device
        )
        k_cache = torch.zeros(
            num_pages,
            block_size,
            num_kv_heads,
            head_dim,
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        v_cache = torch.zeros_like(k_cache)
        page_table = torch.arange(num_pages, dtype=torch.int32, device=device).unsqueeze(0)
        cache_seqlens = torch.tensor([cache_seqlen], dtype=torch.int32, device=device)
        cu_seqlens_q = torch.tensor([0, NUM_QUERY_PER_REQ], dtype=torch.int32, device=device)
        return create_paged_plan(
            q,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            mode="verify",
            enable_cuda_graph=enable_cuda_graph,
            window_left=window_left,
        )

    # CG's own worst-case sizing call -- mirrors LagunaCudaGraphVerify.
    # _init_workspaces exactly (max_pages, max_kv formulas copied verbatim).
    cg_max_pages = blocks_per_slot + 16
    cg_max_kv = cg_max_pages * block_size - 1
    cg_plan = make_plan(cache_seqlen=cg_max_kv, enable_cuda_graph=True, num_pages=cg_max_pages)
    cg_chunk_pages = max((int(cg_plan.kv_chunk_size) + block_size - 1) // block_size, 1)

    rows = []
    for kv_len in probe_kv_lens:
        num_pages_needed = max((kv_len + block_size - 1) // block_size, 1)
        eager_plan = make_plan(
            cache_seqlen=kv_len, enable_cuda_graph=False, num_pages=num_pages_needed
        )
        eager_num_tiles_q = max(
            (NUM_QUERY_PER_REQ * int(eager_plan.gqa_group_size) + int(eager_plan.cta_tile_q) - 1)
            // int(eager_plan.cta_tile_q),
            1,
        )
        eager_num_chunks_kv = int(eager_plan.new_batch_size) // eager_num_tiles_q
        cg_num_chunks_kv_here = max((num_pages_needed + cg_chunk_pages - 1) // cg_chunk_pages, 1)
        rows.append(
            {
                "kv_len": kv_len,
                "eager_kv_chunk_size": int(eager_plan.kv_chunk_size),
                "eager_split_kv": bool(eager_plan.split_kv),
                "eager_num_chunks_kv": eager_num_chunks_kv,
                "eager_cta_tile_q": int(eager_plan.cta_tile_q),
                "cg_num_chunks_kv_if_replayed_here": cg_num_chunks_kv_here,
                "chunk_count_mismatch": eager_num_chunks_kv != cg_num_chunks_kv_here,
            }
        )

    report = {
        "window_left": window_left,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "block_size": block_size,
        "blocks_per_slot": blocks_per_slot,
        "num_query_per_req": NUM_QUERY_PER_REQ,
        "cg_max_kv": cg_max_kv,
        "cg_kv_chunk_size": int(cg_plan.kv_chunk_size),
        "cg_chunk_pages": cg_chunk_pages,
        "cg_split_kv": bool(cg_plan.split_kv),
        "cg_cta_tile_q": int(cg_plan.cta_tile_q),
        "rows": rows,
    }
    with run_record(
        script=__file__,
        workload={
            "contract": "dflash_verify_split_kv_chunking",
            "block_size": block_size,
            "blocks_per_slot": blocks_per_slot,
            "num_query_per_req": NUM_QUERY_PER_REQ,
        },
    ) as rec:
        artifacts = default_store().artifacts_dir(rec.run_id)
        artifacts.mkdir(parents=True, exist_ok=True)
        report_path = artifacts / "verify_split_kv_chunking.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        rec.artifact("verify_split_kv_chunking", report_path)
        rec.metric("cg_kv_chunk_size", report["cg_kv_chunk_size"])
        rec.metric("mismatch_count", sum(1 for row in rows if row["chunk_count_mismatch"]))
        run_id = rec.run_id
    return {"run_id": run_id, **report}


def diagnose_dflash_verify_dense_oracle(
    engine: Any,
    *,
    kv_lens: tuple[int, ...] = (64, 400, 500),
) -> dict[str, Any]:
    """Three-way compare CG-verify / eager-verify / a dense causal attention
    oracle at the same real ``kv_len``, to settle which (if either) matches
    ground truth.

    ``diagnose_dflash_verify_split_kv_chunking`` established a structural
    fact: CG and eager compute genuinely different split-KV chunk counts
    for the same real ``kv_len`` (CG collapses to 1 chunk almost always;
    eager splits into 4-16 chunks once ``kv_len`` >= ~400). That does not
    say which chunk count -- if either -- is *correct*. This function
    answers that by adding a third, independent measurement: a
    from-scratch dense causal attention (``bfdiag.dense_attention_oracle.
    DenseCausalOracleImpl``) that never calls sparkinfer's
    ``create_paged_plan``/``paged_attention_forward`` at all, so it cannot
    share a bug with either split-KV path.

    Method, mirroring the prior note's already-validated "sequential with
    full reset in between" protocol (the note's double-slot-isolation test
    confirmed this gives the same numbers as fully independent slots, so
    reusing it here is not re-opening a settled methodology question):
    for each ``kv_len``, three independent fresh prefills of slot 0 (reset
    -> ``prefill_with_aux`` -> the same fixed verify window), one per
    variant (cg / eager / oracle). Greedy, deterministic model -> the three
    prefills produce identical KV state and an identical first ("bonus")
    token, asserted below before trusting any logits comparison.

    GQA head-mapping calibration: sparkinfer's actual query-head-to-kv-head
    convention lives in compiled CuTe device code, not visible from Python,
    so ``DenseCausalOracleImpl`` cannot simply assume one. This function
    calibrates it empirically at kv_len=64 -- the one point the prior note
    already proved CG and eager agree on bit-exactly -- by running the
    oracle with both ``"interleaved"`` (HF/vLLM ``repeat_kv`` convention)
    and ``"blocked"`` mappings and keeping whichever one agrees with the
    CG/eager consensus at that point (reported under ``"calibration"`` in
    the returned report, so the choice is visible, not asserted). Every
    other ``kv_len`` in ``kv_lens`` is then scored only against the
    calibrated mapping.

    Judgement to apply to the ``rows`` this returns:
      - CG close to oracle, eager far -> eager is the wrong one (fix
        direction 1 in the note: make eager collapse to one chunk too --
        and go find out *why* sparkinfer's multi-chunk merge is wrong).
      - eager close to oracle, CG far -> CG is wrong, and CG is what
        production actually serves today -- treat as urgent, do not
        silently proceed to a fix.
      - both far from oracle -> the bug is not (only) in chunk count;
        report as-is, do not force a conclusion.
      - all three agree at 0.999991+ cosine (the historical "correct
        kernel variant" bar recorded in
        notes/2026-07-27-bfdiag-oracle-divergence.md) but argmax still
        flips -> that is a near-tied-logit argmax flip, not the same
        severity class as a real >>noise-level logit gap; say so
        explicitly rather than let it read as confirming a bug.
    """
    import torch

    from bfdiag.dense_attention_oracle import DenseCausalOracleImpl
    from oracle.comparator import compare_values
    from runtime.backends.dflash_constants import NUM_QUERY_PER_REQ

    backend = engine.backend
    captured_verify = engine._verify_cg
    if captured_verify is None:
        raise RuntimeError("DFlash verify CUDA Graph must be captured")

    full_groups = [key for key in backend._layer_groups if key[0] == -1]
    if not full_groups:
        raise RuntimeError(
            "diagnose_dflash_verify_dense_oracle requires a full-attention "
            "(window_left=-1) layer group -- none found on this backend"
        )
    full_key = full_groups[0]
    full_layer_names = backend._layer_groups[full_key]
    _window_left, num_q_heads, num_kv_heads = full_key
    head_dim = 128
    scale = head_dim**-0.5

    sfc = backend.static_forward_context
    original_impls = {name: sfc[name].impl for name in full_layer_names}

    def run_variant(*, kv_len: int, use_cg: bool, gqa_head_mapping: str | None) -> dict[str, Any]:
        prompt_ids = historical_m1_prompt_ids(kv_len)
        reset_dflash_workload_state(engine)
        first_token, _aux = backend.prefill_with_aux(0, prompt_ids)
        verify_tokens = [first_token] + [11] * (NUM_QUERY_PER_REQ - 1)
        patched = gqa_head_mapping is not None
        try:
            if patched:
                for name in full_layer_names:
                    sfc[name].impl = DenseCausalOracleImpl(
                        num_heads=num_q_heads,
                        head_size=head_dim,
                        scale=scale,
                        num_kv_heads=num_kv_heads,
                        gqa_head_mapping=gqa_head_mapping,
                    )
            if use_cg:
                logits, _aux2 = engine._verify_cg.replay_with_aux(0, verify_tokens, kv_len)
            else:
                logits, _aux2 = engine._forward_verify_with_aux(
                    0, verify_tokens, kv_len, len(verify_tokens)
                )
            torch.cuda.synchronize()
        finally:
            if patched:
                for name in full_layer_names:
                    sfc[name].impl = original_impls[name]
        return {
            "first_token": first_token,
            "verify_tokens": list(verify_tokens),
            "logits": logits.detach().float().cpu(),
        }

    def vs_reference(
        reference: torch.Tensor, candidate: torch.Tensor, label: str
    ) -> dict[str, Any]:
        cmp = compare_values(reference.reshape(-1), candidate.reshape(-1), top_k=10)
        argmax_ref = reference.argmax(dim=-1)
        argmax_cand = candidate.argmax(dim=-1)
        mismatch_positions = (argmax_ref != argmax_cand).nonzero().flatten().tolist()
        return {
            "label": label,
            "cosine": cmp.cosine_similarity,
            "max_abs_error": cmp.max_abs_error,
            "mean_abs_error": cmp.mean_abs_error,
            "top10_agreement": cmp.top_k_agreement,
            "argmax_mismatch_positions": mismatch_positions,
        }

    try:
        # -- Calibration at kv_len=64 (the prior note's proven bit-exact point) --
        calib_cg = run_variant(kv_len=64, use_cg=True, gqa_head_mapping=None)
        calib_eager = run_variant(kv_len=64, use_cg=False, gqa_head_mapping=None)
        if calib_cg["verify_tokens"] != calib_eager["verify_tokens"]:
            raise AssertionError("calibration: cg/eager prefill state diverged before kv_len=64")
        calib_interleaved = run_variant(kv_len=64, use_cg=False, gqa_head_mapping="interleaved")
        calib_blocked = run_variant(kv_len=64, use_cg=False, gqa_head_mapping="blocked")
        calib_scores = {
            "interleaved": compare_values(
                calib_cg["logits"].reshape(-1), calib_interleaved["logits"].reshape(-1)
            ).cosine_similarity,
            "blocked": compare_values(
                calib_cg["logits"].reshape(-1), calib_blocked["logits"].reshape(-1)
            ).cosine_similarity,
        }
        gqa_head_mapping = max(calib_scores, key=calib_scores.__getitem__)
        calibration = {
            "kv_len": 64,
            "cg_vs_eager_cosine": compare_values(
                calib_cg["logits"].reshape(-1), calib_eager["logits"].reshape(-1)
            ).cosine_similarity,
            "oracle_interleaved_vs_cg_cosine": calib_scores["interleaved"],
            "oracle_blocked_vs_cg_cosine": calib_scores["blocked"],
            "chosen_gqa_head_mapping": gqa_head_mapping,
        }

        rows = []
        for kv_len in kv_lens:
            cg = run_variant(kv_len=kv_len, use_cg=True, gqa_head_mapping=None)
            eager = run_variant(kv_len=kv_len, use_cg=False, gqa_head_mapping=None)
            oracle = run_variant(kv_len=kv_len, use_cg=False, gqa_head_mapping=gqa_head_mapping)
            if not (cg["verify_tokens"] == eager["verify_tokens"] == oracle["verify_tokens"]):
                raise AssertionError(
                    f"kv_len={kv_len}: cg/eager/oracle prefill state diverged before "
                    "the comparison even started -- not a verify-path bug, a test-harness one"
                )
            rows.append(
                {
                    "kv_len": kv_len,
                    "first_token": cg["first_token"],
                    "cg_vs_oracle": vs_reference(oracle["logits"], cg["logits"], "cg"),
                    "eager_vs_oracle": vs_reference(oracle["logits"], eager["logits"], "eager"),
                    "cg_vs_eager": vs_reference(eager["logits"], cg["logits"], "cg_vs_eager"),
                }
            )
    finally:
        for name in full_layer_names:
            sfc[name].impl = original_impls[name]
        reset_dflash_workload_state(engine)

    report = {
        "window_left": _window_left,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "calibration": calibration,
        "rows": rows,
    }
    with run_record(
        script=__file__,
        workload={
            "contract": "dflash_verify_dense_oracle",
            "kv_lens": list(kv_lens),
            "gqa_head_mapping": gqa_head_mapping,
        },
    ) as rec:
        artifacts = default_store().artifacts_dir(rec.run_id)
        artifacts.mkdir(parents=True, exist_ok=True)
        report_path = artifacts / "verify_dense_oracle.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        rec.artifact("verify_dense_oracle", report_path)
        rec.metric("calibration_cg_vs_eager_cosine", calibration["cg_vs_eager_cosine"])
        for row in rows:
            kv = row["kv_len"]
            rec.metric(f"kv{kv}_cg_vs_oracle_cosine", row["cg_vs_oracle"]["cosine"])
            rec.metric(f"kv{kv}_eager_vs_oracle_cosine", row["eager_vs_oracle"]["cosine"])
        run_id = rec.run_id
    return {"run_id": run_id, **report}


def run_historical_m1_decode_cg(
    backend: Any,
    *,
    rounds: int = 3,
) -> dict[str, Any]:
    """Run the frozen no-DFlash 64K M=1 CUDA-graph decode contract.

    This intentionally mirrors ``66d5913``'s measurement order: capture on
    the 64K base prompt, warm the captured graph, then for each measured round
    fully prefill ``base + suffix`` and time exactly 127 graph replays.  The
    measurement has no prefix-cache admission/reuse and does not touch DFlash.
    It is therefore comparable to
    ``benchmarks/fixtures/repro_80tok_20260727_0009.json`` (76.8 tok/s mean).
    """
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    if backend.block_size != HISTORICAL_M1_BLOCK_SIZE:
        raise ValueError(
            f"{HISTORICAL_M1_CONTRACT} requires block_size="
            f"{HISTORICAL_M1_BLOCK_SIZE}, got {backend.block_size}"
        )
    if backend.blocks_per_slot != HISTORICAL_M1_BLOCKS_PER_SLOT:
        raise ValueError(
            f"{HISTORICAL_M1_CONTRACT} requires blocks_per_slot="
            f"{HISTORICAL_M1_BLOCKS_PER_SLOT}, got {backend.blocks_per_slot}"
        )

    import torch

    base_ids = historical_m1_prompt_ids(HISTORICAL_M1_CONTEXT_TOKENS)
    suffix_ids = [token + 50_000 for token in historical_m1_prompt_ids(HISTORICAL_M1_SUFFIX_TOKENS)]
    prompt_hash = _token_ids_hash(base_ids)

    # The graph is a one-time warm-engine expense.  Its capture input and
    # warmup order are part of the historical contract, so do not fold this
    # into the timed rounds below.
    try:
        backend.reset_slot(0)
        backend._unpatch_impls_for_prefill()
        first_token = backend.prefill(0, base_ids)
        backend._ensure_decode_cg()
        cg = backend._decode_cg
        if cg is None:
            raise RuntimeError("historical M=1 decode CUDA Graph capture failed")
        kv_len = backend.slot_kv_len[0]
        for _ in range(20):
            first_token = cg.replay([0], [first_token], [kv_len])[0]
            kv_len += 1

        results: list[dict[str, float]] = []
        with run_record(
            script=__file__,
            workload={
                "contract": HISTORICAL_M1_CONTRACT,
                "prompt_hash": prompt_hash,
                "prompt_len": HISTORICAL_M1_CONTEXT_TOKENS + HISTORICAL_M1_SUFFIX_TOKENS,
                "greedy": True,
                "block_size": HISTORICAL_M1_BLOCK_SIZE,
                "blocks_per_slot": HISTORICAL_M1_BLOCKS_PER_SLOT,
                "max_model_len": HISTORICAL_M1_MAX_MODEL_LEN,
                "dflash": False,
                "prefix_cache": False,
                "b1_metadata_fastpath": bool(getattr(cg, "_b1_metadata_fastpath_enabled", False)),
            },
        ) as rec:
            for index in range(rounds):
                backend.reset_slot(0)
                backend._unpatch_impls_for_prefill()
                torch.cuda.synchronize()
                prefill_start = time.perf_counter()
                token = backend.prefill(0, base_ids + suffix_ids)
                torch.cuda.synchronize()
                prefill_s = time.perf_counter() - prefill_start

                backend._repatch_impls_for_cg()
                kv_len = backend.slot_kv_len[0]
                torch.cuda.synchronize()
                decode_start = time.perf_counter()
                for _ in range(HISTORICAL_M1_NEW_TOKENS - 1):
                    token = cg.replay([0], [token], [kv_len])[0]
                    kv_len += 1
                torch.cuda.synchronize()
                decode_s = time.perf_counter() - decode_start
                row = {
                    "prefill_s": prefill_s,
                    "step_ms": decode_s / (HISTORICAL_M1_NEW_TOKENS - 1) * 1000,
                    "tok_s": (HISTORICAL_M1_NEW_TOKENS - 1) / decode_s,
                }
                results.append(row)
                for name, value in row.items():
                    rec.metric(f"round_{index + 1}_{name}", value)
            average_tok_s = sum(row["tok_s"] for row in results) / len(results)
            rec.metric("avg_tok_s", average_tok_s)
            rec.metric("best_tok_s", max(row["tok_s"] for row in results))
            run_id = rec.run_id
    finally:
        backend._unpatch_impls_for_prefill()
        backend.reset_slot(0)
    return {
        "run_id": run_id,
        "contract": HISTORICAL_M1_CONTRACT,
        "rounds": results,
        "avg_tok_s": average_tok_s,
        "best_tok_s": max(row["tok_s"] for row in results),
    }


def profile_historical_m1_decode_cg(
    backend: Any,
    *,
    replay_count: int = 32,
) -> dict[str, Any]:
    """Capture a CUDA-profiler artifact for the frozen M=1 decode path.

    Profiler overhead makes this unsuitable for tok/s comparisons.  It is a
    diagnosis-only companion to :func:`run_historical_m1_decode_cg`, with the
    same 64K prompt, graph, and slot-reset sequence.  The returned run record
    holds both Chrome trace and a CUDA-time-sorted text table.
    """
    if replay_count <= 0:
        raise ValueError("replay_count must be positive")
    if backend._decode_cg is None:
        raise RuntimeError("capture the M=1 decode graph before profiling it")

    import torch

    base_ids = historical_m1_prompt_ids(HISTORICAL_M1_CONTEXT_TOKENS)
    suffix_ids = [token + 50_000 for token in historical_m1_prompt_ids(HISTORICAL_M1_SUFFIX_TOKENS)]
    cg = backend._decode_cg
    try:
        backend.reset_slot(0)
        backend._unpatch_impls_for_prefill()
        token = backend.prefill(0, base_ids + suffix_ids)
        backend._repatch_impls_for_cg()
        kv_len = backend.slot_kv_len[0]

        with run_record(
            script=__file__,
            workload={
                "contract": HISTORICAL_M1_CONTRACT,
                "prompt_len": HISTORICAL_M1_CONTEXT_TOKENS + HISTORICAL_M1_SUFFIX_TOKENS,
                "greedy": True,
                "block_size": HISTORICAL_M1_BLOCK_SIZE,
                "blocks_per_slot": HISTORICAL_M1_BLOCKS_PER_SLOT,
                "max_model_len": HISTORICAL_M1_MAX_MODEL_LEN,
                "dflash": False,
                "prefix_cache": False,
                "profile_only": True,
            },
        ) as rec:
            activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            with torch.profiler.profile(activities=activities) as profiler:
                for _ in range(replay_count):
                    token = cg.replay([0], [token], [kv_len])[0]
                    kv_len += 1
            torch.cuda.synchronize()
            artifacts = default_store().artifacts_dir(rec.run_id)
            artifacts.mkdir(parents=True, exist_ok=True)
            table_path = artifacts / "cuda_profile.txt"
            table_path.write_text(
                profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=80)
            )
            trace_path = artifacts / "cuda_profile.json"
            profiler.export_chrome_trace(str(trace_path))
            rec.artifact("cuda_profile_table", table_path)
            rec.artifact("cuda_profile_trace", trace_path)
            rec.metric("profiled_replays", replay_count)
            run_id = rec.run_id
    finally:
        backend._unpatch_impls_for_prefill()
        backend.reset_slot(0)
    return {"run_id": run_id, "replay_count": replay_count}


def check_b1_metadata_fastpath_parity(backend: Any, *, tokens: int = 32) -> dict[str, Any]:
    """Compare scalar and fused B=1 metadata updates on the frozen prompt."""
    if tokens <= 0:
        raise ValueError("tokens must be positive")

    base_ids = historical_m1_prompt_ids(HISTORICAL_M1_CONTEXT_TOKENS)
    suffix_ids = [token + 50_000 for token in historical_m1_prompt_ids(HISTORICAL_M1_SUFFIX_TOKENS)]

    def decode_with_fastpath(enabled: bool) -> list[int]:
        backend.reset_slot(0)
        backend._unpatch_impls_for_prefill()
        first = backend.prefill(0, base_ids + suffix_ids)
        backend._ensure_decode_cg()
        cg = backend._decode_cg
        if cg is None:
            raise RuntimeError("M=1 decode CUDA Graph capture failed")
        cg.reset()
        cg._b1_metadata_fastpath_enabled = enabled
        output = [first]
        kv_len = backend.slot_kv_len[0]
        for _ in range(tokens - 1):
            output.append(cg.replay([0], [output[-1]], [kv_len])[0])
            kv_len += 1
        return output

    try:
        scalar = decode_with_fastpath(False)
        fused = decode_with_fastpath(True)
        if scalar != fused:
            mismatch = next(
                (index for index, pair in enumerate(zip(scalar, fused)) if pair[0] != pair[1]),
                min(len(scalar), len(fused)),
            )
            expected = scalar[mismatch] if mismatch < len(scalar) else None
            actual = fused[mismatch] if mismatch < len(fused) else None
            raise AssertionError(
                f"B=1 metadata fast path diverged at token {mismatch}: "
                f"scalar={expected}, fused={actual}"
            )
        return {"tokens": len(fused), "parity": True}
    finally:
        backend._unpatch_impls_for_prefill()
        backend.reset_slot(0)


def run_historical_dflash_m16(
    engine: Any,
    tokenizer: Any,
    *,
    rounds: int = 3,
    max_tokens: int = HISTORICAL_DFLASH_M16_NEW_TOKENS,
) -> dict[str, Any]:
    """Measure the production M=16 DFlash server-step contract at 64K.

    The timed section calls ``dflash_round`` rather than a private forward so
    it includes the same verify, accept, draft-KV, and next-draft work that
    ``ServerEngine`` uses.  Each round starts from a fully reset engine: no
    prefix reuse, draft-KV residue, or prior acceptance sequence is allowed
    to influence the result.
    """
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    if max_tokens < 2:
        raise ValueError("max_tokens must be at least two")

    import torch

    backend = engine.backend
    if backend.block_size != HISTORICAL_M1_BLOCK_SIZE:
        raise ValueError(
            f"{HISTORICAL_DFLASH_M16_CONTRACT} requires block_size="
            f"{HISTORICAL_M1_BLOCK_SIZE}, got {backend.block_size}"
        )
    prompt_ids = historical_dflash_m16_prompt_ids(tokenizer)
    prompt_hash = _token_ids_hash(prompt_ids)

    # Capture is a warm-engine one-time cost, never part of steady-state
    # throughput.  The same production bootstrap also validates that the
    # draft and verify graphs exist before measured rounds start.
    reset_dflash_workload_state(engine)
    engine.dflash_prefill_bootstrap(0, prompt_ids)
    torch.cuda.synchronize()
    reset_dflash_workload_state(engine)

    results: list[dict[str, float]] = []
    try:
        with run_record(
            script=__file__,
            workload={
                "contract": HISTORICAL_DFLASH_M16_CONTRACT,
                "prompt_hash": prompt_hash,
                "prompt_len": len(prompt_ids),
                "max_tokens": max_tokens,
                "k": 15,
                "greedy": True,
                "block_size": backend.block_size,
                # These are hard comparability inputs: they determine the
                # paged-attention workspace geometry and graph eligibility.
                "blocks_per_slot": backend.blocks_per_slot,
                "max_model_len": backend.runtime_config.model_config.max_model_len,
                "capacity": backend.num_slots,
                "dflash": True,
                "prefix_cache": False,
                "server_step_path": True,
            },
        ) as rec:
            for index in range(rounds):
                reset_dflash_workload_state(engine)
                torch.cuda.synchronize()
                prefill_start = time.perf_counter()
                state = engine.dflash_prefill_bootstrap(0, prompt_ids)
                torch.cuda.synchronize()
                prefill_s = time.perf_counter() - prefill_start

                tokens = [state["anchor"]]
                total_draft = 0
                total_accepted = 0
                step_times_s: list[float] = []
                decode_start = time.perf_counter()
                while len(tokens) < max_tokens:
                    torch.cuda.synchronize()
                    step_start = time.perf_counter()
                    decision = engine.dflash_round(0, state["anchor"], state["draft_tokens"])
                    torch.cuda.synchronize()
                    step_times_s.append(time.perf_counter() - step_start)

                    total_draft += len(state["draft_tokens"])
                    total_accepted += decision["num_accepted"]
                    tokens.extend(decision["committed"])
                    state = {
                        "anchor": decision["next_anchor"],
                        "draft_tokens": decision["next_draft_tokens"],
                    }

                decode_s = time.perf_counter() - decode_start
                generated = max_tokens - 1
                row = {
                    "round": float(index + 1),
                    "prefill_s": prefill_s,
                    "decode_s": decode_s,
                    "tok_s": generated / max(decode_s, 1e-6),
                    "steps": float(len(step_times_s)),
                    "avg_round_ms": 1000 * sum(step_times_s) / max(len(step_times_s), 1),
                    "acceptance_rate": total_accepted / max(total_draft, 1),
                }
                results.append(row)
                for name, value in row.items():
                    rec.metric(f"round_{index + 1}_{name}", value)

            avg_tok_s = sum(row["tok_s"] for row in results) / len(results)
            rec.metric("avg_tok_s", avg_tok_s)
            rec.metric("best_tok_s", max(row["tok_s"] for row in results))
            rec.metric(
                "avg_acceptance_rate",
                sum(row["acceptance_rate"] for row in results) / len(results),
            )
    finally:
        reset_dflash_workload_state(engine)

    return {
        "contract": HISTORICAL_DFLASH_M16_CONTRACT,
        "prompt_hash": prompt_hash,
        "rounds": results,
        "avg_tok_s": sum(row["tok_s"] for row in results) / len(results),
    }


def profile_historical_dflash_m16(
    engine: Any,
    tokenizer: Any,
    *,
    profile_rounds: int = 8,
    warmup_rounds: int = 24,
) -> dict[str, Any]:
    """Profile steady-state DFlash rounds for the fixed 64K quick-brown-fox
    throughput contract and its required load-time geometry.

    This deliberately advances through an explicit post-bootstrap warmup
    window before profiling.  DFlash acceptance on this prompt changes
    sharply after its initially high-acceptance rows; profiling only those
    rows would falsely describe the full 256-token throughput contract.
    The artifact therefore answers the useful M=16 question (where
    verify/draft/MoE time goes) at a recorded generation position rather
    than mixing it with cold prompt processing.  It is diagnosis-only:
    profiler overhead means its timings must never be compared with the
    throughput contract.
    """
    if profile_rounds <= 0:
        raise ValueError("profile_rounds must be positive")
    if warmup_rounds < 0:
        raise ValueError("warmup_rounds must be non-negative")

    backend = engine.backend
    expected = {
        "block_size": HISTORICAL_DFLASH_PREFIX_BLOCK_SIZE,
        "blocks_per_slot": HISTORICAL_DFLASH_PREFIX_BLOCKS_PER_SLOT,
        "max_model_len": HISTORICAL_DFLASH_PREFIX_MAX_MODEL_LEN,
    }
    actual = {
        "block_size": backend.block_size,
        "blocks_per_slot": backend.blocks_per_slot,
        "max_model_len": backend.runtime_config.model_config.max_model_len,
    }
    for name, value in expected.items():
        if actual[name] != value:
            raise ValueError(
                f"{HISTORICAL_DFLASH_M16_CONTRACT} profiler requires {name}={value}, "
                f"got {actual[name]}"
            )

    import torch

    prompt_ids = historical_dflash_m16_prompt_ids(tokenizer)
    prompt_hash = _token_ids_hash(prompt_ids)
    reset_dflash_workload_state(engine)
    try:
        state = engine.dflash_prefill_bootstrap(0, prompt_ids)
        warmup_generated_tokens = 0
        for _ in range(warmup_rounds):
            decision = engine.dflash_round(0, state["anchor"], state["draft_tokens"])
            warmup_generated_tokens += len(decision["committed"])
            state = {
                "anchor": decision["next_anchor"],
                "draft_tokens": decision["next_draft_tokens"],
            }
        torch.cuda.synchronize()

        with run_record(
            script=__file__,
            workload={
                "contract": HISTORICAL_DFLASH_M16_CONTRACT,
                "profile_only": True,
                "prompt_hash": prompt_hash,
                "prompt_len": len(prompt_ids),
                "k": 15,
                "greedy": True,
                "block_size": backend.block_size,
                "blocks_per_slot": backend.blocks_per_slot,
                "max_model_len": backend.runtime_config.model_config.max_model_len,
                "capacity": backend.num_slots,
                "dflash": True,
                "cuda_graph": True,
                "prefix_cache": False,
            },
        ) as rec:
            activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            generated: list[int] = []
            profiled_draft_tokens = 0
            profiled_accepted_tokens = 0
            with torch.profiler.profile(activities=activities) as profiler:
                for _ in range(profile_rounds):
                    decision = engine.dflash_round(0, state["anchor"], state["draft_tokens"])
                    generated.extend(decision["committed"])
                    profiled_draft_tokens += len(state["draft_tokens"])
                    profiled_accepted_tokens += decision["num_accepted"]
                    state = {
                        "anchor": decision["next_anchor"],
                        "draft_tokens": decision["next_draft_tokens"],
                    }
            torch.cuda.synchronize()
            artifacts = default_store().artifacts_dir(rec.run_id)
            artifacts.mkdir(parents=True, exist_ok=True)
            table_path = artifacts / "cuda_profile.txt"
            table_path.write_text(
                profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=120)
            )
            trace_path = artifacts / "cuda_profile.json"
            profiler.export_chrome_trace(str(trace_path))
            rec.artifact("cuda_profile_table", table_path)
            rec.artifact("cuda_profile_trace", trace_path)
            rec.metric("warmup_rounds", warmup_rounds)
            rec.metric("warmup_generated_tokens", warmup_generated_tokens)
            rec.metric("profiled_rounds", profile_rounds)
            rec.metric("profiled_generated_tokens", len(generated))
            rec.metric(
                "profiled_acceptance_rate",
                profiled_accepted_tokens / max(profiled_draft_tokens, 1),
            )
            rec.record.fingerprint.extra["profiled_output_hash"] = _token_ids_hash(generated)
            run_id = rec.run_id
    finally:
        reset_dflash_workload_state(engine)
    return {
        "run_id": run_id,
        "profile_rounds": profile_rounds,
        "warmup_rounds": warmup_rounds,
    }


def profile_laguna_target_shape_matrix(
    engine: Any,
    tokenizer: Any,
    *,
    shapes: tuple[int, ...] = (1, 2, 4, 6, 7, 8, 16),
    replays_per_shape: int = 4,
    nvfp4_split_decode: bool | None = None,
) -> dict[str, Any]:
    """Profile eager target forwards across the serving MoE shape regimes.

    This records the component-level target cost after one fixed 64K DFlash
    bootstrap. It deliberately does not report throughput: M=16 production
    verify runs in its captured graph and M<=6 decode has its own graph path.
    The matrix instead prevents a kernel candidate from being justified by a
    cost model sampled at the wrong MoE implementation boundary.
    """
    if not shapes or any(shape <= 0 for shape in shapes):
        raise ValueError("shapes must contain positive token counts")
    if tuple(sorted(set(shapes))) != shapes:
        raise ValueError("shapes must be unique and sorted")
    if replays_per_shape <= 0:
        raise ValueError("replays_per_shape must be positive")
    if nvfp4_split_decode is not None and not isinstance(nvfp4_split_decode, bool):
        raise TypeError("nvfp4_split_decode must be bool or None")

    backend = engine.backend
    required_tokens = HISTORICAL_DFLASH_M16_CONTEXT_TOKENS + max(shapes)
    required_blocks = (required_tokens + backend.block_size - 1) // backend.block_size
    if backend.block_size != HISTORICAL_DFLASH_PREFIX_BLOCK_SIZE:
        raise ValueError(
            "target shape profiler requires "
            f"block_size={HISTORICAL_DFLASH_PREFIX_BLOCK_SIZE}, got {backend.block_size}"
        )
    if backend.blocks_per_slot < required_blocks:
        raise ValueError(
            "target shape profiler requires at least "
            f"{required_blocks} blocks_per_slot for {required_tokens} tokens, "
            f"got {backend.blocks_per_slot}"
        )
    if backend.runtime_config.model_config.max_model_len < required_tokens:
        raise ValueError(
            "target shape profiler requires max_model_len at least "
            f"{required_tokens}, got {backend.runtime_config.model_config.max_model_len}"
        )

    import os

    import torch

    prompt_ids = historical_dflash_m16_prompt_ids(tokenizer)
    prompt_hash = _token_ids_hash(prompt_ids)
    previous_split_decode = os.environ.get("SPARKINFER_NVFP4_SPLIT_DECODE")
    if nvfp4_split_decode is not None:
        os.environ["SPARKINFER_NVFP4_SPLIT_DECODE"] = "1" if nvfp4_split_decode else "0"
    try:
        reset_dflash_workload_state(engine)
        engine.dflash_prefill_bootstrap(0, prompt_ids)
        kv_len = backend.slot_kv_len[0]
        torch.cuda.synchronize()
        with run_record(
            script=__file__,
            workload={
                "contract": "laguna-target-shape-matrix-64k",
                "profile_only": True,
                "prompt_hash": prompt_hash,
                "prompt_len": len(prompt_ids),
                "shapes": list(shapes),
                "replays_per_shape": replays_per_shape,
                "block_size": backend.block_size,
                "blocks_per_slot": backend.blocks_per_slot,
                "max_model_len": backend.runtime_config.model_config.max_model_len,
                "capacity": backend.num_slots,
                "cuda_graph": False,
                "nvfp4_split_decode": nvfp4_split_decode,
            },
        ) as rec:
            activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            artifacts = default_store().artifacts_dir(rec.run_id)
            artifacts.mkdir(parents=True, exist_ok=True)
            for shape in shapes:
                tokens = list(range(1000, 1000 + shape))
                with torch.profiler.profile(activities=activities) as profiler:
                    for _ in range(replays_per_shape):
                        backend._forward(
                            [0], tokens, [kv_len], qo_len=shape, is_decode=False, skip_logits=True
                        )
                torch.cuda.synchronize()
                table_path = artifacts / f"cuda_profile_m{shape}.txt"
                table_path.write_text(
                    profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=120)
                )
                rec.artifact(f"cuda_profile_m{shape}", table_path)
                rec.metric(f"m{shape}_replays", replays_per_shape)
            run_id = rec.run_id
    finally:
        reset_dflash_workload_state(engine)
        if nvfp4_split_decode is not None:
            if previous_split_decode is None:
                os.environ.pop("SPARKINFER_NVFP4_SPLIT_DECODE", None)
            else:
                os.environ["SPARKINFER_NVFP4_SPLIT_DECODE"] = previous_split_decode
    return {
        "run_id": run_id,
        "shapes": shapes,
        "replays_per_shape": replays_per_shape,
        "nvfp4_split_decode": nvfp4_split_decode,
    }


def summarize_moe_route_ids(route_ids: list[list[list[int]]]) -> dict[str, Any]:
    """Return per-layer expert-count evidence from one target forward."""
    layers: list[dict[str, Any]] = []
    for layer_index, layer_ids in enumerate(route_ids, start=1):
        counts = [0] * 256
        for token_ids in layer_ids:
            for expert_id in token_ids:
                if not 0 <= expert_id < len(counts):
                    raise ValueError(f"layer {layer_index} has invalid expert id {expert_id}")
                counts[expert_id] += 1
        routed_pairs = sum(counts)
        layers.append(
            {
                "moe_layer": layer_index,
                "routed_pairs": routed_pairs,
                "unique_experts": sum(count > 0 for count in counts),
                "max_pairs_per_expert": max(counts, default=0),
                "expert_pair_counts": counts,
            }
        )
    return {
        "moe_layer_count": len(layers),
        "layers": layers,
        "mean_unique_experts": sum(layer["unique_experts"] for layer in layers)
        / max(len(layers), 1),
    }


def capture_laguna_route_histograms(
    engine: Any,
    tokenizer: Any,
    *,
    shapes: tuple[int, ...] = (7, 8, 16),
) -> dict[str, Any]:
    """Capture all 47 router outputs for dynamic-serving shapes only."""
    if not shapes or any(shape < 7 or shape > 16 for shape in shapes):
        raise ValueError("shapes must be dynamic serving token counts in [7, 16]")
    if tuple(sorted(set(shapes))) != shapes:
        raise ValueError("shapes must be unique and sorted")

    backend = engine.backend
    prompt_ids = historical_dflash_m16_prompt_ids(tokenizer)
    prompt_hash = _token_ids_hash(prompt_ids)
    import runtime.backends.laguna as laguna_backend_module

    original_capture = laguna_backend_module.capture_routing
    captures: list[list[list[int]]] = []

    def capture(_logits: Any, topk_ids: Any, _weights: Any) -> None:
        captures.append(topk_ids.detach().cpu().tolist())

    reset_dflash_workload_state(engine)
    try:
        engine.dflash_prefill_bootstrap(0, prompt_ids)
        kv_len = backend.slot_kv_len[0]
        reports: dict[str, Any] = {}
        laguna_backend_module.capture_routing = capture
        for shape in shapes:
            captures.clear()
            backend._forward(
                [0],
                list(range(1000, 1000 + shape)),
                [kv_len],
                qo_len=shape,
                is_decode=False,
                skip_logits=True,
            )
            if len(captures) != 47:
                raise RuntimeError(f"M={shape} captured {len(captures)} MoE layers, expected 47")
            reports[f"m{shape}"] = summarize_moe_route_ids(captures)
        with run_record(
            script=__file__,
            workload={
                "contract": "laguna-route-histogram-64k",
                "prompt_hash": prompt_hash,
                "prompt_len": len(prompt_ids),
                "shapes": list(shapes),
                "block_size": backend.block_size,
                "blocks_per_slot": backend.blocks_per_slot,
                "max_model_len": backend.runtime_config.model_config.max_model_len,
            },
        ) as rec:
            artifacts = default_store().artifacts_dir(rec.run_id)
            artifacts.mkdir(parents=True, exist_ok=True)
            path = artifacts / "moe_route_histograms.json"
            path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
            rec.artifact("moe_route_histograms", path)
            for name, report in reports.items():
                rec.metric(f"{name}_mean_unique_experts", report["mean_unique_experts"])
            run_id = rec.run_id
    finally:
        laguna_backend_module.capture_routing = original_capture
        reset_dflash_workload_state(engine)
    return {"run_id": run_id, "shapes": shapes, "reports": reports}


def capture_laguna_target_logits_oracle(
    engine: Any,
    tokenizer: Any,
    *,
    variant: str,
    token_count: int,
) -> dict[str, Any]:
    """Record complete eager target logits for one named serving shape."""
    if not variant:
        raise ValueError("variant must be non-empty")
    if token_count <= 0 or token_count > 16:
        raise ValueError("token_count must be in [1, 16]")
    import torch

    backend = engine.backend
    prompt_ids = historical_dflash_m16_prompt_ids(tokenizer)
    try:
        reset_dflash_workload_state(engine)
        state = engine.dflash_prefill_bootstrap(0, prompt_ids)
        verify_tokens = [state["anchor"], *state["draft_tokens"]][:token_count]
        if len(verify_tokens) != token_count:
            raise RuntimeError(f"target oracle received M={len(verify_tokens)}")
        kv_len = backend.slot_kv_len[0]
        logits = backend._forward(
            [0], verify_tokens, [kv_len], qo_len=token_count, is_decode=False, skip_logits=False
        )
        if logits is None:
            raise RuntimeError("target forward unexpectedly omitted logits")
        torch.cuda.synchronize()
        cpu_logits = logits.detach().cpu().clone()
    finally:
        restore_dflash_daemon_state(engine)
    report = {
        "variant": variant,
        "verify_tokens": verify_tokens,
        "kv_len": kv_len,
        "logits_shape": list(cpu_logits.shape),
        "logits_sha256": hashlib.sha256(
            cpu_logits.contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
        "top1": cpu_logits.argmax(dim=-1).tolist(),
    }
    with run_record(
        script=__file__,
        workload={
            "contract": "laguna-target-eager-logits-oracle-64k",
            "variant": variant,
            "prompt_hash": _token_ids_hash(prompt_ids),
            "prompt_len": len(prompt_ids),
            "target_tokens": token_count,
            "block_size": backend.block_size,
            "blocks_per_slot": backend.blocks_per_slot,
            "max_model_len": backend.runtime_config.model_config.max_model_len,
            "capacity": backend.num_slots,
            "cuda_graph": False,
        },
    ) as rec:
        artifacts = default_store().artifacts_dir(rec.run_id)
        artifacts.mkdir(parents=True, exist_ok=True)
        path = artifacts / f"m{token_count}_logits_oracle.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True))
        rec.artifact("target_logits_oracle", path)
        run_id = rec.run_id
    return {"run_id": run_id, **report}


def profile_rms_norm_m16(
    *,
    iterations: int = 400,
    num_warps_variants: tuple[int, ...] = (4, 8, 16),
) -> dict[str, Any]:
    """Screen the production RMSNorm kernels at DFlash's M=16/H=3072 shape.

    This is deliberately a diagnostics-only direct launch of the production
    Triton kernels. It proves numerical parity and measures CUDA-event time
    before any runtime launch configuration is changed.
    """
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not num_warps_variants or any(warps <= 0 for warps in num_warps_variants):
        raise ValueError("num_warps_variants must contain positive values")

    import torch

    from runtime.kernels.fused_rms_norm import _fused_add_rms_norm_kernel, _rms_norm_kernel

    m, n, eps = 16, 3072, 1e-6
    block_size = 4096
    generator = torch.Generator(device="cuda").manual_seed(17)
    x = torch.randn((m, n), device="cuda", dtype=torch.bfloat16, generator=generator)
    residual = torch.randn((m, n), device="cuda", dtype=torch.bfloat16, generator=generator)
    weight = torch.randn((n,), device="cuda", dtype=torch.bfloat16, generator=generator)
    rms_reference = (
        x.float()
        * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps)
        * weight.float()
    ).to(torch.bfloat16)
    fused_value = x.float() + residual.float()
    residual_reference = fused_value.to(torch.bfloat16)
    fused_reference = (
        fused_value
        * torch.rsqrt(fused_value.square().mean(dim=-1, keepdim=True) + eps)
        * weight.float()
    ).to(torch.bfloat16)

    def relative_rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
        numerator = torch.linalg.vector_norm((actual.float() - expected.float()).reshape(-1))
        denominator = torch.linalg.vector_norm(expected.float().reshape(-1)).clamp_min(1e-12)
        return float((numerator / denominator).item())

    def launch_rms(out: torch.Tensor, warps: int) -> None:
        _rms_norm_kernel[(m,)](
            x,
            weight,
            out,
            x.stride(0),
            out.stride(0),
            N=n,
            eps=eps,
            BLOCK_SIZE=block_size,
            num_warps=warps,
        )

    def launch_fused(out: torch.Tensor, residual_out: torch.Tensor, warps: int) -> None:
        _fused_add_rms_norm_kernel[(m,)](
            x,
            residual,
            weight,
            out,
            residual_out,
            x.stride(0),
            residual.stride(0),
            out.stride(0),
            residual_out.stride(0),
            N=n,
            eps=eps,
            BLOCK_SIZE=block_size,
            num_warps=warps,
        )

    results: list[dict[str, float | int]] = []
    with run_record(
        script=__file__,
        workload={"kind": "rms_norm_m16", "m": m, "hidden_size": n, "greedy": True},
    ) as rec:
        for warps in num_warps_variants:
            rms_out = torch.empty_like(x)
            fused_out = torch.empty_like(x)
            residual_out = torch.empty_like(residual)
            for _ in range(20):
                launch_rms(rms_out, warps)
                launch_fused(fused_out, residual_out, warps)
            torch.cuda.synchronize()

            rms_start, rms_end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            rms_start.record()
            for _ in range(iterations):
                launch_rms(rms_out, warps)
            rms_end.record()

            fused_start, fused_end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            fused_start.record()
            for _ in range(iterations):
                launch_fused(fused_out, residual_out, warps)
            fused_end.record()
            fused_end.synchronize()

            rms_ms = rms_start.elapsed_time(rms_end) / iterations
            fused_ms = fused_start.elapsed_time(fused_end) / iterations
            result = {
                "num_warps": warps,
                "rms_ms": rms_ms,
                "fused_ms": fused_ms,
                "rms_relative_rmse": relative_rmse(rms_out, rms_reference),
                "fused_relative_rmse": relative_rmse(fused_out, fused_reference),
                "residual_relative_rmse": relative_rmse(residual_out, residual_reference),
            }
            if (
                max(
                    result["rms_relative_rmse"],
                    result["fused_relative_rmse"],
                    result["residual_relative_rmse"],
                )
                > 1e-3
            ):
                raise RuntimeError(f"RMSNorm numerical gate failed for num_warps={warps}: {result}")
            results.append(result)
            for name, value in result.items():
                if name != "num_warps":
                    rec.metric(f"warps_{warps}_{name}", float(value))

        artifacts = default_store().artifacts_dir(rec.run_id)
        artifacts.mkdir(parents=True, exist_ok=True)
        result_path = artifacts / "rms_norm_m16.json"
        result_path.write_text(json.dumps(results, indent=2) + "\n")
        rec.artifact("rms_norm_m16", result_path)
        run_id = rec.run_id
    return {"run_id": run_id, "results": results}


def _first_token_divergence(reference: list[int], candidate: list[int]) -> dict[str, int | None]:
    """Return the first exact token difference, including a length-only one."""
    for index, (expected, actual) in enumerate(zip(reference, candidate)):
        if expected != actual:
            return {
                "index": index,
                "reference_token": expected,
                "candidate_token": actual,
            }
    if len(reference) != len(candidate):
        index = min(len(reference), len(candidate))
        return {
            "index": index,
            "reference_token": reference[index] if index < len(reference) else None,
            "candidate_token": candidate[index] if index < len(candidate) else None,
        }
    return {"index": None, "reference_token": None, "candidate_token": None}


def _first_verify_round_divergence(
    reference: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    """Return the first decision-relevant proposal or verifier-top1 difference."""
    for index, (expected, actual) in enumerate(zip(reference, candidate)):
        for field in ("kv_len", "bonus_token"):
            if expected[field] != actual[field]:
                return {
                    "round": index,
                    "kind": field,
                    "reference": expected[field],
                    "candidate": actual[field],
                }
        expected_committed, _ = _verify_committed_tokens(expected)
        actual_committed, _ = _verify_committed_tokens(actual)
        if expected_committed != actual_committed:
            first_position = next(
                (
                    i
                    for i, pair in enumerate(zip(expected_committed, actual_committed))
                    if pair[0] != pair[1]
                ),
                min(len(expected_committed), len(actual_committed)),
            )
            if len(expected_committed) != len(actual_committed) and first_position == min(
                len(expected_committed), len(actual_committed)
            ):
                return {
                    "round": index,
                    "kind": "committed_length",
                    "reference": expected_committed,
                    "candidate": actual_committed,
                }
            prefix_drafts_match = (
                expected["draft_tokens"][:first_position] == actual["draft_tokens"][:first_position]
            )
            expected_top1 = expected["positions"][first_position]["top1_tok"]
            actual_top1 = actual["positions"][first_position]["top1_tok"]
            if prefix_drafts_match and expected_top1 != actual_top1:
                return {
                    "round": index,
                    "kind": "verifier_top1",
                    "position": first_position,
                    "reference": expected_top1,
                    "candidate": actual_top1,
                }
            return {
                "round": index,
                "kind": "draft_tokens",
                "position": first_position,
                "reference": expected_committed[first_position],
                "candidate": actual_committed[first_position],
            }
    if len(reference) != len(candidate):
        return {
            "round": min(len(reference), len(candidate)),
            "kind": "round_count",
            "reference": len(reference),
            "candidate": len(candidate),
        }
    return {"round": None, "kind": None}


def _verify_committed_tokens(round_dump: dict[str, Any]) -> tuple[list[int], int]:
    """Reconstruct the output tokens that one verify-only round can commit."""
    draft_tokens = round_dump["draft_tokens"]
    top1 = [position["top1_tok"] for position in round_dump["positions"]]
    for index, draft_token in enumerate(draft_tokens):
        if top1[index] != draft_token:
            return draft_tokens[:index] + [top1[index]], index
    return draft_tokens + [top1[len(draft_tokens)]], len(draft_tokens)


def diagnose_historical_dflash_partial_prefix_reuse(
    engine: Any,
    tokenizer: Any,
    *,
    max_tokens: int = HISTORICAL_DFLASH_PREFIX_MAX_TOKENS,
    base_generation_tokens: int = 5,
    capture_verify_top2: bool = False,
    capture_target_prefill_state: bool = False,
) -> dict[str, Any]:
    """Compare cold DFlash output with the canonical partial-prefix state path.

    The historical 64K workload first generates from the 55,536-token base,
    then submits the base plus a 10,000-token suffix.  This diagnostic runs
    that exact transition next to a reset cold prefill and writes both token
    streams, state geometry, and trace-round boundaries into a bfdiag record.
    ``capture_verify_top2`` additionally records the existing diagnostic
    verifier dump, separating a changed draft proposal from a changed target
    top-1 without altering the normal performance workload.
    ``capture_target_prefill_state`` hashes target KV and aux state just after
    prefill and before DFlash bootstrap; it is diagnostic-only host I/O.
    It deliberately reports divergence instead of raising: a failed oracle is
    useful evidence and must leave a complete record for ``bf trace show``.
    """
    if max_tokens < 2:
        raise ValueError("max_tokens must be at least two")
    if base_generation_tokens < 2:
        raise ValueError("base_generation_tokens must be at least two")

    backend = engine.backend
    expected = {
        "block_size": HISTORICAL_DFLASH_PREFIX_BLOCK_SIZE,
        "blocks_per_slot": HISTORICAL_DFLASH_PREFIX_BLOCKS_PER_SLOT,
        "max_model_len": HISTORICAL_DFLASH_PREFIX_MAX_MODEL_LEN,
    }
    actual = {
        "block_size": backend.block_size,
        "blocks_per_slot": backend.blocks_per_slot,
        "max_model_len": backend.runtime_config.model_config.max_model_len,
    }
    for name, value in expected.items():
        if actual[name] != value:
            raise ValueError(
                f"{HISTORICAL_DFLASH_PREFIX_CONTRACT} partial-prefix diagnostic requires "
                f"{name}={value}, got {actual[name]}"
            )

    from runtime.backends.dflash_constants import DRAFT_WINDOW
    from runtime.backends.laguna_dflash import _ring_prefix_reuse_is_safe

    base_ids, full_ids = historical_dflash_prefix_prompt_ids(tokenizer)
    prompt_hash = _token_ids_hash(full_ids)
    base_hash = _token_ids_hash(base_ids)
    try:
        with run_record(
            script=__file__,
            workload={
                "contract": HISTORICAL_DFLASH_PREFIX_CONTRACT,
                "prompt_hash": prompt_hash,
                "base_prompt_hash": base_hash,
                "prompt_len": len(full_ids),
                "base_len": len(base_ids),
                "suffix_len": HISTORICAL_DFLASH_PREFIX_SUFFIX_TOKENS,
                "max_tokens": max_tokens,
                "base_generation_tokens": base_generation_tokens,
                "k": 15,
                "greedy": True,
                "block_size": backend.block_size,
                "blocks_per_slot": backend.blocks_per_slot,
                "max_model_len": backend.runtime_config.model_config.max_model_len,
                "capacity": backend.num_slots,
                "dflash": True,
                "cuda_graph": True,
                "prefix_cache": True,
                "partial_prefix_oracle": True,
                "capture_verify_top2": capture_verify_top2,
                "capture_target_prefill_state": capture_target_prefill_state,
            },
        ) as rec:
            artifacts = default_store().artifacts_dir(rec.run_id)
            artifacts.mkdir(parents=True, exist_ok=True)
            verify_dump_path = artifacts / "partial_prefix_verify_top2.jsonl"
            prior_verify_dump = os.environ.get("QSR_DEBUG_VERIFY_LOGITS_FILE")
            if capture_verify_top2 and prior_verify_dump is not None:
                raise RuntimeError(
                    "capture_verify_top2 requires QSR_DEBUG_VERIFY_LOGITS_FILE to be unset"
                )
            if capture_verify_top2:
                os.environ["QSR_DEBUG_VERIFY_LOGITS_FILE"] = str(verify_dump_path)
            prefill_states: dict[str, dict[str, Any]] = {}

            def observe_prefill(label: str):
                def observer(
                    observed_slot: int,
                    observed_prefix_len: int,
                    observed_first_token: int,
                    observed_aux: list[Any] | None,
                ) -> None:
                    prefill_states[label] = capture_laguna_target_prefill_state(
                        backend,
                        observed_slot,
                        observed_prefix_len,
                        observed_first_token,
                        observed_aux,
                    )

                return observer

            try:
                reset_dflash_workload_state(engine)
                cold_tokens, cold_stats = engine.generate_verify_only(
                    full_ids,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    slot=0,
                    enable_prefix_cache=False,
                    prefill_observer=(
                        observe_prefill("cold") if capture_target_prefill_state else None
                    ),
                )

                reset_dflash_workload_state(engine)
                base_tokens, base_stats = engine.generate_verify_only(
                    base_ids,
                    max_tokens=base_generation_tokens,
                    temperature=0.0,
                    slot=0,
                    enable_prefix_cache=False,
                )
                cached_kv_len = backend.slot_kv_len[0]
                prefix_len = backend.find_prefix_match(0, full_ids)
                rewind_tokens = cached_kv_len - prefix_len
                ring_specs = (
                    (backend._ring_slots_per_slot, backend._swa_window),
                    (engine._draft_blocks_per_slot * engine.block_size, DRAFT_WINDOW),
                )
                reuse_is_safe = _ring_prefix_reuse_is_safe(cached_kv_len, prefix_len, ring_specs)
                if prefix_len <= 0 or prefix_len >= len(full_ids) or not reuse_is_safe:
                    raise AssertionError(
                        "canonical partial-prefix setup did not retain a safe partial match: "
                        f"prefix_len={prefix_len}, cached_kv_len={cached_kv_len}, "
                        f"reuse_is_safe={reuse_is_safe}"
                    )
                partial_tokens, partial_stats = engine.generate_verify_only(
                    full_ids,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    slot=0,
                    enable_prefix_cache=True,
                    prefill_observer=(
                        observe_prefill("partial_reuse") if capture_target_prefill_state else None
                    ),
                )
            finally:
                if capture_verify_top2:
                    os.environ.pop("QSR_DEBUG_VERIFY_LOGITS_FILE", None)

            divergence = _first_token_divergence(cold_tokens, partial_tokens)
            verify_divergence = None
            if capture_verify_top2:
                verify_rounds = [
                    json.loads(line)
                    for line in verify_dump_path.read_text().splitlines()
                    if line.strip()
                ]
                cold_count = int(cold_stats["num_steps"])
                base_count = int(base_stats["num_steps"])
                partial_count = int(partial_stats["num_steps"])
                if len(verify_rounds) != cold_count + base_count + partial_count:
                    raise AssertionError(
                        "verify dump round count disagrees with generation stats: "
                        f"dump={len(verify_rounds)}, cold={cold_count}, base={base_count}, "
                        f"partial={partial_count}"
                    )
                verify_divergence = _first_verify_round_divergence(
                    verify_rounds[:cold_count],
                    verify_rounds[cold_count + base_count :],
                )
            target_prefill_comparison = None
            if capture_target_prefill_state:
                if set(prefill_states) != {"cold", "partial_reuse"}:
                    raise AssertionError(
                        f"missing target prefill captures: {sorted(prefill_states)}"
                    )
                target_prefill_comparison = compare_laguna_target_prefill_states(
                    prefill_states["cold"], prefill_states["partial_reuse"]
                )
            report = {
                "contract": HISTORICAL_DFLASH_PREFIX_CONTRACT,
                "cold": {"tokens": cold_tokens, "stats": cold_stats},
                "base_seed": {"tokens": base_tokens, "stats": base_stats},
                "partial_reuse": {"tokens": partial_tokens, "stats": partial_stats},
                "prefix": {
                    "cached_kv_len": cached_kv_len,
                    "prefix_len": prefix_len,
                    "rewind_tokens": rewind_tokens,
                    "ring_specs": [list(spec) for spec in ring_specs],
                    "reuse_is_safe": reuse_is_safe,
                },
                "divergence": divergence,
                "verify_round_divergence": verify_divergence,
                "target_prefill_comparison": target_prefill_comparison,
                "parity": divergence["index"] is None,
                # ``generate_verify_only`` emits one trace row per DFlash round.
                "trace_round_ranges": {
                    "cold": [0, int(cold_stats["num_steps"]) - 1],
                    "base_seed": [
                        int(cold_stats["num_steps"]),
                        int(cold_stats["num_steps"] + base_stats["num_steps"]) - 1,
                    ],
                    "partial_reuse": [
                        int(cold_stats["num_steps"] + base_stats["num_steps"]),
                        int(
                            cold_stats["num_steps"]
                            + base_stats["num_steps"]
                            + partial_stats["num_steps"]
                        )
                        - 1,
                    ],
                },
            }
            report_path = artifacts / "partial_prefix_reuse_oracle.json"
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            rec.artifact("partial_prefix_reuse_oracle", report_path)
            if capture_verify_top2:
                rec.artifact("partial_prefix_verify_top2", verify_dump_path)
            if capture_target_prefill_state:
                target_state_path = artifacts / "partial_prefix_target_prefill_state.json"
                target_state_path.write_text(
                    json.dumps(prefill_states, indent=2, sort_keys=True) + "\n"
                )
                rec.artifact("partial_prefix_target_prefill_state", target_state_path)
            rec.metric("partial_prefix_len", prefix_len)
            rec.metric("partial_rewind_tokens", rewind_tokens)
            rec.metric("partial_reuse_safe", float(reuse_is_safe))
            rec.metric("partial_parity", float(report["parity"]))
            rec.metric(
                "first_token_divergence",
                float(divergence["index"] if divergence["index"] is not None else -1),
            )
            rec.metric("cold_acceptance_rate", cold_stats["acceptance_rate"])
            rec.metric("partial_acceptance_rate", partial_stats["acceptance_rate"])
            run_id = rec.run_id
    finally:
        reset_dflash_workload_state(engine)

    return {"run_id": run_id, **report}


def run_historical_dflash_prefix_cache_m16(
    engine: Any,
    tokenizer: Any,
    *,
    rounds: int = 3,
) -> dict[str, Any]:
    """Reproduce fd33368's 64K cold/warm DFlash benchmark without vLLM.

    This is a direct port of ``benchmarks/full_comparison_ours.py``'s
    workload contract, not a new synthetic benchmark.  In particular, the
    warm phase intentionally preserves both target and draft KV state across
    its rounds, exactly as the archived script did.
    """
    if rounds <= 0:
        raise ValueError("rounds must be positive")

    import torch

    backend = engine.backend
    expected = {
        "block_size": HISTORICAL_DFLASH_PREFIX_BLOCK_SIZE,
        "blocks_per_slot": HISTORICAL_DFLASH_PREFIX_BLOCKS_PER_SLOT,
        "max_model_len": HISTORICAL_DFLASH_PREFIX_MAX_MODEL_LEN,
    }
    actual = {
        "block_size": backend.block_size,
        "blocks_per_slot": backend.blocks_per_slot,
        "max_model_len": backend.runtime_config.model_config.max_model_len,
    }
    for name, value in expected.items():
        if actual[name] != value:
            raise ValueError(
                f"{HISTORICAL_DFLASH_PREFIX_CONTRACT} requires {name}={value}, got {actual[name]}"
            )

    base_ids, full_ids = historical_dflash_prefix_prompt_ids(tokenizer)
    prompt_hash = _token_ids_hash(full_ids)
    base_hash = _token_ids_hash(base_ids)
    results: dict[str, list[dict[str, Any]]] = {"cold": [], "warm": []}

    # fd33368 first ran a five-token short prompt to force one-time JIT/CG
    # work outside the timed workload.  Preserve that ordering verbatim.
    reset_dflash_workload_state(engine)
    engine.generate_verify_only(
        base_ids[:256],
        max_tokens=5,
        temperature=0.0,
        slot=0,
        enable_prefix_cache=False,
    )
    torch.cuda.synchronize()

    try:
        with run_record(
            script=__file__,
            workload={
                "contract": HISTORICAL_DFLASH_PREFIX_CONTRACT,
                "prompt_hash": prompt_hash,
                "base_prompt_hash": base_hash,
                "prompt_len": len(full_ids),
                "base_len": len(base_ids),
                "suffix_len": HISTORICAL_DFLASH_PREFIX_SUFFIX_TOKENS,
                "max_tokens": HISTORICAL_DFLASH_PREFIX_MAX_TOKENS,
                "k": 15,
                "greedy": True,
                "block_size": backend.block_size,
                "blocks_per_slot": backend.blocks_per_slot,
                "max_model_len": backend.runtime_config.model_config.max_model_len,
                "capacity": backend.num_slots,
                "dflash": True,
                "cuda_graph": True,
                "prefix_cache": True,
            },
        ) as rec:
            for scenario in ("cold", "warm"):
                if scenario == "cold":
                    # Archived script reset before each cold round.
                    def setup() -> None:
                        reset_dflash_workload_state(engine)
                else:
                    # Archived script populated the prefix once then retained
                    # target and draft KV for all warm rounds.
                    reset_dflash_workload_state(engine)
                    engine.generate_verify_only(
                        base_ids,
                        max_tokens=5,
                        temperature=0.0,
                        slot=0,
                        enable_prefix_cache=True,
                    )
                    torch.cuda.synchronize()

                    def setup() -> None:
                        pass

                for index in range(rounds):
                    setup()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    tokens, stats = engine.generate_verify_only(
                        full_ids,
                        max_tokens=HISTORICAL_DFLASH_PREFIX_MAX_TOKENS,
                        temperature=0.0,
                        slot=0,
                        enable_prefix_cache=(scenario == "warm"),
                    )
                    torch.cuda.synchronize()
                    row = {
                        "round": index,
                        "wall_s": time.perf_counter() - start,
                        **stats,
                        "output_hash": _token_ids_hash(tokens),
                    }
                    results[scenario].append(row)
                    for name, value in row.items():
                        if isinstance(value, (int, float)):
                            rec.metric(f"{scenario}_{index}_{name}", value)
                    rec.record.fingerprint.extra[f"{scenario}_{index}_output_hash"] = row[
                        "output_hash"
                    ]
    finally:
        reset_dflash_workload_state(engine)

    return {
        "contract": HISTORICAL_DFLASH_PREFIX_CONTRACT,
        "prompt_hash": prompt_hash,
        "base_prompt_hash": base_hash,
        "cold": results["cold"],
        "warm": results["warm"],
    }
