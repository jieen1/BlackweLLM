"""FN6: live Flash-Next K=3 chain + one-pass target batch verification.

This is the first integration gate for the engine-level spec-row design.  It
loads the real RadixArk checkpoint, captures single-token decode and K+1
verify graphs, bootstraps MTP with shifted teacher-forced prompt rows, then
generates greedily without replaying the target after partial acceptance.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys
import time

_ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
sys.path.insert(0, _ROOT)
import runtime  # noqa: E402

assert runtime.__file__.startswith(_ROOT), runtime.__file__

import torch  # noqa: E402

from bfdiag.record import auto_record  # noqa: E402
from runtime.model.flashnext.model import (  # noqa: E402
    FlashNextGraphEngine,
    FlashNextTextConfig,
    load_flashnext_model,
    new_session,
    prepare_graph_buffers,
)
from runtime.model.flashnext.mtp import load_flashnext_mtp  # noqa: E402
from runtime.model.flashnext.spec import FlashNextSpecEngine  # noqa: E402

CKPT = pathlib.Path("/home/bot/models/Qwen3.8-Flash-Next-NVFP4-RadixArk")
MAX_SEQ = int(os.getenv("FN_MAX_SEQ", "4096"))
PROMPT_TOKENS = int(os.getenv("FN_PROMPT_TOKENS", "0"))
PROMPT_KIND = os.getenv("FN_PROMPT_KIND", "tcp")
K = int(os.getenv("FN_SPEC_K", "1"))
ROUNDS = int(os.getenv("FN_SPEC_ROUNDS", "64"))
PROFILE_DIR = os.getenv("FN_TORCH_PROFILE_DIR")
PROFILE_PREFILL_TOKENS = int(os.getenv("FN_PROFILE_PREFILL_TOKENS", "4"))
PROFILE_DECODE_ROUNDS = int(os.getenv("FN_PROFILE_DECODE_ROUNDS", "8"))
PROFILE_CHROME_TRACE = os.getenv("FN_PROFILE_CHROME_TRACE", "0") == "1"
BATCH_PREFILL = os.getenv("FN_BATCH_PREFILL", "1") == "1"
PREFILL_CHUNK = int(os.getenv("FN_PREFILL_CHUNK", "0"))
BATCH_GDN_VERIFY = os.getenv("FN_BATCH_GDN_VERIFY", "1") == "1"
BATCH_GDN_PROJECTIONS = os.getenv("FN_BATCH_GDN_PROJECTIONS", "1") == "1"
EXACT_ROW_MATH = os.getenv("FN_EXACT_ROW_MATH", "0") == "1"
MTP_CONTINUATION_GRAPH = os.getenv("FN_MTP_CONTINUATION_GRAPH", "0") == "1"
MTP_SPARSE_GRAPH = os.getenv("FN_MTP_SPARSE_GRAPH", "0") == "1"
USE_VERIFY_GRAPH = os.getenv("FN_USE_VERIFY_GRAPH", "1") == "1"
PREFILL_LAYER_MAJOR_MODE = os.getenv("FN_PREFILL_LAYER_MAJOR", "0").strip().lower()
PREFILL_SWEEP = tuple(
    int(value) for value in os.getenv("FN_PREFILL_SWEEP", "").split(",") if value
)
COMPARE_PREFILL_CHUNKS = tuple(
    int(value)
    for value in os.getenv("FN_COMPARE_PREFILL_CHUNKS", "").split(",")
    if value
)
VALIDATE_BATCH_PREFILL = os.getenv("FN_VALIDATE_BATCH_PREFILL", "0") == "1"
PREFILL_ONLY = os.getenv("FN_PREFILL_ONLY", "0") == "1"
PREFILL_MLP_CG = os.getenv("FN_PREFILL_MLP_CG", "0") == "1"
COMPARE_MOE_RANKS = os.getenv("FN_COMPARE_MOE_RANKS", "0") == "1"
COMPARE_LAYER_MAJOR = os.getenv("FN_COMPARE_LAYER_MAJOR", "0") == "1"
COMPARE_FLASHINFER_MOE = os.getenv("FN_COMPARE_FLASHINFER_MOE", "0") == "1"
FLASHINFER_MOE_LAYER = int(os.getenv("FN_COMPARE_FLASHINFER_MOE_LAYER", "0"))
FLASHINFER_MOE_ROWS = int(os.getenv("FN_COMPARE_FLASHINFER_MOE_ROWS", "512"))
FLASHINFER_MOE_WARMUP = int(os.getenv("FN_COMPARE_FLASHINFER_MOE_WARMUP", "5"))
FLASHINFER_MOE_ITERS = int(os.getenv("FN_COMPARE_FLASHINFER_MOE_ITERS", "20"))
FLASHINFER_MOE_FUSED_FINALIZE = (
    os.getenv("FN_COMPARE_FLASHINFER_MOE_FUSED_FINALIZE", "1") == "1"
)
PRINT_COMPLETION = os.getenv("FN_PRINT_COMPLETION", "1") == "1"
PLE_RESIDENT = os.getenv("FN_PLE_RESIDENT", "0") == "1"
PLE_CACHE_ROWS = int(os.getenv("FN_PLE_CACHE_ROWS", "131072"))
PLE_CACHE_PAGES = int(os.getenv("FN_PLE_CACHE_PAGES", "0"))
PLE_IO_WORKERS = int(os.getenv("FN_PLE_IO_WORKERS", "32"))
TRIM_INITIAL_MTP_WORKSPACE = os.getenv("FN_TRIM_INITIAL_MTP_WORKSPACE", "1") == "1"
_BF = None


def _profile() -> torch.profiler.profile:
    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    )


def _finish_profile(profiler: torch.profiler.profile, phase: str) -> None:
    assert PROFILE_DIR is not None
    output_dir = pathlib.Path(PROFILE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = profiler.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=80,
        max_name_column_width=180,
    )
    (output_dir / f"{phase}.txt").write_text(table)
    if PROFILE_CHROME_TRACE:
        profiler.export_chrome_trace(str(output_dir / f"{phase}.json"))
    print(f"profile[{phase}]\n{table}", flush=True)


def _memory(label: str) -> None:
    free, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    print(
        f"{label}: used={(total - free) / 2**30:.2f} GiB, "
        f"free={free / 2**30:.2f} GiB, "
        f"allocated={allocated / 2**30:.2f} GiB, "
        f"reserved={reserved / 2**30:.2f} GiB, "
        f"peak_allocated={peak_allocated / 2**30:.2f} GiB, "
        f"peak_reserved={peak_reserved / 2**30:.2f} GiB",
        flush=True,
    )
    if _BF is not None:
        prefix = f"memory.{label.replace(' ', '_')}"
        _BF.metric(f"{prefix}.driver_used_gib", (total - free) / 2**30)
        _BF.metric(f"{prefix}.driver_free_gib", free / 2**30)
        _BF.metric(f"{prefix}.allocated_gib", allocated / 2**30)
        _BF.metric(f"{prefix}.reserved_gib", reserved / 2**30)
        _BF.metric(f"{prefix}.peak_allocated_gib", peak_allocated / 2**30)
        _BF.metric(f"{prefix}.peak_reserved_gib", peak_reserved / 2**30)


def _metrics(**values: float) -> None:
    if _BF is None:
        return
    for name, value in values.items():
        _BF.metric(name, value)


def _write_round_trace(rows: list[dict[str, object]]) -> pathlib.Path | None:
    """Persist detailed FN6 decisions after the timed decode region."""
    if _BF is None or not rows:
        return None
    path = (
        pathlib.Path(os.environ["QSR_BFDIAG_DIR"])
        / "runs"
        / _BF.run_id
        / "artifacts"
        / "fn6_rounds.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    _BF.artifact("fn6_rounds", path)
    _BF.save()
    return path


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _prefill_layer_major(chunk_size: int) -> bool:
    if PREFILL_LAYER_MAJOR_MODE == "auto":
        # Keep auto conservative until the real 23K prompt state/logit gate
        # qualifies large-M MLP numerics against the 512 chunk-major path.
        return False
    if PREFILL_LAYER_MAJOR_MODE in {"1", "true", "yes", "on"}:
        return True
    if PREFILL_LAYER_MAJOR_MODE in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        "FN_PREFILL_LAYER_MAJOR must be one of "
        "{auto,0,1,false,true,no,yes,off,on}, "
        f"got {PREFILL_LAYER_MAJOR_MODE!r}"
    )


def _tensor_similarity(got: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    got_f = got.float().flatten()
    expected_f = expected.float().flatten()
    cosine = torch.nn.functional.cosine_similarity(got_f, expected_f, dim=0)
    return float(cosine), float((got_f - expected_f).abs().max())


def _capture_prefill_state(sess) -> dict[str, torch.Tensor]:
    return {
        **{
            f"{name}.conv": state.conv_state.clone()
            for name, state in sess.gdn.items()
        },
        **{
            f"{name}.recurrent": state.recurrent_state.clone()
            for name, state in sess.gdn.items()
        },
        "ple.conv": sess.ple_conv_state.clone(),
        **{
            f"qsa.{layer}.idx_k": pool[: sess.pos].clone()
            for layer, pool in sess.qsa_idx_k_pool.items()
        },
        **{
            f"qsa.{layer}.pooled_k": pool[: (sess.pos // 4)].clone()
            for layer, pool in sess.qsa_pooled_k_pool.items()
        },
        **{
            f"qsa.{layer}.k": pool[: sess.pos].clone()
            for layer, pool in sess.qsa_k_pool.items()
        },
        **{
            f"qsa.{layer}.v": pool[: sess.pos].clone()
            for layer, pool in sess.qsa_v_pool.items()
        },
    }


def _bench_cuda(
    fn,
    *,
    warmup: int,
    iters: int,
) -> tuple[torch.Tensor, float, float]:
    for _ in range(max(warmup, 0)):
        fn()
    torch.cuda.synchronize()
    baseline_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = None
    for _ in range(max(iters, 1)):
        out = fn()
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    peak_bytes = max(0, torch.cuda.max_memory_allocated() - baseline_bytes)
    assert out is not None
    return out, elapsed_ms / max(iters, 1), peak_bytes / 2**20


def _capture_flashnext_moe_batch(
    target: FlashNextGraphEngine,
    prompt_ids: list[int],
    *,
    layer_idx: int,
    rows: int,
) -> dict[str, torch.Tensor | int]:
    probe_layer = target.model.layers[layer_idx]
    if probe_layer.mlp is None:
        raise ValueError(f"layer {layer_idx} has no MLP")
    original_forward = probe_layer.mlp.expert_layer.forward
    captured: dict[str, torch.Tensor | int] = {}

    def wrapped_forward(hidden: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor):
        out = original_forward(hidden, topk_ids, topk_weights)
        if "hidden" not in captured:
            take = min(rows, hidden.shape[0])
            captured["rows"] = take
            captured["hidden"] = hidden[:take].clone()
            captured["topk_ids"] = topk_ids[:take].clone()
            captured["topk_weights"] = topk_weights[:take].clone()
            captured["b12x_output"] = out[:take].clone()
        return out

    probe_layer.mlp.expert_layer.forward = wrapped_forward
    try:
        probe_tokens = min(len(prompt_ids), max(rows, 1))
        target._zero_state()
        target.prefill(
            prompt_ids[:probe_tokens],
            chunk_size=probe_tokens,
            layer_major=_prefill_layer_major(probe_tokens),
        )
        if "hidden" not in captured:
            raise RuntimeError(
                f"failed to capture MoE batch for layer {layer_idx} with {probe_tokens} tokens"
            )
        return captured
    finally:
        probe_layer.mlp.expert_layer.forward = original_forward
        target._zero_state()


def _compare_flashinfer_moe(
    target: FlashNextGraphEngine,
    cfg: FlashNextTextConfig,
    prompt_ids: list[int],
) -> None:
    if FLASHINFER_MOE_ROWS <= 0:
        raise ValueError(
            f"FN_COMPARE_FLASHINFER_MOE_ROWS must be positive, got {FLASHINFER_MOE_ROWS}"
        )
    if FLASHINFER_MOE_LAYER < 0 or FLASHINFER_MOE_LAYER >= len(target.model.layers):
        raise ValueError(
            f"FN_COMPARE_FLASHINFER_MOE_LAYER must be in [0, {len(target.model.layers) - 1}], "
            f"got {FLASHINFER_MOE_LAYER}"
        )

    os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    from flashinfer.fused_moe import cutlass_fused_moe
    from flashinfer.fused_moe.core import ActivationType

    from runtime.backends.flashnext_moe import load_flashnext_experts
    from runtime.backends.laguna_sparkinfer_moe import SparkinferMoELayer
    from runtime.model.qwen38_moe import QwenMoeGeometry

    probe = _capture_flashnext_moe_batch(
        target,
        prompt_ids,
        layer_idx=FLASHINFER_MOE_LAYER,
        rows=FLASHINFER_MOE_ROWS,
    )
    hidden = probe["hidden"].contiguous()
    topk_ids = probe["topk_ids"].contiguous()
    topk_weights = probe["topk_weights"].contiguous()
    rows = int(probe["rows"])
    b12x_reference = probe["b12x_output"].contiguous()

    geometry = QwenMoeGeometry(
        num_experts=cfg.num_experts,
        top_k=cfg.num_experts_per_tok,
        hidden_size=cfg.hidden_size,
        moe_intermediate_size=cfg.moe_intermediate_size,
        shared_expert_intermediate_size=cfg.shared_expert_intermediate_size,
    )
    raw = load_flashnext_experts(CKPT, FLASHINFER_MOE_LAYER, geometry, hidden.device)
    from b12x._lib.intrinsics import swizzle_block_scale

    gate_scale2 = (1.0 / raw["gate_gs"]).reshape(cfg.num_experts).float().contiguous()
    down_scale2 = (1.0 / raw["down_gs"]).reshape(cfg.num_experts).float().contiguous()
    a1 = float(raw["a1_input_scale"])
    a2 = float(raw["a2_input_scale"])
    if a1 <= 0.0 or a2 <= 0.0:
        raise ValueError(
            f"invalid activation scales for layer {FLASHINFER_MOE_LAYER}: a1={a1} a2={a2}"
        )

    w13_weight = torch.cat([raw["up_w"], raw["gate_w"]], dim=1).contiguous().view(torch.long)
    w2_weight = raw["down_w"].contiguous().view(torch.long)
    w13_blockscale = torch.cat(
        [
            swizzle_block_scale(raw["up_sf"].clone().contiguous()),
            swizzle_block_scale(raw["gate_sf"].clone().contiguous()),
        ],
        dim=1,
    ).contiguous()
    w2_blockscale = swizzle_block_scale(raw["down_sf"].clone().contiguous()).contiguous()
    w13_input_scale_quant = torch.tensor(1.0 / a1, dtype=torch.float32, device=hidden.device)
    w2_input_scale_quant = torch.tensor(1.0 / a2, dtype=torch.float32, device=hidden.device)
    g1_alphas = (a1 * gate_scale2).to(torch.float32)
    g2_alphas = (a2 * down_scale2).to(torch.float32)

    layer = target.model.layers[FLASHINFER_MOE_LAYER]
    if layer.mlp is None or not isinstance(layer.mlp.expert_layer, SparkinferMoELayer):
        raise TypeError(f"layer {FLASHINFER_MOE_LAYER} does not use SparkinferMoELayer")
    b12x_layer = layer.mlp.expert_layer

    def run_b12x() -> torch.Tensor:
        return b12x_layer.forward(hidden, topk_ids, topk_weights)

    def run_flashinfer() -> torch.Tensor:
        output = torch.empty(rows, cfg.hidden_size, dtype=torch.bfloat16, device=hidden.device)
        return cutlass_fused_moe(
            output=output,
            input=hidden,
            token_selected_experts=topk_ids.to(torch.int),
            token_final_scales=topk_weights,
            fc1_expert_weights=w13_weight,
            fc2_expert_weights=w2_weight,
            output_dtype=torch.bfloat16,
            input_sf=None,
            quant_scales=[
                w13_input_scale_quant,
                w13_blockscale.view(torch.int32),
                g1_alphas,
                w2_input_scale_quant,
                w2_blockscale.view(torch.int32),
                g2_alphas,
            ],
            ep_size=1,
            ep_rank=0,
            tp_size=1,
            tp_rank=0,
            tune_max_num_tokens=_next_power_of_two(rows),
            activation_type=ActivationType.Swiglu,
            enable_alltoall=False,
            use_fused_finalize=FLASHINFER_MOE_FUSED_FINALIZE,
        )[0]

    b12x_out = run_b12x()
    flashinfer_out = run_flashinfer()
    ref_cosine, ref_max_abs = _tensor_similarity(b12x_out, b12x_reference)
    fi_cosine, fi_max_abs = _tensor_similarity(flashinfer_out, b12x_reference)
    cross_cosine, cross_max_abs = _tensor_similarity(flashinfer_out, b12x_out)

    _, b12x_ms, b12x_peak_mb = _bench_cuda(
        run_b12x,
        warmup=FLASHINFER_MOE_WARMUP,
        iters=FLASHINFER_MOE_ITERS,
    )
    _, flashinfer_ms, flashinfer_peak_mb = _bench_cuda(
        run_flashinfer,
        warmup=FLASHINFER_MOE_WARMUP,
        iters=FLASHINFER_MOE_ITERS,
    )
    print(
        "flashinfer_moe_comparison="
        f"layer={FLASHINFER_MOE_LAYER} "
        f"rows={rows} "
        f"hidden={cfg.hidden_size} "
        f"topk={cfg.num_experts_per_tok} "
        f"b12x_ms={b12x_ms:.3f} "
        f"b12x_rows_per_s={rows / (b12x_ms / 1000.0):.2f} "
        f"b12x_peak_mb={b12x_peak_mb:.1f} "
        f"flashinfer_ms={flashinfer_ms:.3f} "
        f"flashinfer_rows_per_s={rows / (flashinfer_ms / 1000.0):.2f} "
        f"flashinfer_peak_mb={flashinfer_peak_mb:.1f} "
        f"speedup={b12x_ms / flashinfer_ms:.3f} "
        f"b12x_vs_capture_cosine={ref_cosine:.8f} "
        f"b12x_vs_capture_max_abs={ref_max_abs:.6f} "
        f"flashinfer_vs_capture_cosine={fi_cosine:.8f} "
        f"flashinfer_vs_capture_max_abs={fi_max_abs:.6f} "
        f"flashinfer_vs_b12x_cosine={cross_cosine:.8f} "
        f"flashinfer_vs_b12x_max_abs={cross_max_abs:.6f} "
        f"fused_finalize={int(FLASHINFER_MOE_FUSED_FINALIZE)}",
        flush=True,
    )


def main() -> None:
    global _BF

    from transformers import AutoTokenizer

    graph_status = (
        f"verify={int(USE_VERIFY_GRAPH)},"
        f"continuation={int(MTP_CONTINUATION_GRAPH)},"
        f"sparse={int(MTP_SPARSE_GRAPH)}"
    )
    _BF = auto_record(
        script=__file__,
        model={
            "path": str(CKPT),
            "dtype": "bfloat16",
            "max_model_len": MAX_SEQ,
            "quantization": "modelopt_nvfp4",
        },
        workload={
            "contract": "flashnext_fn6",
            "contract_version": 1,
            "workload_name": PROMPT_KIND,
            "batch": 1,
            "k": K,
            "greedy": True,
            "max_model_len": MAX_SEQ,
            "cuda_graph_status": graph_status,
            "warm_only": bool(PREFILL_SWEEP),
        },
        extra={
            "fn6": {
                "rounds": ROUNDS,
                "batch_prefill": BATCH_PREFILL,
                "prefill_chunk": PREFILL_CHUNK,
                "prefill_sweep": PREFILL_SWEEP,
                "prefill_layer_major": PREFILL_LAYER_MAJOR_MODE,
                "prefill_mlp_cg": PREFILL_MLP_CG,
                "batch_gdn_verify": BATCH_GDN_VERIFY,
                "batch_gdn_projections": BATCH_GDN_PROJECTIONS,
                "exact_row_math": EXACT_ROW_MATH,
                "ple_resident": PLE_RESIDENT,
                "ple_cache_rows": PLE_CACHE_ROWS,
                "ple_cache_pages": PLE_CACHE_PAGES,
                "ple_io_workers": PLE_IO_WORKERS,
                "ple_io": os.getenv("QSR_FLASHNEXT_PLE_IO", "auto"),
                "ple_source_sha256": hashlib.sha256(
                    (pathlib.Path(_ROOT) / "runtime/model/flashnext/ple.py").read_bytes()
                ).hexdigest(),
                "ple_large_batch_pread_pages": int(
                    os.getenv("QSR_FLASHNEXT_PLE_IO_LARGE_BATCH_PREAD_PAGES", "8192")
                ),
                "trim_initial_mtp_workspace": TRIM_INITIAL_MTP_WORKSPACE,
                "mtp_legacy_route_reduce": os.getenv(
                    "QSR_FLASHNEXT_MTP_LEGACY_ROUTE_REDUCE", "0"
                ),
                "stable_ranks": os.getenv(
                    "B12X_DYNAMIC_BATCHED_STABLE_RANKS", "default"
                ),
            }
        },
    )
    trace_ring = None
    trace_events = None
    if os.getenv("QSR_TRACE", "0") == "1":
        # Import only after auto_record exported this run's id.  Importing the
        # process-global ring earlier makes its atexit dump land under a
        # synthetic local-* run instead of this bfdiag record.
        from bfdiag.trace import events as trace_events_module
        from bfdiag.trace import ring as trace_ring_module

        trace_events = trace_events_module
        trace_ring = trace_ring_module
        _BF.record.trace_path = f"runs/{_BF.run_id}/trace.jsonl"
        _BF.save()
    torch.cuda.reset_peak_memory_stats()

    if PROMPT_KIND == "sglang-long":
        if PROMPT_TOKENS:
            raise ValueError(
                "FN_PROMPT_TOKENS cannot modify the exact sglang-long comparison prompt"
            )
        if MAX_SEQ < 32768:
            raise ValueError(
                "FN_PROMPT_KIND=sglang-long requires FN_MAX_SEQ>=32768 "
                f"(got {MAX_SEQ})"
            )

    cfg = FlashNextTextConfig.from_checkpoint(CKPT)
    started = time.perf_counter()
    target_load_started = time.perf_counter()
    model = load_flashnext_model(
        CKPT,
        "cuda",
        progress=lambda _done, _total: None,
        ple_resident=PLE_RESIDENT,
        ple_cache_rows=PLE_CACHE_ROWS,
        ple_cache_pages=PLE_CACHE_PAGES,
        ple_io_workers=PLE_IO_WORKERS,
    )
    torch.cuda.synchronize()
    target_load_seconds = time.perf_counter() - target_load_started
    _memory("target loaded")

    mtp = None
    mtp_load_seconds = 0.0
    if not PREFILL_ONLY:
        mtp_load_started = time.perf_counter()
        mtp = load_flashnext_mtp(CKPT, cfg, model, "cuda")
        torch.cuda.synchronize()
        mtp_load_seconds = time.perf_counter() - mtp_load_started
        _memory("MTP loaded")

    graph_started = time.perf_counter()
    sess = new_session(model, "cuda")
    prepare_graph_buffers(model, sess, "cuda", max_seq=MAX_SEQ)
    sess.want_hc_hidden = True
    target = FlashNextGraphEngine(model, sess, "cuda")
    spec = None
    if PREFILL_MLP_CG:
        if PREFILL_CHUNK <= 0:
            raise ValueError("FN_PREFILL_MLP_CG=1 requires a positive FN_PREFILL_CHUNK")
        target.capture_prefill_mlp_graphs(PREFILL_CHUNK)
    if not PREFILL_ONLY:
        spec = FlashNextSpecEngine(
            model,
            mtp,
            sess,
            max_seq=MAX_SEQ,
            device="cuda",
            k=K,
            exact_row_math=EXACT_ROW_MATH,
            batch_lm_head=True,
            batch_gdn_recurrence=BATCH_GDN_VERIFY,
            batch_gdn_projections=BATCH_GDN_PROJECTIONS,
            mtp_continuation_graph=MTP_CONTINUATION_GRAPH,
            mtp_sparse_graph=MTP_SPARSE_GRAPH,
        )
        # Verify first so every shared capacity-backed scratch arena (notably
        # b12x's micro MoE workspace) is grown to K+1 before the M=1 target
        # graph captures its pointers.  Capturing target first lets verify
        # warm-up replace a live workspace allocation under the target graph.
        target._zero_state()
        spec.capture_verify()
        target.capture()
        target._zero_state()  # capture warmup wrote candidate QSA rows
    torch.cuda.synchronize()
    graph_seconds = time.perf_counter() - graph_started
    _memory("graphs captured")
    # Startup/load peaks and request peaks answer different questions. Reset
    # after all graphs exist so the request snapshot below measures serving
    # work rather than model construction or capture warmup.
    torch.cuda.reset_peak_memory_stats()

    tokenizer_load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(CKPT))
    tokenizer_load_seconds = time.perf_counter() - tokenizer_load_started
    ready_seconds = time.perf_counter() - started
    startup = {
        "target_load": round(target_load_seconds, 3),
        "mtp_load": round(mtp_load_seconds, 3),
        "graphs": round(graph_seconds, 3),
        "tokenizer": round(tokenizer_load_seconds, 3),
        "ready": round(ready_seconds, 3),
    }
    print(f"startup={startup}", flush=True)
    _metrics(
        startup_target_load_s=target_load_seconds,
        startup_mtp_load_s=mtp_load_seconds,
        startup_graphs_s=graph_seconds,
        startup_tokenizer_s=tokenizer_load_seconds,
        startup_ready_s=ready_seconds,
    )

    torch.cuda.synchronize()
    request_started = time.perf_counter()
    if PROMPT_KIND == "sglang-long":
        filler = (
            "The quick brown fox jumps over the lazy dog while the observer records "
            "the exact sequence of events for later analysis of timing and behavior. "
        )
        messages = [
            {
                "role": "user",
                "content": (
                    f"Below is a long document.\n\n{filler * 900}\n\n"
                    "Question: what does the observer record? Answer in one sentence."
                ),
            }
        ]
    elif PROMPT_KIND == "tcp":
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise technical assistant. Give a rigorous, self-contained answer."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Explain how TCP congestion control works, including slow start, "
                    "congestion avoidance, fast retransmit, fast recovery, and how modern "
                    "algorithms such as CUBIC differ from Reno. Use a detailed example."
                ),
            },
        ]
    else:
        raise ValueError(f"unsupported FN_PROMPT_KIND={PROMPT_KIND!r}")
    prompt_encoding = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    prompt_ids = (
        prompt_encoding.input_ids if hasattr(prompt_encoding, "input_ids") else prompt_encoding
    )
    if isinstance(prompt_ids, torch.Tensor):
        prompt_ids = prompt_ids.flatten().tolist()
    if PROMPT_TOKENS:
        if PROMPT_TOKENS <= 0:
            raise ValueError(f"FN_PROMPT_TOKENS must be positive, got {PROMPT_TOKENS}")
        repeats = (PROMPT_TOKENS + len(prompt_ids) - 1) // len(prompt_ids)
        prompt_ids = (prompt_ids * repeats)[:PROMPT_TOKENS]
    if len(prompt_ids) > MAX_SEQ:
        raise ValueError(
            f"prompt has {len(prompt_ids)} tokens but FN_MAX_SEQ={MAX_SEQ}"
        )
    prompt_hash = hashlib.sha256(
        ",".join(str(token) for token in prompt_ids).encode()
    ).hexdigest()
    _BF.record.fingerprint.workload.prompt_len = len(prompt_ids)
    _BF.record.fingerprint.workload.prompt_hash = prompt_hash
    _BF.save()
    print(
        f"benchmark_id=flashnext:{PROMPT_KIND}:{len(prompt_ids)} "
        f"prompt_kind={PROMPT_KIND} prompt_tokens={len(prompt_ids)} "
        f"synthetic_repeat={bool(PROMPT_TOKENS)} max_seq={MAX_SEQ}",
        flush=True,
    )
    print(
        "benchmark_config="
        f"batch_prefill={int(BATCH_PREFILL)} "
        f"prefill_chunk={PREFILL_CHUNK} "
        f"layer_major={int(_prefill_layer_major(PREFILL_CHUNK))} "
        f"prefill_mlp_cg={int(PREFILL_MLP_CG)} "
        f"spec_k={K} rounds={ROUNDS} "
        f"verify_cg={int(USE_VERIFY_GRAPH)} "
        f"mtp_continuation_cg={int(MTP_CONTINUATION_GRAPH)} "
        f"mtp_sparse_cg={int(MTP_SPARSE_GRAPH)} "
        f"batch_gdn_recurrence={int(BATCH_GDN_VERIFY)} "
        f"batch_gdn_projections={int(BATCH_GDN_PROJECTIONS)} "
        f"exact_row_math={int(EXACT_ROW_MATH)} "
        f"moe_backend={os.getenv('QSR_FLASHNEXT_MOE_BACKEND', 'b12x')} "
        f"ple_mode={'resident' if PLE_RESIDENT else 'stream'} "
        f"ple_cache_rows={PLE_CACHE_ROWS} "
        f"ple_cache_pages={PLE_CACHE_PAGES} "
        f"ple_io_workers={PLE_IO_WORKERS} "
        f"ple_io={os.getenv('QSR_FLASHNEXT_PLE_IO', 'auto')} "
        "ple_large_batch_pread_pages="
        f"{os.getenv('QSR_FLASHNEXT_PLE_IO_LARGE_BATCH_PREAD_PAGES', '8192')} "
        f"trim_initial_mtp_workspace={int(TRIM_INITIAL_MTP_WORKSPACE)} "
        f"stable_ranks={os.getenv('B12X_DYNAMIC_BATCHED_STABLE_RANKS', 'default')}",
        flush=True,
    )
    tokenization_seconds = time.perf_counter() - request_started

    if COMPARE_FLASHINFER_MOE:
        _compare_flashinfer_moe(target, cfg, prompt_ids)
        return

    if COMPARE_MOE_RANKS:
        rank_results = {}
        rank_timings = {"sort": [], "batched": []}
        for name, enabled in (
            ("sort", "0"),
            ("batched", "1"),
            ("sort", "0"),
            ("batched", "1"),
        ):
            os.environ["B12X_DYNAMIC_BATCHED_STABLE_RANKS"] = enabled
            target._zero_state()
            torch.cuda.synchronize()
            rank_started = time.perf_counter()
            rank_logits, rank_hiddens = target.prefill(
                prompt_ids,
                chunk_size=PREFILL_CHUNK,
                layer_major=_prefill_layer_major(PREFILL_CHUNK),
            )
            torch.cuda.synchronize()
            rank_timings[name].append(time.perf_counter() - rank_started)
            rank_results[name] = (rank_logits.clone(), rank_hiddens.clone())
        sort_seconds = min(rank_timings["sort"])
        batched_seconds = min(rank_timings["batched"])
        sort_logits, sort_hiddens = rank_results["sort"]
        batched_logits, batched_hiddens = rank_results["batched"]
        print(
            "moe_rank_comparison="
            f"sort_seconds={sort_seconds:.3f} "
            f"sort_throughput={len(prompt_ids) / sort_seconds:.2f} "
            f"batched_seconds={batched_seconds:.3f} "
            f"batched_throughput={len(prompt_ids) / batched_seconds:.2f} "
            f"speedup={sort_seconds / batched_seconds:.3f} "
            f"logits_bitwise={torch.equal(sort_logits, batched_logits)} "
            f"hiddens_bitwise={torch.equal(sort_hiddens, batched_hiddens)}",
            flush=True,
        )
        del (
            rank_results,
            rank_timings,
            sort_logits,
            sort_hiddens,
            batched_logits,
            batched_hiddens,
        )
        target._zero_state()
        request_started = time.perf_counter()
        tokenization_seconds = 0.0

    if COMPARE_LAYER_MAJOR:
        timings = {"chunk": [], "layer": []}
        results = {}
        for name, layer_major in (
            ("chunk", False),
            ("layer", True),
            ("chunk", False),
            ("layer", True),
        ):
            target._zero_state()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            compare_started = time.perf_counter()
            compare_logits, compare_hiddens = target.prefill(
                prompt_ids,
                chunk_size=PREFILL_CHUNK,
                layer_major=layer_major,
            )
            torch.cuda.synchronize()
            timings[name].append(time.perf_counter() - compare_started)
            if len(timings[name]) == 2:
                results[name] = {
                    "seconds": min(timings[name]),
                    "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "logits": compare_logits.clone(),
                    "hiddens": compare_hiddens.clone(),
                    "states": _capture_prefill_state(sess),
                }
            del compare_logits, compare_hiddens

        chunk = results["chunk"]
        layer = results["layer"]
        logits_cosine, logits_max_abs = _tensor_similarity(
            layer["logits"], chunk["logits"]
        )
        hidden_cosine, hidden_max_abs = _tensor_similarity(
            layer["hiddens"], chunk["hiddens"]
        )
        state_metrics = {
            name: _tensor_similarity(layer["states"][name], chunk["states"][name])
            for name in chunk["states"]
        }
        worst_state = min(state_metrics, key=lambda name: state_metrics[name][0])
        max_abs_state = max(state_metrics, key=lambda name: state_metrics[name][1])
        print(
            "prefill_layer_major_comparison="
            f"chunk_seconds={chunk['seconds']:.3f} "
            f"chunk_throughput={len(prompt_ids) / chunk['seconds']:.2f} "
            f"chunk_peak_gib={chunk['peak_gib']:.2f} "
            f"layer_seconds={layer['seconds']:.3f} "
            f"layer_throughput={len(prompt_ids) / layer['seconds']:.2f} "
            f"layer_peak_gib={layer['peak_gib']:.2f} "
            f"speedup={chunk['seconds'] / layer['seconds']:.3f} "
            f"logits_top1_match={int(layer['logits'].argmax()) == int(chunk['logits'].argmax())} "
            f"logits_cosine={logits_cosine:.8f} "
            f"logits_max_abs={logits_max_abs:.6f} "
            f"hidden_cosine={hidden_cosine:.8f} "
            f"hidden_max_abs={hidden_max_abs:.6f} "
            f"min_state_cosine={state_metrics[worst_state][0]:.8f} "
            f"min_state_name={worst_state} "
            f"max_state_abs={state_metrics[max_abs_state][1]:.6f} "
            f"max_state_abs_name={max_abs_state}",
            flush=True,
        )
        del results
        target._zero_state()
        if PREFILL_ONLY:
            return
        request_started = time.perf_counter()
        tokenization_seconds = 0.0

    if COMPARE_PREFILL_CHUNKS:
        if len(COMPARE_PREFILL_CHUNKS) != 2 or any(
            chunk <= 0 for chunk in COMPARE_PREFILL_CHUNKS
        ):
            raise ValueError(
                "FN_COMPARE_PREFILL_CHUNKS must contain two positive sizes, "
                f"got {COMPARE_PREFILL_CHUNKS}"
            )

        def capture_prefill_state() -> dict[str, torch.Tensor]:
            return {
                **{
                    f"{name}.conv": state.conv_state.clone()
                    for name, state in sess.gdn.items()
                },
                **{
                    f"{name}.recurrent": state.recurrent_state.clone()
                    for name, state in sess.gdn.items()
                },
                "ple.conv": sess.ple_conv_state.clone(),
                **{
                    f"qsa.{layer}.idx_k": pool[: sess.pos].clone()
                    for layer, pool in sess.qsa_idx_k_pool.items()
                },
                **{
                    f"qsa.{layer}.pooled_k": pool[: (sess.pos // 4)].clone()
                    for layer, pool in sess.qsa_pooled_k_pool.items()
                },
                **{
                    f"qsa.{layer}.k": pool[: sess.pos].clone()
                    for layer, pool in sess.qsa_k_pool.items()
                },
                **{
                    f"qsa.{layer}.v": pool[: sess.pos].clone()
                    for layer, pool in sess.qsa_v_pool.items()
                },
            }

        def tensor_similarity(
            got: torch.Tensor, expected: torch.Tensor
        ) -> tuple[float, float]:
            got_f = got.float().flatten()
            expected_f = expected.float().flatten()
            cosine = torch.nn.functional.cosine_similarity(got_f, expected_f, dim=0)
            return float(cosine), float((got_f - expected_f).abs().max())

        reference_chunk, candidate_chunk = COMPARE_PREFILL_CHUNKS
        timings = {reference_chunk: [], candidate_chunk: []}
        results = {}
        for chunk_size in (
            reference_chunk,
            candidate_chunk,
            reference_chunk,
            candidate_chunk,
        ):
            target._zero_state()
            torch.cuda.synchronize()
            compare_started = time.perf_counter()
            logits, hiddens = target.prefill(
                prompt_ids,
                chunk_size=chunk_size,
                layer_major=_prefill_layer_major(chunk_size),
            )
            torch.cuda.synchronize()
            timings[chunk_size].append(time.perf_counter() - compare_started)
            if len(timings[chunk_size]) == 2:
                results[chunk_size] = (
                    logits.clone(),
                    hiddens.clone(),
                    capture_prefill_state(),
                )
            del logits, hiddens

        reference_seconds = min(timings[reference_chunk])
        candidate_seconds = min(timings[candidate_chunk])
        reference_logits, reference_hiddens, reference_states = results[reference_chunk]
        candidate_logits, candidate_hiddens, candidate_states = results[candidate_chunk]

        logits_cosine, logits_max_abs = tensor_similarity(
            candidate_logits, reference_logits
        )
        hidden_cosine, hidden_max_abs = tensor_similarity(
            candidate_hiddens, reference_hiddens
        )
        state_metrics = {
            name: tensor_similarity(candidate_states[name], reference_states[name])
            for name in reference_states
        }
        worst_state = min(state_metrics, key=lambda name: state_metrics[name][0])
        max_abs_state = max(state_metrics, key=lambda name: state_metrics[name][1])
        print(
            "prefill_chunk_comparison="
            f"reference_chunk={reference_chunk} "
            f"reference_seconds={reference_seconds:.3f} "
            f"reference_throughput={len(prompt_ids) / reference_seconds:.2f} "
            f"candidate_chunk={candidate_chunk} "
            f"candidate_seconds={candidate_seconds:.3f} "
            f"candidate_throughput={len(prompt_ids) / candidate_seconds:.2f} "
            f"speedup={reference_seconds / candidate_seconds:.3f} "
            f"logits_top1_match={int(candidate_logits.argmax()) == int(reference_logits.argmax())} "
            f"logits_cosine={logits_cosine:.8f} "
            f"logits_max_abs={logits_max_abs:.6f} "
            f"hidden_cosine={hidden_cosine:.8f} "
            f"hidden_max_abs={hidden_max_abs:.6f} "
            f"min_state_cosine={state_metrics[worst_state][0]:.8f} "
            f"min_state_name={worst_state} "
            f"max_state_abs={state_metrics[max_abs_state][1]:.6f} "
            f"max_state_abs_name={max_abs_state}",
            flush=True,
        )
        del results, reference_hiddens, candidate_hiddens, reference_states, candidate_states
        target._zero_state()
        if PREFILL_ONLY:
            return
        request_started = time.perf_counter()
        tokenization_seconds = 0.0

    if VALIDATE_BATCH_PREFILL:
        target._zero_state()
        batch_logits, batch_hiddens = target.prefill(
            prompt_ids,
            chunk_size=PREFILL_CHUNK,
            layer_major=_prefill_layer_major(PREFILL_CHUNK),
        )
        torch.cuda.synchronize()
        batch_states = {
            **{
                f"{name}.conv": state.conv_state.clone()
                for name, state in sess.gdn.items()
            },
            **{
                f"{name}.recurrent": state.recurrent_state.clone()
                for name, state in sess.gdn.items()
            },
            "ple.conv": sess.ple_conv_state.clone(),
            **{
                f"qsa.{layer}.idx_k": pool[: sess.pos].clone()
                for layer, pool in sess.qsa_idx_k_pool.items()
            },
            **{
                f"qsa.{layer}.k": pool[: sess.pos].clone()
                for layer, pool in sess.qsa_k_pool.items()
            },
            **{
                f"qsa.{layer}.v": pool[: sess.pos].clone()
                for layer, pool in sess.qsa_v_pool.items()
            },
        }

        target._zero_state()
        serial_hiddens = []
        serial_logits = None
        for token in prompt_ids:
            serial_logits = target.step(int(token))
            serial_hiddens.append(sess.hc_hidden_buf.clone())
        serial_hiddens_tensor = torch.stack(serial_hiddens)
        torch.cuda.synchronize()
        serial_states = {
            **{
                f"{name}.conv": state.conv_state
                for name, state in sess.gdn.items()
            },
            **{
                f"{name}.recurrent": state.recurrent_state
                for name, state in sess.gdn.items()
            },
            "ple.conv": sess.ple_conv_state,
            **{
                f"qsa.{layer}.idx_k": pool[: sess.pos]
                for layer, pool in sess.qsa_idx_k_pool.items()
            },
            **{
                f"qsa.{layer}.k": pool[: sess.pos]
                for layer, pool in sess.qsa_k_pool.items()
            },
            **{
                f"qsa.{layer}.v": pool[: sess.pos]
                for layer, pool in sess.qsa_v_pool.items()
            },
        }

        def similarity(got: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
            got_f = got.float().flatten()
            expected_f = expected.float().flatten()
            cosine = torch.nn.functional.cosine_similarity(got_f, expected_f, dim=0)
            return float(cosine), float((got_f - expected_f).abs().max())

        hidden_cosine, hidden_max_abs = similarity(batch_hiddens, serial_hiddens_tensor)
        logits_cosine, logits_max_abs = similarity(batch_logits, serial_logits)
        state_metrics = {
            name: similarity(batch_states[name], serial_states[name])
            for name in batch_states
        }
        worst_state = min(state_metrics, key=lambda name: state_metrics[name][0])
        max_abs_state = max(state_metrics, key=lambda name: state_metrics[name][1])
        print(
            "prefill_validation="
            f"batch_top1={int(batch_logits.argmax())} "
            f"serial_top1={int(serial_logits.argmax())} "
            f"logits_cosine={logits_cosine:.6f} "
            f"logits_max_abs={logits_max_abs:.4f} "
            f"hidden_cosine={hidden_cosine:.6f} "
            f"hidden_max_abs={hidden_max_abs:.4f} "
            f"min_state_cosine={state_metrics[worst_state][0]:.6f} "
            f"min_state_name={worst_state} "
            f"max_state_abs={state_metrics[max_abs_state][1]:.4f} "
            f"max_state_abs_name={max_abs_state}",
            flush=True,
        )
        target._zero_state()
        request_started = time.perf_counter()
        tokenization_seconds = 0.0

    if PREFILL_SWEEP:
        for chunk_size in PREFILL_SWEEP:
            target._zero_state()
            torch.cuda.synchronize()
            sweep_started = time.perf_counter()
            if chunk_size == 1:
                for token in prompt_ids:
                    target.step(int(token))
            else:
                target.prefill(
                    prompt_ids,
                    chunk_size=chunk_size,
                    layer_major=_prefill_layer_major(chunk_size),
                )
            torch.cuda.synchronize()
            sweep_seconds = time.perf_counter() - sweep_started
            print(
                f"prefill_sweep chunk={chunk_size} seconds={sweep_seconds:.3f} "
                f"throughput={len(prompt_ids) / sweep_seconds:.2f}",
                flush=True,
            )
        target._zero_state()
        request_started = time.perf_counter()
        tokenization_seconds = 0.0

    target_hiddens = None
    logits = None
    torch.cuda.synchronize()
    prefill_started = time.perf_counter()
    prefill_profiler = _profile() if PROFILE_DIR else None
    if BATCH_PREFILL:
        if prefill_profiler is not None:
            prefill_profiler.start()
        with torch.profiler.record_function("flashnext.target_prefill_batch"):
            logits, target_hiddens = target.prefill(
                prompt_ids,
                chunk_size=PREFILL_CHUNK,
                layer_major=_prefill_layer_major(PREFILL_CHUNK),
            )
    else:
        hidden_rows = []
        if prefill_profiler is not None and PROFILE_PREFILL_TOKENS <= 0:
            prefill_profiler = None
        profile_prefill_from = max(0, len(prompt_ids) - PROFILE_PREFILL_TOKENS)
        for token_index, token in enumerate(prompt_ids):
            if prefill_profiler is not None and token_index == profile_prefill_from:
                prefill_profiler.start()
            with torch.profiler.record_function("flashnext.target_prefill_token"):
                logits = target.step(int(token))
                hidden_rows.append(sess.hc_hidden_buf.clone())
        target_hiddens = torch.stack(hidden_rows)
    if prefill_profiler is not None:
        prefill_profiler.stop()
        _finish_profile(prefill_profiler, "target_prefill")
    anchor = int(logits.argmax())
    torch.cuda.synchronize()
    first_token_at = time.perf_counter()
    prefill_seconds = first_token_at - prefill_started
    ttft_seconds = first_token_at - request_started

    if PREFILL_ONLY:
        ple_table = next(layer.ple.table for layer in model.layers if layer.ple is not None)
        print(
            "prefill_only="
            f"prompt_tokens={len(prompt_ids)} "
            f"seconds={prefill_seconds:.3f} "
            f"throughput={len(prompt_ids) / prefill_seconds:.2f} tok/s "
            f"ttft={ttft_seconds:.3f}s "
            f"layer_major={int(_prefill_layer_major(PREFILL_CHUNK))} "
            f"ple_row_hits={ple_table.cache_hits} "
            f"ple_row_misses={ple_table.cache_misses} "
            f"ple_page_hits={ple_table.page_cache_hits} "
            f"ple_page_misses={ple_table.page_cache_misses}",
            flush=True,
        )
        _metrics(
            prompt_tokens=len(prompt_ids),
            prefill_s=prefill_seconds,
            prefill_tok_s=len(prompt_ids) / prefill_seconds,
            ttft_s=ttft_seconds,
        )
        _memory("done")
        return

    assert spec is not None

    mtp_sync_started = time.perf_counter()
    mtp_profiler = _profile() if PROFILE_DIR else None
    if mtp_profiler is not None:
        mtp_profiler.start()
    with torch.profiler.record_function("flashnext.initial_mtp_sync"):
        drafts = spec.sync_and_propose(
            [*prompt_ids[1:], anchor],
            target_hiddens,
        )
    del target_hiddens
    if mtp_profiler is not None:
        mtp_profiler.stop()
        _finish_profile(mtp_profiler, "initial_mtp")
    torch.cuda.synchronize()
    if TRIM_INITIAL_MTP_WORKSPACE:
        # Teacher-forced sync is the only large-M MTP call in a request.  Its
        # grouped-GEMM workspaces are dead after ``target_hiddens`` is
        # consumed, but the caching allocator otherwise retains roughly
        # 11 GiB on the 23K-token workload.  Graph replay uses its own fixed
        # pools, so returning these one-shot blocks to the driver is safe and
        # prevents a valid long prompt from leaving only ~2.5 GiB headroom.
        torch.cuda.empty_cache()
    initial_mtp_seconds = time.perf_counter() - mtp_sync_started
    if TRIM_INITIAL_MTP_WORKSPACE:
        _memory("initial MTP trimmed")

    generated = [*prompt_ids, anchor]
    accepted = 0
    accepted_by_position = [0] * K
    accepted_lengths = [0] * (K + 1)
    timing_totals = {"ple": 0.0, "verify": 0.0, "mtp": 0.0}
    verify_logits = []
    round_trace_rows: list[dict[str, object]] = []
    eos_round = None
    eos_accepted = None
    eos_accepted_by_position = None
    eos_completion_tokens = None
    eos_decode_seconds = None
    rounds = ROUNDS
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    decode_profiler = _profile() if PROFILE_DIR else None
    if decode_profiler is not None:
        decode_profiler.start()
    for round_index in range(rounds):
        kv_len_before = sess.pos
        anchor_before = int(anchor)
        drafts_before = [int(token) for token in drafts]
        round_started = time.perf_counter()
        trace_row = (
            trace_ring.begin_round(slot=0, kv_len_before=kv_len_before)
            if trace_ring is not None
            else None
        )
        with torch.profiler.record_function("flashnext.spec_round"):
            result = spec.round(
                anchor,
                drafts,
                use_graph=USE_VERIFY_GRAPH,
                return_verify_logits=True,
            )
        committed = result["committed"]
        num_accepted = int(result["num_accepted"])
        reject_position = int(result["reject_position"])
        bonus_token = int(result["bonus_token"])
        round_seconds = time.perf_counter() - round_started
        accepted_before_round = accepted
        accepted_by_position_before_round = accepted_by_position.copy()
        committed_before_round = len(generated) - len(prompt_ids) - 1
        accepted += num_accepted
        accepted_lengths[num_accepted] += 1
        for position in range(num_accepted):
            accepted_by_position[position] += 1
        for name, seconds in result["timing"].items():
            timing_totals[name] += seconds
        generated.extend(committed)
        if eos_round is None and cfg.eos_token_id in committed:
            eos_offset = committed.index(cfg.eos_token_id)
            effective_num_accepted = min(num_accepted, eos_offset + 1)
            eos_round = round_index + 1
            eos_accepted = accepted_before_round + effective_num_accepted
            eos_accepted_by_position = accepted_by_position_before_round
            for position in range(effective_num_accepted):
                eos_accepted_by_position[position] += 1
            eos_completion_tokens = 1 + committed_before_round + eos_offset + 1
            torch.cuda.synchronize()
            eos_decode_seconds = time.perf_counter() - t0
        verify_logits.append(result["verify_logits"])
        round_trace_rows.append(
            {
                "round_idx": round_index,
                "kv_len_before": kv_len_before,
                "anchor_token": anchor_before,
                "draft_tokens": drafts_before,
                "verify_tokens": result["verify_tokens"],
                "verify_prediction_ids": result["verify_prediction_ids"],
                "accepted_n": num_accepted,
                "reject_position": reject_position,
                "bonus_token": bonus_token,
                "teacher_tokens": result["teacher_tokens"],
                "committed_tokens": committed,
                "next_draft_tokens": result["next_draft_tokens"],
                "target_pos_after": sess.pos,
                "mtp_sync_len_after": spec.mtp_session.sync_len,
                "t_ple_ms": float(result["timing"]["ple"]) * 1000.0,
                "t_verify_ms": float(result["timing"]["verify"]) * 1000.0,
                "t_mtp_ms": float(result["timing"]["mtp"]) * 1000.0,
                "t_round_ms": round_seconds * 1000.0,
            }
        )
        if trace_row is not None:
            assert trace_events is not None
            trace_ring.finish_round(
                trace_row,
                trace_events.PHASE_DRAFT,
                path=(
                    trace_events.Path.CG_REPLAY
                    if USE_VERIFY_GRAPH
                    else trace_events.Path.EAGER
                ),
                cg_miss_reason=(
                    trace_events.CgMissReason.NONE
                    if USE_VERIFY_GRAPH
                    else trace_events.CgMissReason.CUDA_GRAPH_DISABLED
                ),
                draft_tokens_n=K,
                accepted_n=num_accepted,
                reject_position=reject_position,
                bonus_token=bonus_token,
            )
        anchor = int(result["next_anchor"])
        drafts = result["next_draft_tokens"]
        if decode_profiler is not None and round_index + 1 == min(
            PROFILE_DECODE_ROUNDS, rounds
        ):
            decode_profiler.stop()
            _finish_profile(decode_profiler, "decode")
            decode_profiler = None
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    round_trace_path = _write_round_trace(round_trace_rows)
    if round_trace_path is not None:
        print(f"round_trace={round_trace_path}", flush=True)
    request_finished = time.perf_counter()
    request_seconds = request_finished - request_started
    cold_completion_seconds = request_finished - started
    committed_tokens = len(generated) - len(prompt_ids) - 1
    completion_tokens = len(generated) - len(prompt_ids)
    print(
        "request="
        f"prompt_tokens={len(prompt_ids)} "
        f"completion_tokens={completion_tokens} "
        f"tokenize={tokenization_seconds:.4f}s "
        f"prefill={prefill_seconds:.3f}s "
        f"prefill_throughput={len(prompt_ids) / prefill_seconds:.2f} tok/s "
        f"ttft={ttft_seconds:.3f}s "
        f"initial_mtp={initial_mtp_seconds:.3f}s "
        f"decode={elapsed:.3f}s "
        f"request_total={request_seconds:.3f}s "
        f"e2e_completion_throughput={completion_tokens / request_seconds:.2f} tok/s "
        f"cold_to_completion={cold_completion_seconds:.3f}s",
        flush=True,
    )
    _metrics(
        prompt_tokens=len(prompt_ids),
        completion_tokens=completion_tokens,
        prefill_s=prefill_seconds,
        prefill_tok_s=len(prompt_ids) / prefill_seconds,
        ttft_s=ttft_seconds,
        initial_mtp_s=initial_mtp_seconds,
        decode_s=elapsed,
        request_total_s=request_seconds,
        e2e_completion_tok_s=completion_tokens / request_seconds,
        cold_to_completion_s=cold_completion_seconds,
    )
    _memory("request done")
    if not verify_logits:
        print(
            "timing="
            f"{ {name: round(seconds, 3) for name, seconds in timing_totals.items()} } "
            "ple_spec_stats={}",
            flush=True,
        )
        _memory("done")
        return
    if eos_round is not None:
        assert eos_accepted is not None
        assert eos_accepted_by_position is not None
        assert eos_completion_tokens is not None
        assert eos_decode_seconds is not None
        request_to_eos = ttft_seconds + initial_mtp_seconds + eos_decode_seconds
        _metrics(
            eos_completion_tokens=eos_completion_tokens,
            eos_rounds=eos_round,
            eos_accepted=eos_accepted,
            eos_accept_rate=eos_accepted / (eos_round * K),
            eos_decode_s=eos_decode_seconds,
            request_to_eos_s=request_to_eos,
            eos_e2e_completion_tok_s=eos_completion_tokens / request_to_eos,
        )
        print(
            "request_to_eos="
            f"completion_tokens={eos_completion_tokens} "
            f"decode={eos_decode_seconds:.3f}s "
            f"request_total={request_to_eos:.3f}s "
            f"e2e_completion_throughput={eos_completion_tokens / request_to_eos:.2f} tok/s "
            f"rounds={eos_round} "
            f"accepted={eos_accepted} "
            f"accept_rate={eos_accepted / (eos_round * K):.3f} "
            f"position_accept="
            f"{[round(value / eos_round, 3) for value in eos_accepted_by_position]}",
            flush=True,
        )
    ple_table = next(layer.ple.table for layer in model.layers if layer.ple is not None)
    spec_ple_stats = {
        "row_hits": ple_table.cache_hits,
        "row_misses": ple_table.cache_misses,
        "page_hits": ple_table.page_cache_hits,
        "page_misses": ple_table.page_cache_misses,
        "pread_pages": ple_table.pread_pages,
        "io_uring_pages": ple_table.io_uring_pages,
    }

    assert sess.pos == len(generated) - 1
    spec_gdn = {
        key: (state.conv_state.clone(), state.recurrent_state.clone())
        for key, state in sess.gdn.items()
    }
    spec_ple = sess.ple_conv_state.clone()
    spec_qsa = {
        layer: (
            sess.qsa_idx_k_pool[layer][: sess.pos].clone(),
            sess.qsa_k_pool[layer][: sess.pos].clone(),
            sess.qsa_v_pool[layer][: sess.pos].clone(),
        )
        for layer in sess.qsa_k_pool
    }
    verify_logits_tensor = torch.cat(verify_logits)
    if not torch.isfinite(verify_logits_tensor).all():
        bad = int((~torch.isfinite(verify_logits_tensor)).sum())
        raise RuntimeError(f"speculative verify produced {bad} non-finite logits")
    speculative_states = {
        **{f"{key}.conv": conv for key, (conv, _recurrent) in spec_gdn.items()},
        **{f"{key}.recurrent": recurrent for key, (_conv, recurrent) in spec_gdn.items()},
        "ple.conv": spec_ple,
        **{
            f"qsa.{layer}.{kind}": tensor
            for layer, tensors in spec_qsa.items()
            for kind, tensor in zip(("idx_k", "k", "v"), tensors, strict=True)
        },
    }
    for name, state in speculative_states.items():
        bad = int((~torch.isfinite(state)).sum())
        if bad:
            raise RuntimeError(f"speculative state {name} produced {bad} non-finite values")

    # Exact greedy/state gate: replay the client-visible stream except its
    # one-token-ahead bonus through the ordinary decode graph.  Every target
    # prediction and every recurrent/cache state must match the speculative
    # commit.  This catches off-by-one accept, PLE window, and GDN row bugs.
    target._zero_state()
    if BATCH_PREFILL:
        _baseline_prompt_logits, _baseline_prompt_hiddens = target.prefill(
            prompt_ids,
            chunk_size=PREFILL_CHUNK,
            layer_major=_prefill_layer_major(PREFILL_CHUNK),
        )
    else:
        for token in prompt_ids:
            target.step(int(token))
    baseline_logits_rows = []
    greedy_mismatches = []
    for index, token in enumerate(
        generated[len(prompt_ids) : -1],
        start=len(prompt_ids),
    ):
        baseline_logits = target.step(int(token))
        predicted = int(baseline_logits.argmax())
        baseline_logits_rows.append(baseline_logits.clone())
        if predicted != generated[index + 1]:
            greedy_mismatches.append((index, predicted, int(generated[index + 1])))
    torch.cuda.synchronize()
    state_cosines = []
    state_max_abs = []

    def record_state(name: str, got: torch.Tensor, expected: torch.Tensor) -> None:
        got_bad = int((~torch.isfinite(got)).sum())
        expected_bad = int((~torch.isfinite(expected)).sum())
        if got_bad or expected_bad:
            raise RuntimeError(
                f"state comparison {name} encountered non-finite values: "
                f"baseline={got_bad}, speculative={expected_bad}"
            )
        got_f = got.float().flatten()
        expected_f = expected.float().flatten()
        state_cosines.append(float(torch.nn.functional.cosine_similarity(got_f, expected_f, dim=0)))
        state_max_abs.append(float((got_f - expected_f).abs().max()))

    for key, state in sess.gdn.items():
        conv, recurrent = spec_gdn[key]
        record_state(f"{key}.conv", state.conv_state, conv)
        record_state(f"{key}.recurrent", state.recurrent_state, recurrent)
    record_state("ple.conv", sess.ple_conv_state, spec_ple)
    for layer, (idx_k, k, v) in spec_qsa.items():
        record_state(f"qsa.{layer}.idx_k", sess.qsa_idx_k_pool[layer][: sess.pos], idx_k)
        record_state(f"qsa.{layer}.k", sess.qsa_k_pool[layer][: sess.pos], k)
        record_state(f"qsa.{layer}.v", sess.qsa_v_pool[layer][: sess.pos], v)

    baseline_logits_tensor = torch.stack(baseline_logits_rows)
    assert verify_logits_tensor.shape == baseline_logits_tensor.shape
    if not torch.isfinite(baseline_logits_tensor).all():
        bad = int((~torch.isfinite(baseline_logits_tensor)).sum())
        raise RuntimeError(f"ordinary decode produced {bad} non-finite logits")
    cosine = torch.nn.functional.cosine_similarity(
        verify_logits_tensor,
        baseline_logits_tensor,
        dim=-1,
    )
    abs_error = (verify_logits_tensor - baseline_logits_tensor).abs()
    top1_agreement = (
        (verify_logits_tensor.argmax(dim=-1) == baseline_logits_tensor.argmax(dim=-1))
        .float()
        .mean()
    )
    _metrics(
        accepted=accepted,
        accept_rate=accepted / (rounds * K),
        committed_tokens=committed_tokens,
        committed_tok_s=committed_tokens / elapsed,
        top1_agreement=float(top1_agreement),
        min_logits_cosine=float(cosine.min()),
        max_abs_error=float(abs_error.max()),
        min_state_cosine=min(state_cosines),
        max_state_abs=max(state_max_abs),
        greedy_mismatches=len(greedy_mismatches),
    )

    if eos_completion_tokens is not None:
        eos_rows = min(eos_completion_tokens - 1, int(verify_logits_tensor.shape[0]))
        eos_top1_agreement = (
            (
                verify_logits_tensor[:eos_rows].argmax(dim=-1)
                == baseline_logits_tensor[:eos_rows].argmax(dim=-1)
            )
            .float()
            .mean()
        )
        eos_cosine = cosine[:eos_rows]
        eos_abs_error = abs_error[:eos_rows]
        eos_generated_index = len(prompt_ids) + eos_completion_tokens - 1
        eos_greedy_mismatches = [
            mismatch
            for mismatch in greedy_mismatches
            if mismatch[0] < eos_generated_index
        ]
        _metrics(
            eos_top1_agreement=float(eos_top1_agreement),
            eos_min_logits_cosine=float(eos_cosine.min()),
            eos_max_abs_error=float(eos_abs_error.max()),
            eos_greedy_mismatches=len(eos_greedy_mismatches),
        )
        print(
            "quality_to_eos="
            f"rows={eos_rows} "
            f"top1_agreement={float(eos_top1_agreement):.3f} "
            f"min_logits_cosine={float(eos_cosine.min()):.6f} "
            f"max_abs_error={float(eos_abs_error.max()):.4f} "
            f"greedy_mismatches={eos_greedy_mismatches[:8]}",
            flush=True,
        )

    if PRINT_COMPLETION:
        print(tokenizer.decode(generated), flush=True)
    print(
        f"K={K} rounds={rounds} accepted={accepted} "
        f"accept_rate={accepted / (rounds * K):.3f} "
        f"position_accept={[round(value / rounds, 3) for value in accepted_by_position]} "
        f"accepted_lengths={accepted_lengths} "
        f"committed={committed_tokens} elapsed={elapsed:.3f}s "
        f"throughput={committed_tokens / elapsed:.2f} tok/s "
        f"top1_agreement={float(top1_agreement):.3f} "
        f"min_logits_cosine={float(cosine.min()):.6f} "
        f"max_abs_error={float(abs_error.max()):.4f} "
        f"min_state_cosine={min(state_cosines):.6f} "
        f"max_state_abs={max(state_max_abs):.4f} "
        f"greedy_mismatches={greedy_mismatches[:8]}",
        flush=True,
    )
    print(
        "timing="
        f"{ {name: round(seconds, 3) for name, seconds in timing_totals.items()} } "
        f"ple_spec_stats={spec_ple_stats}",
        flush=True,
    )
    _memory("done")


if __name__ == "__main__":
    main()
