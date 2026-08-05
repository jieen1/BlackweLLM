"""Qwen3.6 MTP CUDA Graph capture + replay smoke check.

**不要把图重放与 eager 的逐位相同当作本脚本的判据**：本仓库早已证明
这个比较做不到，与 MTP 无关：
``notes/2026-08-02-eager-verify-cg-verify-divergence.md`` 的结论是
**CG 冻结在 1 个 KV 分块、eager 用 4~16 个**，而"块数不一致本身不是错误，
两种块数在注意力算子这一层都算对了"（cos ≥ 0.999997，kv_len=64/400/500 全成立）。
全模型 logits 上看到的 argmax 翻转是**近似平局位置的翻转**，不是某条路径算错了。

2026-08-03 的实测复现了同一机制：跑到第 4 轮（首次全接受）结束时，
``accepts``/``anchor``/``live_col``/committed token **两条路径完全一致**，
而 48 个 GDN 层里 45 个的状态差在 **conv 3.1e-02 / recurrent 2e-03**——
**bf16 精度尺度，不是写错行的量级**（写错行会给出完全不同的值）。
到第 5 轮累积漂移把一个近似平局翻了过去，于是本脚本判失败。

**这个失败三次误导了修复方向**：先后被归因为"地址被烤死"和"anchor 写进候选行"，
两个假设都按历史代码认真查证过、也都改过，而签名分毫未变——**因为没有东西可修，是尺子错了。**

**正确的判据是两条**，都已在本仓库建立：

1. **B1-R 的 gap-error**（``docs/b1-correctness-criterion.md``，对照已校准的 bar）
   —— W4A4 / FP8 W8A8 / FP8 KV 都是这么判的。这个判据存在的理由，
   恰恰就是 bit-exactness 被证明达不到。
2. **投机 vs 非投机在同一条路径内逐 token 一致**
   （``scripts/b3_mtp_e2e_acceptance_throughput.py``）——这才是投机解码的正确性定义。

本脚本只确认两件可二值判定的事实：draft/verify 图都被捕获，且一次真实单槽
MTP round 确实 replay 了 verify 图。它不启动 HTTP 服务、不创建第二个模型实例，
默认一个 slot、一个短 prompt，以适合共享单卡上的安全验证。
"""


from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _ROOT)
import runtime  # noqa: E402

assert runtime.__file__.startswith(_ROOT), (
    f"editable install shadowed the worktree: runtime.__file__={runtime.__file__} "
    f"-- rerun with PYTHONPATH including {_ROOT}"
)

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from runtime.backends.qwen36 import Qwen36Backend  # noqa: E402
from runtime.checkpoints import standard_checkpoint_path  # noqa: E402
from runtime.model.compressed_tensors_linear import (  # noqa: E402
    CompressedTensorsFP8ChannelLinear,
    fp8_channel_raw_execution_uses_all_layers,
)
from runtime.model_loading import load_qwen36_model  # noqa: E402
from runtime.sampling import SamplingParams  # noqa: E402

DEVICE = "cuda"
MODEL_PATH = standard_checkpoint_path()
MAX_SEQ_LEN = 512
# Single-slot by default.  Set ``QSR_MTP_SMOKE_SLOTS=2`` only when checking
# M-2's real multi-request verify graph; this still remains one process on
# one GPU, never two services.
NUM_SLOTS = int(os.environ.get("QSR_MTP_SMOKE_SLOTS", "1"))
K = 4
# One round is the default correctness smoke.  Raise it explicitly for the
# steady-state MTP number; it stays one process, one model, and one GPU.
NUM_ROUNDS = int(os.environ.get("QSR_MTP_SMOKE_ROUNDS", "1"))
RESULT_PATH = os.environ.get("QSR_MTP_SMOKE_RESULT_PATH")
# Optional evidence mode for the already-captured production MTP body. It
# profiles *additional*, post-timing steady rounds, so profiler bookkeeping
# never contaminates the reported throughput measurement above.
PROFILE_ROUNDS = int(os.environ.get("QSR_MTP_SMOKE_PROFILE_ROUNDS", "0"))
PROFILE_TRACE_PATH = os.environ.get("QSR_MTP_SMOKE_PROFILE_TRACE")
# Optional real checkpoint/GDN/MTP prefix-cache exercise.  It reuses the
# already-loaded model and the same slot, never a second service or model.
PREFIX_REUSE = os.environ.get("QSR_MTP_SMOKE_PREFIX_REUSE", "0") == "1"
PROMPT = "The capital of France is"
# Preserve the established default ``Request {slot}.`` prompts.  An exact
# prompt is opt-in solely for an apples-to-apples historical target check.
EXACT_PROMPT = os.environ.get("QSR_MTP_SMOKE_EXACT_PROMPT")


def _assert_raw_fp8_weights_remain_packed(backend: Qwen36Backend) -> None:
    """Fail B1 if the raw W8A8 production path created a BF16 weight cache.

    The model needs to execute a real prefill and MTP round before this is
    meaningful: lazy fallback paths otherwise cannot be distinguished from an
    unexercised layer.  In the raw all-layer contract, every channel-FP8
    linear must continue to own only its packed FP8 tensor and scale.
    """
    if not fp8_channel_raw_execution_uses_all_layers():
        return
    fp8_modules = [
        module
        for module in backend.model.modules()
        if isinstance(module, CompressedTensorsFP8ChannelLinear)
    ]
    assert fp8_modules, "expected channel-scaled FP8 linears in the Qwen3.6 checkpoint"
    materialized = [
        module for module in fp8_modules if getattr(module, "_weight_bf16", None) is not None
    ]
    assert not materialized, (
        "raw W8A8 execution materialized BF16 weight cache for "
        f"{len(materialized)}/{len(fp8_modules)} channel-FP8 linears"
    )
    print(f"raw W8A8 retained packed weights for all {len(fp8_modules)} channel-FP8 linears")


def _run(
    backend: Qwen36Backend, slots: list[int], *, prompt_ids_per_slot: list[list[int]]
) -> tuple[list[dict[int, dict]], float]:
    for slot in slots:
        backend.reset_slot(slot)
    state = backend.prefill_chunked_begin(slots, prompt_ids_per_slot, params_per_slot={})
    anchors = {slot: state.result[slot]["anchor"] for slot in slots}
    drafts = {slot: list(state.result[slot]["draft_tokens"]) for slot in slots}
    trace: list[dict[int, dict]] = []
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(NUM_ROUNDS):
        # Use the public production entrypoint even for B=1.  Historical
        # Qwen3.6 routed this through the shared batch verify path too; a
        # direct ``engine.round`` call would miss exactly that regression.
        result = backend.mtp_verify_and_commit_batch(slots, anchors, drafts)
        trace.append(result)
        anchors = {slot: result[slot]["next_anchor"] for slot in slots}
        drafts = {slot: list(result[slot]["next_draft_tokens"]) for slot in slots}
    torch.cuda.synchronize()
    return trace, time.perf_counter() - started


def _run_plain_decode_baseline(
    backend: Qwen36Backend,
    slots: list[int],
    *,
    prompt_ids_per_slot: list[list[int]],
    target_committed_per_slot: dict[int, int],
) -> tuple[dict[int, list[int]], float]:
    """Measure plain greedy decode for the same committed-token budget.

    This reuses the exact same backend/model instance and times only the
    post-prefill decode rounds, mirroring :func:`_run`'s steady-state
    measurement. Each slot drops out once it has produced the same number of
    committed tokens that the preceding MTP run produced for that slot.
    """
    params = SamplingParams()
    for slot in slots:
        backend.reset_slot(slot)
    state = backend.prefill_chunked_begin(slots, prompt_ids_per_slot, params_per_slot={})
    current = {slot: state.result[slot]["anchor"] for slot in slots}
    committed = {slot: [] for slot in slots}
    active = [slot for slot in slots if target_committed_per_slot[slot] > 0]
    torch.cuda.synchronize()
    started = time.perf_counter()
    while active:
        next_tokens = backend.decode_batch_sampled(
            active,
            [current[slot] for slot in active],
            [backend.slot_state(slot).kv_len for slot in active],
            [params] * len(active),
        )
        next_active: list[int] = []
        for slot, token in zip(active, next_tokens, strict=True):
            token_i = int(token)
            committed[slot].append(token_i)
            current[slot] = token_i
            if len(committed[slot]) < target_committed_per_slot[slot]:
                next_active.append(slot)
        active = next_active
    torch.cuda.synchronize()
    return committed, time.perf_counter() - started


def _profile_steady_mtp_rounds(
    backend: Qwen36Backend,
    slots: list[int],
    anchors: dict[int, int],
    drafts: dict[int, list[int]],
) -> dict[str, list[dict[str, float | str]]]:
    """Profile post-warmup MTP rounds without changing the timed smoke metric.

    This is deliberately attached to the existing B1 runner rather than a
    separate benchmark: it proves the trace comes from this worktree's
    no-vLLM backend, one model instance, and the same captured graph bodies
    checked by the smoke.  ``self_*`` durations avoid double-counting nested
    CUDA/CPU operator rows.
    """
    if PROFILE_ROUNDS <= 0:
        return {}
    from torch.profiler import ProfilerActivity, profile

    engine = backend._mtp
    assert engine is not None and engine._verify_cg is not None
    current_anchors = dict(anchors)
    current_drafts = {slot: list(tokens) for slot, tokens in drafts.items()}
    phase_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}

    def instrument(obj: object, name: str, label: str):
        original = getattr(obj, name)

        def wrapped(*args, **kwargs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.profiler.record_function(label):
                result = original(*args, **kwargs)
            end.record()
            phase_events.setdefault(label, []).append((start, end))
            return result

        setattr(obj, name, wrapped)
        return original

    # These are mutually exclusive phases of the production MTP round. The
    # CUDA events deliberately measure elapsed stream time, while profiler
    # rows expose the host work inside each phase.
    restores = [
        (engine._verify_cg, "replay", instrument(engine._verify_cg, "replay", "mtp.verify_graph")),
        (engine.model, "compute_logits", instrument(engine.model, "compute_logits", "mtp.lm_head")),
        (
            engine,
            "_sync_real_suffix_batch_ragged",
            instrument(engine, "_sync_real_suffix_batch_ragged", "mtp.sync"),
        ),
        (
            engine,
            "_continue_draft_batch",
            instrument(engine, "_continue_draft_batch", "mtp.draft"),
        ),
    ]
    torch.cuda.synchronize()
    try:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
        ) as prof:
            for _ in range(PROFILE_ROUNDS):
                result = backend.mtp_verify_and_commit_batch(slots, current_anchors, current_drafts)
                current_anchors = {slot: result[slot]["next_anchor"] for slot in slots}
                current_drafts = {
                    slot: list(result[slot]["next_draft_tokens"]) for slot in slots
                }
            torch.cuda.synchronize()
    finally:
        for obj, name, original in restores:
            setattr(obj, name, original)
    if PROFILE_TRACE_PATH:
        prof.export_chrome_trace(PROFILE_TRACE_PATH)
        print(f"wrote MTP CPU/CUDA trace: {PROFILE_TRACE_PATH}")

    events = prof.key_averages()

    def top_self(attr: str) -> list[dict[str, float | str]]:
        rows = [
            {"name": event.key, "ms": float(getattr(event, attr, 0.0)) / 1000.0}
            for event in events
            if getattr(event, attr, 0.0) > 0
        ]
        return sorted(rows, key=lambda row: float(row["ms"]), reverse=True)[:20]

    summary = {
        "self_cpu_ms": top_self("self_cpu_time_total"),
        "self_cuda_ms": top_self("self_device_time_total"),
        "phase_cuda_ms": {
            name: sum(start.elapsed_time(end) for start, end in pairs) / PROFILE_ROUNDS
            for name, pairs in phase_events.items()
        },
    }
    print(f"MTP profiler: {PROFILE_ROUNDS} post-timing steady round(s)")
    for label, rows in summary.items():
        if label == "phase_cuda_ms":
            print(f"  {label}:")
            for name, elapsed_ms in rows.items():
                print(f"    {elapsed_ms:8.3f} ms/round  {name}")
            continue
        print(f"  {label}:")
        for row in rows[:10]:
            print(f"    {row['ms']:8.3f} ms  {row['name']}")
    return summary


def _check_batched_sync_graph_against_eager(backend: Qwen36Backend, slots: list[int]) -> None:
    """Compare only the MTP sync graph with its identical eager B-wide body.

    This isolates the newly captured ``m+1`` teacher-forcing operation from
    the known target verify-vs-decode attention-plan divergence.  The probe
    overwrites only the already-disposable speculative MTP tail; the first
    real round immediately rewrites that tail from target hidden states.
    """
    engine = backend._mtp
    assert engine is not None
    sync = engine._batched_sync
    assert sync is not None and engine.cg_status.get("sync") == "captured"
    starts = [engine._sync_len[slot] for slot in slots]
    previous_lengths = [engine._caches[slot].seq_len for slot in slots]
    hidden_size = engine.model.model.hidden_size
    tokens = [[0] for _ in slots]
    hidden = torch.zeros(
        len(slots), 1, hidden_size, dtype=engine.dtype, device=engine.device
    )
    graph_logits, graph_hidden = sync.replay(slots, tokens, hidden, starts)
    torch.cuda.synchronize()
    graph_logits = graph_logits.clone()
    graph_hidden = graph_hidden.clone()
    for slot, start in zip(slots, starts, strict=True):
        engine._caches[slot].seq_len = start
    eager_logits, eager_hidden = sync.replay_eager(slots, tokens, hidden, starts)
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_logits, eager_logits, rtol=0, atol=0)
    torch.testing.assert_close(graph_hidden, eager_hidden, rtol=0, atol=0)
    for slot, previous_length in zip(slots, previous_lengths, strict=True):
        engine._caches[slot].seq_len = previous_length
    print("MTP sync CUDA Graph is bit-exact with its eager body")


def _check_mtp_prefix_reuse(backend: Qwen36Backend, tokenizer: AutoTokenizer) -> None:
    """Exercise a real block-boundary prefix hit through all MTP state.

    With two slots this also verifies the production persistent restore after
    the original slot has already begun an unrelated request.  The backbone
    paged KV, GDN checkpoint, and MTP causal KV must then come from the
    scratch arena, not from an idle source slot.  The follow-up verify round
    is deliberately essential: matching cache lengths alone would not prove
    that the restored MTP rows are usable.
    """
    engine = backend._mtp
    assert engine is not None
    slot = 0
    prefix_token = tokenizer(" prefix", add_special_tokens=False)["input_ids"][0]
    suffix_token = tokenizer(" continuation", add_special_tokens=False)["input_ids"][0]
    prefix = [prefix_token] * 64
    follow_up = [*prefix, suffix_token]

    backend.reset_slot(slot)
    first = backend.prefill_chunked_begin([slot], [prefix], params_per_slot={})
    assert first.done
    assert backend.slot_state(slot).kv_len == 64
    assert engine._sync_len[slot] == 64
    backend.reset_slot(slot)

    hit = backend.reconcile_prefix_hit(follow_up)
    assert hit.kv_hit == 64 and hit.state_hit == 64, hit
    resumed = backend.prefill_chunked_begin([slot], [follow_up], params_per_slot={})
    assert resumed.done
    assert backend.slot_state(slot).kv_len == len(follow_up)
    assert engine._sync_len[slot] == len(follow_up)
    assert engine._caches[slot].seq_len == len(follow_up) + K - 1

    decision = backend.mtp_verify_and_commit_batch(
        [slot],
        {slot: resumed.result[slot]["anchor"]},
        {slot: list(resumed.result[slot]["draft_tokens"])},
    )[slot]
    assert len(decision["next_draft_tokens"]) == K
    print(
        "MTP prefix cache restored target/GDN/MTP state at 64 tokens and "
        "completed a real verify+re-draft round"
    )

    if backend.num_slots > 1:
        target_slot = 1
        backend.reset_slot(slot)
        # Make the former source ineligible for ordinary cross-slot reuse.
        # A successful target restore below can therefore only use the
        # slot-independent scratch arena, not stale source-slot metadata.
        unrelated = [tokenizer(" unrelated", add_special_tokens=False)["input_ids"][0]]
        backend.prefill_chunked_begin([slot], [unrelated], params_per_slot={})
        assert backend.slot_state(slot).kv_len == 1
        backend.reset_slot(target_slot)
        remote_hit = backend.reconcile_prefix_hit(follow_up)
        remote = backend.prefill_chunked_begin(
            [target_slot], [follow_up], params_per_slot={}
        )
        assert remote.done
        assert remote_hit.kv_hit == 64 and remote_hit.state_hit == 64, remote_hit
        assert backend.stats["prefix_persistent_restores"] >= 2
        assert backend.stats["prefix_cross_slot_restores"] == 0
        assert backend.slot_state(target_slot).kv_len == len(follow_up)
        assert engine._sync_len[target_slot] == len(follow_up)
        assert engine._caches[target_slot].seq_len == len(follow_up) + K - 1
        remote_decision = backend.mtp_verify_and_commit_batch(
            [target_slot],
            {target_slot: remote.result[target_slot]["anchor"]},
            {target_slot: list(remote.result[target_slot]["draft_tokens"])},
        )[target_slot]
        assert len(remote_decision["next_draft_tokens"]) == K
        print(
            "MTP persistent scratch prefix restored target/GDN/MTP KV after the source "
            "slot was reused, then completed a real verify+re-draft round"
        )

        # Publish a second, different prefix into another scratch page and
        # restore it after overwriting its source too.  This proves the arena
        # is page-partitioned content storage, not a renamed one-entry slot.
        backend.reset_slot(slot)
        alternate_token = tokenizer(" alternate", add_special_tokens=False)["input_ids"][0]
        alternate = [alternate_token] * 64
        alternate_follow_up = [*alternate, suffix_token]
        second = backend.prefill_chunked_begin([slot], [alternate], params_per_slot={})
        assert second.done and backend.slot_state(slot).kv_len == 64
        backend.reset_slot(slot)
        assert len(backend._persistent_prefixes) >= 2  # noqa: SLF001 - real arena proof
        backend.prefill_chunked_begin([slot], [unrelated], params_per_slot={})
        backend.reset_slot(target_slot)
        alternate_hit = backend.reconcile_prefix_hit(alternate_follow_up)
        alternate_remote = backend.prefill_chunked_begin(
            [target_slot], [alternate_follow_up], params_per_slot={}
        )
        assert alternate_hit.kv_hit == 64 and alternate_hit.state_hit == 64, alternate_hit
        assert backend.stats["prefix_persistent_restores"] >= 3
        alternate_decision = backend.mtp_verify_and_commit_batch(
            [target_slot],
            {target_slot: alternate_remote.result[target_slot]["anchor"]},
            {target_slot: list(alternate_remote.result[target_slot]["draft_tokens"])},
        )[target_slot]
        assert len(alternate_decision["next_draft_tokens"]) == K
        print("MTP scratch arena retained and restored two independent prefixes")

    # Return to a cold state before the timed smoke and erase the retained
    # synthetic prefix so it cannot affect its prompt.
    for active_slot in range(backend.num_slots):
        backend.reset_slot(active_slot)
        backend.drop_prefix_cache(active_slot)
    for stats in (backend.stats, engine.stats):
        for name in stats:
            stats[name] = 0


def main() -> None:
    print(f"checkpoint: {MODEL_PATH}")
    model = load_qwen36_model(
        MODEL_PATH,
        device=DEVICE,
        dtype=torch.bfloat16,
        max_seq_len=MAX_SEQ_LEN,
        enable_mtp=True,
    )
    backend = Qwen36Backend(
        model,
        num_slots=NUM_SLOTS,
        max_seq_len=MAX_SEQ_LEN,
        block_size=64,
        device=DEVICE,
        dtype=torch.bfloat16,
        enable_prefix_cache=PREFIX_REUSE,
    )
    backend.enable_mtp(num_speculative_tokens=K, enable_resync=False)
    engine = backend._mtp
    graph_batch = backend.capture_decode_cuda_graph()
    print(f"backbone decode CUDA Graph captured up to batch={graph_batch}")
    print("MTP cg_status:", engine.cg_status)
    assert engine.cg_status.get("anchor") == "unused", (
        f"anchor must be folded into the K+1 verify graph: {engine.cg_status}"
    )
    assert engine.cg_status.get("draft") == "captured", (
        f"draft CUDA Graph did not capture: {engine.cg_status}"
    )
    assert engine.cg_status.get("verify") == "captured", (
        f"verify CUDA Graph did not capture: {engine.cg_status}"
    )
    assert engine.cg_status.get("sync") == "captured", (
        "every active batch width, including B=1, must capture the equal-m+1 "
        f"MTP synchronisation graph: {engine.cg_status}"
    )
    assert engine._anchor_cg is None
    assert engine._draft_cg is not None
    assert engine._verify_cg is not None

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if PREFIX_REUSE:
        _check_mtp_prefix_reuse(backend, tok)
    slots = list(range(NUM_SLOTS))
    prompt_ids_per_slot = []
    for slot in slots:
        prompt = EXACT_PROMPT if EXACT_PROMPT is not None else f"{PROMPT} Request {slot}."
        prompt_ids_per_slot.append(tok(prompt, return_tensors=None)["input_ids"])
    print(f"slots: {NUM_SLOTS}; prompt_ids: {[len(ids) for ids in prompt_ids_per_slot]}")

    _check_batched_sync_graph_against_eager(backend, slots)

    trace, elapsed_s = _run(backend, slots, prompt_ids_per_slot=prompt_ids_per_slot)
    _assert_raw_fp8_weights_remain_packed(backend)
    for i, decisions in enumerate(trace):
        for slot in slots:
            decision = decisions[slot]
            print(
                f"  round {i}, slot {slot}: accepted={decision['num_accepted']}/{K} "
                f"committed={decision['committed']}"
            )

    # Print replay counters before asserting them: capture health only says a
    # graph exists, while these prove which production body actually ran.
    print("MTP runtime stats:", engine.stats)
    print("backend MTP graph stats:", {
        name: value for name, value in backend.stats.items() if name.startswith("mtp_")
    })

    assert backend.stats["mtp_verify_graph_replays"] == NUM_ROUNDS, (
        "captured verify graph was not replayed by the real MTP round: "
        f"backend={backend.stats}, engine={engine.stats}"
    )
    assert backend.stats["mtp_verify_graph_slots"] == NUM_ROUNDS * NUM_SLOTS
    assert backend.stats["mtp_batched_verify_replays"] == (NUM_ROUNDS if NUM_SLOTS > 1 else 0)
    # Prefill's teacher-forced final chunk also uses the K-1 continuation
    # graph once per slot.  Rounds use one B-wide replay, so replay *count*
    # is ``slots + rounds`` while replayed slot rows are ``slots * (1 +
    # rounds)``.  Keeping those distinct is important: count proves the
    # graph lifecycle; rows prove the real batch width.
    assert backend.stats["mtp_draft_graph_replays"] == NUM_SLOTS + NUM_ROUNDS
    assert backend.stats["mtp_draft_graph_slots"] == (NUM_ROUNDS + 1) * NUM_SLOTS
    assert backend.stats["mtp_batched_draft_replays"] == (NUM_ROUNDS if NUM_SLOTS > 1 else 0)
    committed = sum(
        len(decision["committed"]) for decisions in trace for decision in decisions.values()
    )
    committed_per_slot = {
        slot: sum(len(decisions[slot]["committed"]) for decisions in trace) for slot in slots
    }
    profile_summary = _profile_steady_mtp_rounds(
        backend,
        slots,
        {slot: trace[-1][slot]["next_anchor"] for slot in slots},
        {slot: list(trace[-1][slot]["next_draft_tokens"]) for slot in slots},
    )
    print(
        f"MTP steady rounds: {committed} committed tokens in {elapsed_s:.3f}s "
        f"= {committed / elapsed_s:.2f} committed tok/s "
        f"({elapsed_s * 1000 / NUM_ROUNDS:.2f} ms/round)"
    )
    baseline_tokens, baseline_s = _run_plain_decode_baseline(
        backend,
        slots,
        prompt_ids_per_slot=prompt_ids_per_slot,
        target_committed_per_slot=committed_per_slot,
    )
    baseline_committed = sum(len(tokens) for tokens in baseline_tokens.values())
    assert baseline_committed == committed, (
        "plain decode baseline must match the MTP committed-token budget: "
        f"mtp={committed}, baseline={baseline_committed}, per_slot={committed_per_slot}"
    )
    mtp_tokens = {
        slot: [
            token
            for decisions in trace
            for token in decisions[slot]["committed"]
        ]
        for slot in slots
    }
    # The target's fixed K+1 verify graph and the plain one-token decode
    # graph intentionally have different paged-attention reduction plans.
    # On this model those numerically valid plans can eventually flip a
    # near-tie after MoE amplification; treating a later token mismatch as
    # a graph-addressing failure would repeat the false diagnosis recorded
    # in notes/2026-08-02-eager-verify-cg-verify-divergence.md.  Preserve
    # the comparison as evidence, while the real graph correctness gates
    # remain replay/addressing assertions here plus the calibrated gap-error
    # criterion in docs/b1-correctness-criterion.md.
    token_stream_match = baseline_tokens == mtp_tokens
    if not token_stream_match:
        print(
            "MTP/plain token streams diverged after the shared prefix; "
            "recording it for gap-error diagnosis rather than mislabelling "
            "a distinct attention reduction plan as an address failure."
        )
    assert backend.stats["mtp_batched_sync_replays"] >= 1, (
        "MTP synchronisation graph was captured but no equal-m+1 group "
        f"replayed it: {backend.stats}"
    )
    baseline_tok_s = baseline_committed / baseline_s
    mtp_tok_s = committed / elapsed_s
    print(
        f"Plain decode baseline: {baseline_committed} committed tokens in {baseline_s:.3f}s "
        f"= {baseline_tok_s:.2f} committed tok/s"
    )
    print(f"MTP / plain decode throughput ratio: {mtp_tok_s / baseline_tok_s:.2f}x")
    result = {
        "checkpoint": str(MODEL_PATH),
        "slots": NUM_SLOTS,
        "k": K,
        "rounds": NUM_ROUNDS,
        "prefix_reuse_checked": PREFIX_REUSE,
        "committed_tokens": committed,
        "elapsed_s": elapsed_s,
        "committed_tokens_per_s": mtp_tok_s,
        "ms_per_round": elapsed_s * 1000 / NUM_ROUNDS,
        "plain_decode_elapsed_s": baseline_s,
        "plain_decode_committed_tokens_per_s": baseline_tok_s,
        "mtp_vs_plain_decode_ratio": mtp_tok_s / baseline_tok_s,
        "plain_decode_token_stream_match": token_stream_match,
        "mtp_tokens": mtp_tokens,
        "plain_decode_tokens": baseline_tokens,
        "cg_status": engine.cg_status,
        "backend_stats": backend.stats,
        "profile": profile_summary,
    }
    if RESULT_PATH:
        Path(RESULT_PATH).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote verification evidence: {RESULT_PATH}")
    print(f"PASS: MTP draft/verify graphs captured and verify replayed {NUM_ROUNDS} time(s), K={K}")


if __name__ == "__main__":
    main()
