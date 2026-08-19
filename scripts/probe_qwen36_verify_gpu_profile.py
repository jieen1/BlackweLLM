"""Per-kernel CUDA profile of the Qwen3.6 MTP verify body at 128K/c4.

Production geometry: 4 slots x 131072-token prefix, MTP K=3, verify query
``[4, anchor+3]`` against the pooled full-attention KV and GDN rows.  The
verify graph's captured body is :meth:`Qwen36Model.verify_batch`; this probe
stages the graph-owned descriptor with the same :meth:`_fill` the replay
uses, then runs the body eagerly under ``torch.profiler`` so every kernel is
individually visible (CUDA-graph replay flattens them into one launch node).
Kernel execution times are the same eager-vs-captured; only launch overhead
differs, which is exactly what the graph is for.

Usage (mirror the serving env, single process):
    source /tmp/qwen36_server_env.sh
    export QSR_SERVER_CAPACITY=4 QSR_SERVER_NUM_SLOTS=5
    /home/bot/.venvs/torch-nightly/bin/python scripts/probe_qwen36_verify_gpu_profile.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _ROOT)
import runtime  # noqa: E402

assert runtime.__file__.startswith(_ROOT), f"shadowed import: {runtime.__file__}"

import torch  # noqa: E402

from runtime.backends.qwen36 import Qwen36Backend  # noqa: E402
from runtime.checkpoints import standard_checkpoint_path  # noqa: E402
from runtime.model_loading import load_qwen36_model  # noqa: E402

MAX_SEQ_LEN = int(os.environ.get("QSR_PROBE_MAX_SEQ_LEN", "262144"))
NUM_SLOTS = int(os.environ.get("QSR_PROBE_NUM_SLOTS", "5"))
CTX_LEN = int(os.environ.get("QSR_PROBE_CTX_LEN", "131072"))
ACTIVE = int(os.environ.get("QSR_PROBE_ACTIVE", "4"))
K = int(os.environ.get("QSR_PROBE_K", "3"))
WARM_ROUNDS = int(os.environ.get("QSR_PROBE_WARM_ROUNDS", "5"))
PROFILE_ITERS = int(os.environ.get("QSR_PROBE_PROFILE_ITERS", "10"))


def _ramp_prompt(length: int) -> list[int]:
    base = list(range(1, 257))
    prompt = (base * (length // len(base) + 1))[:length]
    return prompt


def main() -> None:
    torch.set_grad_enabled(False)
    model = load_qwen36_model(
        standard_checkpoint_path(),
        device="cuda",
        dtype=torch.bfloat16,
        max_seq_len=MAX_SEQ_LEN,
        enable_mtp=True,
    )
    backend = Qwen36Backend(
        model,
        num_slots=NUM_SLOTS,
        max_seq_len=MAX_SEQ_LEN,
        block_size=16,
        device="cuda",
        dtype=torch.bfloat16,
        enable_prefix_cache=True,
    )
    backend.enable_mtp(num_speculative_tokens=K)
    print("MTP cg_status:", backend._mtp.cg_status)  # noqa: SLF001

    slots = list(range(ACTIVE))
    prompts = [_ramp_prompt(CTX_LEN) for _ in slots]
    print(f"prefilling {ACTIVE} slots x {CTX_LEN} tokens ...", flush=True)
    # Match the server's own ``_prefill_chunk_tokens()`` (8192) instead of
    # the protocol default of 512: the probe prefills 4 x 128K and a
    # 512-token chunk loop issues ~1000 separate forwards (measured
    # 2026-08-06: >25 min just to prefill).
    state = backend.prefill_chunked_begin(
        slots, prompts, chunk_size=8192, params_per_slot={}
    )
    while not state.done:
        backend.prefill_chunked_step(state)
    anchors = {slot: state.result[slot]["anchor"] for slot in slots}
    drafts = {
        slot: (
            state.result[slot]["draft_tokens"]
            if isinstance(state.result[slot]["draft_tokens"], torch.Tensor)
            else list(state.result[slot]["draft_tokens"])
        )
        for slot in slots
    }
    torch.cuda.synchronize()

    print(f"warming {WARM_ROUNDS} MTP rounds ...", flush=True)
    for _ in range(WARM_ROUNDS):
        result = backend.mtp_verify_and_commit_batch(slots, anchors, drafts)
        anchors = {slot: result[slot]["next_anchor"] for slot in slots}
        drafts = {
            slot: (
                result[slot]["next_draft_tokens"]
                if isinstance(result[slot]["next_draft_tokens"], torch.Tensor)
                else list(result[slot]["next_draft_tokens"])
            )
            for slot in slots
        }
    torch.cuda.synchronize()

    engine = backend._mtp  # noqa: SLF001
    verify_cg = engine._verify_cg  # noqa: SLF001
    assert verify_cg is not None and verify_cg._captured  # noqa: SLF001
    past_lens = [backend.slot_state(slot).kv_len for slot in slots]
    tokens = torch.tensor(
        [[1000 + slot, 1001 + slot, 1002 + slot, 1003 + slot] for slot in slots],
        dtype=torch.long,
        device="cuda",
    )
    verify_cg._fill(slots, tokens, past_lens)  # noqa: SLF001
    descriptor = verify_cg._batches[ACTIVE]  # noqa: SLF001
    torch.cuda.synchronize()

    print(
        f"profiling eager verify_batch [{ACTIVE}, {K+1}] @KV={CTX_LEN} "
        f"x {PROFILE_ITERS} iters ...",
        flush=True,
    )
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for _ in range(PROFILE_ITERS):
            hidden = model.verify_batch(descriptor)
            _ = model.compute_logits(hidden)
    trace_path = os.environ.get("QSR_PROBE_TRACE")
    if trace_path:
        prof.export_chrome_trace(trace_path)
        print(f"wrote chrome trace: {trace_path}")
    ev1.record()
    ev1.synchronize()
    print(f"verify_batch+lm_head wall/iter: {ev0.elapsed_time(ev1) / PROFILE_ITERS:.2f} ms")

    events = prof.key_averages()
    cuda = sorted(
        (
            (event.key, event.self_device_time_total / 1000.0, event.count)
            for event in events
            if event.self_device_time_total > 0
        ),
        key=lambda row: row[1],
        reverse=True,
    )
    total = sum(row[1] for row in cuda)
    print(
        f"\ntop-40 CUDA kernels (self device ms over {PROFILE_ITERS} iters, "
        f"total {total:.1f} ms):"
    )
    print(f"{'kernel':<72} {'ms':>10} {'%':>6} {'calls':>7}")
    for name, ms, count in cuda[:40]:
        print(f"{name:<72} {ms:10.2f} {100 * ms / total:6.1f} {count:7d}")

    cpu = sorted(
        (
            (event.key, event.self_cpu_time_total / 1000.0, event.count)
            for event in events
            if event.self_cpu_time_total > 0
        ),
        key=lambda row: row[1],
        reverse=True,
    )
    print("\ntop-25 CPU ops (self ms):")
    for name, ms, count in cpu[:25]:
        print(f"{name:<72} {ms:10.2f} {count:7d}")


if __name__ == "__main__":
    main()
