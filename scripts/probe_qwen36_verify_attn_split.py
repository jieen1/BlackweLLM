#!/usr/bin/env python3
"""Probe: what do Qwen3.6 MTP-verify paged-attention plans actually do at
128K, and what does forcing a different split cost/save -- measured, not
assumed.

Background: the 2026-08-05 nsys trace attributes ~37.5% of GPU time to
sparkinfer paged attention during 128K MTP rounds. Qwen3.6 has only 4 KV
heads, so an unsplit call launches few CTAs on a 188-SM card. Planner code
reading says verify *defaults* force_split_kv=True but budget-constrained
binary search may still land on one chunk; decode (draft) disables split
unless forced. This probe measures the real plans and kernel timings
instead of trusting either the code reading or the historical 0.988 ms
split-KV number (different kernel generation, Laguna geometry).

Guardrails honored: standalone single-layer-style probe (shared-card
convention, same shape as scripts/b3_probe_gdn_spec_rollback.py); asserts
this worktree's code is imported; takes the GPU lock; CUDA-event timing
after warmup (each kernel variant pays its one-time CuTe compile during
warmup, exactly as production does); split-vs-unsplit output cross-check
before trusting any speed number.

Run: /home/bot/.venvs/torch-nightly/bin/python scripts/probe_qwen36_verify_attn_split.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, _ROOT)

import runtime  # noqa: E402

assert runtime.__file__.startswith(_ROOT), (
    f"editable install shadowed the worktree: runtime.__file__={runtime.__file__}"
)

import torch  # noqa: E402

assert torch.cuda.is_available(), "probe requires the GPU"

from runtime.backends._sparkinfer_import import ensure_sparkinfer_path  # noqa: E402

ensure_sparkinfer_path()

from b12x.attention.paged._forward import paged_attention_forward  # noqa: E402
from b12x.attention.paged._scratch import build_paged_attention_binding  # noqa: E402
from b12x.attention.paged.planner import create_paged_plan  # noqa: E402
from b12x.attention.paged.workspace import PagedAttentionWorkspace  # noqa: E402

# ---- real Qwen3.6-27B full-attention geometry (verified from config.json) ----
NUM_Q_HEADS = 24
NUM_KV_HEADS = 4
HEAD_DIM = 256
# sparkinfer's paged backend only accepts page 64/128; the runtime maps
# its 16-token storage blocks onto 128-token attention pages
# (_PAGED_ATTENTION_PAGE_SIZE, runtime/model/qwen36_model.py).
PAGE_SIZE = 128
NUM_PAGES = 1024  # 128K tokens per slot
MAX_KV = NUM_PAGES * PAGE_SIZE
K = 3  # production MTP depth
DEVICE = torch.device("cuda")
LOCK_NAME = "verify-attn-split-probe"
torch.manual_seed(7)
torch.set_grad_enabled(False)


def _gpu_lock(action: str) -> bool:
    """Serialize the probe when the optional local lock helper exists."""

    lock_tool = Path("/tmp/gpu_lock.sh")
    if not lock_tool.exists():
        print("[probe] /tmp/gpu_lock.sh unavailable; proceeding on an idle GPU", flush=True)
        return action == "acquire"
    result = subprocess.run(
        [str(lock_tool), action, LOCK_NAME, "120"],
        capture_output=True,
        text=True,
    )
    print(result.stdout.strip() or result.stderr.strip(), flush=True)
    return result.returncode == 0


def describe(plan, label: str) -> None:
    print(
        f"    [{label}] split_kv={plan.split_kv} new_batch_size={plan.new_batch_size} "
        f"kv_chunk_size={getattr(plan, 'kv_chunk_size', '?')} "
        f"partial_rows={plan.total_num_partial_rows} cta_tile_q={plan.cta_tile_q}"
    )


def run_once(ws, plan, q, kc, vc, pt, cs, cu, out):
    ws._ensure_capacity(plan)
    ws._copy_runtime_metadata(pt, cs, cu)
    ws._copy_plan_metadata(plan)
    ws._plan = plan
    binding = build_paged_attention_binding(
        scratch=ws,
        q=q,
        k_cache=kc,
        v_cache=vc,
        output=out,
        k_descale=torch.ones(1, device=DEVICE),
        v_descale=torch.ones(1, device=DEVICE),
    )
    paged_attention_forward(binding=binding)


def time_plan(make_plan, q, kc, vc, pt, cs, cu, out, iters=50):
    plan = make_plan()
    ws = PagedAttentionWorkspace.for_fixed_capacity(
        mode="verify",
        device=DEVICE,
        dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        num_q_heads=NUM_Q_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim_qk=HEAD_DIM,
        head_dim_vo=HEAD_DIM,
        page_size=PAGE_SIZE,
        max_total_q=q.shape[0],
        max_batch=pt.shape[0],
        max_page_table_width=pt.shape[1],
        max_work_items=max(plan.new_batch_size, 1),
        max_partial_rows=max(plan.total_num_partial_rows, 0),
        num_cache_pages=NUM_PAGES,
        use_cuda_graph=False,
    )
    for _ in range(5):  # warmup: pays the one-time compile for this variant
        run_once(ws, plan, q, kc, vc, pt, cs, cu, out)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        run_once(ws, plan, q, kc, vc, pt, cs, cu, out)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters, out.detach().clone(), plan


def main() -> None:
    locked = _gpu_lock("acquire")
    try:
        kc = (torch.randn(NUM_PAGES, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, device=DEVICE) * 0.5).to(
            torch.float8_e4m3fn
        )
        vc = (torch.randn(NUM_PAGES, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, device=DEVICE) * 0.5).to(
            torch.float8_e4m3fn
        )
        for batch, qo_len in ((3, K + 1), (1, K + 1), (3, 1)):
            mode = "verify" if qo_len > 1 else "decode"
            print(f"\n=== batch={batch} qo_len={qo_len} mode={mode} kv={MAX_KV} ===")
            total_q = batch * qo_len
            q = torch.randn(total_q, NUM_Q_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
            pt = (
                torch.arange(NUM_PAGES, dtype=torch.int32, device=DEVICE)
                .unsqueeze(0)
                .repeat(batch, 1)
            )
            cs = torch.tensor([MAX_KV] * batch, dtype=torch.int32, device=DEVICE)
            cu = torch.tensor(
                [qo_len * i for i in range(batch + 1)], dtype=torch.int32, device=DEVICE
            )
            out = torch.empty(total_q, NUM_Q_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)

            def eager_default():
                return create_paged_plan(
                    q, kc, vc, pt, cs, cu, mode=mode, enable_cuda_graph=False, window_left=-1
                )

            try:
                p = eager_default()
                describe(p, "eager default")
                ms_a, out_a, _ = time_plan(eager_default, q, kc, vc, pt, cs, cu, out)
                print(f"    [eager default] {ms_a:.3f} ms/call")
            except Exception as e:  # noqa: BLE001
                print(f"    [eager default] FAILED: {type(e).__name__}: {e}")
                ms_a, out_a = None, None

            # Graph-style plans with generous work-item capacity: the shape
            # of plan the MTP verify GRAPH capture can reach when its
            # workspace capacity admits chunks.  Compare the current
            # split-KV policy against a true unsplit plan; the production
            # adapter exposes the same switch through
            # QSR_QWEN36_VERIFY_FLASHINFER_DISABLE_SPLIT_KV.
            for disable_split in (False, True):
                for budget in (64, 192):

                    def graph_plan(b=budget, unsplit=disable_split):
                        return create_paged_plan(
                            q,
                            kc,
                            vc,
                            pt,
                            cs,
                            cu,
                            mode=mode,
                            disable_split_kv=unsplit,
                            enable_cuda_graph=True,
                            graph_chunk_policy=True,
                            max_batch_size_if_split=b,
                            window_left=-1,
                        )

                    try:
                        p = graph_plan()
                        label = f"graph unsplit={disable_split} cap={budget}"
                        describe(p, label)
                        ms_b, out_b, _ = time_plan(graph_plan, q, kc, vc, pt, cs, cu, out)
                        print(f"    [{label}] {ms_b:.3f} ms/call")
                        if out_a is not None:
                            diff = (out_a.float() - out_b.float()).abs().max().item()
                            denom = out_a.float().abs().max().item()
                            rel = diff / max(denom, 1e-9)
                            print(
                                f"    [cross-check vs eager] max_abs={diff:.3e} rel={rel:.3e}"
                            )
                        if ms_a:
                            print(f"    [speedup vs eager] {ms_a / ms_b:.2f}x")
                    except Exception as e:  # noqa: BLE001
                        print(f"    [{label}] FAILED: {type(e).__name__}: {e}")
    finally:
        if locked:
            _gpu_lock("release")


if __name__ == "__main__":
    main()
