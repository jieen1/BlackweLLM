"""Real numerical eager-vs-graph parity check for
``CapturedBatchDecodeGraph`` (qo_len=1) -- NOT signal-probe. Built per an
explicit coordinator requirement, following an independent (Codex)
correctness review that found the earlier signal-probe-only verification
insufficient to catch GDN state pollution: this project's own identity-
recall task is plausibly dominated by full-attention layers' copy
mechanism, masking a GDN-specific perturbation a more GDN-sensitive task
might not tolerate.

This test drives the SAME real input (identical prompt, identical
kv_len, identical draft token) through TWO independent, identically
prefilled physical-slot groups:
  - ``eager_slots``: the already-verified single-call
    ``DirectModelRunner._forward_batch`` path (no CUDA graph at all).
  - ``graph_slots``: ``CapturedBatchDecodeGraph.replay()``.

and compares, directly and numerically:
  1. Full output logits -- max abs diff, cosine similarity, and
     ``torch.allclose`` at a generous-but-meaningful tolerance (eager
     uses a per-call, TIGHT kv_split_size; the graph uses a build-time
     FIXED, larger one -- see build_attention_metadata_batch's docstring
     -- so a different attention split/merge reduction order is
     EXPECTED to introduce some floating-point noise, the same
     "different shape/path -> different rounding" phenomenon already
     established for batch-composition effects. The check here is that
     the difference stays SMALL, not that it's exactly zero).
  2. Top-1/top-5 predicted token id agreement.
  3. The GDN recurrent state tensors themselves (``conv_state``/
     ``ssm_state``, read directly out of ``runner.kv_caches`` for the
     physical row each group's slot maps to) -- for a SINGLE token at
     IDENTICAL input, GDN's own math does not depend on attention's
     kv_split_size at all, so eager and graph should agree here far
     more tightly than the logits comparison. This is the single most
     direct test for the "capture's warmup polluted the real GDN state"
     bug class this round's fix targets: a real state-neutral capture
     should show these tensors bytewise-identical (or extremely close);
     a regression would show them meaningfully diverged even though the
     signal-probe-level text output might still look fine.

Usage:
    python -m benchmarks.cudagraph_eager_parity_check
    python -m benchmarks.cudagraph_eager_parity_check --repeat 3
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_KERNEL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SM120_VLLM_INTEGRATION = "/home/bot/project/sm120-flash-attention/vllm_integration"
MODEL = "unsloth/Qwen3.6-27B-NVFP4"
PROMPT = "The capital of France is"

# Tolerances: generous relative to typical BF16/FP8 noise (this project's
# own established atol=1e-2/rtol=1e-2 convention for BF16 numerical
# checks), but still tight enough to catch a real several-extra-step
# state perturbation, which would show up as a much larger, obviously
# not-noise-shaped divergence.
LOGITS_ATOL = 0.5
LOGITS_RTOL = 0.05
GDN_STATE_ATOL = 1e-2
GDN_STATE_RTOL = 1e-2

# P3.3a INV5 hit-table case tolerances. The hit slot's decode runs at a larger
# kv_len than the base case's 5-token prompt, so the eager path's per-call TIGHT
# kv_split_size (max_num_splits=1) and the graph's build-time FIXED one
# (ceil(capacity/64)=32 => several splits) genuinely differ in reduction order
# -- the SAME documented fp8 attention-split non-associativity the base case's
# docstring calls out (negligible at its tiny kv_len, visible here). That noise
# is NOT an INV5 violation: INV5 is "the captured graph replays a non-contiguous
# (hit-populated) block table correctly", whose DECISIVE proof is GDN layer 0
# BYTEWISE-exact (a wrong-block read would corrupt layer 0 first, far above
# noise). The deeper-stack gate is therefore near-tie (generous enough for the
# split noise, tight enough that a real wrong-block/pollution diff -- tens of
# units -- still fails), and the logits gate is cosine-near-1.0 + top-1 match.
HIT_LOGITS_COSINE_MIN = 0.99
# Full-stack near-tie tolerance: admits the accumulated eager-vs-graph
# attention-split fp8 noise (observed max conv diff ~2.0 at the deepest of 48
# layers at this decode kv_len) while still failing a REAL wrong-block-read or
# capture-warmup-pollution diff, which is tens of units (the benign dead-spec-row
# artifact alone is ~46-57; notes/2026-07-20-cold-prefill-rootcause-plan.md sec 3).
# The DECISIVE INV5 proof is gdn_layer0_exact (bytewise), not this loose gate.
HIT_GDN_FULL_STACK_ATOL = 5.0
HIT_GDN_FULL_STACK_RTOL = 0.05

# P3.3a INV5: a longer prompt so a cold-prefilled producer publishes a
# MULTI-BLOCK cached prefix; a hit slot then restores those shared [0, L)
# block ids and appends a freshly-allocated private tail -- a genuinely
# NON-CONTIGUOUS block table (the shared ids are the producer's old physical
# blocks, not the hit slot's own contiguous range) for the captured graph to
# replay through. K=3 MTP so mtp_prefill_with_cache (the production populate /
# hit path) is configured.
HIT_PROMPT = (
    "The capital of France is Paris. The capital of Germany is Berlin. "
    "The capital of Italy is Rome. The capital of Spain is Madrid. "
    "The capital of Portugal is Lisbon. The capital of Belgium is Brussels. "
    "The capital of the Netherlands is Amsterdam. The capital of Austria is Vienna. "
    "The capital of Switzerland is Bern. The capital of Sweden is Stockholm. "
    "The capital of Norway is Oslo. The capital of Denmark is Copenhagen."
)
HIT_SPECULATIVE_CONFIG = {
    "method": "mtp",
    "num_speculative_tokens": 3,
    "attention_backend": "CUSTOM",
}


def _run_once() -> dict:
    import torch

    sys.path.insert(0, SM120_VLLM_INTEGRATION)
    import register_sm120_backend  # noqa: F401
    from transformers import AutoTokenizer

    from oracle.qwen36_vllm.direct_model_runner import (
        CapturedBatchDecodeGraph,
        DirectModelRunner,
        _physical_slot,
        build_vllm_config,
    )

    vllm_config = build_vllm_config(
        model=MODEL, kv_cache_dtype="fp8_e4m3", max_model_len=2048, gpu_memory_utilization=0.5
    )
    block_size, blocks_per_slot = 16, 128
    batch = 2
    # eager_slots (batch) + graph_slots (batch) + CapturedBatchDecodeGraph's
    # own internally reserved warmup slots (batch) = 3*batch.
    runner = DirectModelRunner(
        vllm_config, num_slots=3 * batch, block_size=block_size, blocks_per_slot=blocks_per_slot
    )
    tok = AutoTokenizer.from_pretrained(MODEL)
    eager_slots = list(range(batch))
    graph_slots = list(range(batch, 2 * batch))

    prompt_ids = tok.encode(PROMPT, add_special_tokens=False)

    eager_next = [runner.prefill(s, prompt_ids) for s in eager_slots]
    graph_next = [runner.prefill(s, prompt_ids) for s in graph_slots]
    if eager_next != graph_next:
        return {"passed": False, "error": "prefill greedy tokens diverged between eager_slots and graph_slots"}

    eager_kv = [runner.slot_kv_len[s] for s in eager_slots]
    graph_kv = [runner.slot_kv_len[s] for s in graph_slots]
    if eager_kv != graph_kv:
        return {"passed": False, "error": "kv_len diverged between eager_slots and graph_slots after prefill"}

    # Identical draft token for both groups (the greedy next token from
    # the identical prefill).
    draft_tokens_eager = list(eager_next)
    draft_tokens_graph = list(graph_next)

    # --- Eager path (already-verified, no CUDA graph at all). ---
    eager_logits = runner._forward_batch(eager_slots, draft_tokens_eager, eager_kv).clone()

    # --- Graph path: capture (self-contained, uses its own reserved
    # warmup slots -- never touches eager_slots or graph_slots) then
    # replay at the IDENTICAL input. ---
    graph = CapturedBatchDecodeGraph(runner, batch_size=batch, qo_len=1)
    graph.capture()
    graph_logits = graph.replay(graph_slots, draft_tokens_graph, graph_kv).clone()

    # --- 1. Logits comparison. ---
    max_abs_diff = (eager_logits.float() - graph_logits.float()).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        eager_logits.float().flatten(), graph_logits.float().flatten(), dim=0
    ).item()
    allclose = torch.allclose(eager_logits.float(), graph_logits.float(), atol=LOGITS_ATOL, rtol=LOGITS_RTOL)

    eager_top5 = eager_logits.float().topk(5, dim=-1)
    graph_top5 = graph_logits.float().topk(5, dim=-1)
    top1_match = [
        int(eager_top5.indices[i, 0].item()) == int(graph_top5.indices[i, 0].item()) for i in range(batch)
    ]
    top5_overlap = [
        len(set(eager_top5.indices[i].tolist()) & set(graph_top5.indices[i].tolist())) for i in range(batch)
    ]

    logits_result = {
        "max_abs_diff": max_abs_diff,
        "cosine_similarity": cos_sim,
        "allclose": allclose,
        "top1_match": top1_match,
        "top5_overlap_count": top5_overlap,
    }

    # --- 2. Direct GDN state tensor comparison -- the decisive check for
    # capture-warmup state pollution. Compare EVERY GDN layer's
    # conv_state/ssm_state row for the corresponding eager/graph physical
    # slots (both received the IDENTICAL single-token update above). ---
    gdn_state_results = []
    all_gdn_close = True
    for layer_name in runner.gdn_layer_names:
        conv_state, ssm_state = runner.kv_caches[layer_name]
        for i in range(batch):
            eager_phys = _physical_slot(eager_slots[i])
            graph_phys = _physical_slot(graph_slots[i])
            conv_e = conv_state[eager_phys].float()
            conv_g = conv_state[graph_phys].float()
            ssm_e = ssm_state[eager_phys].float()
            ssm_g = ssm_state[graph_phys].float()
            conv_close = torch.allclose(conv_e, conv_g, atol=GDN_STATE_ATOL, rtol=GDN_STATE_RTOL)
            ssm_close = torch.allclose(ssm_e, ssm_g, atol=GDN_STATE_ATOL, rtol=GDN_STATE_RTOL)
            conv_max_diff = (conv_e - conv_g).abs().max().item()
            ssm_max_diff = (ssm_e - ssm_g).abs().max().item()
            gdn_state_results.append(
                {
                    "layer": layer_name,
                    "pair_idx": i,
                    "conv_close": conv_close,
                    "conv_max_diff": conv_max_diff,
                    "ssm_close": ssm_close,
                    "ssm_max_diff": ssm_max_diff,
                }
            )
            if not (conv_close and ssm_close):
                all_gdn_close = False

    passed = bool(allclose) and all(top1_match) and all_gdn_close
    return {
        "passed": passed,
        "logits": logits_result,
        "gdn_states_all_close": all_gdn_close,
        "gdn_state_sample": gdn_state_results[:4],  # first few layers, avoid huge output
        "num_gdn_layers_checked": len(runner.gdn_layer_names),
    }


def _run_hit_table_once() -> dict:
    """INV5 eager-vs-graph parity over a HIT-POPULATED, NON-CONTIGUOUS block
    table. A producer slot cold-prefills ``HIT_PROMPT`` via the production
    ``mtp_prefill_with_cache`` (populating the persistent cache: the [0, L)
    attention blocks + a completion GDN checkpoint at L); after a ``reset_slot``
    (R10 -- blocks stay hit-able at ref_cnt == 0) two fresh slots HIT that
    prefix (restore the shared [0, L) block ids + GDN checkpoint, continue-
    prefill the short suffix). Each hit slot's block table is therefore the
    producer's old [0, L) physical block ids followed by a freshly-allocated
    private tail -- NOT a contiguous range from the slot's own physical slot.
    One identical decode token is then driven through the eager
    ``_forward_batch`` path (eager slot) and ``CapturedBatchDecodeGraph.replay``
    (graph slot); logits + every GDN layer's conv/ssm row must agree tightly.
    This proves the captured decode graph replays correctly regardless of how
    many of a slot's blocks came from the cache (``_fill_buffers`` sources page
    ids from ``runner.block_table[slot]`` every replay; no id is baked in)."""
    import torch

    sys.path.insert(0, SM120_VLLM_INTEGRATION)
    import register_sm120_backend  # noqa: F401
    from transformers import AutoTokenizer

    from oracle.qwen36_vllm.direct_model_runner import (
        CapturedBatchDecodeGraph,
        DirectModelRunner,
        _physical_slot,
        build_vllm_config,
    )

    vllm_config = build_vllm_config(
        model=MODEL,
        kv_cache_dtype="fp8_e4m3",
        max_model_len=2048,
        gpu_memory_utilization=0.55,
        speculative_config=HIT_SPECULATIVE_CONFIG,
    )
    block_size, blocks_per_slot = 16, 128
    batch = 1
    # producer (1) + eager (batch) + graph (batch) + the graph's own reserved
    # warmup (batch, the LAST slot) + 1 spare = 1 + 3*batch + 1.
    runner = DirectModelRunner(
        vllm_config,
        num_slots=1 + 3 * batch + 1,
        block_size=block_size,
        blocks_per_slot=blocks_per_slot,
        enable_block_table=True,
        enable_prefix_cache=True,
        enable_persistent_prefix_cache=True,
        enable_cudagraph=False,  # capture the graph manually below
    )
    tok = AutoTokenizer.from_pretrained(MODEL)
    producer_slot = 0
    eager_slots = [1]
    graph_slots = [2]

    prompt_ids = tok.encode(HIT_PROMPT, add_special_tokens=False)

    # --- Produce a cached prefix (cold), then reset the producer (R10: the
    #     published [0, L) blocks stay hit-able at ref_cnt == 0). ---
    runner.mtp_prefill_with_cache([producer_slot], [prompt_ids])
    L = runner.reconcile_prefix_hit(prompt_ids)
    if L < block_size:
        return {"passed": False, "error": f"producer did not publish a full cached block (L={L})"}
    num_L_blocks = L // block_size
    cached_ids = list(runner.block_table[producer_slot][:num_L_blocks])
    runner.reset_slot(producer_slot)

    # --- Hit-populate the eager and graph slots from the cached prefix. ---
    eager_pr = runner.mtp_prefill_with_cache(eager_slots, [prompt_ids])
    graph_pr = runner.mtp_prefill_with_cache(graph_slots, [prompt_ids])
    eager_anchor = eager_pr[eager_slots[0]]["anchor"]
    graph_anchor = graph_pr[graph_slots[0]]["anchor"]
    if eager_anchor != graph_anchor:
        return {
            "passed": False,
            "error": f"hit anchors diverged eager={eager_anchor} graph={graph_anchor}",
        }

    # The hit table must be genuinely NON-CONTIGUOUS: its [0, L) page ids are
    # the producer's cached blocks (shared), and the slot's own "natural"
    # contiguous range (from its physical slot) must NOT already equal the
    # table -- otherwise this test would silently reduce to the contiguous
    # case the base ``_run_once`` already covers.
    eager_table = list(runner.block_table[eager_slots[0]])
    eager_natural = list(
        range(_physical_slot(eager_slots[0]) * blocks_per_slot,
              _physical_slot(eager_slots[0]) * blocks_per_slot + len(eager_table))
    )
    non_contiguous = (
        eager_table[:num_L_blocks] == cached_ids and eager_table != eager_natural
    )
    if not non_contiguous:
        return {
            "passed": False,
            "error": "hit table is not the expected non-contiguous shared+tail layout",
            "eager_table": eager_table,
            "cached_ids": cached_ids,
        }

    eager_kv = [runner.slot_kv_len[s] for s in eager_slots]
    graph_kv = [runner.slot_kv_len[s] for s in graph_slots]
    if eager_kv != graph_kv:
        return {"passed": False, "error": "kv_len diverged between eager/graph hit slots"}

    draft_token = eager_anchor  # identical greedy next token for both groups

    # --- Eager decode (no graph) on the hit-populated, non-contiguous table. ---
    eager_logits = runner._forward_batch(eager_slots, [draft_token], eager_kv).clone()

    # --- Graph decode: capture (own reserved warmup slot) then replay at the
    #     IDENTICAL input on the graph slot's hit-populated table. ---
    graph = CapturedBatchDecodeGraph(runner, batch_size=batch, qo_len=1)
    graph.capture()
    graph_logits = graph.replay(graph_slots, [draft_token], graph_kv).clone()

    max_abs_diff = (eager_logits.float() - graph_logits.float()).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        eager_logits.float().flatten(), graph_logits.float().flatten(), dim=0
    ).item()
    eager_top1 = int(eager_logits.float().argmax(dim=-1)[0].item())
    graph_top1 = int(graph_logits.float().argmax(dim=-1)[0].item())
    top1_match = eager_top1 == graph_top1

    # --- Direct GDN state comparison (every layer). Both hit slots are fresh
    #     (no prior spec-decode), so their conv spec-extension rows are
    #     identically zero; a full-row compare is valid here. (For a slot-reuse-
    #     after-decode comparison one would mask to the committed rows only --
    #     notes/2026-07-20-cold-prefill-rootcause-plan.md sec 3 -- but that
    #     artifact cannot arise on these never-decoded slots.)
    #
    #     Layer 0 is the DECISIVE INV5 addressing proof: it must be BYTEWISE
    #     exact (a wrong-block read of the non-contiguous table would corrupt it
    #     first, far above fp8 noise). The full stack is gated near-tie at a
    #     tolerance that admits the documented eager-vs-graph attention-split
    #     noise but still fails a real wrong-block/pollution diff (see the
    #     HIT_GDN_FULL_STACK_* constants). ---
    gdn_state_results = []
    layer0_exact = False
    full_stack_near_tie = True
    max_conv_diff = 0.0
    max_ssm_diff = 0.0
    for li, layer_name in enumerate(runner.gdn_layer_names):
        conv_state, ssm_state = runner.kv_caches[layer_name]
        eager_phys = _physical_slot(eager_slots[0])
        graph_phys = _physical_slot(graph_slots[0])
        conv_e = conv_state[eager_phys].float()
        conv_g = conv_state[graph_phys].float()
        ssm_e = ssm_state[eager_phys].float()
        ssm_g = ssm_state[graph_phys].float()
        conv_max = (conv_e - conv_g).abs().max().item()
        ssm_max = (ssm_e - ssm_g).abs().max().item()
        max_conv_diff = max(max_conv_diff, conv_max)
        max_ssm_diff = max(max_ssm_diff, ssm_max)
        if li == 0:
            layer0_exact = bool(
                torch.equal(conv_state[eager_phys], conv_state[graph_phys])
                and torch.equal(ssm_state[eager_phys], ssm_state[graph_phys])
            )
        conv_close = torch.allclose(
            conv_e, conv_g, atol=HIT_GDN_FULL_STACK_ATOL, rtol=HIT_GDN_FULL_STACK_RTOL
        )
        ssm_close = torch.allclose(
            ssm_e, ssm_g, atol=HIT_GDN_FULL_STACK_ATOL, rtol=HIT_GDN_FULL_STACK_RTOL
        )
        if not (conv_close and ssm_close):
            full_stack_near_tie = False
        gdn_state_results.append(
            {
                "layer": layer_name,
                "conv_max_diff": conv_max,
                "ssm_max_diff": ssm_max,
            }
        )

    cosine_ok = cos_sim >= HIT_LOGITS_COSINE_MIN
    passed = layer0_exact and top1_match and cosine_ok and full_stack_near_tie
    return {
        "passed": passed,
        "L": L,
        "non_contiguous_table": non_contiguous,
        "logits": {
            "max_abs_diff": max_abs_diff,
            "cosine_similarity": cos_sim,
            "cosine_ok": cosine_ok,
            "top1_match": top1_match,
        },
        "gdn_layer0_exact": layer0_exact,
        "gdn_full_stack_near_tie": full_stack_near_tie,
        "gdn_max_conv_diff": max_conv_diff,
        "gdn_max_ssm_diff": max_ssm_diff,
        "gdn_state_sample": gdn_state_results[:4],
        "num_gdn_layers_checked": len(runner.gdn_layer_names),
    }


def _run_subprocess() -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "benchmarks.cudagraph_eager_parity_check", "--single-run-json"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        capture_output=True,
        text=True,
        timeout=300,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("SINGLE_RUN_RESULT: "):
            import json

            return json.loads(line[len("SINGLE_RUN_RESULT: ") :])
    return {
        "passed": False,
        "error": "no result line found",
        "returncode": proc.returncode,
        "stderr_tail": proc.stderr[-4000:],
    }


def _run_subprocess_hit() -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "benchmarks.cudagraph_eager_parity_check", "--single-run-hit-json"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        capture_output=True,
        text=True,
        timeout=300,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("SINGLE_RUN_HIT_RESULT: "):
            import json

            return json.loads(line[len("SINGLE_RUN_HIT_RESULT: ") :])
    return {
        "passed": False,
        "error": "no hit result line found",
        "returncode": proc.returncode,
        "stderr_tail": proc.stderr[-4000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--single-run-json", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--single-run-hit-json", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.single_run_json:
        import json

        result = _run_once()
        print(f"SINGLE_RUN_RESULT: {json.dumps(result)}")
        return 0 if result["passed"] else 1

    if args.single_run_hit_json:
        import json

        result = _run_hit_table_once()
        print(f"SINGLE_RUN_HIT_RESULT: {json.dumps(result)}")
        return 0 if result["passed"] else 1

    if args.repeat > 1:
        results = [_run_subprocess() for _ in range(args.repeat)]
        for i, r in enumerate(results):
            status = "PASS" if r.get("passed") else "FAIL"
            print(f"run {i + 1}/{args.repeat}: {status}")
            if not r.get("passed"):
                print(f"  detail: {r}")
        n_pass = sum(1 for r in results if r.get("passed"))
        print(f"\n=== {n_pass}/{args.repeat} passed ===")
        return 0 if n_pass == args.repeat else 1

    # Default: run BOTH parity cases, each in its own fresh subprocess (model-
    # load isolation -- two 27B runners do not coexist in one GPU). The base
    # case covers contiguous tables; the hit case (INV5) covers hit-populated
    # NON-CONTIGUOUS tables. Overall PASS requires both.
    print("=== cudagraph_eager_parity_check ===")
    base = _run_subprocess()
    base_ok = bool(base.get("passed"))
    print(f"eager_vs_graph_parity (contiguous): {'PASS' if base_ok else 'FAIL'}")
    if not base_ok:
        print(f"  detail: {base}")
    else:
        print(f"  logits_cosine={base.get('logits', {}).get('cosine_similarity')} "
              f"gdn_all_close={base.get('gdn_states_all_close')}")

    hit = _run_subprocess_hit()
    hit_ok = bool(hit.get("passed"))
    print(f"hit_table_parity (INV5, non-contiguous L={hit.get('L')}): "
          f"{'PASS' if hit_ok else 'FAIL'}")
    if not hit_ok:
        print(f"  detail: {hit}")
    else:
        print(f"  non_contiguous_table={hit.get('non_contiguous_table')} "
              f"logits_cosine={hit.get('logits', {}).get('cosine_similarity')} "
              f"top1_match={hit.get('logits', {}).get('top1_match')}")
        print(f"  gdn_layer0_exact={hit.get('gdn_layer0_exact')} "
              f"gdn_full_stack_near_tie={hit.get('gdn_full_stack_near_tie')} "
              f"max_conv_diff={hit.get('gdn_max_conv_diff')} "
              f"max_ssm_diff={hit.get('gdn_max_ssm_diff')}")

    overall = base_ok and hit_ok
    print(f"\n=== overall: {'PASS' if overall else 'FAIL'} ===")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
