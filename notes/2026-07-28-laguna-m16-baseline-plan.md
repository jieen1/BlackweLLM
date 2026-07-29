# Laguna 64K M=16 performance baseline and execution plan

The controlled evidence ledger is
`benchmarks/fixtures/laguna_m16_64k_baseline_20260728.json`.  It is the
only source for performance claims in this workstream.

## Measurement rules

- The first target is **300 completion/output tok/s** on the frozen
  `fd33368-full-comparison-ours-64k` workload.  That workload fixes the
  prompt, 64K geometry, K=15, CUDA Graph, prefix cache and all load-time
  paging settings.
- This is only a recovery gate.  The DFlash success criterion is an exact
  prompt/hot-state comparison against vLLM: at least 330 output tok/s as the
  first clear win over the recorded 293 output tok/s vLLM reference, with 350
  output tok/s as the substantial-win target.  It must also materially exceed
  the runtime's no-DFlash lane.
- Always run `bf diff <previous> <candidate>` before interpreting two run
  records.  A record with a known-bad historical fingerprint is useful for
  diagnosis, not for a formal comparison.
- Keep completion/output tok/s and counter accepted tok/s as separate fields.
  Older vLLM fixtures call completion throughput `accepted_tok_s`; do not mix
  that historical label with counter-derived throughput.
- Output hashes and generation semantics are correctness gates.  Acceptance
  is a scenario-quality guard: normal conversation must stay at least 70%,
  with roughly 80% healthy; it is not a number to optimize artificially.

## Ordered work

1. Establish a same-prompt comparison matrix before assigning any DFlash
   regression to acceptance rate or a kernel:

   | Lane | Purpose | Required measurements |
   | --- | --- | --- |
   | vLLM + DFlash | external implementation and kernel reference | output tok/s, counter accepted tok/s, acceptance, M=16 profile |
   | runtime + DFlash | production speculative path | the same fields plus verify/draft/commit trace phases |
   | runtime without DFlash | M=1 target-model floor independent of draft acceptance | output tok/s, M=1 profile, output-quality check |

   Every lane uses the same 64K token IDs, generation length, greedy setting,
   paging geometry and hot/cold designation.  Do not attribute a DFlash loss
   until the no-DFlash lane establishes whether the target-model base path is
   already behind.  vLLM runs are reference captures, not a routine loop.
2. Re-run the canonical contract once with `QSR_TRACE=1`, then inspect the
   existing trace before changing code.  The current output is correct but
   decode is 1.05--1.11 s for 18 M=16 rounds versus the archived 0.83--0.88 s.
   The trace must attribute the extra 12--15 ms/round to verify, draft,
   metadata or commit work.
3. Change one hot-path boundary only, beginning with the no-vLLM
   `laguna_forward_context`/runtime-config replacement boundary if the trace
   points there.  Verify fixed-prompt output, quality gate and the same
   performance contract immediately after each edit.
4. If the recovered path remains below 300, evaluate page size 64 to 128 as
   an isolated load-time experiment.  Existing roofline evidence gives about
   2.1 ms/round (about 5%) attention headroom, but KV/prefix/graph layout make
   this a high-risk change.
5. Do not modify SparkInfer's main MoE kernel without new evidence.  Its
   M=16 main kernel is already HBM-bound; only router/top-k fusion has a small
   evidence-backed residual opportunity.

The vLLM quick-brown trace is retained outside Git because it is a generated
profile artifact.  The versioned fixture stores its contract, summary and
SHA-256 so it remains auditable without restoring a vLLM runtime dependency.
