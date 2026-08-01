# DFlash M=16 acceptance incident — stopped state

Date: 2026-07-29

## Scope and frozen input

- Runtime: current `vllm-removal-phase1` HEAD `6349751` in the no-vLLM environment.
- Model path: Laguna-S-2.1 NVFP4 plus its DFlash draft model.
- Workload: the existing frozen 64K quick-brown-fox DFlash contract
  (`dflash-m16-64k-quick-brown-fox`), greedy, K=15, one slot.
- All GPU checks used the existing `bfdiag` warm daemon and
  `historical_dflash_m16_prompt_ids`; no new benchmark script was created.

## Confirmed observations

1. The low acceptance is real in the current no-vLLM runtime. With verify
   CUDA Graph disabled, 16 rounds produced accepted-draft counts:

   ```text
   [12, 15, 15, 0, 0, 1, 1, 1, 3, 1, 1, 0, 5, 0, 2, 0]
   ```

   This is `0.2375` accepted drafts / proposed drafts. Therefore the issue is
   not solely the verify CUDA Graph.

2. Draft CUDA Graph is not the source. Starting from the same fresh 64K KV
   state, draft-CG and eager-draft produced identical initial draft tokens and
   identical first five acceptance counts:

   ```text
   [12, 15, 15, 0, 15]
   ```

3. Verify CUDA Graph does have a separate correctness defect. Its auxiliary
   hidden states differ numerically from eager verification from the first
   verify round, despite matching top-1 verifier tokens at that point. At
   `kv_len=65536`, the captured-vs-eager diagnostic measured:

   ```text
   logits: max_abs=3.5625, rmse=0.4050
   aux layer RMSE: [0.00066, 0.01274, 0.04136, 0.27204, 0.52137, 1.16134]
   ```

   After three graph-driven rounds, the graph and eager branches have already
   accumulated different draft/KV state. They can then disagree on the
   recovery token at the first rejection (graph: `785`; eager: `350`) even
   when both report `num_accepted=0`.

4. The former captured/eager diagnostic was insufficient for this incident:
   it compared both verify modes from a graph-produced prefix state. That
   proves instantaneous logit parity at that state, but cannot prove
   end-to-end DFlash parity because the auxiliary states are fed back into the
   draft KV cache after every round.

## Current localization boundary

- The immediate acceptance collapse is downstream of the draft proposal graph.
- Disabling verify-CG removes its auxiliary-state drift but does **not** restore
  the expected acceptance level; the remaining primary fault is in the
  no-vLLM/self-built target-model DFlash auxiliary-hidden-state to draft-KV
  path.
- No fix has been applied in this incident pass. Do not claim that verify-CG
  is the sole root cause.

## Relevant local evidence

- `runtime/backends/laguna_dflash.py`: DFlash prefill, verify, acceptance, and
  auxiliary-hidden-state to draft-KV commit path.
- `runtime/backends/laguna_cuda_graph.py`: M=16 verify CUDA Graph path.
- `runtime/backends/laguna_dflash_cudagraph.py`: draft CUDA Graph path
  (ruled out by the A/B above).
- `runtime/model/laguna_model.py`: self-built target-model auxiliary hidden
  state collection.
- `runtime/model/laguna_dflash_model.py`: DFlash hidden-state fusion and
  context-KV precompute.
- `bfdiag/workloads.py`: fixed workload and verify-CG divergence diagnostic.
- Historical commits with directly relevant prior evidence:
  `30675d2` (CG binding address bug; acceptance recovery), `aea4cf4`
  (verify-only state machine), `7327a78`/`e9cf99d`/`92be1c1`
  (verify-CG mode/worklist changes), and `6349751` (shared eager prefill
  workspace; its three-step parity check does not cover post-aux feedback).

## Stop condition

Investigation deliberately stopped here at the user's request. No further
profiling, model reload, benchmark, external lookup, or code change is active.
