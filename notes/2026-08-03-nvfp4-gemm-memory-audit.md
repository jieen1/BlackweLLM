# NVFP4 dense-MLP fused w4a16 kernel: GPU verification (2026-08-03)

Worktree `work/nvfp4-gemm-20260802` @ `47fb868`. Checkpoint
`nvidia/Qwen3.6-27B-NVFP4`. GPU: single RTX PRO 6000 Blackwell Max-Q,
under `/tmp/gpu_lock.sh`. All commands run with
`PYTHONPATH=/home/bot/project/qsr-w-nvfp4` (or `~/.venvs/vllm/bin/python -m
pytest`), each script asserting `runtime.__file__` resolves inside this
worktree first -- the earlier session in this branch's history was burned
once by the editable-install path trap (`python scripts/x.py` resolving
`runtime` to the main worktree).

This replaces the previous attempt on this branch
(`sparkinfer.gemm.blockscaled.mm`, routed through
`ModelOptNVFP4Linear.forward()` directly), which dropped memory but failed
B1-R's calibrated gap-error bars because it turned a W4A16 (weight-only)
checkpoint into an unintended W4A4 approximation. This attempt fuses
`Qwen36MLP`'s three NVFP4 submodules into one call to
`sparkinfer.moe._shared.kernels.w4a16.kernel.run_w4a16_moe` (degenerate
1-expert/top-1 MoE) -- see `runtime/model/qwen36_model.py::Qwen36MLP`'s
docstring for the design.

## 1. Correctness

### Single MLP block (layer 5, real checkpoint weights)

`scripts/verify_nvfp4_gemm_single_layer.py`: fused w4a16 forward vs legacy
per-Linear BF16-dequant forward, same real gate/up/down_proj weights,
random BF16 activations, M=1..512:

| M | cosine | max_abs_err | rel_to_max |
|---:|---:|---:|---:|
| 1 | 0.999990 | 0.000004 | 0.0010 |
| 2 | 0.999988 | 0.000031 | 0.0063 |
| 8 | 0.999988 | 0.000031 | 0.0057 |
| 32 | 0.999983 | 0.000031 | 0.0060 |
| 128 | 0.999984 | 0.000031 | 0.0056 |
| 512 | 0.999984 | 0.000061 | 0.0100 |

For comparison, the previous (blockscaled.mm) attempt measured cosine
~0.9955 on the equivalent per-Linear check. This is a different precision
regime, not an incremental improvement.

### Full-model B1-R gap error

`scripts/verify_nvfp4_gemm_full_model_gap.py` (oracle = `Qwen36MLP`'s
legacy per-Linear BF16-dequant forward, candidate = the new fused
`run_w4a16_moe` forward, step-locked through the oracle's own greedy
tokens, 3 workloads x 65 steps = 195 total steps), judged against
`bfdiag.divergence.logit_agreement.CALIBRATED_THRESHOLDS`:

| metric | measured | bar | historical control (calibration run) |
|---|---:|---:|---:|
| median_gap_error | 0.125 | 0.25 | 0.125 |
| p90_gap_error | 0.25 | 0.5 | 0.250 |
| p99_gap_error | 2.3125 | 6.0 | 3.375 |
| max_gap_error | 4.875 | 20.0 | 10.56 |
| mean_kl_topk | 3.03e-4 | 5e-3 | 1.58e-3 |
| max_tie_slack_ulps | 1.0 | 32.0 | 8 |
| disagreement_rate | 0.0154 | 0.03 | 0.0077 |
| p90_logprob_error | 0.2499 | 0.5 | 0.250 |
| max_drift_ratio | 1.0 | 3.0 | 1.50 |

All 9 measured metrics pass with comfortable margin -- `median_gap_error`
and `p90_gap_error` land almost exactly on the historical **control** run's
own numbers (the "no injected bug" calibration baseline), not just under
the bar. This is a stark contrast with the previous attempt on this branch
(median 0.75 vs bar 0.25, p90 2.125 vs bar 0.5, disagreement_rate 0.074 vs
bar 0.03 -- all real failures, worse than the weakest injected bug B1-R was
calibrated against).

**The script's own overall verdict is still `passed=False`.** Not because
of any of the 9 metrics above -- because `CALIBRATED_THRESHOLDS` also gates
`nll_relative_excess` (R4: teacher-forced NLL excess vs the real HF
reference) and `min_logits_cosine` (R3: prefill logits cosine vs HF), and
`AgreementReport.evaluate_summary` treats an unmeasured (`None`) gated
metric as a hard fail rather than silently passing. This script's
oracle/candidate design only ever compares two paths **internal to this
codebase** (legacy per-Linear dequant vs fused w4a16) -- it never runs the
real HF `transformers` reference forward pass, so it structurally cannot
populate those two fields. This is a pre-existing characteristic of the
script as written in the previous round (unchanged by this round beyond
retargeting the monkeypatch from `ModelOptNVFP4Linear.forward` to
`Qwen36MLP.forward`, since the fused kernel bypasses the former entirely),
not a new regression and not something this round's task scope covers
(reproducing full B1-R calibration with a live HF reference is a
substantially larger, separate harness). Read plainly: every gate this
script is actually wired to measure passes at essentially control-run
precision; two of eleven CALIBRATED_THRESHOLDS gates are structurally
unmeasurable by this harness.

## 2. Memory

`scripts/measure_nvfp4_gemm_memory_and_throughput.py` (external
`nvidia-smi`, matching `notes/2026-08-02-gpu-memory-audit.md`'s
methodology):

```
before_load        1305 MiB
after_load         22014 MiB  (+20709)
after_prefill      68707 MiB  (+46693)   -- 13-token prompt, first fwd
after_warm_decode  68707 MiB  (+0)
after_decode_loop  68706 MiB  (~flat over 30 steps -- no leak)
total resident: 67.10 GiB
```

Well under the fully-dequantized-everywhere baseline (76.34 GiB), but
above the previous (correctness-broken) blockscaled.mm attempt's
36.78 GiB. Root-caused with an isolated diagnostic (build real
`Qwen36MLP` instances one at a time from real checkpoint tensors for all
64 layers, forward once each at M=13, track `torch.cuda.memory_allocated`
deltas): growth is **perfectly linear**, ~288 MiB/layer, 64 layers ->
~18.0 GiB total, no acceleration, no JIT-cache blowup, no leak. This
matches the expected cost of keeping BOTH the raw checkpoint NVFP4
Parameter (`gate_proj.weight`/`up_proj.weight`/`down_proj.weight`, ~9.15
GiB across 64 layers) AND the w4a16 kernel's own repacked `w13`/`w2`
representation (~8.65 GiB) resident simultaneously -- the kernel's packed
format is a genuinely separate byte layout (row-rotated, tile-packed into
int32 words), not a view, so it cannot alias the raw Parameter storage.

The raw Parameter is deliberately **not** freed after building the fused
representation: `ModelOptNVFP4Linear._ensure_ready()` (used directly by
`scripts/b1_verify_greedy_alignment.py` and
`scripts/b3_probe_batching_bar.py`, per that class's module docstring)
needs `.weight`/`.weight_scale`/`.weight_scale_2` to stay populated on
every NVFP4 submodule, including the three inside each `Qwen36MLP`. Freeing
them post-prep would recover roughly half the ~18 GiB (~9.15 GiB), but was
**not attempted this round** -- untested trade-off against those two
scripts' documented contract, flagged as a follow-up, not implemented.

The remainder of the gap vs. the 36.78 GiB attempt (roughly 68.7 - 36.78 -
18.0 ≈ 14 GiB) is `lm_head`'s reintroduced BF16 dequant cache (~1.2 GiB --
`lm_head` went through `blockscaled.mm` too in the previous attempt, so had
no persistent cache there) plus whatever the previous attempt's own
36.78 GiB already carried for FP8 self_attn/GDN dequant caches and
prefill KV-cache/activation overhead -- unchanged this round (task scope:
"先把 NVFP4 这条做对,先别动 FP8 Linear" -- not independently reverified this
session).

## 3. Throughput

Same script, eager (no CUDA graph) greedy decode, 30 steps, one prompt
("Once upon a time, in a small village near the mountains,"):

```
30 eager decode steps in 5.16s -> 5.819 tok/s
```

**Same-harness comparison** (this exact unchanged script, run against both
attempts on this branch): 5.819 tok/s this round vs 2.477 tok/s for the
previous blockscaled.mm attempt -- 2.35x faster, apples-to-apples.

**Not comparable** to the "~4 tok/s" figure named in the task
(`docs/implementation-plan.md` line ~495, `docs/roadmap.md` line ~97,
`notes/2026-08-02-qwen36-dequant-cache-memory-floor.md` line ~84): that
number comes from a different harness entirely -- B2's Laguna-served
`server.app` HTTP measurement, not a bare eager forward loop on the
self-built model object -- and that same note explicitly records "服务进程内
CUDA Graph 是否真的生效未能从日志确认" (whether CUDA graph was actually active
in that measurement could not be confirmed from logs). Different serving
stack, different request path, unconfirmed CUDA graph status on the
baseline side: not a valid comparison without first aligning harnesses,
which is out of this round's scope.

## What was not verified

- Freeing the raw NVFP4 Parameter after fused-weight prep (the ~9 GiB
  memory win described above) -- not attempted, would need re-verifying
  `scripts/b1_verify_greedy_alignment.py` / `scripts/b3_probe_batching_bar.py`
  against the resulting broken `_ensure_ready()` contract.
- `nll_relative_excess` / `min_logits_cosine` against a live HF
  `transformers` reference -- the two CALIBRATED_THRESHOLDS gates this
  session's harness cannot produce (see §1).
- CUDA-graph-captured decode throughput for the fused w4a16 path (this
  round only measured eager). The previous attempt's own throughput number
  was also eager-only.
- lm_head's ~1.2 GiB BF16 dequant cache and the pre-existing FP8
  self_attn/GDN dequant cache were not independently re-measured in
  isolation this session (attributed by subtraction/prior note, not
  directly profiled this round).
