# 2026-08-04 Qwen3.6 W4A16 profile and split-phase plan

## Status

**Current conclusion:** the quality-preserving Qwen3.6 MLP path is raw packed
NVFP4 **W4A16** (BF16 activations).  It must remain so.  The historical custom
NVFP4 path is W4A4 and is not a production candidate: its full-model quality
gate failed.  No production path may lazily materialize all quantized weights
as BF16.

## Comparable evidence

The following two Nsight-style profiles used the same cached
`unsloth/Qwen3.6-27B-NVFP4` checkpoint and the W1-S greedy MTP workload.  They
are useful for phase attribution, not a valid `bf diff` throughput comparison:
the artifacts were not produced through `bf exec`.

| path | verify wall time | W4A16 kernel / verify round | raw FP8 GEMM / verify round |
|---|---:|---:|---:|
| current no-vLLM runtime | 43.148 ms | 18.887 ms | 14.463 ms |
| historical vLLM path | 31.083 ms | about 12.56 ms (historical W4A4) | about 11.81 ms |

The historical W4A4 number is **not** a porting target because it changes
activation precision.  It instead identifies scheduling/launch structure as
the remaining W4A16 investigation target.  Attention is not the immediate
dominant cost in this short-context trace.

Reproduction of the current profile:

```bash
PYTHONPATH=/home/bot/project/qwen-sm120-runtime \
  ~/.venvs/vllm/bin/python -m benchmarks.mtp_w1s_our_runtime_perf \
  --num-requests 1 --concurrency 1 --max-tokens 8 --repeats 1 \
  --profile-rounds 2 --profile-trace-path /tmp/qwen_w1s_current_profile.json \
  --result-path /tmp/qwen_w1s_current_profile_result.json
```

GPU safety rule for this work: run that or any layer oracle only through the
single-job `/tmp/gpu_lock.sh`; never start a second service and never touch the
user-owned server process.

## Landed low-risk change

SparkInfer commit `5539bb292f084b22ea3e5d6360eabdfd4fadbb12` removes two
redundant `barrier_count.zero_()` / `barrier_epoch.zero_()` launches per W4A16
MLP invocation.  The existing resident-grid barrier already resets count and
advances epoch before returning, while persistent workspace construction
initializes both values to zero.  Real Qwen layer-5 oracle validation passed
for all tested M values and the B1 CUDA-graph profile ran successfully.

The follow-up profile reduced W4A16 CUDA self time from 37.775 ms to 37.156 ms
over two verify rounds.  This is a small confirmed launch-overhead removal, not
a claim of end-to-end performance parity.

## Active experiment: opt-in split W4A16 decode

The current direct small-M kernel is a 188-CTA cooperative launch: it computes
FC1, writes the existing BF16 intermediate, crosses a resident-grid barrier,
then computes FC2.  SparkInfer's shared micro-kernel already supports
compile-time phase 1 (FC1 only) and phase 2 (FC2 only).  The active isolated
prototype will invoke those two phases in stream order, preserving exactly the
existing raw packed weights, scales, routes, BF16 intermediate storage and
output ABI.

It is explicitly opt-in (`SPARKINFER_W4A16_SPLIT_DECODE=1`) and must pass, in
order:

1. real-checkpoint single-layer W4A16 vs BF16-dequant oracle across M;
2. CUDA-graph replay correctness; and
3. one locked W1-S A/B trace with identical workload settings.

If any gate fails or the profile does not reduce the W4A16 phase, the prototype
is rejected rather than made default.  A raw-FP8 QKV fusion is a separate,
later candidate; it does not supersede this W4A16 root-cause experiment.
