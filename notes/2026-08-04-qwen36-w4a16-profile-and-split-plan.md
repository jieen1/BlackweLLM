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

## Unreached experiment: opt-in modelopt split W4A16 decode

> **2026-08-04 correction (supersedes the initial result below):** path
> inspection plus Nsight Compute established that the Unsloth checkpoint uses
> `weight_layout="packed"`. Its real M=4 execution is
> `W4A16FusedMoeKernel` / packed TC-decode, whereas this experimental switch
> only gates the `weight_layout="modelopt"` direct micro kernel. Therefore the
> split code was never reached by this checkpoint. The reported 43.148→48.612
> ms cross-process difference must **not** be attributed to the switch and is
> not an A/B result. The branch remains unmerged because it is irrelevant to
> this model's active packed path, not because that invalid A/B established a
> regression.

The current direct small-M kernel is a 188-CTA cooperative launch: it computes
FC1, writes the existing BF16 intermediate, crosses a resident-grid barrier,
then computes FC2.  SparkInfer's shared micro-kernel already supports
compile-time phase 1 (FC1 only) and phase 2 (FC2 only).  The isolated
prototype invoked those two phases in stream order, preserving exactly the
existing raw packed weights, scales, routes, BF16 intermediate storage and
output ABI.

It was explicitly opt-in (`SPARKINFER_W4A16_SPLIT_DECODE=1`) and required to
pass, in order:

1. real-checkpoint single-layer W4A16 vs BF16-dequant oracle across M;
2. CUDA-graph replay correctness; and
3. one locked W1-S A/B trace with identical workload settings.

### Invalid result (retained for audit)

The real-checkpoint single-layer oracle passed for M=1, 2, 4, 8, 32, 128 and
512 (cosine 0.999983–0.999990), and the full W1-S CUDA-graph run captured all
three MTP graphs. Those checks only validate the existing packed path; they do
not validate or time the modelopt split code.

| W1-S two-round profile | fused direct | split phase 1 + phase 2 |
|---|---:|---:|
| verify graph wall time / round | 43.148 ms | **48.612 ms** |
| W4A16 CUDA self time / round | 18.887 ms | **20.691 ms** |

The table is an uncontrolled cross-process observation, not a split-launch
comparison. The branch is retained as SparkInfer commit `23fb17a`, **not
merged and not enabled by default**. Do not spend further Qwen effort on its
modelopt-only FC1/FC2 split; the relevant target is the packed kernel.

Reproduction of the rejected full-model run:

```bash
BF_SPARKINFER_PATH=/tmp/sparkinfer-qwen-w4a16-split \
PYTHONPATH=/tmp/sparkinfer-qwen-w4a16-split:/home/bot/project/qwen-sm120-runtime \
SPARKINFER_W4A16_SPLIT_DECODE=1 \
~/.venvs/vllm/bin/python -m benchmarks.mtp_w1s_our_runtime_perf \
  --num-requests 1 --concurrency 1 --max-tokens 8 --repeats 1 \
  --profile-rounds 2 --profile-trace-path /tmp/qwen_w1s_split_w4a16_profile.json \
  --result-path /tmp/qwen_w1s_split_w4a16_result.json
```

## Actual packed-kernel root cause, profiled

Nsight Compute sampled the real layer-5 M=4 packed `W4A16FusedMoeKernel`:

| metric | measured |
|---|---:|
| dynamic shared memory / CTA | 54.27 KiB |
| blocks per SM | 1 |
| achieved occupancy | 16.67% (8 warps / SM) |
| active warps / scheduler | 1.97 |
| cycles with no eligible warp | 54.30% |
| DRAM throughput | 41.33% of peak |
| SM compute throughput | 38.39% of peak |

This is the actual optimization target: reduce the packed kernel's shared
memory/CTA footprint enough to admit a second resident block, or otherwise
raise eligible-warp availability while preserving raw packed W4A16 weights and
BF16 activations. Any candidate must first demonstrate `>= 2` blocks/SM in the
same M=4 Nsight profile, then pass the oracle and W1-S graph gates.
