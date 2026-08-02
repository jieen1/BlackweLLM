# NVFP4 raw-Parameter free + FP8 W8A8 kernel investigation (2026-08-03)

Worktree `work/nvfp4-gemm-20260802` @ `35d61b3` (this round's starting
commit). Follow-up to `notes/2026-08-03-nvfp4-gemm-memory-audit.md`'s two
flagged "what was not verified" items. GPU: single RTX PRO 6000 Blackwell
Max-Q, under `/tmp/gpu_lock.sh`. All commands run with
`PYTHONPATH=/home/bot/project/qsr-w-nvfp4`, each script asserting
`runtime.__file__` resolves inside this worktree.

## 1. Freed the raw NVFP4 Parameters after the fused w4a16 build

The previous round's audit found `Qwen36MLP._ensure_w4a16_fused_ready`
left `gate_proj`/`up_proj`/`down_proj`'s raw NVFP4 `.weight`/
`.weight_scale`/`.weight_scale_2` (~9.15 GiB/64 layers) resident forever
alongside the kernel's own repacked `w13`/`w2` representation
(~8.65 GiB) -- two independent byte layouts, not aliased, both live at
once. Left unfreed because two scripts allegedly needed the raw
Parameters to stay populated.

**Checked which scripts actually depend on this, and found three, not
two** -- the audit note's two (`scripts/b1_verify_greedy_alignment.py`,
`scripts/b3_probe_batching_bar.py`) plus a third the note didn't
name: `scripts/verify_nvfp4_gemm_full_model_gap.py` (this branch's own
B1-R correctness gate) and `scripts/verify_nvfp4_gemm_single_layer.py`
(this branch's own single-layer cosine check) both monkeypatch/call the
legacy per-Linear path *repeatedly* on the same loaded model/MLP instance
(once per workload, once per M value) -- each repeat needs the raw
Parameters again. `scripts/b1_verify_greedy_alignment.py` turned out NOT
to need any change: its one dequant-and-stage pass runs strictly before
`Qwen36MLP`'s fused forward ever executes for the first time, so nothing
it does is affected by freeing later.

**Fix**: `Qwen36MLP._ensure_w4a16_fused_ready` now frees the three
submodules' raw Parameters (reassigns `.data` to a 0-element tensor, same
dtype/device, so `named_parameters()`/module-tree walks don't hit a
missing attribute) right after building `_w4a16_prepared`, unless
`self._keep_raw_nvfp4_weights` is set. Verified none of
`prepare_w4a16_modelopt_nvfp4_weights`'s internals alias their inputs
(`torch.cat` decouples `gate`/`up_proj`'s storage for `w13`; `w2`'s
`.unsqueeze(0).contiguous()` IS a no-op view of `down_proj.weight`, but
`_repack_weight` always allocates a fresh `torch.empty` output and fills
it via `.copy_()` -- checked the source, not assumed) -- this is a real
free, not a false sense of one. The three affected scripts now opt in via
`_keep_raw_nvfp4_weights = True` right after constructing/loading their
MLP(s), before the first fused forward.

**First surprise**: freeing the Parameter dropped `torch.cuda.
memory_allocated()` as expected but barely moved external `nvidia-smi`
(what every memory-audit script in this repo actually measures) --
67.10 GiB before this fix, 64.58 GiB after, not the ~9.15 GiB expected.
Root cause (verified with a standalone `torch.cuda.memory_allocated()` vs
`torch.cuda.memory_reserved()` probe, not assumed): PyTorch's caching
allocator keeps freed blocks reserved for its own reuse; only the
fraction reused by later allocations before a run's peak shows up
externally. Fixed by calling `torch.cuda.empty_cache()` once per real MLP
layer (inside `_free_raw_nvfp4_weights`, itself only reachable once per
instance -- bounded ~64 calls over the model's lifetime, not a per-token
cost).

### GPU verification (same three metrics as the previous round, same scripts)

**Correctness** -- both scripts re-run byte-for-byte identical to the
previous round's numbers (confirms the free doesn't touch any value the
fused kernel or the legacy oracle reads):

- `scripts/verify_nvfp4_gemm_single_layer.py`: cosine 0.999990 (M=1) down
  to 0.999983 (M=32), max_abs_err 0.000004-0.000061 -- identical to the
  prior round.
- `scripts/verify_nvfp4_gemm_full_model_gap.py` (3 workloads x 65 steps):
  median_gap_error=0.125 (bar 0.25), p90_gap_error=0.25 (bar 0.5),
  p99=2.3125 (bar 6.0), max=4.875 (bar 20.0), mean_kl_topk=3.03e-4 (bar
  5e-3), max_tie_slack_ulps=1.0 (bar 32.0), disagreement_rate=0.0154 (bar
  0.03), p90_logprob_error=0.2499 (bar 0.5), max_drift_ratio=1.0 (bar
  3.0) -- identical to the prior round's numbers. Same overall
  `passed=False` verdict for the same pre-existing, out-of-scope reason
  (2 of 11 CALIBRATED_THRESHOLDS gates need a live HF reference this
  harness structurally cannot produce -- see the prior note's §1 for the
  full explanation; unchanged this round).

**Memory** -- `scripts/measure_nvfp4_gemm_memory_and_throughput.py`
(external `nvidia-smi`, unchanged methodology, default -- i.e. NOT
`_keep_raw_nvfp4_weights` -- so this measures the real production path):

```
before_load        1305 MiB
after_load         22014 MiB
after_prefill      59052 MiB
after_warm_decode  59052 MiB
after_decode_loop  59053 MiB
total resident: 57.67 GiB   (was 67.10 GiB -- -9.43 GiB)
```

**Throughput** -- same script, eager, 30 decode steps, same prompt:

```
30 eager decode steps in 4.58s -> 6.547 tok/s   (was 5.819 tok/s)
```

Faster, not just non-regressed -- plausibly less memory pressure/
fragmentation, though this session did not isolate that as the specific
cause (could be run-to-run variance; both runs are eager, no CUDA graph,
same script, same prompt).

## 2. FP8 self_attn/GDN projections: kernel investigated, not wired in

The remaining ~14 GiB BF16 dequant cache (`ModelOptFP8Linear`'s
`_ensure_ready()`, still hit by every real forward through self_attn's
q/k/v/o_proj and GDN's in_proj_qkv/in_proj_z/out_proj) is the other half
of the original 49.72 GiB dequant-cache audit finding -- NVFP4's half is
now fixed (fusion, §1's free); this is the rest.

### Scheme check first (the NVFP4 round's own lesson, applied)

Checked the real checkpoint's `quantization_config.config_groups` before
touching any kernel, not after: `group_0` (every FP8-targeted layer)
declares **both** `weights` and `input_activations` as `{dynamic: false,
num_bits: 8, type: float}` -- i.e. genuine static per-tensor W8A8, not
weight-only. Confirmed against real safetensors headers, not just the
config: every FP8 layer (checked `self_attn.q_proj` layer 11,
`linear_attn.in_proj_qkv` layer 0) ships an actual `input_scale` tensor
(`F32`, shape `[]`, scalar) alongside `weight`/`weight_scale`. This is
the **opposite** situation from NVFP4's MLP checkpoint (W4A16,
`input_activations: None`, no real activation-scale tensor at all) --
here, quantizing the activation to FP8 using the checkpoint's own static
`input_scale` is the checkpoint's *intended* execution path, not an
invented approximation.

### Kernel match

`sparkinfer.gemm.tensor_fp8_linear` ("static per-tensor FP8 linear for
SM12x", per its own module docstring) matches this scheme exactly: single
combined `output_scale = input_scale * weight_scale`, activation must
already be `float8_e4m3fn` when passed to `mm()`. Checked
`block_fp8_linear`/`mxfp8_linear` too (both exist under
`sparkinfer/gemm/`) but did not evaluate them further once
`tensor_fp8_linear` matched -- they're dynamic per-block schemes, a
different (finer) granularity than what this checkpoint actually
calibrated and ships scales for.

Added `ModelOptFP8Linear.input_scale` (a real `nn.Parameter`, loaded from
the checkpoint's `.input_scale` tensor -- previously silently discarded
via `_IGNORED_WEIGHT_SUFFIXES`) so this data is available. Left
`ModelOptFP8Linear.forward()`/`_ensure_ready()` **completely unchanged**
-- same "leave the submodule's own legacy path alone, route one level up"
split NVFP4 used for `Qwen36MLP`, deliberately, because `_bmm_project`
(`runtime/model/qwen36_model.py` -- real GDN spec-decode code, not a
diagnostic) and `scripts/b3_probe_batching_bar.py`'s part B both read
`module._weight_bf16` directly and would break if this class's own
`forward()` changed meaning.

### Environment gap: sparkinfer's own version gate

`tensor_fp8_linear.is_supported()` returns `False` on this machine:
`sparkinfer._lib.gating.MIN_CUTLASS_DSL = "4.6.0"`, installed
`nvidia-cutlass-dsl` is 4.5.2. Checked whether this reflects a real
functional gap or just an untested-combination pin: `cute.nvgpu.warp.
MmaMXF8Op` (what the kernel's own lower-level
`is_tensor_fp8_linear_supported()` actually probes for) is present at
4.5.2, and a synthetic `pack_weight`+`mm()` call (matching sparkinfer's
own `tests/gemm/test_tensor_fp8_linear.py` fixture, called directly,
bypassing `is_supported()`) produced the expected result within that
test's own tolerance (`rtol=1e-2, atol=2e-3`; measured max abs diff
~1.5e-5 on a `[7,128]x[64,128]` synthetic case). So the kernel genuinely
works here -- `is_supported()`'s pin is conservative for this specific
(untested-by-sparkinfer) version combination, not evidence of breakage.
Did not touch sparkinfer's source (only `origin` is editable per this
session's standing rule, and no fix was warranted here since the
workaround -- calling `pack_weight`/`mm` directly -- is a legitimate use
of sparkinfer's public API, not a bug needing a patch).

### Measured precision: real, but meaningfully worse than NVFP4's fusion

`scripts/verify_fp8_tensor_gemm_single_layer.py` -- one real
`self_attn.q_proj` (layer 3) and one real `linear_attn.in_proj_qkv`
(layer 0), legacy BF16xBF16 forward vs. FP8xFP8 `tensor_fp8_linear.mm`
(activation quantized with the checkpoint's real static `input_scale`),
M=1..512:

| target | M | cosine | max_abs_err | rel_to_max |
|---|---:|---:|---:|---:|
| self_attn.q_proj | 1 | 0.999675 | 0.002686 | 0.0263 |
| self_attn.q_proj | 512 | 0.999646 | 0.004517 | 0.0287 |
| linear_attn.in_proj_qkv | 1 | 0.999653 | 0.003662 | 0.0270 |
| linear_attn.in_proj_qkv | 512 | 0.999646 | 0.007324 | 0.0230 |

For comparison, NVFP4's fused kernel measured cosine 0.999983-0.999990
(the same single-layer-style check, previous round). FP8's `(1 -
cosine)` here is roughly 30-40x larger than NVFP4's -- genuinely working,
not broken, but a real, measured, and non-trivial precision cost, unlike
NVFP4's fusion (which introduced no new error source at all: it
dequantizes weight *inside* the kernel against an un-quantized BF16
activation, exactly matching what B1-R's HF calibration reference also
does). Quantizing the activation to FP8 too -- which this checkpoint's
scheme calls for and B1-R's calibration baseline never does -- is real
additional lossiness relative to what B1-R was calibrated against, even
though it is what the checkpoint format was designed for.

### Decision: not wired into production this round

Did not modify `Qwen36Attention.forward`/`decode_batch` or
`Qwen36GatedDeltaNet.forward`/`spec_forward` to route through this
kernel. Reasoning:

1. The measured per-layer error is meaningfully worse than NVFP4's near-
   perfect fusion, and would apply across a **larger** footprint --
   every layer's self_attn or GDN projections, not just the MLP.
2. NVFP4's own full-model gap-error result (§1 above) already sits
   almost exactly at its calibration bar's own control-run noise floor
   (median/p90 land on the control run's own numbers, not just under the
   bar) -- there is very little headroom left before a change with
   materially worse per-layer precision risks tipping the combined
   (NVFP4 fusion + FP8 kernel) full-model gap error over CALIBRATED_
   THRESHOLDS.
3. Verifying whether it actually does exceed those bars requires the
   same full-model, step-locked, oracle/candidate B1-R harness the NVFP4
   fusion used -- but wiring the candidate side requires real changes to
   `decode_batch` (CUDA-graph-capture-safety-critical: fixed buffers, no
   shape-varying allocations across replays) and `spec_forward`, not a
   one-level-up fusion like `Qwen36MLP`'s. That is substantially more
   surface area and risk than this round's remaining scope supports
   responsibly.

Given the task's explicit bar ("正确性优先...这轮不能比这个差"), reporting
this as a measured, evidence-backed finding -- kernel exists, scheme
matches, functionally verified, but its precision profile makes wiring
it in a real correctness risk that needs its own dedicated
verification round -- was judged the right call over forcing it in
un-verified.

### What a follow-up round would need

- Wire `tensor_fp8_linear` into `Qwen36Attention`/`Qwen36GatedDeltaNet`
  the same "read raw Parameters directly, build packed weights once,
  lazily, one level up" way `Qwen36MLP` did for NVFP4 -- q/k/v/o_proj and
  in_proj_qkv/in_proj_z/out_proj are independent GEMMs (no gated-MLP-
  style fusion needed, simpler than the NVFP4 case in that respect).
  `decode_batch`'s CUDA-graph-safety contract needs explicit attention
  (fixed-size activation-quantization buffers, no per-replay allocation).
- Re-run `scripts/verify_nvfp4_gemm_full_model_gap.py`-style step-locked
  oracle/candidate comparison with the FP8 kernel as candidate (on top of
  the already-fused NVFP4 MLP, since both would be live in production
  simultaneously) against `CALIBRATED_THRESHOLDS` before adopting.
- If it fails, `block_fp8_linear`/`mxfp8_linear` (dynamic per-block
  activation scales, finer granularity than this checkpoint's static
  per-tensor scale) are the next things to evaluate -- not evaluated this
  round.

## What was not verified this round

- Whether the FP8 kernel's full-model gap error actually passes or fails
  CALIBRATED_THRESHOLDS -- not measured (would need the production wiring
  described above).
- CUDA-graph-captured decode throughput for either the NVFP4 free or a
  hypothetical FP8 kernel path -- this round, like the previous one, only
  measured eager decode.
- Whether `torch.cuda.empty_cache()`'s cost (bounded to ~64 calls, once
  per real MLP layer, over the model's lifetime) is negligible under a
  real concurrent-serving workload rather than this session's single-
  request eager-decode benchmark.
