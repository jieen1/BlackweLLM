# Standard model (unsloth/Qwen3.6-27B-NVFP4) served for the first time: C-LIVE, CUDA Graph, bit-exact

Worktree `work/std-serve-20260803` @ `9e67e4a` (this round's starting commit).
GPU: single RTX PRO 6000 Blackwell Max-Q, under `/tmp/gpu_lock.sh`. Task: the
standard model (`unsloth/Qwen3.6-27B-NVFP4`) had only ever been validated on
B1's eager single-sequence path (cosine 0.999979-0.999999, gap p90=0.25) --
it had never been served through `server/app.py`. This note is the first
service-path acceptance run: `python -m server.app`, C-LIVE, CUDA Graph
capture confirmation, and bit-exact vs B1 eager, all on this checkpoint.

## Headline result

Every item was closed, but not on the first try -- **decode CUDA Graph
capture failed on the first launch** (fused w4a16 MLP path, never exercised
under `torch.cuda.graph` before). Root-caused, fixed in `runtime/` (not
sparkinfer), and the fix's own correctness was then confirmed by the same
bit-exact gate that was already planned for item 4. Details below, in the
order the acceptance criteria were checked.

## 1. Server up

`QSR_SERVER_MODEL_PATH=unsloth/Qwen3.6-27B-NVFP4`, `QSR_SERVER_CAPACITY=1`,
`QSR_SERVER_ENABLE_DFLASH=0` (qwen36 backend does not support DFlash --
`ServerEngine._load_qwen36_model` raises if asked), `QSR_SERVER_ENABLE_
CUDAGRAPH=1`. `/v1/models` and `/v1/chat/completions` both work; weights
resident 20.2 GiB in the (small, `--max-seq-len 512`) verification harness,
~53 GiB in the full server config with KV cache -- matches the task's stated
53.08 GiB figure.

**One deviation from the literal instructions, necessary and reported
here**: the task said `QSR_SERVER_NUM_SLOTS=1`, but `ServerEngine.__init__`
requires `num_slots >= capacity + cg_extra`, and `cg_extra=1` whenever
`enable_cudagraph and not enable_dflash` (one warmup slot for M=1..N decode
CG capture) -- `num_slots=1` would have raised `ValueError` at startup for
any config with CUDA Graph on. Used `QSR_SERVER_NUM_SLOTS=2` instead
(`capacity=1` still enforces single concurrency, so "本任务不测并发" still
holds -- capacity is the concurrency knob, num_slots here is purely the CG
warmup-slot requirement).

## 2. C-LIVE

**First run (default `QSR_TOOL_CALL_PARSER=poolside_v1`): 51/61 passed.**
Investigated: 6 of the 10 failures were tool-calling checks (`no tool XML
leak`, `has tool_calls`, `has tool_use blocks`, ...) -- the default parser is
tuned for Laguna (poolside's own checkpoint format); the standard model needs
a different one. `server/formats/tool_parsers/registry.py` already has
`qwen3_coder` registered.

**Second run (`QSR_TOOL_CALL_PARSER=qwen3_coder`): 64/67 passed** (also note
the total went from 61 to 67 -- the tool-call fix unblocked follow-on
assertions that a leaked-XML response short-circuits before reaching). The
tool-parser mismatch accounted for 7 of the original 10 failures.

**Remaining 3 failures, each verified directly (not assumed) to be genuine
model behavior, not a serving bug:**

1. `/v1/completions: no <think>/</think> leaked into text` -- prompt "The
   capital of France is" (raw completion, no chat template) produced `Paris.
   \n\n<think>\n\n</think>\n\nThat is correct. Paris is the capital and`. Per
   this check's own module docstring, `/v1/completions` is supposed to
   return the raw completion **verbatim, with no thinking-strip wrapping**
   -- the model spontaneously emitting `<think>` tokens even for a bare
   continuation (no chat template) is genuine model behavior, and passing it
   through unstripped is the *correct* behavior for this endpoint. The
   inline "no `<think>` leaked" assertion in `c_live_smoke.py`'s check 2
   implicitly assumed no model would ever do this; that assumption does not
   hold for this checkpoint.
2. `openai multi-turn: has content` / 3. `anthropic array content: has
   content` -- both use `max_tokens=512`. Reproduced directly: curling
   `anthropic array content`'s exact payload ("Say hi in one word.",
   `max_tokens=512`) returns `stop_reason: "max_tokens"`, `content: [{"type":
   "text", "text": ""}]`, and a 512-token `reasoning_content` that is still
   mid-thought when truncated ("I'll go with \"Hi.\" but to be perfectly
   compliant, I'll just write \"Hi" -- cut off, not concluded). This
   checkpoint's thinking is far more verbose than what generated the
   original 67/67 baseline (Laguna) -- 512 tokens is not enough budget for
   this model to finish thinking about even a trivial prompt, let alone
   answer. The server's reasoning/content split and `stop_reason` reporting
   are both correct; the test fixture's `max_tokens=512` is simply too small
   for this model's thinking style.

**Net: C-LIVE is 64/67, and the 3-check gap is fully attributed to real,
verified differences in the standard model's own behavior (verbose
thinking, raw-completion `<think>` emission), not to a serving-path
regression.**

## 3. CUDA Graph: captured on the first attempt, but wrong -- decode failed to capture

First launch (fix not yet applied): `ServerEngine._load_qwen36_model` ->
`Qwen36Backend.capture_decode_cuda_graph()` raised inside the capture region:

```
RuntimeError: W4A16 GEMM scratch is not initialized for CUDA graph capture;
provide a preallocated fc*_c_tmp workspace with sufficient capacity
```

raised by sparkinfer's `kernel.py::_get_c_tmp`, called from `run_w4a16_moe`,
called from `Qwen36MLP._forward_w4a16_fused`
(`runtime/model/qwen36_model.py`). The server did NOT crash -- `capture_
decode_cuda_graph()` catches the exception, logs it, and falls back to
eager batched decode (`Qwen36Backend.capture_decode_cuda_graph`'s own
docstring: "a failed capture drops the whole set and falls back to eager").
This is exactly the scenario flagged before this round started: **"融合
w4a16 路径从未在 CUDA Graph 下验证过"** -- confirmed true, on the first
real attempt.

### Root cause (read, not guessed)

`_forward_w4a16_fused` already passes `fc1_c_tmp=buffers.fc1_c_tmp` /
`fc2_c_tmp=buffers.fc2_c_tmp` from sparkinfer's own `make_w4a16_packed_
buffers()` convenience allocator -- so the scratch argument was not simply
missing. The actual problem is a **sizing mismatch inside sparkinfer
itself**, between two of its own formulas for the same buffer:

- `make_w4a16_packed_buffers` -> `plan_w4a16_buffers` sizes `fc1_c_tmp`/
  `fc2_c_tmp` via `max_packed_route_slots(routed_rows, block_size_m,
  num_experts)`. For this deployment's decode (num_experts=1, degenerate
  1-expert/top-1 MoE -- see `Qwen36MLP`'s class docstring; topk=1;
  `select_route_block_size_m` picks `block_size_m=8` for small M): decode
  batch=2 gives `route_slots=9`.
- `run_w4a16_moe`'s own decode fast path (`use_direct_topk_routes`/
  TC-decode -- exactly what small-M bf16 gated decode with int32 `topk_ids`
  and no `expert_map` always takes) computes its scratch requirement
  **internally**, independently, as `route_slots_for_scratch = m * topk *
  block_size_m`. Same inputs (m=2, topk=1, block_size_m=8): **16**.

9 < 16 -- the caller-provided scratch is undersized for what the kernel's
own fast path will ask for. In eager mode `_get_c_tmp` silently absorbs
this: `if scratch is not None and scratch.numel() >= elements: return
scratch[:elements]` else (if not `is_current_stream_capturing()`) falls back
to a fresh `torch.empty(elements, ...)` -- no error, no wrong output, just a
throwaway allocation every call. That is why B1-R's eager gap-error
validation (cosine 0.999979-0.999999, gap p90=0.25) never observed this:
the undersized scratch was invisible in eager mode. `torch.cuda.graph`
capture refuses the same fallback outright (correctly -- allocating fresh
memory mid-capture with addresses not baked into the graph would be
unsound), which is how this surfaced.

This is inside sparkinfer's `moe/_shared/kernels/w4a16/{host,kernel}.py`
(`plan_w4a16_buffers` vs `run_w4a16_moe`'s inline `route_slots_for_scratch`
disagree with each other), not in this repo's call pattern. Per this
round's constraint ("sparkinfer 可以直接改, 本任务应该不需要") this was
**not** patched in sparkinfer -- worked around entirely in `runtime/model/
qwen36_model.py` instead (see `Qwen36MLP._w4a16_c_tmp_scratch`'s docstring
for the full derivation). Note for the record: `AGENTS.md`'s "External
dependencies" section says sparkinfer has been directly editable (`origin`
remote) since 2026-08-02, which contradicts an older, more restrictive
"read-only, hand off to the sparkinfer team" convention recorded elsewhere
-- the newer, dated statement in `AGENTS.md` should be treated as current.

### The fix (`runtime/model/qwen36_model.py`)

`Qwen36MLP` now owns two persistent, monotonically-grown scratch buffers
(`_w4a16_c_tmp_fc1`/`_w4a16_c_tmp_fc2`), sized via sparkinfer's own public
`packed_gemm_scratch_elements(size_n=..., route_slots=m*64, moe_
block_size=64, sms=...)` -- deliberately using `route_slots=m*64` /
`moe_block_size=64` (64 is the max of sparkinfer's own `_W4A16_ALLOWED_
ROUTED_SIZES=(8,16,32,48,64)`) rather than replicating `run_w4a16_moe`'s
internal fast-path selection logic. Proved (algebraically, in the method's
own docstring) that this dominates `route_slots_for_scratch=m*topk*
block_size_m_actual` for every real `block_size_m` sparkinfer can select:
both terms of `packed_gemm_scratch_elements`'s `min(size_n*route_slots,
sms*4*moe_block_size*256)` scale linearly with `moe_block_size`, and using
64 gives 4x headroom over block_size_m=8's extra 2x doubling. Grown lazily
and ONLY outside capture (`torch.cuda.is_current_stream_capturing()` guards
growth; decode always warms up eagerly at the exact same batch size
immediately before capturing it -- `Qwen36Backend._capture_decode_graphs`
-- so the scratch is already correctly sized by the time real capture
starts). For prefill (never graph-captured, `m` can be very large) this
changes nothing observable: if the persistent buffer is too small for a
given prefill `m`, `_get_c_tmp` falls back to a fresh `torch.empty(...)`
exactly as it did before this change (the "not capturing" branch is
untouched).

**Result after the fix**: `Qwen3.6 decode CUDA Graph captured at load (max
batch_size=2)`, `/debug/stats`'s `_cuda_graph_dbg` = `{"decode":
"captured"}`. Ran the full C-LIVE suite (64/67, above) against this build
and re-checked `_backend_stats_dbg` afterward:

```
decode_rounds: 4364
decode_graph_replays: 4364
```

**Every decode round in the entire C-LIVE run replayed the captured CUDA
Graph -- 4364/4364, 100%.** (The task's own reference point was Laguna's B2
result of 23/23 on a single request; this is the same signal at C-LIVE
scale.)

## 4. Bit-exact vs B1 eager, same checkpoint

`scripts/b2_verify_serving.py` (the existing B2 gate script, previously only
ever run against nvidia's checkpoint) -- parameterized by `--model-path`
(same convention as `verify_nvfp4_gemm_full_model_gap.py` /
`measure_nvfp4_gemm_memory_and_throughput.py`, commit `9e67e4a`; `_ROOT` also
switched from a hardcoded worktree path to `Path(__file__).resolve().
parent.parent`, so this script now runs correctly from whichever worktree
invokes it). Run against unsloth's checkpoint, `--slots 2 --steps 16
--max-seq-len 512`:

```
== 1. serial serving vs B1 eager (bit-exact gate) ==
  [PASS] prompt[0] greedy tokens identical (17 tokens)
  [PASS] prompt[1] greedy tokens identical (17 tokens)
== 2. batched decode vs serial decode ==       2/2 PASS
== 3. concurrency: 2 slots, isolation ==       2/2 PASS
== 4. CUDA Graph decode capture + replay ==
  [PASS] capture returned a batch size -- max_batch=2
  [PASS] capture left no recurrent state behind -- max|state|=0.0
  [PASS] prompt[0] graph replay == eager batched
  [PASS] prompt[1] graph replay == eager batched
  graph replays used: 32
== 5. prefix cache: (kv_hit, state_hit) and a real resume ==   3/3 PASS

14/14 checks passed
```

Serial serving is bit-exact vs B1 eager (greedy token ids identical).
**CUDA Graph replay is bit-exact vs eager batched decode** -- this is also
the correctness confirmation for the scratch-sizing fix above: if the
conservative-but-not-formula-exact scratch size were somehow wrong in a way
that corrupted output rather than merely erroring, this check would have
caught it (different tokens), not just "capture succeeded."

## Gates

- `/tmp/ci-sim/bin/python -m ruff check .` -- clean.
- `/tmp/ci-sim/bin/python -m pytest -q` -- 1036 passed, 172 skipped (exact
  baseline match). The fix touches only a GPU-only code path
  (`torch.cuda.*`) with no CPU-testable call sites in `tests/` (confirmed by
  grep before and after), so zero CPU test impact was expected and observed.
- `PYTHONPATH=<worktree> timeout 500 ~/.venvs/vllm/bin/python -m pytest -q`
  -- 1576 passed (exact baseline match).

## What was NOT verified

- **Concurrency beyond capacity=1** -- out of scope per the task
  ("本任务不测并发"); `b2_verify_serving.py`'s own check 3 (2-slot isolation)
  ran as part of the bit-exact gate above and passed, but the live server
  itself was only ever run at `capacity=1`.
- **The sparkinfer-side route_slots formula mismatch itself** -- not fixed
  upstream (see "Root cause" above); flagged here as a genuine finding for
  whoever next touches `sparkinfer/moe/_shared/kernels/w4a16/host.py`'s
  `plan_w4a16_buffers` or `kernel.py`'s `route_slots_for_scratch`. The
  `runtime/`-side workaround in this commit is a correct, verified
  (bit-exact) fix for THIS repo's call pattern, but every other caller of
  `make_w4a16_packed_buffers` for a small-M/direct-topk-eligible shape has
  the same latent undersizing if it is ever run under CUDA Graph capture.
- **Long-context / large-prefill CUDA Graph behavior** -- CUDA Graph in this
  backend only ever wraps decode (batch size 1..num_slots), never prefill;
  not exercised here beyond what B1-R/B2 already cover for prefill
  correctness (unchanged by this round).
- **The C-LIVE lost-wakeup check (item 5 in `c_live_smoke.py`)** -- passed
  (20/20 rapid-fire rounds, no stalls), but per that script's own docstring
  this is a known-weak check that would not reliably catch the specific
  race it targets even if present; a green run here is not strong evidence
  either way for this checkpoint specifically.
