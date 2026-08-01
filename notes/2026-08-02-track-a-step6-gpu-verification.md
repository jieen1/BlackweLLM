# Track A step 6 GPU verification (2026-08-02)

Evidence for the two gates `docs/implementation-plan.md` §6 step 6 writes down for
the A4 loader adapter split (compressed-tensors adapter carved out of
`runtime/model_loading.py`/`runtime/model/laguna_model.py`/`runtime/model/
_weight_loading.py` into `runtime/loading/{common,compressed_tensors,
language_model_only}.py`, no behavior change for Laguna): **per-tensor checksum
equality** and **greedy bit-exact**.

## 0. Method: same commit, two worktrees, not "trust the diff"

The refactor is uncommitted at verification time, so "old" is simply `HEAD`
(`d87c7ef`) and "new" is this worktree's working tree. Rather than compare
against a written-down expectation, both were run for real:

- A throwaway detached worktree was created at `HEAD` (`git worktree add
  --detach <tmp> HEAD`), same pattern as
  `2026-08-02-track-a-step5-gpu-verification.md` used for step 5's before/after
  comparison.
- The gitignored, prebuilt `runtime/kernels/_generated/laguna_router_sm120.so` +
  its manifest (not tracked in git, normally produced by a build step) were
  copied from this worktree into the throwaway one so its server could start at
  all -- justified because this step's diff touches zero kernel source, so the
  binary is identical either way (its own SHA256-against-manifest check, which
  `runtime/laguna_router.py::LagunaRouterLibrary.load()` enforces unconditionally,
  confirms this rather than assuming it).
- A real footgun caught and fixed before trusting any result: this host's
  `~/.venvs/vllm` has `blackwellm` installed *editable*, and that editable
  install's finder (`__editable___blackwellm_0_1_0_finder.py`) hard-wires
  `runtime`/`server`/etc. to a **static path pointing at the main worktree**
  (`/home/bot/project/qwen-sm120-runtime`), independent of cwd. Python's
  `sys.meta_path` checks the standard `PathFinder` (cwd-based) before that
  custom finder, so a script run with the correct cwd/`sys.path[0]` resolves
  correctly -- but `python <script.py>` puts the *script's own directory* on
  `sys.path[0]`, not cwd, so the first version of the per-tensor checksum
  script below silently would have loaded and hashed the **main worktree's**
  code with no error at all. Caught by a defensive assertion
  (`runtime.__file__.startswith(expected_root)`) that failed loudly on the
  first attempt; fixed by explicitly `sys.path.insert(0, expected_root)`
  rather than trusting cwd. Recorded here so the next person who writes a
  standalone verification script against this repo doesn't repeat it.

## 1. Per-tensor checksum equality

Script: `runtime.laguna_config.build_laguna_config` + `init_laguna_distributed_
environment` + `runtime.model_loading.load_laguna_model` (the same production
call sequence `LagunaBackend.__init__` uses), then SHA256 of the raw bytes of
every `model.named_parameters()` and `model.named_buffers()` tensor (bit-cast via
`.view(torch.uint8)` so bf16/f8_e4m3/f32 all hash uniformly), plus a float64 sum
as a secondary, human-readable number.

Checkpoint: `poolside/Laguna-S-2.1-NVFP4`
(`.../snapshots/07614121b31898586430f189d27a25a0be310843`), the real production
checkpoint, not a synthetic fixture.

Ran once per worktree, `torch.set_grad_enabled(False)`, single real CUDA device.

**Result: 724/724 tensors present in both, 0 shape/dtype mismatches, 0 SHA256
mismatches, 0 sum mismatches.** Every single loaded parameter and buffer
(attention QKV/O, dense MLP, embeddings, `lm_head`, RMSNorm weights, and the
`k_scale`/`v_scale`/`_k_scale`/`_v_scale` KV-cache-scale Parameters/buffers the
adapter split's `apply_kv_cache_scale_post_load` touches) is byte-for-byte
identical between the pre-refactor and post-refactor code. (MoE expert weights
are not in this count -- confirmed by design, not a gap: they never go through
`load_weights()`/`model_loading.py` at all, loaded directly by
`runtime/backends/laguna_sparkinfer_moe.py` instead, untouched by this step --
see `runtime/loading/compressed_tensors.py`'s module docstring for the scope
decision.)

This is the gate that actually exercises the split: `iterate_safetensors_
checkpoint`/`assert_all_params_loaded`/`apply_kv_cache_scale_post_load` moved to
`runtime/loading/common.py`, `IGNORE_WEIGHT_SUFFIXES`/`remap_kv_scale_name` moved
to `runtime/loading/compressed_tensors.py`, and `load_laguna_model`/
`load_laguna_dflash_draft_model` gained a `language_model_only` parameter whose
default (`False`) wraps the same stream in a no-op pass-through generator. A
silent value change anywhere in that move would show up here as a SHA256
mismatch; none did.

## 2. Greedy bit-exact (real HTTP server, before vs. after)

Same pattern as step 5's verification: started the real production server
(`scripts/blackwellm_ctl.sh start`, default config -- `QSR_SERVER_ENABLE_DFLASH=1`,
so this also exercises `load_laguna_dflash_draft_model`'s draft-model loading
path, not just the main model) on this worktree's code, sent five fixed
`/v1/completions` requests (`temperature=0`, `max_tokens=64`), saved
`choices[0].text` and `finish_reason` for each. Stopped it, started the same
script from the throwaway `HEAD` worktree, sent the identical five requests.

Prompts: "The capital of France is", "Explain the theory of relativity in one
paragraph:", "def fibonacci(n):", "Write a haiku about the ocean.", "The three
laws of thermodynamics are:".

**Result: all five `choices[0].text` values are byte-for-byte identical between
the two servers, and `finish_reason` matches (`length` all five times).**

## What this does and does not cover

Covered directly: the dense/non-MoE weight-loading path for both the main model
and the DFlash draft model, through the actual production entry points, against
the real checkpoint, end to end through a real HTTP server with speculative
decoding enabled.

Not covered, honestly:

- **The `language_model_only=True` branch was never exercised on GPU, or
  against any real checkpoint at all.** Laguna has no vision tower, so every
  real call in both of the above runs used the default (`False`), a no-op
  pass-through. The `True` branch (B0-1a's vision-tensor filter) is verified
  only by the constructed-tensor-name CPU tests in
  `tests/test_loading_language_model_only.py`; see that module's docstring and
  the accompanying implementation-plan-step report for exactly what is and is
  not claimed there.
- **MoE expert weight loading** (`runtime/backends/laguna_sparkinfer_moe.py`)
  is untouched by this step and therefore not part of what this verification
  needed to re-prove -- called out above only so its absence from the 724-tensor
  count isn't mistaken for a gap.
- Performance and acceptance-rate regression were not re-measured here: this
  step's diff is confined to loader plumbing that runs once at startup, before
  any decode step, so it has no plausible mechanism to move either number, and
  step 5's verification already established the harness/config this would use.
  If that assumption is ever in doubt, `bf exec benchmarks/acceptance_regression.py`
  against a warm daemon is the established way to check it (see step 5's note).
