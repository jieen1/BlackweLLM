# Repository Guidelines

## Project Structure & Module Organization

Actual structure (post-B5 modularization, 2026-07-22):

- `runtime/`: Core inference engine (B5 模块化后拆分为多个域):
  - `direct_model_runner.py`: Qwen3.6 main runner class (MTP/prefill/decode)
  - `block_pool.py`: Paging/prefix-cache infrastructure (Block, BlockPool, hash)
  - `metadata_builders.py`: Attention/GDN metadata construction
  - `cuda_graphs.py`: CUDA Graph capture/replay (CapturedBatchDecodeGraph, CapturedMTPDraftStepGraph)
  - `mtp_accept.py`: MTP accept/reject logic (pure functions)
  - `compat_vllm.py`: B7-V1 single-point vLLM dependency consolidation
  - `sampling.py`: Temperature/top-k/top-p sampling primitives
  - `backends/laguna.py`: Laguna-S-2.1 backend (sparkinfer MoE kernel, SWA ring KV, DFlash)
- `server/`: OpenAI + Anthropic dual-protocol API (streaming, tools, thinking)
  - `app.py`: FastAPI application
  - `engine.py`: Continuous-batching server engine
  - `formats/`: Protocol adapters (openai, anthropic, stream, tools, thinking)
- `benchmarks/`: Reproducible performance measurements + fixtures
  - `fixtures/`: speed_baseline.json, golden/, laguna_vllm_baseline.json
- `tests/`: CPU-only unit tests, no model weights required
- `notes/`: Design documents and investigation records
- `docs/`: roadmap.md, architecture.md

Keep components small and layer boundaries explicit. Do not embed
backend-specific calls throughout model code; route all vLLM imports through
`runtime/compat_vllm.py`.

## Build, Test, and Development Commands

Install the lightweight development tools, then run these repository-root
commands:

```bash
python -m pip install -e '.[dev]'        # development dependencies
python -m pip install -e '.[cuda]'       # PyTorch CUDA runtime
python -m pytest -q                      # correctness suite
python -m pytest tests/test_gdn.py -q   # focused regression
python -m benchmarks.workloads            # print frozen W1/W2 contracts
```

Keep environment setup, CUDA/toolchain versions, and model-location variables
in `README.md` or dedicated developer documentation. Do not make unit tests require
downloading model weights unless explicitly marked as integration tests.

## Coding Style & Naming Conventions

Use Python with four-space indentation, `snake_case` for modules/functions,
`PascalCase` for classes, and type annotations on public interfaces. Name CUDA
sources descriptively (for example, `nvfp4_gemm_sm120.cu`). Favor clear,
fixed-scope interfaces over generic abstractions: this runtime supports one
model family, one GPU architecture, one GPU, and at most four concurrent slots.

Add a formatter and linter with the first Python code (for example, Ruff), and
run them before submitting changes. Keep generated packed weights, profiles,
and large checkpoints out of Git.

## Testing Guidelines

Add tests beside each capability using `test_<unit>.py` and
`test_<behavior>` names. Compare model layers, logits, MTP acceptance, and GDN
state against the vLLM oracle. Cover prefill, multi-step decode, slot reset and
reuse, batches 1--4, and CUDA Graph replay. For quantized kernels, record
error metrics and top-k-logit agreement; greedy fixtures must not show
systematic token drift.

## Commits & Pull Requests

There is no Git history yet, so use concise imperative commit subjects, e.g.
`Add GDN state reset test`. Keep commits narrowly scoped. Pull requests should
state the affected workload, correctness evidence, benchmark comparison (using
accepted tokens/s and ITL), hardware/CUDA environment, and any changed runtime
assumptions. Include profiler screenshots only when they substantiate a
performance claim.
