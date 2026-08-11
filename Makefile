# BlackForge (qwen-sm120-runtime) — developer & ops tasks.
#
# Run `make help` for an overview. Server tuning is done through QSR_*
# environment variables (see README "Configuration"); `make serve` only
# sets the listen address.

PYTHON ?= python
NVCC ?= nvcc
HOST ?= 0.0.0.0
PORT ?= 8000
# Packages that hold production code (formatted + lint-strict). benchmarks/
# is diagnostic scratch and is lint-relaxed via pyproject per-file-ignores.
PKGS = runtime server loader model oracle tests tools
# `make smoke` targets an already-running server started the operational way
# (scripts/blackwellm_ctl.sh start / `make serve` with PORT=8100), which is
# why this does NOT default to the $(PORT) above: blackwellm_ctl.sh pins
# 8100 because 8000 on this host is already held by an unrelated service
# (see its own comment). Override at the command line if your server is
# somewhere else: `make smoke SMOKE_BASE_URL=http://host:port`.
SMOKE_BASE_URL ?= http://127.0.0.1:8100

.DEFAULT_GOAL := help

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install package with dev + serving extras (editable)
	$(PYTHON) -m pip install -e '.[dev,serving]'

install-cuda: ## Install the CUDA runtime extra (torch + kernel deps; see pyproject.toml)
	$(PYTHON) -m pip install -e '.[cuda]'

lint: ## Ruff lint gate for the whole repo (must stay green)
	$(PYTHON) -m ruff check .

format: ## Auto-fix lint issues and format the production packages
	$(PYTHON) -m ruff check . --fix
	$(PYTHON) -m ruff format $(PKGS)

format-check: ## Verify production packages are formatted (no writes)
	$(PYTHON) -m ruff format --check $(PKGS)

test: ## Run the CPU-only unit test suite
	$(PYTHON) -m pytest -q

verify-cuda: ## Smoke-test that an SM120 CUDA op executes
	$(PYTHON) -m tools.verify_cuda

smoke: ## P0-B C-LIVE: run the live-server smoke gate (server must already be running -- see scripts/blackwellm_ctl.sh smoke)
	$(PYTHON) scripts/c_live_smoke.py --base-url $(SMOKE_BASE_URL)

workloads: ## Print the frozen Phase-0 W1/W2 workload contracts
	$(PYTHON) -m benchmarks.workloads

ROUTER_SOURCE = runtime/kernels/laguna_router_sm120.cu
ROUTER_EXPORTS = runtime/kernels/laguna_router_sm120.exports
ROUTER_GENERATED_DIR = runtime/kernels/_generated
ROUTER_LIBRARY = $(ROUTER_GENERATED_DIR)/laguna_router_sm120.so
ROUTER_MANIFEST = $(ROUTER_GENERATED_DIR)/laguna_router_sm120.manifest.json
# SM120 target family: use the `f` (family-specific, forward-compatible) virtual
# architecture and matching real target, not the generic virtual arch paired with
# the `a`-suffixed real target. Feature gating for SM120's block-scaled MMA (NVFP4)
# lives on the *virtual* arch (`compute_120` vs `compute_120f`/`compute_120a`), not
# on `code=`. `arch=compute_120,code=sm_120a` (the old form) elaborates device code
# against the generic compute_120 feature set, then labels the cubin `sm_120a`; any
# kernel that needs the family-gated block-scaled MMA path compiles clean but the
# kernel body degrades to a `BPT.TRAP`/`EXIT` stub that faults at launch (verified
# 2026-08-01 against runtime/kernels/nvfp4_gemm_sm120.cu — see docs/investigation-queue.md
# B-5). `compute_120f,code=sm_120f` gets the same block-scaled MMA feature access as
# `compute_120a,code=sm_120a` (SASS-instruction-identical, verified byte-for-byte) while
# staying loadable across the whole 120 family (sm_121 / DGX Spark included), where
# `sm_120a` cubins are rejected. Today's router only uses baseline warp-shuffle ISA, so
# this change is a no-op for it (verified byte-identical SASS across both flag forms).
ROUTER_FLAGS = -std=c++17 -O3 --shared -Xcompiler -fPIC -Xcompiler -fvisibility=hidden -gencode arch=compute_120f,code=sm_120f -cudart static -Xlinker --version-script=$(ROUTER_EXPORTS)

build-laguna-router: ## Build the fixed-contract SM120 Laguna router artifact
	@mkdir -p $(ROUTER_GENERATED_DIR)
	@set -eu; tmp_library="$(ROUTER_LIBRARY).tmp"; \
	$(NVCC) $(ROUTER_FLAGS) $(ROUTER_SOURCE) -o "$$tmp_library"; \
	mv "$$tmp_library" "$(ROUTER_LIBRARY)"
	@$(PYTHON) -c 'import hashlib,json,subprocess,sys; from pathlib import Path; library=Path(sys.argv[1]); manifest=Path(sys.argv[2]); source=Path(sys.argv[3]); flags=sys.argv[4]; payload={"abi_version":1,"target_sm":"sm_120f","nvcc":subprocess.check_output([sys.argv[5],"--version"],text=True).strip(),"ptxas":subprocess.check_output(["ptxas","--version"],text=True).strip(),"compile_flags":flags,"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"runtime_git_sha":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"library_sha256":hashlib.sha256(library.read_bytes()).hexdigest(),"provenance":{"license":"Apache-2.0","upstream":"vLLM csrc/libtorch_stable/moe/topk_softmax_kernels.cu","specialization":"256-expert BF16/FP32 sigmoid top-k router"}}; temporary=manifest.with_suffix(".tmp"); temporary.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); temporary.replace(manifest)' "$(ROUTER_LIBRARY)" "$(ROUTER_MANIFEST)" "$(ROUTER_SOURCE)" "$(ROUTER_FLAGS)" "$(NVCC)"
	@$(MAKE) verify-laguna-router

verify-laguna-router: ## Verify the generated router ABI and dynamic dependencies
	@test -f $(ROUTER_LIBRARY)
	@test -f $(ROUTER_MANIFEST)
	@nm -D --defined-only $(ROUTER_LIBRARY) | awk '{print $$3}' | grep -Ex 'qsr_laguna_router_(abi_version|bf16|f32)' | wc -l | grep -qx 3
	@! readelf -d $(ROUTER_LIBRARY) | grep -Ei 'libtorch|vllm'
	@! ldd $(ROUTER_LIBRARY) | grep -Ei 'libtorch|vllm'

FP8_W8A8_SOURCE = runtime/kernels/fp8_w8a8_sm120.cu
FP8_W8A8_EXPORTS = runtime/kernels/fp8_w8a8_sm120.exports
FP8_W8A8_GENERATED_DIR = runtime/kernels/_generated
FP8_W8A8_LIBRARY = $(FP8_W8A8_GENERATED_DIR)/fp8_w8a8_sm120.so
FP8_W8A8_MANIFEST = $(FP8_W8A8_GENERATED_DIR)/fp8_w8a8_sm120.manifest.json
CUTLASS_ROOT ?= /home/bot/project/cutlass-4.6.1
FP8_W8A8_FLAGS = -std=c++17 -O3 --shared -Xcompiler -fPIC -Xcompiler -fvisibility=hidden -diag-suppress=20012,20013,20015 -gencode arch=compute_120f,code=sm_120f -cudart static -Xlinker --version-script=$(FP8_W8A8_EXPORTS) -I$(CUTLASS_ROOT)/include -I$(CUTLASS_ROOT)/tools/util/include

build-fp8-w8a8: ## Build the self-owned SM120 E4M3 W8A8 scaled-GEMM artifact
	@mkdir -p $(FP8_W8A8_GENERATED_DIR)
	@set -eu; tmp_library="$(FP8_W8A8_LIBRARY).tmp"; \
	$(NVCC) $(FP8_W8A8_FLAGS) $(FP8_W8A8_SOURCE) -o "$$tmp_library"; \
	mv "$$tmp_library" "$(FP8_W8A8_LIBRARY)"
	@$(PYTHON) -c 'import hashlib,json,subprocess,sys; from pathlib import Path; library=Path(sys.argv[1]); manifest=Path(sys.argv[2]); source=Path(sys.argv[3]); flags=sys.argv[4]; payload={"abi_version":2,"target_sm":"sm_120f","nvcc":subprocess.check_output([sys.argv[5],"--version"],text=True).strip(),"ptxas":subprocess.check_output(["ptxas","--version"],text=True).strip(),"compile_flags":flags,"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"runtime_git_sha":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"library_sha256":hashlib.sha256(library.read_bytes()).hexdigest(),"provenance":{"license":"Apache-2.0","upstream":"vLLM csrc/libtorch_stable/quantization/w8a8/cutlass/c3x","specialization":"Qwen FP8-channel no-bias E4M3 W8A8 per-token/per-channel scaled GEMM"}}; temporary=manifest.with_suffix(".tmp"); temporary.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); temporary.replace(manifest)' "$(FP8_W8A8_LIBRARY)" "$(FP8_W8A8_MANIFEST)" "$(FP8_W8A8_SOURCE)" "$(FP8_W8A8_FLAGS)" "$(NVCC)"
	@$(MAKE) verify-fp8-w8a8

verify-fp8-w8a8: ## Verify the generated W8A8 ABI and dynamic dependencies
	@test -f $(FP8_W8A8_LIBRARY)
	@test -f $(FP8_W8A8_MANIFEST)
	@nm -D --defined-only $(FP8_W8A8_LIBRARY) | awk '{print $$3}' | grep -Ex 'qsr_fp8_w8a8_(abi_version|dynamic_per_token_quant_sm120|workspace_size_sm120|scaled_mm_sm120)' | wc -l | grep -qx 4
	@! readelf -d $(FP8_W8A8_LIBRARY) | grep -Ei 'libtorch|vllm'
	@! ldd $(FP8_W8A8_LIBRARY) | grep -Ei 'libtorch|vllm'

serve: ## Start the OpenAI/Anthropic-compatible server (tune via QSR_* env)
	$(PYTHON) -m server.app --host $(HOST) --port $(PORT)

IQ2_MMA16_SOURCE = runtime/kernels/iq2_mma16.cu
IQ2_MMA16_EXPORTS = runtime/kernels/iq2_mma16.exports
IQ2_MMA16_GENERATED_DIR = runtime/kernels/_generated
IQ2_MMA16_LIBRARY = $(IQ2_MMA16_GENERATED_DIR)/iq2_mma16.so
IQ2_MMA16_MANIFEST = $(IQ2_MMA16_GENERATED_DIR)/iq2_mma16.manifest.json
IQ2_MMA16_FLAGS = -std=c++17 -O3 --shared -Xcompiler -fPIC -Xcompiler -fvisibility=hidden -gencode arch=compute_120f,code=sm_120f -cudart static -Xlinker --version-script=$(IQ2_MMA16_EXPORTS)

build-iq2-mma16: ## Build the self-owned SM120 IQ2_XS -> INT8 MMA grouped-MoE artifact
	@mkdir -p $(IQ2_MMA16_GENERATED_DIR)
	@set -eu; tmp_library="$(IQ2_MMA16_LIBRARY).tmp"; \
	$(NVCC) $(IQ2_MMA16_FLAGS) $(IQ2_MMA16_SOURCE) -o "$$tmp_library"; \
	mv "$$tmp_library" "$(IQ2_MMA16_LIBRARY)"
	@$(PYTHON) -c 'import hashlib,json,subprocess,sys; from pathlib import Path; library=Path(sys.argv[1]); manifest=Path(sys.argv[2]); source=Path(sys.argv[3]); flags=sys.argv[4]; payload={"abi_version":1,"target_sm":"sm_120f","nvcc":subprocess.check_output([sys.argv[5],"--version"],text=True).strip(),"ptxas":subprocess.check_output(["ptxas","--version"],text=True).strip(),"compile_flags":flags,"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"runtime_git_sha":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"library_sha256":hashlib.sha256(library.read_bytes()).hexdigest(),"provenance":{"license":"Apache-2.0","upstream":"self-owned runtime/kernels/iq2_mma16.cu","specialization":"DSV4 IQ2_XS routed-expert gate+up fused m16n8k16 INT8 MMA"}}; temporary=manifest.with_suffix(".tmp"); temporary.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); temporary.replace(manifest)' "$(IQ2_MMA16_LIBRARY)" "$(IQ2_MMA16_MANIFEST)" "$(IQ2_MMA16_SOURCE)" "$(IQ2_MMA16_FLAGS)" "$(NVCC)"
	@$(MAKE) verify-iq2-mma16

verify-iq2-mma16: ## Verify the generated IQ2 MMA16 ABI and dynamic dependencies
	@test -f $(IQ2_MMA16_LIBRARY)
	@test -f $(IQ2_MMA16_MANIFEST)
	@nm -D --defined-only $(IQ2_MMA16_LIBRARY) | awk '{print $$3}' | grep -Ex 'iq2_mma16_launch' | wc -l | grep -qx 1
	@! readelf -d $(IQ2_MMA16_LIBRARY) | grep -Ei 'libtorch|vllm'
	@! ldd $(IQ2_MMA16_LIBRARY) | grep -Ei 'libtorch|vllm'


verify-sparkinfer: ## Report which SparkInfer checkout the warm bfdiag daemon actually loaded
	@bf exec scripts/verify_sparkinfer_load.py --timeout-s 60

clean: ## Remove build and test caches
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true

.PHONY: help install install-cuda lint format format-check test verify-cuda smoke workloads build-laguna-router verify-laguna-router build-fp8-w8a8 verify-fp8-w8a8 serve build-iq2-mma16 verify-iq2-mma16 verify-sparkinfer clean
