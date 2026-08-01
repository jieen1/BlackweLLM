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

workloads: ## Print the frozen Phase-0 W1/W2 workload contracts
	$(PYTHON) -m benchmarks.workloads

ROUTER_SOURCE = runtime/kernels/laguna_router_sm120.cu
ROUTER_EXPORTS = runtime/kernels/laguna_router_sm120.exports
ROUTER_GENERATED_DIR = runtime/kernels/_generated
ROUTER_LIBRARY = $(ROUTER_GENERATED_DIR)/laguna_router_sm120.so
ROUTER_MANIFEST = $(ROUTER_GENERATED_DIR)/laguna_router_sm120.manifest.json
ROUTER_FLAGS = -std=c++17 -O3 --shared -Xcompiler -fPIC -Xcompiler -fvisibility=hidden -gencode arch=compute_120,code=sm_120a -cudart static -Xlinker --version-script=$(ROUTER_EXPORTS)

build-laguna-router: ## Build the fixed-contract SM120 Laguna router artifact
	@mkdir -p $(ROUTER_GENERATED_DIR)
	@set -eu; tmp_library="$(ROUTER_LIBRARY).tmp"; \
	$(NVCC) $(ROUTER_FLAGS) $(ROUTER_SOURCE) -o "$$tmp_library"; \
	mv "$$tmp_library" "$(ROUTER_LIBRARY)"
	@$(PYTHON) -c 'import hashlib,json,subprocess,sys; from pathlib import Path; library=Path(sys.argv[1]); manifest=Path(sys.argv[2]); source=Path(sys.argv[3]); flags=sys.argv[4]; payload={"abi_version":1,"target_sm":"sm_120a","nvcc":subprocess.check_output([sys.argv[5],"--version"],text=True).strip(),"ptxas":subprocess.check_output(["ptxas","--version"],text=True).strip(),"compile_flags":flags,"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"runtime_git_sha":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"library_sha256":hashlib.sha256(library.read_bytes()).hexdigest(),"provenance":{"license":"Apache-2.0","upstream":"vLLM csrc/libtorch_stable/moe/topk_softmax_kernels.cu","specialization":"256-expert BF16/FP32 sigmoid top-k router"}}; temporary=manifest.with_suffix(".tmp"); temporary.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); temporary.replace(manifest)' "$(ROUTER_LIBRARY)" "$(ROUTER_MANIFEST)" "$(ROUTER_SOURCE)" "$(ROUTER_FLAGS)" "$(NVCC)"
	@$(MAKE) verify-laguna-router

verify-laguna-router: ## Verify the generated router ABI and dynamic dependencies
	@test -f $(ROUTER_LIBRARY)
	@test -f $(ROUTER_MANIFEST)
	@nm -D --defined-only $(ROUTER_LIBRARY) | awk '{print $$3}' | grep -Ex 'qsr_laguna_router_(abi_version|bf16|f32)' | wc -l | grep -qx 3
	@! readelf -d $(ROUTER_LIBRARY) | grep -Ei 'libtorch|vllm'
	@! ldd $(ROUTER_LIBRARY) | grep -Ei 'libtorch|vllm'

serve: ## Start the OpenAI/Anthropic-compatible server (tune via QSR_* env)
	$(PYTHON) -m server.app --host $(HOST) --port $(PORT)

verify-sparkinfer: ## Report which SparkInfer checkout the warm bfdiag daemon actually loaded
	@bf exec scripts/verify_sparkinfer_load.py --timeout-s 60

clean: ## Remove build and test caches
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true

.PHONY: help install install-cuda lint format format-check test verify-cuda workloads build-laguna-router verify-laguna-router serve verify-sparkinfer clean
