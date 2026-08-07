# Qwen3.6-27B-NVFP4 128K/c4 benchmark server environment.
# Reproduces the exact config used for the 2026-08-06 M32/no-profiling runs
# (capacity=4, num_slots=5, block_size=16, 256K tokens/slot, MTP K=3,
# FP8 e4m3 KV, CUDA Graph + persistent prefix cache, 92% GPU mem util).
#
# Usage:
#   set -a; source scripts/qwen36_128k_bench_env.sh; set +a
#
# QSR_PROFILE_ROUNDS is deliberately NOT set here: it defaults to off.
# Set QSR_PROFILE_ROUNDS=1 before sourcing only for the per-round phase
# diagnostic (it forces per-round GPU drains and is NOT a benchmark config).

export QSR_SERVER_MODEL_PATH=/home/bot/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-NVFP4/snapshots/ccdaab7e68af2409599b8949a8f2685703c9bae5
export QSR_SERVER_PRODUCTION=1
export QSR_SERVER_CAPACITY=4
export QSR_SERVER_NUM_SLOTS=5
export QSR_SERVER_BLOCK_SIZE=16
export QSR_SERVER_BLOCKS_PER_SLOT=16384
export QSR_SERVER_ENABLE_CUDAGRAPH=1
export QSR_SERVER_ENABLE_PREFIX_CACHE=1
export QSR_SERVER_ENABLE_SESSION_AFFINITY=0
export QSR_SERVER_ENABLE_DFLASH=0
export QSR_SERVER_ENABLE_MTP=1
export QSR_SERVER_MTP_K=3
export QSR_SERVER_KV_CACHE_DTYPE=fp8_e4m3
export QSR_SERVER_GPU_MEM_UTIL=0.92
export QSR_SERVER_REQUEST_TIMEOUT_S=0
export QSR_TOOL_CALL_PARSER=qwen3_coder
export QSR_SERVED_MODEL_NAME=qwen3.6
export QSR_DEBUG_REQUESTS=0
export HF_HUB_OFFLINE=1

# Verify-attention numerics mode (2026-08-07, notes S19): this is the
# THROUGHPUT-HEADLINE profile, so it explicitly opts into the fast mode
# (M32 raw-FP8 verifier + adaptive re-chunking). The runtime DEFAULT is
# the 08-05 quality mode; fast mode costs code quality (0.8902/0.8659 vs
# 0.9268/0.8902) and is for perf measurement only. sparkinfer treats "1"
# as the only enable value, so "0" opts out of each half.
export SPARKINFER_QWEN36_VERIFY_M16=0
export SPARKINFER_QWEN36_VERIFY_NO_ADAPTIVE=0
