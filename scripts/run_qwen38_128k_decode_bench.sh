#!/usr/bin/env bash
# Reproduce the 2026-08-15 Qwen3.8 128K decode matrix.
#
# Start the server in one terminal:
#   scripts/run_qwen38_128k_decode_bench.sh server
# Then run either or both measurement cells in another terminal:
#   scripts/run_qwen38_128k_decode_bench.sh c1
#   scripts/run_qwen38_128k_decode_bench.sh c4

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${QSR_BENCH_PYTHON:-/home/bot/.venvs/vllm/bin/python}
base_url=${QSR_BENCH_BASE_URL:-http://127.0.0.1:8300}
model_path=${QSR_BENCH_MODEL_PATH:-/home/bot/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/9c73e2daee1d0fd494ffbd1d8753f2174a953796}
# The recorded 2026-08-15 run generated its digit filler with this tokenizer.
# The server independently tokenized it to exactly 131072 Qwen3.8 tokens.
tokenizer_path=${QSR_BENCH_TOKENIZER_PATH:-/home/bot/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-NVFP4/snapshots/ccdaab7e68af2409599b8949a8f2685703c9bae5}
run_tag=${QSR_BENCH_TAG:-$(date -u +%Y%m%d_%H%M%S)}
result_dir=${QSR_BENCH_RESULT_DIR:-${repo_root}/benchmarks/fixtures}
server_label="Qwen3.8-27B-NVFP4; qwen36 backend; MTP K=3; CUDA Graph; FP8 KV; fused QKV W8A8; block_size=32; dynamic 19629342720-byte/4160-bundle pool; capacity=4; num_slots=4; logical max=256K/slot"

mkdir -p "${result_dir}"

run_cell() {
    local concurrency=$1
    local warm_rounds=$2
    local stem="qwen38_dynamic_128k_c${concurrency}_${run_tag}"

    "${python_bin}" "${repo_root}/benchmarks/server_perf_grid.py" \
        --base-url "${base_url}" \
        --model qwen3.8 \
        --tokenizer-path "${tokenizer_path}" \
        --server-label "${server_label}" \
        --endpoint completions \
        --contexts 128k \
        --concurrency "${concurrency}" \
        --max-tokens 256 \
        --warm-rounds "${warm_rounds}" \
        --out "${result_dir}/server_perf_grid_${stem}.json"
    curl --fail --silent --show-error "${base_url}/debug/traces" \
        --output "${result_dir}/server_trace_${stem}.json"
    curl --fail --silent --show-error "${base_url}/debug/stats" \
        --output "${result_dir}/server_stats_${stem}.json"
}

case ${1:-} in
    server)
        export HF_HUB_OFFLINE=1
        export QSR_SERVER_MODEL_PATH="${model_path}"
        export QSR_SERVER_BACKEND=qwen36
        export QSR_SERVED_MODEL_NAME=qwen3.8
        # block_size 32 (2026-08-16): measured strictly better than 16 on the
        # 131072-token workload (c1 107.99 vs 105.14 tok/s, TTFT 56.95 vs
        # 60.17 s) and far better than 64 (c4 warm 51.8 at bs64 vs 70.0 at
        # bs32 -- 64-token pages degrade the decode attention page walk).
        export QSR_SERVER_BLOCK_SIZE=32
        export QSR_SERVER_KV_CACHE_DTYPE=fp8_e4m3
        export QSR_SERVER_ENABLE_CUDAGRAPH=1
        export QSR_SERVER_ENABLE_PREFIX_CACHE=${QSR_BENCH_PREFIX_CACHE:-1}
        export QSR_SERVER_REQUEST_TIMEOUT_S=900
        export QSR_THINKING_CAPABLE=1
        export QSR_TRACE=1
        export QSR_DEBUG_REQUESTS=0
        unset QSR_PROFILE_ROUNDS
        cd "${repo_root}"
        exec "${python_bin}" -m server.app \
            --host 127.0.0.1 \
            --port 8300 \
            --capacity 4 \
            --num-slots 4 \
            --blocks-per-slot 16384 \
            --qwen-kv-mode elastic \
            --qwen-kv-pool-bytes 19629342720 \
            --qwen-kv-watermark-bundles 8 \
            --mtp \
            --mtp-k 3 \
            --tool-call-parser qwen3_coder
        ;;
    c1)
        run_cell 1 0
        ;;
    c4)
        run_cell 4 2
        ;;
    *)
        echo "usage: $0 {server|c1|c4}" >&2
        exit 2
        ;;
esac
