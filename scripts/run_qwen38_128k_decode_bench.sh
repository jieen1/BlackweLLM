#!/usr/bin/env bash
# Reproduce the latest Qwen3.8 DSpark 128K decode matrix.
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
speculative=${QSR_BENCH_SPECULATIVE:-dspark}
dspark_draft_model=${QSR_BENCH_DSPARK_MODEL_PATH:-/home/bot/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-DSpark/snapshots/85ef153be924f17ce4bf62726954eeaa4a73e854}
block_size=${QSR_BENCH_BLOCK_SIZE:-128}
blocks_per_slot=${QSR_BENCH_BLOCKS_PER_SLOT:-$((262144 / block_size))}

case "${speculative}" in
    mtp)
        server_label="Qwen3.8-27B-NVFP4; qwen36 backend; MTP K=3; CUDA Graph; FP8 KV; fused QKV W8A8; block_size=${block_size}; dynamic 19629342720-byte/4160-bundle pool; capacity=4; num_slots=4; logical max=256K/slot"
        server_spec_args=(--mtp --mtp-k 3)
        ;;
    dspark)
        server_label="Qwen3.8-27B-NVFP4; qwen36 backend; DSpark K=7; ragged verify + CUDA Graph accept; FP8 KV; fused QKV W8A8; block_size=${block_size}; dynamic 19629342720-byte/4160-bundle pool; capacity=4; num_slots=4; logical max=256K/slot; per-slot + persistent prefix cache; SGLang-compatible eager accepted-prefix KV injection"
        server_spec_args=(
            --dspark
            --dspark-k "${QSR_BENCH_DSPARK_K:-7}"
            --dspark-draft-model "${dspark_draft_model}"
        )
        ;;
    plain)
        server_label="Qwen3.8-27B-NVFP4; qwen36 backend; plain decode; CUDA Graph; FP8 KV; fused QKV W8A8; block_size=${block_size}; dynamic 19629342720-byte/4160-bundle pool; capacity=4; num_slots=4; logical max=256K/slot; prefix cache=off"
        server_spec_args=()
        ;;
    *)
        echo "QSR_BENCH_SPECULATIVE must be mtp, dspark, or plain (got ${speculative@Q})" >&2
        exit 2
        ;;
esac

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
        # The current DSpark parity profile (2026-08-19) uses block_size 128,
        # the same page shape as the retained SGLang comparison. Override with
        # QSR_BENCH_BLOCK_SIZE for an explicit historical A/B.
        export QSR_SERVER_BLOCK_SIZE="${block_size}"
        export QSR_SERVER_KV_CACHE_DTYPE=fp8_e4m3
        export QSR_SERVER_ENABLE_CUDAGRAPH=1
        export QSR_SERVER_ENABLE_PREFIX_CACHE=${QSR_BENCH_PREFIX_CACHE:-1}
        export QSR_SERVER_ENABLE_MTP=0
        export QSR_SERVER_ENABLE_DSPARK=0
        export QSR_PREFILL_CHUNK=${QSR_PREFILL_CHUNK:-8192}
        export QSR_ADMISSION_COALESCE_MS=${QSR_ADMISSION_COALESCE_MS:-10}
        export QSR_PREFIX_CACHE_IN_BATCH_DEDUP=${QSR_PREFIX_CACHE_IN_BATCH_DEDUP:-1}
        export QSR_QWEN36_MLP_FP4_QUANT=${QSR_QWEN36_MLP_FP4_QUANT:-flashinfer}
        export QSR_QWEN36_MLP_W4A4=${QSR_QWEN36_MLP_W4A4:-1}
        export QSR_QWEN36_MLP_W4A4_ALL=${QSR_QWEN36_MLP_W4A4_ALL:-1}
        export QSR_QWEN36_PREFILL_ATTN_BACKEND=${QSR_QWEN36_PREFILL_ATTN_BACKEND:-flashinfer}
        export QSR_QWEN36_GDN_PREFILL_BACKEND=${QSR_QWEN36_GDN_PREFILL_BACKEND:-flashinfer}
        export QSR_QWEN36_DSPARK_CUDA_GRAPH=1
        export QSR_QWEN36_DSPARK_REQUIRE_CG=0
        export QSR_SERVER_REQUEST_TIMEOUT_S=900
        export QSR_THINKING_CAPABLE=1
        export QSR_TRACE=1
        export QSR_DEBUG_REQUESTS=0
        if [[ -n "${QSR_BENCH_PROFILE_ROUNDS:-}" ]]; then
            export QSR_PROFILE_ROUNDS="${QSR_BENCH_PROFILE_ROUNDS}"
        else
            unset QSR_PROFILE_ROUNDS
        fi
        if [[ "${speculative}" == "dspark" ]]; then
            export QSR_SERVER_ENABLE_DSPARK=1
            export QSR_QWEN36_DSPARK_VERIFY_MODE=${QSR_BENCH_DSPARK_VERIFY_MODE:-compact}
            export QSR_QWEN36_DSPARK_REQUIRE_CG=1
        elif [[ "${speculative}" == "plain" ]]; then
            export QSR_SERVER_ENABLE_PREFIX_CACHE=0
        fi
        cd "${repo_root}"
        exec "${python_bin}" -m server.app \
            --host 127.0.0.1 \
            --port 8300 \
            --capacity 4 \
            --num-slots 4 \
            --blocks-per-slot "${blocks_per_slot}" \
            --qwen-kv-mode elastic \
            --qwen-kv-pool-bytes 19629342720 \
            --qwen-kv-watermark-bundles 8 \
            "${server_spec_args[@]}" \
            --tool-call-parser qwen3_coder
        ;;
    c1)
        run_cell 1 0
        ;;
    c4)
        run_cell 4 2
        ;;
    *)
        echo "usage: $0 {server|c1|c4} (QSR_BENCH_SPECULATIVE=dspark|mtp|plain)" >&2
        exit 2
        ;;
esac
