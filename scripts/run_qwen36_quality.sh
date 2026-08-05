#!/usr/bin/env bash
# Qwen3.6-27B quality-suite rerun orchestrator (parallel + resumable) -- our runtime only.
#
# Reproduces the quality evidence recorded in README.md ("Quality validation",
# 2026-07-21/22) on the current build, served by this repo's own runtime
# (server.app / qwen36 backend). No external inference engine is involved.
#
#   * MMLU-Pro  : 414 stratified, thinking, 5-shot CoT, greedy,
#                 max_tokens=32768        (historical: 84.54% vs official 86.2)
#   * HumanEval : evalplus HumanEval+, greedy, max_tokens=768, concurrency 16
#                 (historical: our runtime 44.5%/43.3%)
#   * four dims : tool / agent / longctx / code (quality_regression.py,
#                 concurrency 4, code max_tokens=4096)
#
# Server profiles (2026-08-05 project decision): context is sized to what the
# benchmark actually needs instead of the historical 256K ceiling, and
# concurrency is raised as far as GPU memory allows:
#   * suite-fast (tool/agent/code + HumanEval 768): 32K slots, capacity 8
#     (blocks_per_slot=2048 x block_size=16; longest case is code 4096 +
#     ~160 prompt tokens, so 32K is 7x headroom)
#   * longctx: 139264-token slots (blocks_per_slot=8704), capacity 4 --
#     just above the 128K needle + 2048 answer tokens, instead of 256K
#   * mmlu : 64K slots, capacity 8 (historical MMLU profile: blocks_per_slot
#     4096 x block_size=16)
# Both with prefix_cache=on, fp8_e4m3 KV, and MTP speculative decoding
# (historical launch: --speculative-config '{"method":"mtp",
# "num_speculative_tokens": 3}', see /home/bot/vllm_server/vllm_ctl.sh).
# This rerun enables MTP (K=3, matching the historical speculative config)
# and BOTH MTP's own anchor/draft/sync/verify CUDA Graphs and the decode CUDA
# Graph (QSR_SERVER_ENABLE_CUDAGRAPH=1 / QSR_QWEN36_MTP_CUDA_GRAPH default on)
# -- end-to-end proof on the self-built runtime, per project decision.
# NOTE: enabling decode CUDA Graph requires one extra physical slot for the
# capture warmup (ServerEngine invariant: num_slots >= capacity + 1), so
# num_slots = capacity + 1 in every profile below.
# capacity -- the concurrency knob -- is unchanged; pool sizing does not
# affect outputs, only memory headroom.
# Request timeout is DISABLED (QSR_SERVER_REQUEST_TIMEOUT_S=0) to match the
# historical server: the 600s cap (added 2026-07-22 19:46, commit a1fac04,
# AFTER the July quality runs) aborts 128K longctx and max_tokens=4096 code
# requests mid-generation on the slower current build. Generation parameters
# are unchanged. Client-side harness timeouts are set to 3600s as a defensive
# floor (a live request that is merely slow must never be double-killed).
#
# Parallelism:
#   * the four quality dims run as 4 independent processes against the same
#     server (each resumable via <out>.work/*.jsonl), merged at the end;
#   * MMLU-Pro runs as N shard processes (default 4 x concurrency 2 = the
#     historical concurrency 8), each with its own .shard*.jsonl, merged by
#     `mmlu_pro_eval.py --merge`.
#
# Resume: every command is idempotent. Re-run the same command after an
# interruption; completed dims/shard questions are skipped from checkpoints.
#
# Usage:
#   scripts/run_qwen36_quality.sh env-check
#   scripts/run_qwen36_quality.sh server start suite|mmlu
#   scripts/run_qwen36_quality.sh server stop [suite|mmlu]
#   scripts/run_qwen36_quality.sh ours-suite        # 4-dim quality
#   scripts/run_qwen36_quality.sh ours-longctx      # longctx on 128K+ server
#   scripts/run_qwen36_quality.sh ours-humaneval    # HumanEval+ 768 (README row)
#   scripts/run_qwen36_quality.sh mmlu              # MMLU-Pro 414 sharded
#   scripts/run_qwen36_quality.sh compare
#   scripts/run_qwen36_quality.sh all
#
# Env overrides: RUN_LABEL, MMLU_SHARDS, KEEP_SERVER, PORT.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# NOTE: the venv is named after the pre-removal era; it hosts the repo's
# editable install + eval deps. `server.app` and the qwen36 backend import no vllm.
PY=/home/bot/.venvs/vllm/bin/python
LOCK_TOOL=/tmp/gpu_lock.sh
LOCK_NAME=qwen36-quality-rerun

PORT="${PORT:-8300}"
ROOT_URL="http://127.0.0.1:${PORT}"
BASE_URL="${ROOT_URL}/v1"
MODEL="${MODEL:-qwen3.6}"
RUN_LABEL="${RUN_LABEL:-qwen36_$(date +%Y%m%d)}"
MMLU_SHARDS="${MMLU_SHARDS:-4}"
KEEP_SERVER="${KEEP_SERVER:-0}"
# MTP + CUDA Graph on by default for this rerun (project decision). MTP K=3
# matches the historical vLLM speculative config exactly.
CUDAGRAPH="${CUDAGRAPH:-1}"
MTP="${MTP:-1}"
MTP_K="${MTP_K:-3}"

MODEL_SNAPSHOT="/home/bot/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-NVFP4/snapshots/ccdaab7e68af2409599b8949a8f2685703c9bae5"

OUT_DIR="evalplus_results/quality"
OFFICIAL_DIR="evalplus_results/official"
HUMANEVAL_DIR="evalplus_results/humaneval"
SUITE_OUT="$OUT_DIR/${RUN_LABEL}.json"
MMLU_OUT="$OFFICIAL_DIR/mmlu_pro_think_${RUN_LABEL}.json"
HE_OUT="$HUMANEVAL_DIR/our_runtime_${RUN_LABEL}.jsonl"
LOG_DIR="logs/quality"
mkdir -p "$LOG_DIR" "$OUT_DIR" "$OFFICIAL_DIR" "$HUMANEVAL_DIR"

say() { echo "[run-qwen36-quality] $*"; }

cmd_env_check() {
    say "== environment check =="
    "$PY" -c "import evalplus, aiohttp, datasets; print('deps ok: evalplus/aiohttp/datasets')" \
        || { say "FATAL: missing deps in $PY"; exit 1; }
    [ -f "$MODEL_SNAPSHOT/config.json" ] || { say "FATAL: unsloth snapshot missing"; exit 1; }
    [ -d "/home/bot/.cache/huggingface/datasets/TIGER-Lab___mmlu-pro" ] \
        || say "WARN: MMLU-Pro cache not found (will download on first eval)"
    local mem_free
    mem_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
    say "GPU free memory: ${mem_free} MiB (suite needs ~55GiB, mmlu needs ~45GiB)"
    [ -n "$(ss -ltn 2>/dev/null | rg ":$PORT\b")" ] && say "WARN: port $PORT already bound"
    "$LOCK_TOOL" status 2>/dev/null || true
    say "ok"
}

cmd_lock_acquire() {
    "$LOCK_TOOL" acquire "$LOCK_NAME" 3600
}

cmd_lock_release() {
    "$LOCK_TOOL" release "$LOCK_NAME"
}

server_stop_port() {
    local port="$1"
    local pids
    pids=$(pgrep -f "server\.app.*--port $port" 2>/dev/null | sort -u)
    local f pid
    for f in "$LOG_DIR"/suite.pid "$LOG_DIR"/mmlu.pid "$LOG_DIR"/longctx.pid; do
        if [ -f "$f" ]; then
            pid=$(cat "$f")
            [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && pids="$pids $pid"
        fi
    done
    pids=$(printf '%s\n' $pids | sort -u | tr -s '\n' ' ')
    if [ -n "$pids" ]; then
        say "stopping pid(s): $pids"
        kill -9 $pids 2>/dev/null || true
        sleep 3
    else
        say "no server process on port $port"
    fi
}

server_alive() {
    curl -fs -m 3 --noproxy '*' "$ROOT_URL/v1/models" 2>/dev/null \
        | grep -q '"id": *"qwen3.6"'
}

server_maybe_stop() {
    if [ "$KEEP_SERVER" = 1 ]; then
        say "KEEP_SERVER=1: leaving server up on port $PORT"
        return 0
    fi
    if [ "$SERVER_STARTED" = 1 ]; then
        cmd_server stop
    fi
}

server_start_suite() {
    QSR_SERVER_MODEL_PATH="$MODEL_SNAPSHOT" \
    QSR_SERVER_PRODUCTION=1 \
    QSR_SERVER_CAPACITY=8 \
    QSR_SERVER_NUM_SLOTS=9 \
    QSR_SERVER_BLOCK_SIZE=16 \
    QSR_SERVER_BLOCKS_PER_SLOT=2048 \
    QSR_SERVER_ENABLE_CUDAGRAPH="$CUDAGRAPH" \
    QSR_SERVER_ENABLE_PREFIX_CACHE=1 \
    QSR_SERVER_ENABLE_SESSION_AFFINITY=0 \
    QSR_SERVER_ENABLE_DFLASH=0 \
    QSR_SERVER_ENABLE_MTP="$MTP" \
    QSR_SERVER_MTP_K="$MTP_K" \
    QSR_SERVER_KV_CACHE_DTYPE=fp8_e4m3 \
    QSR_SERVER_GPU_MEM_UTIL=0.85 \
    QSR_SERVER_REQUEST_TIMEOUT_S=0 \
    QSR_TOOL_CALL_PARSER=qwen3_coder \
    QSR_SERVED_MODEL_NAME="qwen3.6" \
    QSR_DEBUG_REQUESTS=0 \
    HF_HUB_OFFLINE=1 \
    setsid nohup "$PY" -m server.app --host 127.0.0.1 --port "$PORT" \
        > "$LOG_DIR/server_suite_${RUN_LABEL}.log" 2>&1 < /dev/null &
    echo $! > "$LOG_DIR/suite.pid"
    say "suite-fast server starting (8 slots x 32K, port $PORT), log: $LOG_DIR/server_suite_${RUN_LABEL}.log"
}

server_start_longctx() {
    QSR_SERVER_MODEL_PATH="$MODEL_SNAPSHOT" \
    QSR_SERVER_PRODUCTION=1 \
    QSR_SERVER_CAPACITY=4 \
    QSR_SERVER_NUM_SLOTS=5 \
    QSR_SERVER_BLOCK_SIZE=16 \
    QSR_SERVER_BLOCKS_PER_SLOT=8704 \
    QSR_SERVER_ENABLE_CUDAGRAPH="$CUDAGRAPH" \
    QSR_SERVER_ENABLE_PREFIX_CACHE=1 \
    QSR_SERVER_ENABLE_SESSION_AFFINITY=0 \
    QSR_SERVER_ENABLE_DFLASH=0 \
    QSR_SERVER_ENABLE_MTP="$MTP" \
    QSR_SERVER_MTP_K="$MTP_K" \
    QSR_SERVER_KV_CACHE_DTYPE=fp8_e4m3 \
    QSR_SERVER_GPU_MEM_UTIL=0.85 \
    QSR_SERVER_REQUEST_TIMEOUT_S=0 \
    QSR_TOOL_CALL_PARSER=qwen3_coder \
    QSR_SERVED_MODEL_NAME="qwen3.6" \
    QSR_DEBUG_REQUESTS=0 \
    HF_HUB_OFFLINE=1 \
    setsid nohup "$PY" -m server.app --host 127.0.0.1 --port "$PORT" \
        > "$LOG_DIR/server_longctx_${RUN_LABEL}.log" 2>&1 < /dev/null &
    echo $! > "$LOG_DIR/longctx.pid"
    say "longctx server starting (4 slots x 139264 tokens, port $PORT), log: $LOG_DIR/server_longctx_${RUN_LABEL}.log"
}

server_start_mmlu() {
    QSR_SERVER_MODEL_PATH="$MODEL_SNAPSHOT" \
    QSR_SERVER_PRODUCTION=1 \
    QSR_SERVER_CAPACITY=8 \
    QSR_SERVER_NUM_SLOTS=9 \
    QSR_SERVER_BLOCK_SIZE=16 \
    QSR_SERVER_BLOCKS_PER_SLOT=4096 \
    QSR_SERVER_ENABLE_CUDAGRAPH="$CUDAGRAPH" \
    QSR_SERVER_ENABLE_PREFIX_CACHE=1 \
    QSR_SERVER_ENABLE_SESSION_AFFINITY=0 \
    QSR_SERVER_ENABLE_DFLASH=0 \
    QSR_SERVER_ENABLE_MTP="$MTP" \
    QSR_SERVER_MTP_K="$MTP_K" \
    QSR_SERVER_KV_CACHE_DTYPE=fp8_e4m3 \
    QSR_SERVER_GPU_MEM_UTIL=0.85 \
    QSR_SERVER_REQUEST_TIMEOUT_S=0 \
    QSR_TOOL_CALL_PARSER=qwen3_coder \
    QSR_SERVED_MODEL_NAME="qwen3.6" \
    QSR_DEBUG_REQUESTS=0 \
    HF_HUB_OFFLINE=1 \
    setsid nohup "$PY" -m server.app --host 127.0.0.1 --port "$PORT" \
        > "$LOG_DIR/server_mmlu_${RUN_LABEL}.log" 2>&1 < /dev/null &
    echo $! > "$LOG_DIR/mmlu.pid"
    say "mmlu server starting (8 slots x 64K, port $PORT), log: $LOG_DIR/server_mmlu_${RUN_LABEL}.log"
}

server_wait() {
    local url="$1" tries="$2" what="$3"
    local i=0
    while [ "$i" -lt "$tries" ]; do
        if curl -fs -m 3 --noproxy '*' "$url/v1/models" >/dev/null 2>&1; then
            say "$what is UP"
            return 0
        fi
        i=$((i + 1))
        sleep 5
    done
    say "FATAL: $what did not come up in $((tries * 5))s; tail of log:"
    ls -t "$LOG_DIR"/server_*"${RUN_LABEL}"*.log 2>/dev/null | head -1 | xargs tail -40 2>/dev/null || true
    return 1
}

cmd_server() {
    local action="${1:-}" profile="${2:-}"
    case "$action" in
        start)
            if server_alive; then
                say "server already up on $PORT (qwen3.6) -- reusing it"
                return 0
            fi
            case "$profile" in
                suite) server_start_suite ;;
                longctx) server_start_longctx ;;
                mmlu) server_start_mmlu ;;
                *) say "usage: $0 server start suite|longctx|mmlu"; exit 2 ;;
            esac
            SERVER_STARTED=1
            server_wait "$ROOT_URL" 120 "$profile server" || return 1
            ;;
        stop)
            server_stop_port "$PORT"
            ;;
        *)
            say "usage: $0 server start|stop suite|mmlu"; exit 2 ;;
    esac
}

run_dim() {
    local dim="$1" conc="$2"
    say "  [dim] $dim (concurrency $conc) -> $OUT_DIR/${RUN_LABEL}.part.${dim}.json"
    "$PY" -u benchmarks/quality_regression.py \
        --base-url "$BASE_URL" --model "$MODEL" \
        --label "$RUN_LABEL" --dims "$dim" \
        --concurrency "$conc" --max-tokens-code 4096 \
        --code-workdir "$OUT_DIR/${RUN_LABEL}.work" \
        --workdir "$OUT_DIR/${RUN_LABEL}.work" \
        --out "$OUT_DIR/${RUN_LABEL}.part.${dim}.json" \
        > "$LOG_DIR/suite_${dim}_${RUN_LABEL}.log" 2>&1
}

phase_ours_suite() {
    say "== phase: our-runtime suite fast dims (tool/agent/code, parallel, resumable) =="
    local SERVER_STARTED=0
    cmd_env_check
    cmd_lock_acquire || return 1
    cmd_server start suite || { cmd_lock_release; return 1; }

    local pids=()
    run_dim tool 8 & pids+=($!)
    run_dim agent 8 & pids+=($!)
    run_dim code 8 & pids+=($!)

    local fail=0
    for p in "${pids[@]}"; do
        wait "$p" || fail=1
    done
    if [ "$fail" = 1 ]; then
        say "one or more dims incomplete/failed -- rerun this command to resume from checkpoints"
        tail -n 20 "$LOG_DIR"/suite_*"${RUN_LABEL}"*.log
        server_maybe_stop
        cmd_lock_release
        return 1
    fi

    local inputs=("$OUT_DIR/${RUN_LABEL}.part.tool.json"
                  "$OUT_DIR/${RUN_LABEL}.part.agent.json"
                  "$OUT_DIR/${RUN_LABEL}.part.code.json")
    if [ -f "$OUT_DIR/${RUN_LABEL}.part.longctx.json" ]; then
        inputs+=("$OUT_DIR/${RUN_LABEL}.part.longctx.json")
    else
        say "note: longctx part missing -- run 'ours-longctx' first, then re-run ours-suite to merge"
    fi
    "$PY" benchmarks/quality_merge.py --label "$RUN_LABEL" --out "$SUITE_OUT" \
        "${inputs[@]}"
    say "suite report -> $SUITE_OUT (longctx merged when available)"
    server_maybe_stop
    cmd_lock_release
}

phase_ours_longctx() {
    say "== phase: our-runtime longctx (needles 8K..128K, dedicated 128K+ server) =="
    local SERVER_STARTED=0
    cmd_env_check
    cmd_lock_acquire || return 1
    cmd_server start longctx || { cmd_lock_release; return 1; }

    run_dim longctx 4
    local rc=$?
    if [ "$rc" != 0 ]; then
        say "longctx phase failed (rc=$rc) -- rerun this command to resume from checkpoints"
        tail -n 20 "$LOG_DIR"/suite_longctx_*"${RUN_LABEL}"*.log
        server_maybe_stop
        cmd_lock_release
        return "$rc"
    fi
    say "longctx part -> $OUT_DIR/${RUN_LABEL}.part.longctx.json"
    server_maybe_stop
    cmd_lock_release
}

phase_ours_humaneval() {
    say "== phase: our-runtime HumanEval+ 768 (README 2026-07-21 row) =="
    local SERVER_STARTED=0
    cmd_env_check
    cmd_lock_acquire || return 1
    cmd_server start suite || { cmd_lock_release; return 1; }

    "$PY" -u benchmarks/quality_eval.py \
        --base-url "$BASE_URL" --model "$MODEL" \
        --output "$HE_OUT" \
        --concurrency 16 --max-tokens 768 --evaluate \
        > "$LOG_DIR/humaneval_${RUN_LABEL}.log" 2>&1
    local rc=$?
    if [ "$rc" != 0 ]; then
        say "HumanEval phase failed (rc=$rc) -- rerun this command to resume"
        tail -30 "$LOG_DIR/humaneval_${RUN_LABEL}.log"
    else
        say "HumanEval 768 -> $HE_OUT (+ _eval_results.json)"
    fi
    server_maybe_stop
    cmd_lock_release
    return "$rc"
}

phase_mmlu() {
    say "== phase: MMLU-Pro 414 stratified (${MMLU_SHARDS} shards, resumable) =="
    local SERVER_STARTED=0
    cmd_env_check
    cmd_lock_acquire || return 1
    cmd_server start mmlu || { cmd_lock_release; return 1; }

    local pids=()
    local i
    for i in $(seq 0 $((MMLU_SHARDS - 1))); do
        say "  [mmlu] shard $i/$MMLU_SHARDS"
        HF_HUB_OFFLINE=1 "$PY" -u benchmarks/official/mmlu_pro_eval.py \
            --base-url "$BASE_URL" --model "$MODEL" \
            --limit 414 --concurrency 2 --max-tokens 32768 \
            --out "$MMLU_OUT" --shards "$MMLU_SHARDS" --shard-idx "$i" \
            > "$LOG_DIR/mmlu_shard${i}_${RUN_LABEL}.log" 2>&1 &
        pids+=($!)
    done

    local fail=0
    for p in "${pids[@]}"; do
        wait "$p" || fail=1
    done
    if [ "$fail" = 1 ]; then
        say "one or more MMLU shards failed -- rerun this command to resume"
        tail -n 20 "$LOG_DIR"/mmlu_shard*_"${RUN_LABEL}".log
        server_maybe_stop
        cmd_lock_release
        return 1
    fi

    HF_HUB_OFFLINE=1 "$PY" -u benchmarks/official/mmlu_pro_eval.py \
        --merge --model "$MODEL" --base-url "$BASE_URL" \
        --limit 414 --concurrency 2 --max-tokens 32768 \
        --out "$MMLU_OUT"
    say "mmlu report -> $MMLU_OUT"
    server_maybe_stop
    cmd_lock_release
}

cmd_compare() {
    say "== compare (current vs README historical values) =="
    "$PY" - "$SUITE_OUT" "$MMLU_OUT" "${HE_OUT%.jsonl}_eval_results.json" <<'PYEOF'
import json, os, sys

def load(p):
    return json.load(open(p)) if p and os.path.exists(p) else None

def pass_rates(path):
    d = load(path)
    if not d:
        return None
    nb = np_ = pb = pp = 0
    for reslist in d.get("eval", {}).values():
        for r in reslist:
            nb += 1; np_ += 1
            pb += r.get("base_status") == "pass"
            pp += r.get("plus_status") == "pass"
    return (pb / nb if nb else 0.0), (pp / np_ if np_ else 0.0), nb

suite, mmlu, cur_he = (load(sys.argv[1]), load(sys.argv[2]),
                       pass_rates(sys.argv[3]))
print("\n=== current vs README historical ===")
print("MMLU-Pro 414 thinking: historical 84.54 vs current "
      f"{mmlu['accuracy'] if mmlu else 'N/A'} "
      f"(n={mmlu.get('n') if mmlu else '-'}, max_tokens={mmlu.get('max_tokens') if mmlu else '-'}, "
      f"thinking={mmlu.get('thinking') if mmlu else '-'})")
if cur_he:
    print(f"HumanEval 768: historical 44.5/43.3 vs current "
          f"{cur_he[0]*100:.1f}/{cur_he[1]*100:.1f} ({cur_he[2]} problems)")
else:
    print(f"HumanEval 768: historical 44.5/43.3 vs current N/A "
          f"(missing {sys.argv[3]})")
if suite:
    d = suite.get("dims", {})
    print("tool/agent/longctx: historical 1.0/1.0/1.0 vs current "
          f"{d.get('tool', {}).get('accuracy', 'N/A')}/"
          f"{d.get('agent', {}).get('accuracy', 'N/A')}/"
          f"{d.get('longctx', {}).get('accuracy', 'N/A')}")
    c = d.get("code", {})
    if c:
        print("code 4096 HumanEval/HumanEval+: current "
              f"{c.get('humaneval_pass_at_1')}/{c.get('humaneval_plus_pass_at_1')} "
              f"(generated {c.get('generated')}/{c.get('n')})")
    else:
        print("code 4096: incomplete (rerun ours-suite to resume)")
PYEOF
}

cmd_all() {
    local KEEP_SERVER=0
    phase_ours_suite
    phase_ours_longctx
    phase_ours_humaneval
    phase_mmlu
    cmd_compare
}

case "${1:-all}" in
    env-check) cmd_env_check ;;
    server) cmd_server "${@:2}" ;;
    ours-suite) phase_ours_suite ;;
    ours-longctx) phase_ours_longctx ;;
    ours-humaneval) phase_ours_humaneval ;;
    mmlu) phase_mmlu ;;
    compare) cmd_compare ;;
    all)
        cmd_all
        ;;
    *)
        echo "usage: $0 {env-check|server start|stop suite|longctx|mmlu|ours-suite|ours-longctx|ours-humaneval|mmlu|compare|all}"
        exit 2
        ;;
esac
