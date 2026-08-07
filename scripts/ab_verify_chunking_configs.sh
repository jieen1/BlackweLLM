#!/usr/bin/env bash
# A/B harness: verify split-KV chunking configs on the production 128K/c4
# profile (scripts/qwen36_128k_bench_env.sh + QSR_PROFILE_ROUNDS=1).
# Sequential, single-GPU. Emits per-config: server log, perf-grid JSON,
# aggregated round profile.
set -u
cd /home/bot/project/qwen-sm120-runtime
PY=$HOME/.venvs/vllm/bin/python
PORT=8300
LOGDIR=logs/quality
mkdir -p "$LOGDIR"
SRV_PID=0

ready_wait() {
  local log=$1
  for _ in $(seq 1 240); do
    if grep -q "engine ready" "$log" 2>/dev/null; then return 0; fi
    if [ "$SRV_PID" -ne 0 ] && ! kill -0 "$SRV_PID" 2>/dev/null; then
      echo "server process died during startup"; return 1
    fi
    sleep 2
  done
  echo "ready timeout"; return 1
}

gpu_drain_wait() {
  local used=99999
  for _ in $(seq 1 90); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 4000 ] && break
    sleep 2
  done
  echo "$used"
}

run_config() {
  local label=$1; shift
  local log=$LOGDIR/ab_server_$label.log
  echo "=== CONFIG $label env: $* ==="
  setsid bash -c "
    set -a; source scripts/qwen36_128k_bench_env.sh; set +a
    export QSR_PROFILE_ROUNDS=1 $*
    exec $PY -m server.app --host 127.0.0.1 --port $PORT
  " > "$log" 2>&1 < /dev/null &
  SRV_PID=$!
  if ! ready_wait "$log"; then
    echo "READY FAIL $label -- last log lines:"; tail -5 "$log"
    kill "$SRV_PID" 2>/dev/null; pkill -f "server\\.app --host 127\\.0\\.0\\.1 --port $PORT" 2>/dev/null
    gpu_drain_wait >/dev/null
    return 1
  fi
  echo "server ready ($label)"
  $PY benchmarks/server_perf_grid.py --base-url http://127.0.0.1:$PORT --model qwen3.6 \
      --contexts 131072 --concurrency 4 --max-tokens 256 --warm-rounds 3 \
      --out "$LOGDIR/ab_grid_$label.json" 2>&1 | tail -8
  sleep 3
  echo "--- round profile ($label) ---"
  $PY scripts/aggregate_round_profile.py "$log" 2>/dev/null | tee "$LOGDIR/ab_rounds_$label.txt" | head -25
  pkill -f "server\\.app --host 127\\.0\\.0\\.1 --port $PORT" 2>/dev/null
  SRV_PID=0
  local used; used=$(gpu_drain_wait)
  echo "=== DONE $label (gpu drained to ${used}MiB) ==="
}

/tmp/gpu_lock.sh acquire ab-verify-chunking 7200 || { echo "GPU lock busy -- aborting"; exit 1; }
trap '/tmp/gpu_lock.sh release ab-verify-chunking' EXIT

run_config baseline
run_config m16frozen SPARKINFER_QWEN36_VERIFY_M16=1 SPARKINFER_QWEN36_VERIFY_NO_ADAPTIVE=1
run_config fixed32 SPARKINFER_QWEN36_VERIFY_FIXED_SPLITS=32
run_config m32frozen SPARKINFER_QWEN36_VERIFY_NO_ADAPTIVE=1

echo "ALL CONFIGS DONE"
