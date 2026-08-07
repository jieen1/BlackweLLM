#!/usr/bin/env bash
# Phase C: short-context throughput A/B (suite-fast profile, 32K slots, cap 8):
# baseline vs m16frozen, plus no-profile 128K headline check for m16frozen.
set -u
cd /home/bot/project/qwen-sm120-runtime
PY=$HOME/.venvs/vllm/bin/python
PORT=8300
LOGDIR=logs/quality
MODEL_SNAPSHOT="/home/bot/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-NVFP4/snapshots/ccdaab7e68af2409599b8949a8f2685703c9bae5"
SRV_PID=0

start_suite_server() {
  local label=$1; shift
  local log=$LOGDIR/abc_server_$label.log
  setsid bash -c "
    export QSR_SERVER_MODEL_PATH=$MODEL_SNAPSHOT QSR_SERVER_PRODUCTION=1 \
      QSR_SERVER_CAPACITY=8 QSR_SERVER_NUM_SLOTS=9 QSR_SERVER_BLOCK_SIZE=16 \
      QSR_SERVER_BLOCKS_PER_SLOT=2048 QSR_SERVER_ENABLE_CUDAGRAPH=1 \
      QSR_SERVER_ENABLE_PREFIX_CACHE=1 QSR_SERVER_ENABLE_SESSION_AFFINITY=0 \
      QSR_SERVER_ENABLE_DFLASH=0 QSR_SERVER_ENABLE_MTP=1 QSR_SERVER_MTP_K=3 \
      QSR_SERVER_KV_CACHE_DTYPE=fp8_e4m3 QSR_SERVER_GPU_MEM_UTIL=0.85 \
      QSR_SERVER_REQUEST_TIMEOUT_S=0 QSR_TOOL_CALL_PARSER=qwen3_coder \
      QSR_SERVED_MODEL_NAME=qwen3.6 QSR_DEBUG_REQUESTS=0 HF_HUB_OFFLINE=1 $*
    exec $PY -m server.app --host 127.0.0.1 --port $PORT
  " > "$log" 2>&1 < /dev/null &
  SRV_PID=$!
  for _ in $(seq 1 240); do
    grep -q "engine ready" "$log" 2>/dev/null && return 0
    sleep 2
  done
  echo "READY FAIL $label"; tail -5 "$log"; return 1
}

stop_server() {
  pkill -f "server[.]app --host 127.0.0.1 --port $PORT" 2>/dev/null
  SRV_PID=0
  for _ in $(seq 1 60); do
    local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 4000 ] && return 0
    sleep 2
  done
}

/tmp/gpu_lock.sh acquire ab-shortctx 7200 || { echo "lock busy"; exit 1; }
trap '/tmp/gpu_lock.sh release ab-shortctx' EXIT

# C1: short-context grid, baseline vs m16frozen (suite profile)
for cfg in "shortctx_baseline:" "shortctx_m16frozen:SPARKINFER_QWEN36_VERIFY_M16=1 SPARKINFER_QWEN36_VERIFY_NO_ADAPTIVE=1"; do
  label="${cfg%%:*}"; envs="${cfg#*:}"
  echo "=== $cfg ==="
  start_suite_server "$label" $envs || continue
  $PY benchmarks/server_perf_grid.py --base-url http://127.0.0.1:$PORT \
      --contexts 4k,32k --concurrency 8 --max-tokens 256 --warm-rounds 3 \
      --out $LOGDIR/ab_grid_$label.json 2>&1 | grep -E "cell|COLD|WARM" 
  stop_server
done

# C2: 128K no-profile headline, m16frozen (bench profile, 256K slots)
echo "=== c2_128k_m16frozen_noprof ==="
setsid bash -c "
  set -a; source scripts/qwen36_128k_bench_env.sh; set +a
  export SPARKINFER_QWEN36_VERIFY_M16=1 SPARKINFER_QWEN36_VERIFY_NO_ADAPTIVE=1
  exec $PY -m server.app --host 127.0.0.1 --port $PORT
" > $LOGDIR/abc_server_c2.log 2>&1 < /dev/null &
SRV_PID=$!
ok=1
for _ in $(seq 1 240); do grep -q "engine ready" $LOGDIR/abc_server_c2.log 2>/dev/null && { ok=0; break; }; sleep 2; done
if [ $ok -eq 0 ]; then
  $PY benchmarks/server_perf_grid.py --base-url http://127.0.0.1:$PORT \
      --contexts 131072 --concurrency 4 --max-tokens 256 --warm-rounds 5 \
      --out $LOGDIR/ab_grid_c2_128k_m16frozen_noprof.json 2>&1 | grep -E "cell|COLD|WARM"
else
  echo "READY FAIL c2"; tail -5 $LOGDIR/abc_server_c2.log
fi
stop_server
echo "PHASE C DONE"
