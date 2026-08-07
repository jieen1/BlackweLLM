#!/usr/bin/env bash
# Phase B: code-dim quality anchor for one verify-chunking config.
# Same protocol as notes S18 (suite-fast server profile, code dim, conc 8).
# Usage: ab_quality_code_dim.sh <label> [SPARKINFER_ENV=VAL ...]
set -u
cd /home/bot/project/qwen-sm120-runtime
PY=$HOME/.venvs/vllm/bin/python
PORT=8300
LABEL=${1:?label required}; shift
LOGDIR=logs/quality
OUT=evalplus_results/quality/code_${LABEL}_20260807.part.code.json
SRV_LOG=$LOGDIR/abq_server_${LABEL}.log
mkdir -p "$LOGDIR" evalplus_results/quality

MODEL_SNAPSHOT="/home/bot/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-NVFP4/snapshots/ccdaab7e68af2409599b8949a8f2685703c9bae5"

/tmp/gpu_lock.sh acquire ab-quality-$LABEL 7200 || { echo "GPU lock busy"; exit 1; }
trap '/tmp/gpu_lock.sh release ab-quality-'$LABEL EXIT

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
" > "$SRV_LOG" 2>&1 < /dev/null &
SRV_PID=$!

for _ in $(seq 1 240); do
  grep -q "engine ready" "$SRV_LOG" 2>/dev/null && break
  if ! kill -0 $SRV_PID 2>/dev/null; then echo "server died"; tail -8 "$SRV_LOG"; exit 1; fi
  sleep 2
done
grep -q "engine ready" "$SRV_LOG" || { echo "ready timeout"; exit 1; }
echo "server ready ($LABEL)"

$PY -u benchmarks/quality_regression.py \
  --base-url http://127.0.0.1:$PORT/v1 --model qwen3.6 \
  --label "code_${LABEL}_20260807" --dims code \
  --concurrency 8 --max-tokens-code 4096 \
  --code-workdir "evalplus_results/quality/code_${LABEL}_20260807.work" \
  --workdir "evalplus_results/quality/code_${LABEL}_20260807.work" \
  --out "$OUT" 2>&1 | tail -15

pkill -f "server\\.app --host 127\\.0\\.0\\.1 --port $PORT" 2>/dev/null
echo "=== QUALITY DONE $LABEL -> $OUT ==="
