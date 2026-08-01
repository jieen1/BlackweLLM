#!/usr/bin/env bash
# Manage the BlackweLLM server (this repo's own SM120 runtime — Laguna
# backend, custom CUDA attention, FP8 KV, DFlash speculative decoding).
#
# Modeled on /home/bot/vllm_server/vllm_ctl.sh, trimmed down to what this
# repo needs: one model (poolside/Laguna-S-2.1-NVFP4, hardcoded in
# ServerEngine.MODEL), one backend, single-process (no forked EngineCore
# like vLLM -- the engine runs in a background thread of the same
# `python -m server.app` process, so a plain PID/pgrep match is enough).
#
# Usage: scripts/blackwellm_ctl.sh {start|stop|restart|status|logs [n]|config|relay|smoke}
#
# `start` also brings up a WSL2 relay (0.0.0.0:9000 -> 127.0.1.1:$PORT) so
# external scrapers (Prometheus included) hit a stable port across server
# restarts; `relay` (re)starts just the relay against an already-running
# server.
#
# All QSR_SERVER_* knobs can be overridden by exporting them (or editing the
# defaults below) before calling this script -- see README.md#configuration
# for what each one does.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_BIN=/home/bot/.venvs/vllm/bin
LOG_DIR="$REPO_ROOT/logs"
PID_FILE="$LOG_DIR/server.pid"

# -- listen address --------------------------------------------------------
# NOTE: port 8000 on this host is already held by an unrelated service
# (cognode/backfiller uvicorn app) -- do NOT default to 8000, it is not ours
# to take. 8100 was confirmed free at the time this script was written.
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8100}"

# -- WSL2 relay -------------------------------------------------------------
# Same workaround /home/bot/vllm_server/vllm_ctl.sh uses: WSL2's mirrored
# networking mode + repeated rebinds of the same port trigger SO_REUSEPORT
# RST on 127.0.0.1 for stale connections. The relay listens on a stable port
# and forwards to 127.0.1.1 (an un-hijacked loopback alias), so clients
# (Prometheus included) that keep hitting the same port across server
# restarts don't see connection resets. Reuses /home/bot/vllm_relay.py as-is.
RELAY_SCRIPT=/home/bot/vllm_relay.py
RELAY_PORT="${RELAY_PORT:-9000}"
RELAY_PID_FILE="$LOG_DIR/relay.pid"

mkdir -p "$LOG_DIR"

# -- server configuration (QSR_* env vars consumed by server/app.py) ------
# `: "${VAR:=default}"` only sets the default if the caller hasn't already
# exported an override, so `FOO=bar scripts/blackwellm_ctl.sh start` works.
: "${QSR_SERVER_PRODUCTION:=1}"
: "${QSR_SERVER_CAPACITY:=3}"          # 3 concurrent requests
: "${QSR_SERVER_NUM_SLOTS:=3}"         # capacity(3) + cg_extra(0, dflash owns its own CG scratch)
: "${QSR_SERVER_BLOCK_SIZE:=64}"
: "${QSR_SERVER_BLOCKS_PER_SLOT:=4096}" # 4096*64 = 262144 tokens = 256K ctx/slot
: "${QSR_SERVER_ENABLE_CUDAGRAPH:=1}"
: "${QSR_SERVER_ENABLE_PREFIX_CACHE:=1}"
: "${QSR_SERVER_ENABLE_DFLASH:=1}"
: "${QSR_SERVER_ENABLE_SESSION_AFFINITY:=0}"
: "${QSR_SERVER_GPU_MEM_UTIL:=0.95}"    # 3x256K is memory-heavy; see `config` output
: "${QSR_SERVED_MODEL_NAME:=laguna-s-2.1 qwen3.6}"
: "${QSR_DEBUG_REQUESTS:=1}"
export QSR_SERVER_PRODUCTION QSR_SERVER_CAPACITY QSR_SERVER_NUM_SLOTS \
    QSR_SERVER_BLOCK_SIZE QSR_SERVER_BLOCKS_PER_SLOT QSR_SERVER_ENABLE_CUDAGRAPH \
    QSR_SERVER_ENABLE_PREFIX_CACHE QSR_SERVER_ENABLE_DFLASH \
    QSR_SERVER_ENABLE_SESSION_AFFINITY QSR_SERVER_GPU_MEM_UTIL \
    QSR_SERVED_MODEL_NAME QSR_DEBUG_REQUESTS

ctx_per_slot() { echo $(( QSR_SERVER_BLOCK_SIZE * QSR_SERVER_BLOCKS_PER_SLOT )); }

cmd_config() {
    echo "-- effective launch config --"
    printf '  %-32s %s\n' \
        "host:port" "$HOST:$PORT" \
        "QSR_SERVER_CAPACITY" "$QSR_SERVER_CAPACITY" \
        "QSR_SERVER_NUM_SLOTS" "$QSR_SERVER_NUM_SLOTS" \
        "QSR_SERVER_BLOCK_SIZE" "$QSR_SERVER_BLOCK_SIZE" \
        "QSR_SERVER_BLOCKS_PER_SLOT" "$QSR_SERVER_BLOCKS_PER_SLOT" \
        "context/slot" "$(ctx_per_slot) tokens ($(( $(ctx_per_slot) / 1024 ))K)" \
        "QSR_SERVER_ENABLE_CUDAGRAPH" "$QSR_SERVER_ENABLE_CUDAGRAPH" \
        "QSR_SERVER_ENABLE_PREFIX_CACHE" "$QSR_SERVER_ENABLE_PREFIX_CACHE" \
        "QSR_SERVER_ENABLE_DFLASH" "$QSR_SERVER_ENABLE_DFLASH" \
        "QSR_SERVER_ENABLE_SESSION_AFFINITY" "$QSR_SERVER_ENABLE_SESSION_AFFINITY" \
        "QSR_SERVER_GPU_MEM_UTIL" "$QSR_SERVER_GPU_MEM_UTIL" \
        "QSR_SERVED_MODEL_NAME" "$QSR_SERVED_MODEL_NAME" \
        "QSR_SERVER_PRODUCTION" "$QSR_SERVER_PRODUCTION"
}

find_pids() {
    # Single-process server (engine runs on a background thread, not a
    # forked subprocess) -- match either the module invocation or the bound
    # port, then intersect so we never accidentally kill an unrelated
    # process that happens to also mention "server.app".
    pgrep -f "$VENV_BIN/python -m server\.app.*--port $PORT" 2>/dev/null | sort -u
}

find_relay_pids() {
    pgrep -f "python3? .*vllm_relay\.py $RELAY_PORT $PORT" 2>/dev/null | sort -u
}

launch_relay() {
    local relay_log="$LOG_DIR/relay.log"
    setsid nohup python3 "$RELAY_SCRIPT" "$RELAY_PORT" "$PORT" \
        > "$relay_log" 2>&1 < /dev/null &
    local relay_pid=$!
    disown "$relay_pid" 2>/dev/null || disown
    echo "$relay_pid" > "$RELAY_PID_FILE"
    echo "started relay pid=$relay_pid (0.0.0.0:$RELAY_PORT -> 127.0.1.1:$PORT), log: $relay_log"
}

cmd_start() {
    local existing
    existing=$(find_pids)
    if [ -n "$existing" ]; then
        echo "already running (pid(s): $(echo "$existing" | tr '\n' ' '))"
        echo "use '$0 restart' to relaunch"
        return 1
    fi

    # Refuse to steal a port some other, unrelated service is already
    # holding (this host runs other apps -- e.g. 8000 is a pre-existing
    # cognode/backfiller uvicorn instance, not ours).
    if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
        echo "ERROR: port $PORT is already bound by another process (not ours):" >&2
        ss -ltnp 2>/dev/null | grep ":$PORT " >&2
        echo "set PORT=<free port> and retry." >&2
        return 1
    fi

    cmd_config
    echo

    local ts log_file
    ts=$(date +%Y%m%d_%H%M%S)
    log_file="$LOG_DIR/server_${ts}.log"

    cd "$REPO_ROOT" || { echo "ERROR: cannot cd to $REPO_ROOT" >&2; return 1; }

    PATH="$VENV_BIN:$PATH" USE_LIBUV=0 setsid nohup "$VENV_BIN/python" -m server.app \
        --host "$HOST" --port "$PORT" \
        > "$log_file" 2>&1 < /dev/null &
    local pid=$!
    disown "$pid" 2>/dev/null || disown

    echo "$pid" > "$PID_FILE"
    ln -sf "$log_file" "$LOG_DIR/current.log"
    echo "started pid=$pid, log: $log_file"

    echo "waiting for startup (model load + KV cache alloc + paged-attention JIT"
    echo "warmup can take a few extra minutes on a clean ~/.cache/sparkinfer,"
    echo "well under 3min once that on-disk compile cache is warm)..."
    local miss_count=0
    for _ in $(seq 1 480); do
        if curl -s -m 3 "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
            echo "server is UP on $HOST:$PORT"
            local relay_pids
            relay_pids=$(find_relay_pids)
            if [ -z "$relay_pids" ]; then
                launch_relay
            else
                echo "relay already running (pid(s): $(echo "$relay_pids" | tr '\n' ' '))"
            fi
            return 0
        fi
        if [ -z "$(find_pids)" ]; then
            miss_count=$((miss_count + 1))
            if [ "$miss_count" -ge 2 ]; then
                echo "process died during startup -- check $LOG_DIR/current.log" >&2
                tail -n 80 "$LOG_DIR/current.log" >&2
                return 1
            fi
        else
            miss_count=0
        fi
        sleep 5
    done
    echo "timed out waiting for startup (still may come up -- check $LOG_DIR/current.log)" >&2
    return 1
}

cmd_stop() {
    local pids relay_pids
    pids=$(find_pids)
    relay_pids=$(find_relay_pids)
    if [ -z "$pids" ] && [ -z "$relay_pids" ]; then
        echo "not running"
        rm -f "$PID_FILE" "$RELAY_PID_FILE"
        return 0
    fi
    if [ -n "$pids" ]; then
        echo "killing pid(s): $(echo "$pids" | tr '\n' ' ')"
        echo "$pids" | xargs -r kill -9
    fi
    if [ -n "$relay_pids" ]; then
        echo "killing relay pid(s): $(echo "$relay_pids" | tr '\n' ' ')"
        echo "$relay_pids" | xargs -r kill -9
    fi
    sleep 2
    local remaining remaining_relay
    remaining=$(find_pids)
    remaining_relay=$(find_relay_pids)
    if [ -n "$remaining" ] || [ -n "$remaining_relay" ]; then
        echo "still alive after kill -9, retrying: $remaining $remaining_relay"
        echo "$remaining $remaining_relay" | xargs -r kill -9
        sleep 2
    fi
    rm -f "$PID_FILE" "$RELAY_PID_FILE"
    echo "stopped"
}

cmd_restart() {
    cmd_stop
    sleep 2
    cmd_start
}

cmd_status() {
    local pids
    pids=$(find_pids)
    if [ -z "$pids" ]; then
        echo "status: NOT RUNNING"
        return 1
    fi
    echo "status: RUNNING (pid(s): $(echo "$pids" | tr '\n' ' '))"
    local relay_pids
    relay_pids=$(find_relay_pids)
    if [ -n "$relay_pids" ]; then
        echo "relay: RUNNING (pid(s): $(echo "$relay_pids" | tr '\n' ' '), 0.0.0.0:$RELAY_PORT -> 127.0.1.1:$PORT)"
    else
        echo "relay: NOT RUNNING"
    fi
    echo
    cmd_config
    echo
    echo "-- process uptime/cpu/mem --"
    local all_pids_csv
    all_pids_csv=$(printf '%s\n%s\n' "$pids" "$relay_pids" | tr -s ' \n' ',' | sed 's/^,//; s/,$//')
    ps -o pid,etime,pcpu,pmem,cmd -p "$all_pids_csv" 2>/dev/null
    echo
    echo "-- GPU memory --"
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv 2>/dev/null
    echo
    echo "-- health check --"
    if curl -s -m 5 "http://localhost:$PORT/health" 2>/dev/null | python3 -m json.tool 2>/dev/null; then
        :
    else
        echo "port $PORT: NOT responding (process may still be starting up)"
    fi
    if [ -n "$relay_pids" ]; then
        if curl -s -m 5 "http://localhost:$RELAY_PORT/v1/models" >/dev/null 2>&1; then
            echo "relay port $RELAY_PORT: OK"
        else
            echo "relay port $RELAY_PORT: NOT responding"
        fi
    fi
    echo
    echo "-- /v1/models --"
    curl -s -m 5 "http://localhost:$PORT/v1/models" 2>/dev/null | python3 -m json.tool 2>/dev/null | head -20
    echo
    echo "-- current metrics snapshot --"
    curl -s -m 5 "http://localhost:$PORT/metrics" 2>/dev/null | \
        grep -E "^blackwellm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc|prefix_cache_hit_rate|requests_completed_total)\{" || \
        echo "(no blackwellm: metrics yet -- server may be idle/just started)"
}

cmd_logs() {
    local log_file="$LOG_DIR/current.log"
    if [ ! -e "$log_file" ]; then
        echo "no log file yet"
        return 1
    fi
    tail -n "${1:-100}" -f "$log_file"
}

cmd_relay() {
    local relay_pids
    relay_pids=$(find_relay_pids)
    if [ -n "$relay_pids" ]; then
        echo "relay already running (pid(s): $(echo "$relay_pids" | tr '\n' ' '))"
        return 0
    fi
    if [ -z "$(find_pids)" ]; then
        echo "ERROR: server not running on port $PORT -- start it first" >&2
        return 1
    fi
    launch_relay
}

# P0-B/C-LIVE (docs/implementation-plan.md §3): run the smoke gate against
# an already-running server. Deliberately does NOT start/stop the server
# itself -- reloading a fresh model is a multi-minute GPU operation (see
# AGENTS.md's warm-engine-vs-cold-start distinction), and this is meant to
# run after every change to server/ or runtime/backends/, not once. Restart
# first if you want the cold-start /metrics check (scripts/c_live_smoke.py
# check 1) to mean anything -- it only tests the real cold-start branch on
# the FIRST generation request the server has ever served.
cmd_smoke() {
    if [ -z "$(find_pids)" ]; then
        echo "ERROR: server not running on port $PORT -- start it first ('$0 start')" >&2
        return 1
    fi
    cd "$REPO_ROOT" || { echo "ERROR: cannot cd to $REPO_ROOT" >&2; return 1; }
    "$VENV_BIN/python" scripts/c_live_smoke.py --base-url "http://localhost:$PORT"
}

case "${1:-}" in
    start)   cmd_start ;;
    stop)    cmd_stop ;;
    restart) cmd_restart ;;
    status)  cmd_status ;;
    logs)    cmd_logs "${2:-100}" ;;
    config)  cmd_config ;;
    relay)   cmd_relay ;;
    smoke)   cmd_smoke ;;
    *)
        echo "usage: $0 {start|stop|restart|status|logs [n_lines]|config|relay|smoke}"
        exit 1
        ;;
esac
