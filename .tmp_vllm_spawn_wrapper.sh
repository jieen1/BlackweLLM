#!/usr/bin/env bash
set -euo pipefail
echo "pid=$$ pythonpath=${PYTHONPATH-}" >> /tmp/qwen38-vllm-spawn-env
export PYTHONPATH="/home/bot/project/qwen-sm120-runtime${PYTHONPATH:+:${PYTHONPATH}}"
exec /home/bot/.venvs/vllm/bin/python "$@"
