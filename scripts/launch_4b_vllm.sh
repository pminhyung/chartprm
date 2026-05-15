#!/usr/bin/env bash
# Launch Qwen3.5-VL-4B vLLM server (policy / D1 trace generation / D2 MC rollout).
#
# Usage: ./scripts/launch_4b_vllm.sh <port> <cuda_devices> [max_model_len]
#   ./scripts/launch_4b_vllm.sh 8000 0       16384   # TP=1
#   ./scripts/launch_4b_vllm.sh 8001 1       16384
#
# Required flags baked in:
#   --reasoning-parser qwen3 : parses <think>...</think> into reasoning_content
#   --generation-config vllm : ignores model's generation_config.json so
#                              client-side sampling is the single source.
#   --enable-prefix-caching  : reuse KV-cache across same-prefix samples
#                              (D2 MC rollout = same prefix × K continuations).
#   setsid ... </dev/null & disown : survives session compaction (SIGHUP).
#
# Sampling params per-request (NOT here): chartvr/config.py:SAMPLING_PARAMS["4b"]
#   thinking_coding : temp=0.6 top_p=0.95 top_k=20 min_p=0 presence_penalty=0

set -euo pipefail

PORT="${1:?PORT required (e.g. 8000)}"
CUDA_DEVICES="${2:?CUDA_VISIBLE_DEVICES required (e.g. 0 or 0,1)}"
MAX_MODEL_LEN="${3:-16384}"

MODEL="${MODEL:-/ex_disk2/mhpark/poc/chartvr/models/qwen3.5-4b}"
SERVED_NAME="${SERVED_NAME:-Qwen3.5-VL-4B}"
GPU_UTIL="${GPU_UTIL:-0.85}"
PY="${PY:-/ex_disk2/mhpark/poc/vllm_nightly_env/bin/python}"

TP=$(echo "$CUDA_DEVICES" | awk -F',' '{print NF}')

LOG_DIR="${LOG_DIR:-/tmp/vllm_logs}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/qwen4b_${PORT}_$(date +%Y%m%d_%H%M%S).log"

echo "[launch] port=$PORT gpus=$CUDA_DEVICES TP=$TP max_len=$MAX_MODEL_LEN"
echo "[launch] model=$MODEL"
echo "[launch] log=$LOG"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" setsid nohup "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --port "$PORT" \
    --trust-remote-code \
    --enable-prefix-caching \
    --reasoning-parser qwen3 \
    --generation-config vllm \
    </dev/null > "$LOG" 2>&1 &
LAUNCHED_PID=$!
disown

echo "[launch] started. Parent PID=$LAUNCHED_PID"
echo "[launch] health: curl -s http://localhost:$PORT/v1/models | jq -r '.data[0].id'"
echo "[launch] tail:   tail -f $LOG"
