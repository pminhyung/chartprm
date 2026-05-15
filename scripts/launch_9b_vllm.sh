#!/usr/bin/env bash
# Launch Qwen3.5-VL-9B vLLM server (D2 perception verifier / image re-query).
#
# Usage: ./scripts/launch_9b_vllm.sh <port> <cuda_devices> [max_model_len]
#   ./scripts/launch_9b_vllm.sh 8100 0      8192    # TP=1 (preferred, 9B fits A100-40GB)
#   ./scripts/launch_9b_vllm.sh 8100 0,1    8192    # TP=2 fallback if TP=1 OOM
#
# Notes:
#   - Verifier uses short prompts (single claim query, max_tokens=128) — 8192 ample.
#   - --enable-prefix-caching helps when many claims share same image.
#   - 9B @ TP=1 weight ≈ 18GB, fits 40GB A100 with gpu-util 0.85.

set -euo pipefail

PORT="${1:?PORT required}"
CUDA_DEVICES="${2:?CUDA_VISIBLE_DEVICES required}"
MAX_MODEL_LEN="${3:-8192}"

MODEL="${MODEL:-/ex_disk2/mhpark/poc/chartvr/models/qwen3.5-9b}"
SERVED_NAME="${SERVED_NAME:-Qwen3.5-VL-9B}"
GPU_UTIL="${GPU_UTIL:-0.85}"
PY="${PY:-/ex_disk2/mhpark/poc/vllm_nightly_env/bin/python}"

TP=$(echo "$CUDA_DEVICES" | awk -F',' '{print NF}')

LOG_DIR="${LOG_DIR:-/tmp/vllm_logs}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/qwen9b_${PORT}_$(date +%Y%m%d_%H%M%S).log"

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
