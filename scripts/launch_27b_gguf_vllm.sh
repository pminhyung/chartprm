#!/usr/bin/env bash
# Launch Qwen3.6-27B GGUF (UD-Q6_K_XL by default) vLLM server.
#
# Model: unsloth/Qwen3.6-27B-GGUF (UD-Q6_K_XL = 25.6 GB weights + mmproj 0.9 GB)
#   - Fits TP=1 on A100-40GB → 12 parallel hosts on GPU 4-15.
#   - vLLM 0.17 mmproj auto-detect via gguf_utils.detect_gguf_multimodal().
#   - vLLM Qwen3.5 GGUF VLM path is UNVALIDATED — smoke test before scale-out.
#
# Usage: ./scripts/launch_27b_gguf_vllm.sh <port> <cuda_devices> [max_model_len]
# Examples:
#   ./scripts/launch_27b_gguf_vllm.sh 9101 4  16384
#   ./scripts/launch_27b_gguf_vllm.sh 9102 5  16384
#
# Sampling params (DO NOT duplicate here — set per-request via MultiHostClient):
#   chartvr/config.py:SAMPLING_PARAMS["27b"]

set -euo pipefail

PORT="${1:?PORT required (e.g. 9101)}"
CUDA_DEVICES="${2:?CUDA_VISIBLE_DEVICES required (comma-separated, e.g. 4 or 4,5)}"
MAX_MODEL_LEN="${3:-16384}"

MODEL="${MODEL:-/ex_disk2/mhpark/poc/chartvr/models/qwen3.6-27b-gguf-q6/Qwen3.6-27B-UD-Q6_K_XL.gguf}"
SERVED_NAME="${SERVED_NAME:-Qwen3.6-27B}"
GPU_UTIL="${GPU_UTIL:-0.92}"
PY="${PY:-/ex_disk2/mhpark/poc/vllm_nightly_env/bin/python}"

TP=$(echo "$CUDA_DEVICES" | awk -F',' '{print NF}')

LOG_DIR="${LOG_DIR:-/tmp/vllm_logs}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/qwen27b_gguf_${PORT}_$(date +%Y%m%d_%H%M%S).log"

echo "[launch] port=$PORT gpus=$CUDA_DEVICES TP=$TP max_len=$MAX_MODEL_LEN gguf"
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
    --enforce-eager \
    --reasoning-parser qwen3 \
    --generation-config vllm \
    </dev/null > "$LOG" 2>&1 &
LAUNCHED_PID=$!
disown

echo "[launch] started. Parent PID=$LAUNCHED_PID"
echo "[launch] health check: curl -s http://localhost:$PORT/v1/models | jq"
echo "[launch] tail log:     tail -f $LOG"
