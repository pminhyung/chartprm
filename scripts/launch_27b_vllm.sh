#!/usr/bin/env bash
# Launch Qwen3.6-27B VLM vLLM server (judge + teacher distillation).
#
# Usage: ./scripts/launch_27b_vllm.sh <port> <cuda_devices> [max_model_len]
# Examples:
#   ./scripts/launch_27b_vllm.sh 9101 4,5,6,7      16384
#   ./scripts/launch_27b_vllm.sh 9102 8,9,10,11    16384
#   ./scripts/launch_27b_vllm.sh 9103 12,13,14,15  16384
#
# Required flags baked in:
#   --reasoning-parser qwen3   : parses <think>...</think> into reasoning_content
#                                (raw <think> tag in content otherwise)
#   --generation-config vllm   : ignores model's generation_config.json so
#                                client-side sampling (chartvr/config.py
#                                SAMPLING_PARAMS["27b"]) is the single source.
#   --enforce-eager            : 27B VLM TP=4 stability (CUDA graph capture OOM
#                                on A100-40GB confirmed 2026-05-08).
#   setsid ... </dev/null & disown : survives claude-code session compaction.
#
# Sampling params (DO NOT duplicate here — set per-request via MultiHostClient):
#   chartvr/config.py:SAMPLING_PARAMS["27b"]
#     thinking : temperature=1.0 top_p=0.95 top_k=20 min_p=0.0
#                presence_penalty=0.0 repetition_penalty=1.0
#     instruct : temperature=0.7 top_p=0.80 top_k=20 min_p=0.0
#                presence_penalty=1.5 repetition_penalty=1.0

set -euo pipefail

PORT="${1:?PORT required (e.g. 9101)}"
CUDA_DEVICES="${2:?CUDA_VISIBLE_DEVICES required (comma-separated, e.g. 4,5,6,7)}"
MAX_MODEL_LEN="${3:-131072}"

MODEL="${MODEL:-/ex_disk2/mhpark/poc/chartvr/models/qwen3.6-27b}"
SERVED_NAME="${SERVED_NAME:-Qwen3.6-27B}"
GPU_UTIL="${GPU_UTIL:-0.85}"
PY="${PY:-/ex_disk2/mhpark/poc/vllm_nightly_env/bin/python}"

# TP = number of CUDA devices
TP=$(echo "$CUDA_DEVICES" | awk -F',' '{print NF}')

LOG_DIR="${LOG_DIR:-/tmp/vllm_logs}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/qwen27b_${PORT}_$(date +%Y%m%d_%H%M%S).log"

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
    --enforce-eager \
    --reasoning-parser qwen3 \
    --generation-config vllm \
    </dev/null > "$LOG" 2>&1 &
LAUNCHED_PID=$!
disown

echo "[launch] started. Parent PID=$LAUNCHED_PID"
echo "[launch] health check: curl -s http://localhost:$PORT/v1/models | jq"
echo "[launch] tail log:     tail -f $LOG"
