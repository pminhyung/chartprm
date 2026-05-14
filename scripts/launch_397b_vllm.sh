#!/usr/bin/env bash
# Launch Qwen3.5-397B-A17B vLLM server with recommended defaults.
#
# Usage: ./scripts/launch_397b_vllm.sh <port> <cuda_devices> [max_model_len]
# Examples:
#   ./scripts/launch_397b_vllm.sh 9200 0,1,2,3,4,5,6,7 4096
#   ./scripts/launch_397b_vllm.sh 9201 8,9,10,11,12,13,14,15 65536
#
# Required flags baked in:
#   --reasoning-parser qwen3   : parses <think>...</think> into reasoning_content
#   --generation-config vllm   : ignores model's generation_config.json
#                                (chartvr/llm_client.py injects correct sampling
#                                 params per enable_thinking flag)
#   --enforce-eager            : GPTQ-int4 + TP=8 stability
#   setsid ... </dev/null & disown : survives claude-code session compaction
#
# NUMA policy:
#   DO NOT pin CPU affinity via taskset/numactl. The 2026-04-10 root-cause
#   analysis showed that forcing NUMA 0 placement when the model weights are
#   already cached in NUMA 1 page cache (from a co-resident 9201 run) caused
#   cross-socket UPI reads during loading and inference path pinned buffers,
#   yielding 2-3x loading slowdown (906s vs 398s) and 4x throughput slowdown.
#   Let the kernel scheduler place worker threads freely; it will prefer the
#   NUMA node where the relevant page cache / GPU-adjacent memory lives.
#   vLLM already detects NUMA-local CPU count and uses it for its thread pool.

set -euo pipefail

PORT="${1:?PORT required (e.g. 9200)}"
CUDA_DEVICES="${2:?CUDA_VISIBLE_DEVICES required (comma-separated, e.g. 0,1,2,3,4,5,6,7)}"
MAX_MODEL_LEN="${3:-4096}"

MODEL="${MODEL:-/ex_disk2/mhpark/poc/chartvr/models/qwen3.5-397b-gptq-int4}"
SERVED_NAME="${SERVED_NAME:-Qwen3.5-397B-A17B-FP8}"
GPU_UTIL="${GPU_UTIL:-0.90}"
PY="${PY:-/ex_disk2/mhpark/poc/vllm_nightly_env/bin/python}"

# TP = number of CUDA devices
TP=$(echo "$CUDA_DEVICES" | awk -F',' '{print NF}')

LOG_DIR="${LOG_DIR:-/tmp/vllm_logs}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/qwen397b_${PORT}_$(date +%Y%m%d_%H%M%S).log"

echo "[launch] port=$PORT gpus=$CUDA_DEVICES TP=$TP max_len=$MAX_MODEL_LEN"
echo "[launch] model=$MODEL"
echo "[launch] log=$LOG"

# Daemonize with setsid + nohup + </dev/null + disown
# Prevents SIGHUP on claude-code session compaction (state.md issue)
# NO taskset / numactl: let kernel scheduler choose NUMA placement
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
