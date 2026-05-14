#!/usr/bin/env bash
# Post-SFT pipeline: wait for SFT to finish, merge LoRA, spin up eval
# vLLM servers on available GPUs, then run 5-bench eval.
#
# Usage:
#   ./scripts/post_sft_pipeline.sh <CKPT_DIR> <RUN_NAME>

set -euo pipefail

CKPT="${1:?CKPT_DIR required}"
RUN_NAME="${2:?RUN_NAME required}"
PY="${PY:-/ex_disk2/mhpark/poc/vllm_nightly_env/bin/python}"
BASE="${BASE:-/ex_disk2/mhpark/poc/chartvr/models/qwen3.5-4b}"
LOG_DIR="${LOG_DIR:-/tmp/post_sft_logs}"
mkdir -p "$LOG_DIR"
PIPE_LOG="$LOG_DIR/${RUN_NAME}_pipeline.log"

{
echo "=== post-SFT pipeline for $RUN_NAME ($(date +%F\ %T)) ==="
echo "  ckpt=$CKPT"

# 1a) Wait for SFT to START (the polling script may still be waiting for GPU)
echo "[1a] waiting for SFT to start (accelerate launch)..."
START_WAIT=0
while ! pgrep -f "accelerate launch.*train_sft.*${CKPT##*/}" >/dev/null 2>&1; do
    sleep 60
    START_WAIT=$((START_WAIT + 1))
    if [ $((START_WAIT % 10)) -eq 1 ]; then
        echo "    still waiting at $(date +%T) (poll ${START_WAIT})"
    fi
done
echo "    SFT started at $(date +%T)"

# 1b) Wait for SFT to COMPLETE
echo "[1b] waiting for SFT to complete..."
while pgrep -f "accelerate launch.*train_sft.*${CKPT##*/}" >/dev/null 2>&1; do
    sleep 60
done
echo "    SFT process exited at $(date +%T)"

# Verify ckpt written
if [ ! -d "$CKPT" ]; then
    echo "    ERROR: ckpt dir $CKPT not found — aborting"
    exit 1
fi

# 2) Merge LoRA into a standalone inference model
MERGED="${CKPT}_merged"
if [ ! -d "$MERGED" ]; then
    echo "[2] merging LoRA → $MERGED"
    "$PY" scripts/merge_lora.py \
        --base "$BASE" \
        --lora "$CKPT" \
        --output "$MERGED"
else
    echo "[2] merged dir already exists: $MERGED (skipping merge)"
fi

# 3) Launch vLLM servers on whatever GPUs are free right now
echo "[3] detecting free GPUs for eval servers..."
FREE_GPUS=$(
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | awk -F', ' '$2+0 < 2000 { print $1 }'
)
if [ -z "$FREE_GPUS" ]; then
    echo "    no fully-free GPU detected — will use GPU 0-7 (assume SFT-trained GPUs now free)"
    FREE_GPUS="0 1 2 3 4 5 6 7"
fi
echo "    using GPUs: $(echo $FREE_GPUS | tr '\n' ' ')"

PORTS=()
for i in $FREE_GPUS; do
    PORT=$((8000 + i))
    PORTS+=("$PORT")
    CUDA_VISIBLE_DEVICES=$i setsid nohup "$PY" -m vllm.entrypoints.openai.api_server \
        --model "$MERGED" \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 8192 \
        --port "$PORT" \
        --trust-remote-code \
        --reasoning-parser qwen3 \
        --generation-config vllm \
        </dev/null > "$LOG_DIR/vllm_eval_gpu${i}.log" 2>&1 &
    disown
done

# 4) Wait for all servers to be ready
echo "[4] waiting for vLLM servers to become ready..."
for PORT in "${PORTS[@]}"; do
    for wait in $(seq 1 60); do
        if curl -sS -m 2 "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
            echo "    port $PORT ready"
            break
        fi
        sleep 15
    done
done

# 5) Run eval_multi_server.py
PORTS_CSV=$(IFS=','; echo "${PORTS[*]}")
OUTDIR="results/$RUN_NAME"
echo "[5] running 5-bench eval: ports=$PORTS_CSV outdir=$OUTDIR"
"$PY" eval_multi_server.py \
    "$PORTS_CSV" \
    "$MERGED" \
    "$OUTDIR" \
    "chartqa_human,chartqa_augmented,charxiv_reasoning,chartqa_pro,chartmuseum"

echo "=== pipeline done at $(date +%F\ %T) ==="
} 2>&1 | tee "$PIPE_LOG"
