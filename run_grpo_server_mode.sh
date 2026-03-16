#!/bin/bash
# ChartVCR GRPO Training with vLLM Server Mode
#
# Setup per model registry:
#   GPUs 8-11: Training (DeepSpeed ZeRO-2, 4 processes)
#   GPUs 12-15: vLLM generation server (TP=4)
#
# Usage: bash run_grpo_server_mode.sh [baseline|cvr]

set -e

REWARD_MODE=${1:-baseline}
BASE="/ex_disk2/mhpark/poc/chartvr"
NIGHTLY="/ex_disk2/mhpark/poc/vllm_nightly_env/bin"
TRAIN_GPUS="4,5,6,7,8,9,10,11"
VLLM_GPUS="0,1,2,3"
VLLM_PORT=9100

case "$REWARD_MODE" in
  baseline|outcome_only) REWARD_ARG="outcome_only"; OUTPUT_DIR="$BASE/ckpt/baseline" ;;
  cvr|chartvr)           REWARD_ARG="chartvr";       OUTPUT_DIR="$BASE/ckpt/cvr" ;;
  *)                     REWARD_ARG="outcome_only";  OUTPUT_DIR="$BASE/ckpt/baseline" ;;
esac

mkdir -p "$OUTPUT_DIR" "$BASE/logs"
export PYTHONPATH="$BASE:$PYTHONPATH"
export CHARTVR_ROOT="$BASE"

echo "=== Starting vLLM server on GPUs $VLLM_GPUS (port $VLLM_PORT) ==="
CUDA_VISIBLE_DEVICES=$VLLM_GPUS $NIGHTLY/trl vllm-serve \
    --model "$BASE/models/qwen3vl-8b-thinking" \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.85 \
    --max_model_len 8192 \
    --port $VLLM_PORT \
    --trust_remote_code \
    2>&1 | tee "$BASE/logs/vllm_server_${REWARD_MODE}.log" &
VLLM_PID=$!
echo "vLLM server PID: $VLLM_PID"

# Wait for server to be healthy
echo "Waiting for vLLM server (max 300s)..."
for i in $(seq 1 60); do
    if curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
        echo "vLLM server ready after ${i}x5s"
        break
    fi
    sleep 5
done

echo "=== Starting GRPO training on GPUs $TRAIN_GPUS ==="
cat > /tmp/accel_config_4gpu.yaml << 'EOF'
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
num_machines: 1
num_processes: 4
mixed_precision: bf16
deepspeed_config:
  zero_optimization:
    stage: 2
  gradient_clipping: 1.0
  train_batch_size: auto
  train_micro_batch_size_per_gpu: auto
EOF

CUDA_VISIBLE_DEVICES=$TRAIN_GPUS $NIGHTLY/accelerate launch \
    --config_file /tmp/accel_config_4gpu.yaml \
    "$BASE/train_grpo.py" \
    --reward_type "$REWARD_ARG" \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$BASE/logs/grpo_${REWARD_MODE}.log"

echo "=== Training complete. Stopping vLLM server ==="
kill $VLLM_PID 2>/dev/null
