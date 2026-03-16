#!/bin/bash
# GRPO with vLLM server mode
# Step 1: Launch vLLM server on GPUs 8-11
# Step 2: Run GRPO training on GPUs 0-7
set -e

REWARD_MODE=${1:-baseline}
BASE="/ex_disk2/mhpark/poc/chartvr"
VENV="/ex_disk2/mhpark/poc/vllm_env/bin"

if [ "$REWARD_MODE" = "baseline" ]; then
    OUTPUT_DIR="$BASE/checkpoints/grpo_baseline"
elif [ "$REWARD_MODE" = "cvr" ]; then
    OUTPUT_DIR="$BASE/checkpoints/grpo_cvr"
fi

export PYTHONPATH="$BASE:$PYTHONPATH"
export CHARTVR_ROOT="$BASE"

echo "=== Step 1: Launch vLLM server on GPUs 8,9,10,11 ==="
CUDA_VISIBLE_DEVICES=8,9,10,11 $VENV/trl vllm-serve \
    --model "$BASE/checkpoints/sft" \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.85 \
    --max_model_len 2048 &
VLLM_PID=$!
echo "vLLM server PID: $VLLM_PID"

# Wait for server to be ready
echo "Waiting for vLLM server to start..."
sleep 60

echo "=== Step 2: Launch GRPO training on GPUs 0-7 ==="
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $VENV/accelerate launch \
    --config_file "$BASE/scripts/deepspeed_zero3_8gpu.yaml" \
    "$BASE/scripts/run_grpo_trl.py" \
    --reward-mode "$REWARD_MODE" \
    --output-dir "$OUTPUT_DIR"

echo "=== Training complete, stopping vLLM server ==="
kill $VLLM_PID 2>/dev/null
