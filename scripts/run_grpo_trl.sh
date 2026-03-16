#!/bin/bash
# GRPO Training using TRL's GRPOTrainer
# Usage:
#   bash scripts/run_grpo_trl.sh baseline   # GRPO baseline (cerm + format)
#   bash scripts/run_grpo_trl.sh cvr        # GRPO with ChartVCR reward
#   bash scripts/run_grpo_trl.sh cvr_equal  # GRPO with equal penalty ablation
set -e

REWARD_MODE=${1:-baseline}
BASE="/ex_disk2/mhpark/poc/chartvr"
VENV="/ex_disk2/mhpark/poc/vllm_env/bin"
NUM_GPUS=${NUM_GPUS:-16}

if [ "$REWARD_MODE" = "baseline" ]; then
    OUTPUT_DIR="$BASE/checkpoints/grpo_baseline"
elif [ "$REWARD_MODE" = "cvr" ]; then
    OUTPUT_DIR="$BASE/checkpoints/grpo_cvr"
elif [ "$REWARD_MODE" = "cvr_equal" ]; then
    OUTPUT_DIR="$BASE/checkpoints/grpo_cvr_equal"
fi

export PYTHONPATH="$BASE:$PYTHONPATH"
export CHARTVR_ROOT="$BASE"

echo "=== GRPO Training: $REWARD_MODE ==="
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NUM_GPUS"

$VENV/torchrun --nproc_per_node=$NUM_GPUS \
    --nnodes=1 --node_rank=0 \
    --master_addr="127.0.0.1" --master_port=12350 \
    "$BASE/scripts/run_grpo_trl.py" \
    --reward-mode "$REWARD_MODE" \
    --output-dir "$OUTPUT_DIR"
