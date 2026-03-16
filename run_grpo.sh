#!/bin/bash
# ChartVCR GRPO Training Runner
# Usage: bash run_grpo.sh [baseline|cvr|cvr_equal] [output_dir]
#
# Uses vllm_nightly_env for training + vLLM generation
# GPUs 0-7: training (DeepSpeed ZeRO-2)
# vLLM generation: colocated (use_vllm=True)

set -e

REWARD_MODE=${1:-baseline}
BASE="/ex_disk2/mhpark/poc/chartvr"
NIGHTLY_PYTHON="/ex_disk2/mhpark/poc/vllm_nightly_env/bin/python"
NIGHTLY_ACCEL="/ex_disk2/mhpark/poc/vllm_nightly_env/bin/accelerate"

case "$REWARD_MODE" in
  baseline|outcome_only) REWARD_MODE="outcome_only"; OUTPUT_DIR="$BASE/ckpt/baseline" ;;
  cvr|chartvr)           REWARD_MODE="chartvr";       OUTPUT_DIR="$BASE/ckpt/cvr" ;;
  cvr_equal)             REWARD_MODE="chartvr";       OUTPUT_DIR="$BASE/ckpt/cvr_equal" ;;
  abl_no_proc)           REWARD_MODE="chartvr";       OUTPUT_DIR="$BASE/ckpt/abl_no_process" ;;
  *)                     OUTPUT_DIR="$BASE/ckpt/$REWARD_MODE" ;;
esac

mkdir -p "$OUTPUT_DIR" "$BASE/logs"

echo "=== GRPO Training: $REWARD_MODE ==="
echo "Output: $OUTPUT_DIR"
echo "Python: $NIGHTLY_PYTHON"

export PYTHONPATH="$BASE:$PYTHONPATH"
export CHARTVR_ROOT="$BASE"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
$NIGHTLY_ACCEL launch \
    --config_file "$BASE/accelerate_config.yaml" \
    "$BASE/train_grpo.py" \
    --reward_type "$REWARD_MODE" \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$BASE/logs/grpo_${REWARD_MODE}.log"
