#!/bin/bash
# Phase 7: Ablation run scripts
# Each ablation is a separate GRPO run from the same SFT checkpoint

set -e

BASE="/ex_disk2/mhpark/poc/chartvr"
BIGCHARTS="$BASE/code/bigcharts-r1/src/open-r1-multimodal"
export PYTHONPATH="$BASE:$PYTHONPATH"
export DEBUG_MODE="true"

cd "$BIGCHARTS"

run_ablation() {
    local RUN_NAME=$1
    local REWARD_FUNCS=$2
    local OUTPUT_DIR="$BASE/checkpoints/ablations/$RUN_NAME"

    echo "=== Running ablation: $RUN_NAME ==="
    export LOG_PATH="$BASE/logs/debug_log_${RUN_NAME}.txt"

    torchrun --nproc_per_node=8 \
        --nnodes=1 --node_rank=0 \
        --master_addr="127.0.0.1" --master_port=12346 \
        src/open_r1/grpo_bigcharts.py \
        --deepspeed local_scripts/zero3.json \
        --output_dir "$OUTPUT_DIR" \
        --model_name_or_path "$BASE/checkpoints/sft" \
        --dataset_name data_config/chartvr_rl_data.yaml \
        --image_root "/" \
        --reward_funcs $REWARD_FUNCS \
        --max_prompt_length 1024 \
        --max_completion_length 2048 \
        --num_generations 8 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 2 \
        --learning_rate 1e-6 \
        --logging_steps 1 \
        --bf16 --torch_dtype bfloat16 \
        --data_seed 42 --report_to none \
        --gradient_checkpointing true \
        --attn_implementation flash_attention_2 \
        --num_train_epochs 1 \
        --run_name "$RUN_NAME" \
        --save_steps 100 --save_only_model true

    echo "=== Ablation $RUN_NAME complete ==="
}

# Priority order:
# A2b: Equal penalty (no causal distinction) — most critical for narrative
# run_ablation "ablation-a2b-equal-penalty" "chart_vcr_equal_penalty"

# A1b: R_acc + R_proc only (no format) — proves R_process value
# run_ablation "ablation-a1b-no-format" "chart_vcr_no_format"

# A1c: R_acc + R_format (no R_process) — isolates format contribution
# run_ablation "ablation-a1c-acc-format" "cerm" "format"

echo "Uncomment the ablation you want to run."
echo "Run one at a time to avoid GPU contention."
