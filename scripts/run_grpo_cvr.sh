#!/bin/bash
# Phase 5: ChartVCR-GRPO — ChartVCR reward
set -e

export DEBUG_MODE="true"
RUN_NAME="chartvr-grpo-cvr"
BASE="/ex_disk2/mhpark/poc/chartvr"
BIGCHARTS="$BASE/code/bigcharts-r1/src/open-r1-multimodal"
LOG_DIR="$BASE/logs"
mkdir -p "$LOG_DIR"
export LOG_PATH="$LOG_DIR/debug_log_${RUN_NAME}.txt"
export PYTHONPATH="$BASE:$BIGCHARTS/src:$PYTHONPATH"
VENV="/ex_disk2/mhpark/poc/vllm_env/bin"
cd "$BIGCHARTS"

$VENV/torchrun --nproc_per_node=8 \
    --nnodes=1 --node_rank=0 \
    --master_addr="127.0.0.1" --master_port=12346 \
    src/open_r1/grpo_bigcharts.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir "$BASE/checkpoints/grpo_cvr" \
    --model_name_or_path "$BASE/checkpoints/sft" \
    --dataset_name data_config/chartvr_rl_data.yaml \
    --image_root "/" \
    --reward_funcs "chart_vcr" \
    --max_prompt_length 1024 \
    --max_completion_length 1024 \
    --num_generations 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-6 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to none \
    --gradient_checkpointing true \
    --attn_implementation sdpa \
    --num_train_epochs 1 \
    --run_name "$RUN_NAME" \
    --save_steps 100 \
    --save_only_model true
