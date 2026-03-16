#!/bin/bash
# Phase 2: SFT warm-up on ChartQA data
# Teaches Qwen2.5-VL-7B the <thinking>/<answer> format

set -e

export DISABLE_VERSION_CHECK=1
BASE="/ex_disk2/mhpark/poc/chartvr"
cd "$BASE/code/llama-factory"

# 8 GPUs via DeepSpeed
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
llamafactory-cli train \
    --stage sft \
    --do_train true \
    --model_name_or_path "$BASE/models/qwen25vl-7b" \
    --dataset chartvr_sft \
    --dataset_dir "$BASE/code/llama-factory/data" \
    --template qwen2_vl \
    --output_dir "$BASE/checkpoints/sft" \
    --overwrite_output_dir true \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 500 \
    --bf16 true \
    --gradient_checkpointing true \
    --max_length 2048 \
    --deepspeed "$BASE/code/bigcharts-r1/src/open-r1-multimodal/local_scripts/zero3.json" \
    --report_to none
