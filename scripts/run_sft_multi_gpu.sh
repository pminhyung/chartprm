#!/bin/bash
# Phase 2: SFT training on 8 GPUs with DeepSpeed ZeRO-3
set -e

BASE="/ex_disk2/mhpark/poc/chartvr"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
/ex_disk2/mhpark/poc/vllm_env/bin/torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=29500 \
    "$BASE/scripts/run_sft_direct.py"
