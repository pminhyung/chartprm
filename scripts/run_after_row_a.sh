#!/bin/bash
# After Row A completes: eval Row A, then start Row B
# Run this script after Row A training finishes

set -e
PYTHON=/ex_disk2/mhpark/poc/vllm_nightly_env/bin/python
BASE=/ex_disk2/mhpark/poc/chartvr
cd $BASE

echo "=== Step 1: Kill vLLM training server ==="
for pid in $(lsof -ti :9100 2>/dev/null); do
    kill $pid 2>/dev/null && echo "Killed PID $pid on port 9100"
done
sleep 5

echo "=== Step 2: Merge LoRA ==="
$PYTHON scripts/merge_lora.py \
    --base models/qwen3.5-4b \
    --lora ckpt/row_a_dapo_outcome \
    --output ckpt/row_a_merged

echo "=== Step 3: Start eval servers (GPU 2-11, TP=1, 10 servers) ==="
for i in 2 3 4 5 6 7 8 9 10 11; do
    port=$((8000 + i))
    CUDA_VISIBLE_DEVICES=$i nohup $PYTHON -m vllm.entrypoints.openai.api_server \
        --model $BASE/ckpt/row_a_merged \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 8192 \
        --port $port \
        --trust-remote-code \
        --reasoning-parser qwen3 \
        > /tmp/vllm_eval_$i.log 2>&1 &
    echo "Started eval server on GPU $i, port $port"
done

echo "=== Step 4: Wait for eval servers ==="
for port in 8002 8003 8004 8005 8006 8007 8008 8009 8010 8011; do
    until curl -s http://localhost:$port/v1/models | grep -q model; do sleep 3; done
    echo "Server $port ready"
done

echo "=== Step 5: Run eval (5 benchmarks) ==="
$PYTHON eval_multi_server.py \
    "8002,8003,8004,8005,8006,8007,8008,8009,8010,8011" \
    "$BASE/ckpt/row_a_merged" \
    "results/v7/row_a" \
    "chartqa_human,chartqa_augmented,charxiv_reasoning,chartqa_pro,chartmuseum"

echo "=== Step 6: Kill eval servers ==="
for port in 8002 8003 8004 8005 8006 8007 8008 8009 8010 8011; do
    pid=$(lsof -ti :$port 2>/dev/null)
    if [ -n "$pid" ]; then kill $pid 2>/dev/null; fi
done
sleep 5

echo "=== Step 7: Start vLLM for Row B training ==="
CUDA_VISIBLE_DEVICES=2 nohup $PYTHON -c "
from trl.scripts.vllm_serve import main; import sys; sys.argv = [
    'vllm-serve', '--model', '$BASE/models/qwen3.5-4b',
    '--tensor_parallel_size', '1', '--gpu_memory_utilization', '0.85',
    '--max_model_len', '8192', '--port', '9100', '--trust_remote_code'
]
main()
" > /tmp/trl_vllm_serve_b.log 2>&1 &
# Wait for server
sleep 30
until curl -s http://localhost:9100/ 2>/dev/null; do sleep 5; done
echo "vLLM server for Row B ready"

echo "=== Step 8: Start Row B training (no CPU offload) ==="
CUDA_VISIBLE_DEVICES=4,5,6,7,8,9,10,11 \
VLLM_PORT=9100 \
NUM_TRAIN_GPUS=8 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup $PYTHON -m accelerate.commands.launch \
    --num_processes 8 \
    --config_file scripts/deepspeed_zero3_nooffload_8gpu.yaml \
    train_grpo_dapo.py \
    --reward_type conditional_cvr \
    --verifier_type rule \
    --data $BASE/data/charts_v2/chartvr_train_final.jsonl \
    --output_dir ckpt/row_b_dapo_cvr \
    --use_lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --num_generations 4 \
    --max_completion_length 1024 \
    --learning_rate 1e-5 \
    --per_device_batch_size 1 \
    --gradient_accumulation_steps 4 \
    > logs/grpo_row_b.log 2>&1 &

echo "Row B training started, PID $!"
echo "=== Pipeline complete ==="
