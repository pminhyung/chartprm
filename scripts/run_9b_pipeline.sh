#!/bin/bash
# 9B Row C + Row D pipeline — robust version with proper cleanup
set -e

PYTHON=/ex_disk2/mhpark/poc/vllm_nightly_env/bin/python
BASE=/ex_disk2/mhpark/poc/chartvr
cd $BASE

cleanup_port() {
    local port=$1
    local pids=$(lsof -ti :$port 2>/dev/null)
    if [ -n "$pids" ]; then
        echo "  Killing processes on port $port: $pids"
        kill -9 $pids 2>/dev/null || true
        sleep 3
    fi
}

cleanup_all_servers() {
    echo "$(date): Cleaning up all servers..."
    cleanup_port 9100
    for i in $(seq 8001 8011); do cleanup_port $i; done
    sleep 2
}

wait_for_server() {
    local port=$1
    local max_attempts=$2
    for i in $(seq 1 $max_attempts); do
        if curl -s --max-time 3 http://localhost:$port/ 2>/dev/null | grep -q "detail\|Not Found"; then
            echo "  Server on port $port ready (attempt $i)"
            return 0
        fi
        sleep 5
    done
    echo "  ERROR: Server on port $port not ready after $max_attempts attempts"
    return 1
}

start_trl_vllm() {
    local gpu=$1
    local model=$2
    echo "$(date): Starting TRL vLLM on GPU $gpu..."
    cleanup_port 9100
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON -m trl.scripts.vllm_serve \
        --model $model \
        --tensor_parallel_size 1 --gpu_memory_utilization 0.85 \
        --max_model_len 8192 --port 9100 --trust_remote_code &
    VLLM_PID=$!
    echo "  vLLM PID: $VLLM_PID"
    wait_for_server 9100 120 || { echo "FATAL: vLLM server failed to start"; kill $VLLM_PID 2>/dev/null; exit 1; }
}

train_row() {
    local row_name=$1
    local reward_type=$2
    local output_dir=$3
    local log_file=$4
    local extra_args=$5

    echo ""
    echo "============================================================"
    echo "$(date): Training $row_name ($reward_type)"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=3,4,5,6,7,8,9,10 \
    VLLM_PORT=9100 NUM_TRAIN_GPUS=8 \
    $PYTHON -m accelerate.commands.launch \
        --num_processes 8 \
        --config_file scripts/deepspeed_zero3_nooffload_8gpu.yaml \
        train_grpo_dapo.py \
        --reward_type $reward_type \
        --data data/charts_v2/chartvr_train_final.jsonl \
        --output_dir $output_dir \
        --model_path models/qwen3.5-9b \
        --use_lora --lora_rank 64 --lora_alpha 128 \
        --num_generations 4 --max_completion_length 4096 \
        --per_device_batch_size 1 --gradient_accumulation_steps 4 \
        --learning_rate 1e-5 \
        $extra_args \
        2>&1 | tee $log_file

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "FATAL: Training $row_name failed!"
        return 1
    fi
    echo "$(date): $row_name training complete!"
}

eval_model() {
    local model_path=$1
    local result_dir=$2
    local row_name=$3

    echo ""
    echo "$(date): Evaluating $row_name..."
    cleanup_all_servers
    sleep 5

    # Start eval servers on GPU 2-11 (skip GPU 0-1)
    for i in 2 3 4 5 6 7 8 9 10 11; do
        port=$((8000 + i))
        CUDA_VISIBLE_DEVICES=$i nohup $PYTHON -m vllm.entrypoints.openai.api_server \
            --model $model_path \
            --tensor-parallel-size 1 --gpu-memory-utilization 0.85 \
            --max-model-len 8192 --port $port --trust-remote-code \
            --reasoning-parser qwen3 > /tmp/vllm_${row_name}_$i.log 2>&1 &
    done

    echo "  Waiting for eval servers..."
    for attempt in $(seq 1 120); do
        ready=0
        for port in 8002 8003 8004 8005 8006 8007 8008 8009 8010 8011; do
            if curl -s --max-time 2 http://localhost:$port/v1/models | grep -q model 2>/dev/null; then
                ready=$((ready + 1))
            fi
        done
        if [ "$ready" -ge 8 ]; then
            echo "  $ready/10 servers ready!"
            break
        fi
        if [ $((attempt % 20)) -eq 0 ]; then echo "  Attempt $attempt: $ready/10 ready"; fi
        sleep 5
    done

    mkdir -p $result_dir
    $PYTHON eval_multi_server.py \
        "8002,8003,8004,8005,8006,8007,8008,8009,8010,8011" \
        "$model_path" \
        "$result_dir" \
        "chartqa_human,chartqa_augmented,charxiv_reasoning,chartqa_pro,chartmuseum"

    echo "=== $row_name Results ==="
    for f in $result_dir/*.jsonl; do
        bench=$(basename "$f" .jsonl)
        total=$(wc -l < "$f")
        correct=$(grep -c '"accuracy": 1.0' "$f" 2>/dev/null || echo 0)
        pct=$($PYTHON -c "print(f'{$correct/$total*100:.1f}%') if $total>0 else print('N/A')" 2>/dev/null)
        echo "  $bench: $correct/$total = $pct"
    done

    cleanup_all_servers
}

# ═══════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════
echo "$(date): === 9B Pipeline Start ==="
cleanup_all_servers

# ── Row C: DAPO + outcome ──
start_trl_vllm 2 models/qwen3.5-9b
train_row "Row C" "outcome_only" "ckpt/row_c_9b_outcome" "logs/grpo_row_c.log"
kill $VLLM_PID 2>/dev/null; cleanup_port 9100; sleep 5

echo "$(date): Merging Row C LoRA..."
$PYTHON scripts/merge_lora.py --base models/qwen3.5-9b --lora ckpt/row_c_9b_outcome --output ckpt/row_c_merged

eval_model "$BASE/ckpt/row_c_merged" "results/v7/row_c" "row_c"

# ── Row D: DAPO + process-augmented ──
start_trl_vllm 2 models/qwen3.5-9b
train_row "Row D" "conditional_cvr" "ckpt/row_d_9b_cvr" "logs/grpo_row_d.log" "--verifier_type rule"
kill $VLLM_PID 2>/dev/null; cleanup_port 9100; sleep 5

echo "$(date): Merging Row D LoRA..."
$PYTHON scripts/merge_lora.py --base models/qwen3.5-9b --lora ckpt/row_d_9b_cvr --output ckpt/row_d_merged

eval_model "$BASE/ckpt/row_d_merged" "results/v7/row_d" "row_d"

# ── Final Report ──
echo ""
echo "============================================================"
echo "  FINAL 9B RESULTS ($(date))"
echo "============================================================"
$PYTHON -c "
import json, os

rows = {'zeroshot_9b': '9B Zero-shot', 'row_c': '9B Row C (outcome)', 'row_d': '9B Row D (proc-aug)'}
benchmarks = ['chartqa_human', 'chartqa_augmented', 'charxiv_reasoning', 'chartqa_pro', 'chartmuseum']

print(f'{\"\":<24}', '  '.join(f'{b:<12}' for b in ['CQA-H','CQA-A','CharXiv','CQA-Pro','ChartMus','AVG']))
print('-'*100)

for row_dir, label in rows.items():
    accs = []
    vals = []
    for bench in benchmarks:
        path = f'results/v7/{row_dir}/{bench}.jsonl'
        if not os.path.exists(path):
            vals.append('N/A'); continue
        total, correct = 0, 0
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                total += 1
                correct += r.get('accuracy', 0)
        pct = correct/total*100 if total > 0 else 0
        accs.append(pct)
        vals.append(f'{pct:.1f}%')
    avg = f'{sum(accs)/len(accs):.1f}%' if accs else 'N/A'
    vals.append(avg)
    print(f'{label:<24}', '  '.join(f'{v:<12}' for v in vals))
"

echo ""
echo "$(date): Pipeline complete!"
