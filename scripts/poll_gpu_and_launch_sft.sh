#!/usr/bin/env bash
# Poll GPU 0-7 until completely free of compute processes, then launch SFT.
# Runs in background. Never kills other users' processes.
#
# Usage:
#   ./scripts/poll_gpu_and_launch_sft.sh <DATA_JSONL> <OUTPUT_DIR>

set -euo pipefail

DATA="${1:?DATA jsonl path required}"
OUTDIR="${2:?OUTPUT_DIR required}"
PY="${PY:-/ex_disk2/mhpark/poc/vllm_nightly_env/bin/python}"
ACC="${ACC:-/ex_disk2/mhpark/poc/vllm_nightly_env/bin/accelerate}"
LOG_DIR="${LOG_DIR:-/tmp/train_logs}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/sft_$(basename "$OUTDIR")_$(date +%Y%m%d_%H%M%S).log"

echo "[poll] target data=$DATA outdir=$OUTDIR log=$LOG"

poll=0
while true; do
    # Build the set of GPU UUIDs that belong to indices 0..7.
    # Format: "index, uuid" → keep uuid only where index < 8.
    idxs_gpus07=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader \
        | awk -F', ' '$1+0 < 8 { print $2 }')

    # Count compute processes whose gpu_uuid is in the 0-7 set. --query-compute-apps
    # columns: "pid, gpu_uuid, used_memory". CSV, so split on comma + optional space.
    n_procs=$(
        nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader 2>/dev/null \
            | awk -F', ' -v uuids="$idxs_gpus07" '
                BEGIN {
                    n = split(uuids, arr, "\n")
                    for (i = 1; i <= n; i++) set[arr[i]] = 1
                }
                { if ($2 in set) c++ }
                END { print (c ? c : 0) }
            '
    )

    if [ "$n_procs" -eq 0 ]; then
        # Extra safety: also require memory.used < 2GB on every GPU 0-7.
        # Covers the case where nvidia-smi compute-apps misses a process but
        # the allocator state still has weights resident.
        mem_ok=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1,2,3,4,5,6,7 \
            | awk 'BEGIN {ok=1} { if ($1+0 > 2000) ok=0 } END { print ok }')
        if [ "$mem_ok" -eq 1 ]; then
            echo "[$(date +%H:%M:%S)] GPU 0-7 fully free — launching SFT"
            break
        else
            if [ $((poll % 5)) -eq 0 ]; then
                echo "[$(date +%H:%M:%S)] poll #$poll: 0 compute procs but memory still resident on GPU 0-7"
            fi
        fi
    else
        if [ $((poll % 5)) -eq 0 ]; then
            echo "[$(date +%H:%M:%S)] poll #$poll: GPU 0-7 has $n_procs compute processes"
        fi
    fi
    poll=$((poll + 1))
    sleep 60
done

# Launch SFT in background
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 setsid nohup \
    "$ACC" launch \
        --num_processes 8 \
        --config_file scripts/deepspeed_zero3_nooffload_8gpu.yaml \
        train_sft.py \
        --model_path /ex_disk2/mhpark/poc/chartvr/models/qwen3.5-4b \
        --data "$DATA" \
        --output_dir "$OUTDIR" \
        --use_lora --lora_rank 64 --lora_alpha 128 --max_length 4096 \
    </dev/null > "$LOG" 2>&1 &
SFT_PID=$!
disown
echo "[$(date +%H:%M:%S)] SFT launched: PID=$SFT_PID log=$LOG"
