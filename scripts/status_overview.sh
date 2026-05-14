#!/usr/bin/env bash
# Single-screen snapshot of the overnight research pipeline state.
# Prints: distillation progress, GPU 0-7 occupancy, SFT status.

set -u

echo "========================================"
echo "ChartVCR overnight status   $(date +%Y-%m-%dT%H:%M:%S)"
echo "========================================"

# Teacher distillation
echo
echo "[1] Teacher distillation"
DISTILL_LOG=/tmp/teacher_distill_full.log
if pgrep -fa "teacher_distill.py --source" >/dev/null 2>&1; then
    PID=$(pgrep -f "teacher_distill.py --source" | head -1)
    ELAPSED=$(ps -p "$PID" -o etime= 2>/dev/null | xargs)
    echo "  process: RUNNING pid=$PID elapsed=$ELAPSED"
else
    echo "  process: NOT RUNNING"
fi
if [ -f /ex_disk2/mhpark/poc/chartvr/data/sft_hq_teacher.jsonl ]; then
    LINES=$(wc -l < /ex_disk2/mhpark/poc/chartvr/data/sft_hq_teacher.jsonl)
    echo "  output lines: $LINES"
fi
if [ -f "$DISTILL_LOG" ]; then
    tail -c 500 "$DISTILL_LOG" | tr '\r' '\n' | grep -E 'Distill:.*ok=' | tail -1 \
      | sed 's/^/  pbar: /'
fi

# GPU polling + SFT
echo
echo "[2] GPU 0-7 polling + SFT"
if pgrep -f "poll_gpu_and_launch_sft" >/dev/null 2>&1; then
    echo "  poll script: RUNNING"
    tail -3 /tmp/gpu_poll.log 2>/dev/null | sed 's/^/    /'
fi
SFTPID=$(
    ps -eo pid,cmd --no-headers \
      | awk '/accelerate launch/ && /train_sft\.py/ && !/awk/ && !/grep/ && !/bash -c/ { print $1; exit }'
)
if [ -n "$SFTPID" ]; then
    SFTELAPSED=$(ps -p "$SFTPID" -o etime= 2>/dev/null | xargs)
    echo "  SFT: RUNNING pid=$SFTPID elapsed=$SFTELAPSED"
    LATEST_LOG=$(ls -t /tmp/train_logs/sft_*.log 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        echo "  SFT log: $LATEST_LOG"
        tail -5 "$LATEST_LOG" 2>/dev/null | sed 's/^/    /'
    fi
else
    echo "  SFT: NOT RUNNING"
fi

# GPU snapshot
echo
echo "[3] GPU state"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader \
  | awk -F', ' '{ printf "  gpu %-2d  %7d MiB  %3d%%\n", $1, $2, $3 }'

# vLLM host health
echo
echo "[4] Qwen on-prem hosts"
for H in "http://10.1.211.147:8000/v1" "http://10.1.211.148:8000/v1" "http://localhost:9201/v1"; do
    MODEL=$(curl -s -m 3 "$H/models" 2>/dev/null | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["data"][0]["id"])' 2>/dev/null || echo "DOWN")
    printf "  %-40s %s\n" "$H" "$MODEL"
done

echo
echo "========================================"
