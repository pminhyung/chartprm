#!/usr/bin/env bash
# Run standard-protocol eval across all comparison models × 5 benches × thinking on/off.
# Strategy: load ONE model on all 16 GPUs (TP=1 per GPU = 16 vLLM servers), run all
# benches in both thinking modes, kill servers, swap model, repeat.
#
# Usage: ./scripts/run_standard_eval_all.sh
#   logs:   logs/standard_eval/
#   results: results/standard/<model_tag>/think_{on,off}/<bench>.jsonl
#
# GPU policy: user explicitly authorized full 0-15 utilization for this run.

set -uo pipefail

ROOT="${CHARTVR_ROOT:-/ex_disk2/mhpark/poc/chartvr}"
PY="/ex_disk2/mhpark/poc/vllm_nightly_env/bin/python"
GPUS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
PORTS=(); for g in "${GPUS[@]}"; do PORTS+=( $((8000 + g)) ); done
PORTS_CSV=$(IFS=,; echo "${PORTS[*]}")

LOG_DIR="$ROOT/logs/standard_eval"
mkdir -p "$LOG_DIR"

BENCHES="chartqa_human,chartqa_augmented,chartqa_pro,charxiv_reasoning,chartmuseum"

# Models: tag → checkpoint path. Order from cheapest/most-validated to newest.
declare -a MODELS=(
  "zero_shot_qwen3_5_4b:$ROOT/models/qwen3.5-4b"
  "v8_sft:$ROOT/ckpt/sft_v8_improved_v2_4b_merged"
  "v_hq_lite_sft:$ROOT/ckpt/sft_hq_lite_clean_4b_8gpu_merged"
  "v9_sft:$ROOT/ckpt/sft_v9_4b_8gpu_merged"
  "v9_1_sft:$ROOT/ckpt/sft_v9_1_4b_8gpu_merged"
  "row_a_outcome:$ROOT/ckpt/row_a_merged"
  "row_b_vapv:$ROOT/ckpt/row_b_merged"
  "row3_dapo_outcome:$ROOT/ckpt/row3_dapo_outcome_v2_merged"
  "row4_dapo_vapv:$ROOT/ckpt/row4_dapo_vapv_merged"
)

start_servers() {
  local model="$1"
  echo "[launcher] starting 16 vLLM servers for: $model" | tee -a "$LOG_DIR/launcher.log"
  for g in "${GPUS[@]}"; do
    local port=$((8000 + g))
    CUDA_VISIBLE_DEVICES=$g setsid nohup "$PY" -m vllm.entrypoints.openai.api_server \
      --model "$model" \
      --tensor-parallel-size 1 \
      --gpu-memory-utilization 0.85 \
      --max-model-len 8192 \
      --port "$port" \
      --trust-remote-code \
      --reasoning-parser qwen3 \
      </dev/null > "$LOG_DIR/vllm_g${g}.log" 2>&1 &
    disown
  done
}

wait_servers() {
  local deadline=$(( $(date +%s) + 600 ))  # 10 min hard cap
  local ready
  while true; do
    ready=0
    for p in "${PORTS[@]}"; do
      if curl -s -o /dev/null -w '%{http_code}' "http://localhost:$p/v1/models" 2>/dev/null | grep -q '^200$'; then
        ready=$((ready + 1))
      fi
    done
    echo "[launcher] $(date +%H:%M:%S) ready=$ready/${#PORTS[@]}" | tee -a "$LOG_DIR/launcher.log"
    [[ "$ready" -eq "${#PORTS[@]}" ]] && return 0
    [[ $(date +%s) -ge $deadline ]] && { echo "[launcher] timeout waiting for servers"; return 1; }
    sleep 15
  done
}

stop_servers() {
  echo "[launcher] stopping vLLM servers" | tee -a "$LOG_DIR/launcher.log"
  pkill -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true
  pkill -f 'VLLM::EngineCore' 2>/dev/null || true
  sleep 5
  # confirm
  local left
  left=$(ps -eo pid,cmd | grep -iE 'vllm.entrypoints|VLLM::EngineCore' | grep -v grep | wc -l)
  echo "[launcher] residual vllm procs: $left" | tee -a "$LOG_DIR/launcher.log"
}

run_one_model() {
  local tag="$1"; local mpath="$2"
  echo "===================================================================" | tee -a "$LOG_DIR/launcher.log"
  echo "[$(date +%F\ %T)] MODEL=$tag PATH=$mpath" | tee -a "$LOG_DIR/launcher.log"
  start_servers "$mpath"
  if ! wait_servers; then
    echo "[launcher] servers failed to come up — skipping $tag" | tee -a "$LOG_DIR/launcher.log"
    stop_servers
    return 1
  fi
  for mode in off on; do
    local out="$ROOT/results/standard/$tag/think_${mode}"
    mkdir -p "$out"
    echo "[$(date +%F\ %T)] $tag thinking=$mode → $out" | tee -a "$LOG_DIR/launcher.log"
    "$PY" "$ROOT/eval_standard.py" "$PORTS_CSV" "$mpath" "$out" "$BENCHES" \
      --concurrency 5 --thinking "$mode" \
      >> "$LOG_DIR/eval_${tag}_think_${mode}.log" 2>&1 || \
      echo "[launcher] WARN: eval failed for $tag/$mode" | tee -a "$LOG_DIR/launcher.log"
  done
  stop_servers
}

main() {
  echo "[launcher] start $(date)  models=${#MODELS[@]}  gpus=${#GPUS[@]}" | tee -a "$LOG_DIR/launcher.log"
  for entry in "${MODELS[@]}"; do
    tag="${entry%%:*}"
    mpath="${entry#*:}"
    if [[ ! -d "$mpath" ]]; then
      echo "[launcher] SKIP missing model: $tag → $mpath" | tee -a "$LOG_DIR/launcher.log"
      continue
    fi
    run_one_model "$tag" "$mpath"
  done
  echo "[launcher] done $(date)" | tee -a "$LOG_DIR/launcher.log"
}

main "$@"
