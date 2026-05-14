#!/usr/bin/env bash
# Score all 9 models × 2 modes with deterministic + Claude judge.
# Phase A: deterministic chartqa_h/a/pro (parallel, instant).
# Phase B: Claude Sonnet judge for charxiv_reasoning + chartmuseum (4 concurrent groups).
set -uo pipefail
ROOT="${CHARTVR_ROOT:-/ex_disk2/mhpark/poc/chartvr}"
PY="/ex_disk2/mhpark/poc/vllm_nightly_env/bin/python"
LOG_DIR="$ROOT/logs/standard_eval/scoring"
mkdir -p "$LOG_DIR"

MODELS=(zero_shot_qwen3_5_4b v8_sft v_hq_lite_sft v9_sft v9_1_sft row_a_outcome row_b_vapv row3_dapo_outcome row4_dapo_vapv)
MODES=(off on)

echo "[scoring] Phase A — deterministic chartqa_{human,augmented,pro}"
for m in "${MODELS[@]}"; do
  for mode in "${MODES[@]}"; do
    dir="$ROOT/results/standard/$m/think_$mode"
    [[ -d "$dir" ]] || continue
    "$PY" "$ROOT/score_standard.py" "$dir" \
      --benches chartqa_human,chartqa_augmented,chartqa_pro \
      > "$LOG_DIR/det_${m}_${mode}.log" 2>&1 &
  done
done
wait
echo "[scoring] Phase A done"

echo "[scoring] Phase B — Claude Sonnet judge for charxiv_reasoning + chartmuseum"
# Split 18 (model,mode) tasks into groups of 4 concurrent
BATCH_SIZE=4
i=0
pids=()
for m in "${MODELS[@]}"; do
  for mode in "${MODES[@]}"; do
    dir="$ROOT/results/standard/$m/think_$mode"
    [[ -d "$dir" ]] || continue
    "$PY" "$ROOT/score_standard.py" "$dir" \
      --benches charxiv_reasoning,chartmuseum \
      --use_judge --concurrency 6 \
      > "$LOG_DIR/judge_${m}_${mode}.log" 2>&1 &
    pids+=( $! )
    i=$((i + 1))
    if [[ $((i % BATCH_SIZE)) -eq 0 ]]; then
      for p in "${pids[@]}"; do wait "$p"; done
      pids=()
      echo "[scoring] judge batch done ($i tasks)"
    fi
  done
done
# wait remaining
for p in "${pids[@]}"; do wait "$p"; done
echo "[scoring] Phase B done $(date)"
