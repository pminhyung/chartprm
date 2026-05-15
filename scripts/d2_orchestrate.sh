#!/bin/bash
# D2 orchestration — chain extract→verify→alignment for main d2_pilot,
# then segment+extract+verify for each baseline, then 4-cell distribution.
#
# Usage: scripts/d2_orchestrate.sh [main|baselines|all|4cell]
#
# Assumes vLLM servers running on ports 8100 (verifier A), 8101 (verifier B),
# 8200 (extractor), 8300 (chart_r1), 8301 (chartgemma).

set -euo pipefail

export CHARTVR_ROOT=/ex_disk2/mhpark/poc/chartvr
PY=/ex_disk2/mhpark/poc/vllm_nightly_env/bin/python
ROOT=/ex_disk2/mhpark/poc/chartvr
WT=$ROOT/.worktrees/research-chartprm

MODE=${1:-all}

run_main_verify() {
  echo "[orchestrate] === main d2_pilot: perception_verify_v2 ==="
  $PY -u $WT/scripts/d2_perception_verify_v2.py \
    --input $ROOT/data/d2_pilot/claims_v2.jsonl \
    --output $ROOT/data/d2_pilot/perception_v2.jsonl \
    --ports 8100,8101 \
    --model internvl_verifier \
    --concurrent_per_host 6
}

run_main_alignment() {
  echo "[orchestrate] === main d2_pilot: alignment_v2 ==="
  $PY -u $WT/scripts/d2_alignment_v2.py
}

run_baseline_pipeline() {
  local NAME=$1
  echo "[orchestrate] === baseline ${NAME}: segment ==="
  if [ "$NAME" = "zeroshot_4b" ]; then
    # Already segmented from d1_pilot_4b/segmented_v3.jsonl, just symlink
    : # baseline_zeroshot_4b_segmented.jsonl already created by d2_prep_zeroshot_4b.py
  else
    $PY -u $WT/scripts/d2_segment_baseline.py \
      --input $ROOT/data/d2_pilot/baseline_${NAME}.jsonl \
      --output $ROOT/data/d2_pilot/baseline_${NAME}_segmented.jsonl
  fi

  echo "[orchestrate] === baseline ${NAME}: extract_v2 ==="
  $PY -u $WT/scripts/d2_claim_extract_v2.py \
    --input $ROOT/data/d2_pilot/baseline_${NAME}_segmented.jsonl \
    --output $ROOT/data/d2_pilot/baseline_${NAME}_claims.jsonl \
    --ports 8200 \
    --model qwen3_6_27b_extractor \
    --concurrent_per_host 8

  echo "[orchestrate] === baseline ${NAME}: verify_v2 ==="
  $PY -u $WT/scripts/d2_perception_verify_v2.py \
    --input $ROOT/data/d2_pilot/baseline_${NAME}_claims.jsonl \
    --output $ROOT/data/d2_pilot/baseline_${NAME}_perception.jsonl \
    --ports 8100,8101 \
    --model internvl_verifier \
    --concurrent_per_host 6
}

run_4cell() {
  echo "[orchestrate] === 4-cell distribution ==="
  $PY -u $WT/scripts/d2_4cell_distribution.py
}

case "$MODE" in
  main)
    run_main_verify
    run_main_alignment
    ;;
  baselines)
    for B in zeroshot_4b chart_r1 chartgemma; do
      run_baseline_pipeline $B
    done
    ;;
  4cell)
    run_4cell
    ;;
  all)
    run_main_verify
    run_main_alignment
    for B in zeroshot_4b chart_r1 chartgemma; do
      run_baseline_pipeline $B
    done
    run_4cell
    ;;
  *)
    echo "Usage: $0 [main|baselines|all|4cell]"
    exit 1
    ;;
esac
echo "[orchestrate] DONE"
