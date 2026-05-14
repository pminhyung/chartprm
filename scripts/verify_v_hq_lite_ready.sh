#!/usr/bin/env bash
# Verify v_hq-lite training readiness. Run this before launching SFT.
# Produces a ready/not-ready verdict at the end.

set -u
cd /ex_disk2/mhpark/poc/chartvr

EXIT=0
ok() { printf "  [OK] %s\n" "$1"; }
fail() { printf "  [FAIL] %s\n" "$1"; EXIT=1; }

echo "=== v_hq-lite Training Readiness Check ==="
echo

echo "[1/6] Data files"
for F in data/sft_hq_lite.jsonl data/sft_hq_lite_no_chartqa.jsonl data/distill_source.jsonl; do
  if [ -f "$F" ]; then
    L=$(wc -l < "$F")
    if [ "$L" -gt 100 ]; then
      ok "$F ($L lines)"
    else
      fail "$F exists but only $L lines"
    fi
  else
    fail "$F missing"
  fi
done

echo
echo "[2/6] Modified/new code files"
for F in train_sft.py eval_multi_server.py scripts/teacher_distill.py scripts/build_distill_source.py scripts/prepare_sft_hq_lite.py scripts/audit_sft_hq_lite.py scripts/audit_distill.py scripts/merge_distilled_into_lite.py code/rewards/llm_verifier.py scripts/generate_qa.py scripts/generate_cot.py scripts/verify_qa_answers.py; do
  if [ -f "$F" ]; then
    if /ex_disk2/mhpark/poc/vllm_nightly_env/bin/python -c "import ast; ast.parse(open('$F').read())" 2>/dev/null; then
      ok "$F syntax valid"
    else
      fail "$F syntax error"
    fi
  else
    fail "$F missing"
  fi
done

echo
echo "[3/6] train_sft.py prompt unification"
if grep -q "EVAL_SYSTEM_PROMPT" train_sft.py && grep -q "Look at this chart and answer the question" train_sft.py; then
  ok "train_sft.py uses EVAL_SYSTEM_PROMPT + eval user template"
else
  fail "train_sft.py prompt unification missing"
fi

echo
echo "[4/6] max_tokens removed from 397B callers (active code only)"
# Only flag lines that are function arguments (contain `max_tokens=` not preceded by `#`)
# and are not inside a comment block. Use python AST-ish inspection via grep on non-comment lines.
MT_COUNT=0
for F in scripts/teacher_distill.py scripts/generate_qa.py scripts/generate_cot.py scripts/verify_qa_answers.py code/rewards/llm_verifier.py eval_multi_server.py; do
  # Strip comments (simple: lines whose first non-space char is #)
  HITS=$(awk '!/^\s*#/ && /max_tokens\s*=/ {print NR":"$0}' "$F" 2>/dev/null | grep -v "^.*:\s*#" || true)
  if [ -n "$HITS" ]; then
    fail "$F has active max_tokens= line(s):"
    echo "$HITS" | sed 's/^/      /'
    MT_COUNT=$((MT_COUNT + 1))
  fi
done
if [ "$MT_COUNT" -eq 0 ]; then
  ok "all 6 files: no active max_tokens= assignments in code"
fi

echo
echo "[5/6] Docs present"
for F in docs/v_hq_pilot_audit.md docs/v_hq_training_readiness.md docs/session_2026-04-09_summary.md; do
  if [ -f "$F" ]; then
    ok "$F"
  else
    fail "$F missing"
  fi
done

echo
echo "[6/6] GPU availability quick check"
FREE_GPUS=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | awk -F, '$2+0 > 70000 {print $1}' | head -8 | wc -l)
echo "  GPUs with >70GB free memory: $FREE_GPUS (need 8 for TP=1 training)"
if [ "$FREE_GPUS" -ge 8 ]; then
  ok "enough free GPUs for training"
else
  printf "  [INFO] not enough free GPUs — 9200/9201 still holding. User must free one group before training\n"
fi

echo
if [ "$EXIT" -eq 0 ]; then
  echo "=== READY ==="
  echo "Next: kill 9200 or 9201, then launch training per docs/v_hq_training_readiness.md"
else
  echo "=== NOT READY — fix issues above ==="
fi
exit "$EXIT"
