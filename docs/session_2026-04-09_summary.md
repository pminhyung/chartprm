# Session Summary — 2026-04-09

## Goal
Execute v_hq teacher-distillation plan to recover from v9.1 NO-GO (CharXiv 39.50%).

## Outcome
- **Teacher distillation abandoned** due to infeasible throughput on current hardware
- **Pivoted to v_hq-lite**: use `data/sft_hq_lite.jsonl` with existing v9/v8 reasoning + prompt unification + rule exclusion
- **All v_hq-lite deliverables prepared and validated**; training awaits user GPU allocation approval
- **Alternative A backup prepared**: `data/sft_hq_lite_no_chartqa.jsonl` (25,744 samples, 100% reasoning)

## What got built

### Data
| File | Samples | Description |
|---|---|---|
| `data/distill_source.jsonl` | 43,744 | Clean source: v9 non-rule + v8 sft_30k + block_c_api_qa_v2, with reasoning_steps preserved |
| `data/sft_hq_lite.jsonl` | 43,744 | Training-ready dataset (primary) |
| `data/sft_hq_lite_no_chartqa.jsonl` | 25,744 | Alternative A: chartqa_train dropped, 100% reasoning |

### Code
| File | Change |
|---|---|
| `train_sft.py` | Patched: `EVAL_SYSTEM_PROMPT`, eval user template, `assistant_text` priority field |
| `scripts/build_distill_source.py` | NEW — source merge, filter, dedup, reasoning carryover |
| `scripts/prepare_sft_hq_lite.py` | NEW — sft_hq_lite builder with optional empty-reason drop |
| `scripts/teacher_distill.py` | NEW — OpenAI SDK async distiller (not usable at scale) |
| `scripts/audit_distill.py` | NEW — distilled output auditor |
| `scripts/audit_sft_hq_lite.py` | NEW — lite dataset auditor |
| `scripts/merge_distilled_into_lite.py` | NEW — merger for opportunistic distillation |
| `scripts/generate_qa.py` | Modified: `max_tokens` removed (2 sites) |
| `scripts/generate_cot.py` | Modified: `max_tokens` removed (2 sites) |
| `scripts/verify_qa_answers.py` | Modified: `max_tokens` removed (1 site) |
| `code/rewards/llm_verifier.py` | Modified: `max_tokens` removed (3 sites + `__init__` param) |

### Docs
| File | Purpose |
|---|---|
| `docs/v_hq_pilot_audit.md` | Full diagnostic: throughput measurements, root causes, pivot rationale |
| `docs/v_hq_training_readiness.md` | Training command recipes + go/no-go criteria |
| `docs/v_hq_lite_audit.txt` | audit_sft_hq_lite.py stdout capture |
| `docs/session_2026-04-09_summary.md` | This file |

### Memory
| File | Content |
|---|---|
| `feedback_gpu_session_scope.md` | Session-scoped GPU policy (9200/9201 serve, training waits) |
| `feedback_qwen_server_immutable.md` | Fix at call/data layer only, never touch server |
| `project_397b_throughput.md` | 397B throughput findings, infeasibility conclusion |

## Key measurements

### 397B server (9200/9201) throughput (thinking mode)
- Aggregate: 6-14 tok/s per host with 4-7 concurrent requests
- Per-request: ~0.3-2.3 tok/s (high variance due to stragglers)
- Thinking response length: 1000-5000+ tokens per sample (from v9.1 distribution)
- Per-sample wall clock: 500-2500 seconds
- **9200 reached fully-hung state** (0% GPU util, 0.3 tok/s for single req) — lingering-request queue stalled, no client can clear it without restart

### sft_hq_lite.jsonl composition (43,744 samples)
- v8 sft_30k origin: 29,549 (67.5%) — 11,549 with reasoning + 18,000 chartqa_train placeholder
- v9 origin (non-rule): 11,017 (25.2%) — all with reasoning
- block_c_api_qa_v2: 3,178 (7.3%) — all with reasoning
- **58.9% with real reasoning, 41.1% placeholder**
- Sources: chartqa_train, scientific_ext, owid, synthetic, scientific, plotly_complex, kaggle_like, additional, worldbank
- Assistant response token length: p50=52, p95=178, max=1446 (Qwen3.5-4B tokenizer)
- Full sequence length (inc. ~800 for sys+user+image): max=2246, p95=978 — fits easily in `max_length=4096`
- All 43,744 images exist and load correctly

## Key decisions (for user review)

1. **Abandon teacher distillation** — justified by measured throughput (see `docs/v_hq_pilot_audit.md`)
2. **Use sft_hq_lite.jsonl as primary training data** — addresses root causes #3 (bimodal) and #4 (prompt mismatch) from v9.1 analysis; indirectly may address #1 (runaway) via cleaner distribution
3. **Keep 9200 and 9201 servers untouched** — 9200 is hung but restart requires user approval; leaving in place
4. **Prepared Alternative A** — drop chartqa_train entirely if v_hq-lite underperforms

## What the user must decide when returning

1. **Approve GPU for training**: 9200 is fully-hung and essentially unusable. Killing 9200 frees GPU 0-7 with minimal real cost (no ongoing work).
2. **Approve first training run**: v_hq-lite (recommended primary) OR go directly to Alternative A (cleaner but smaller)
3. **Re-evaluate 397B teacher distillation as long-tail project**: possibly viable with many days of compute, but not for this experiment iteration

## Next concrete commands (after user approves)

```bash
# 1. Free GPU 0-7 (kill hung 9200; 9201 untouched)
kill 1154658

# 2. Verify GPUs are clean
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -8

# 3. Launch SFT training
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 setsid nohup \
  /ex_disk2/mhpark/poc/vllm_nightly_env/bin/accelerate launch \
    --num_processes 8 --config_file scripts/deepspeed_zero3_nooffload_8gpu.yaml \
    train_sft.py \
    --model_path /ex_disk2/mhpark/poc/chartvr/models/qwen3.5-4b \
    --data data/sft_hq_lite.jsonl \
    --output_dir ckpt/sft_hq_lite_4b_8gpu \
    --use_lora --lora_rank 64 --lora_alpha 128 \
    --max_length 4096 \
    </dev/null > /tmp/train_logs/sft_hq_lite.log 2>&1 & disown

# 4. Monitor: tail -f /tmp/train_logs/sft_hq_lite.log
# 5. After ~5-6h training + merge + eval per docs/v_hq_training_readiness.md
```
