# v_hq SFT Training Readiness — 2026-04-09

## Pre-training checklist (v_hq-lite)

This document records the state needed for a user to launch `train_sft.py` once GPU policy allows new training. No training is executed in this session (GPU 0-7 held by 9200, GPU 8-15 held by 9201).

## Data

### Primary: `data/sft_hq_lite.jsonl` (43,744 samples) — RECOMMENDED
### Alternative A: `data/sft_hq_lite_no_chartqa.jsonl` (25,744 samples)
  - chartqa_train entirely dropped, 100% samples have real reasoning
  - Use if primary underperforms on CharXiv (hypothesis: chartqa_train with placeholder reasoning dilutes signal)

### Primary composition
- **Composition**:
  - v9 high-signal sources (11,017 with `reasoning_steps`): plotly_complex, scientific_ext, owid, synthetic, worldbank
  - v8 sft_30k (29,549 after dedup, of which 18,000 chartqa_train have placeholder reasoning, 11,549 others have `reasoning_steps`)
  - block_c_api_qa_v2 (3,178 with `reasoning_steps`)
- **Rejected from v9**: 30,923 rule-template rows (block_a/b/c/d/f), addressing the v9.1 bimodal reasoning failure mode
- **With real reasoning**: 25,744 / 43,744 = 58.9% (non-chartqa_train)
- **Placeholder reasoning**: 18,000 / 43,744 = 41.1% (chartqa_train only)
- **Assistant response token length** (Qwen3.5-4B tokenizer):
  - `p50=52  p90=141  p95=178  p99=446  max=1446 tokens`
- **Full sequence length** (sys+user+assistant+image≈800 tokens):
  - `p50=852  p95=978  p99=1246  max=2246 tokens`
- **Fit budget**: 100% of samples fit within `max_length=2048` and `max_length=4096`
- **Full load verified**: 43,744/43,744 samples loaded by `train_sft.py::load_sft_dataset` — 0 skipped, format checks pass end-to-end

## Code

- **`train_sft.py`** — patched:
  - `SYSTEM_PROMPT` → `EVAL_SYSTEM_PROMPT` (L32, L71 aliased as messages system)
  - user content: `f"Question: {q}"` → `f"Look at this chart and answer the question.\n\nQuestion: {q}"` (L71-73)
  - Added `assistant_text` priority field for teacher-distilled samples (L66-70)
  - Legacy fallback: existing `reasoning_steps` (list or str) wrapped into `<think>\n{r}\n</think>\n<answer>{ans}</answer>`
  - Placeholder fallback for samples with no reasoning (chartqa_train): `<think>\nLet me analyze the chart to answer this question.\nThe answer is {ans}.\n</think>\n<answer>{ans}</answer>`

- **`scripts/build_distill_source.py`** — built, includes reasoning_steps carryover fix
- **`scripts/prepare_sft_hq_lite.py`** — built, packaged sft_hq_lite.jsonl
- **`scripts/teacher_distill.py`** — built, `max_tokens` omitted (opportunistic augmentation only)
- **`scripts/audit_distill.py`** — built, use for quality audit of distilled output

## Recommended training command (for user when GPU is available)

```bash
# GPU 0-7 or 8-15 must be freed first — user decision
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 setsid nohup \
  /ex_disk2/mhpark/poc/vllm_nightly_env/bin/accelerate launch \
    --num_processes 8 \
    --config_file scripts/deepspeed_zero3_nooffload_8gpu.yaml \
    train_sft.py \
    --model_path /ex_disk2/mhpark/poc/chartvr/models/qwen3.5-4b \
    --data data/sft_hq_lite.jsonl \
    --output_dir ckpt/sft_hq_lite_4b_8gpu \
    --use_lora --lora_rank 64 --lora_alpha 128 \
    --max_length 4096 \
    </dev/null > /tmp/train_logs/sft_hq_lite.log 2>&1 & disown
```

Notes:
- `max_length=4096` is generous (p99=1246, max=2246). Could drop to 2048 if memory-constrained.
- LoRA rank=64/alpha=128 matches v9.1 (stable config).
- ZeRO-3 no-offload per project policy.
- Must verify GPU availability before launch — do NOT kill 9200/9201 unless user approves.

## Post-training

```bash
# Merge LoRA
python scripts/merge_lora.py \
    --base models/qwen3.5-4b \
    --lora ckpt/sft_hq_lite_4b_8gpu \
    --output ckpt/sft_hq_lite_4b_8gpu_merged

# 5-bench eval (8 GPU TP=1)
for i in 0 1 2 3 4 5 6 7; do
    port=$((8000+i))
    CUDA_VISIBLE_DEVICES=$i setsid nohup \
      /ex_disk2/mhpark/poc/vllm_nightly_env/bin/python -m vllm.entrypoints.openai.api_server \
        --model ckpt/sft_hq_lite_4b_8gpu_merged \
        --tensor-parallel-size 1 --gpu-memory-utilization 0.85 \
        --max-model-len 8192 --port $port \
        --trust-remote-code --reasoning-parser qwen3 --generation-config vllm \
        --served-model-name hqlite \
        </dev/null > /tmp/train_logs/vllm_eval_$i.log 2>&1 & disown
done

python eval_multi_server.py \
    "8000,8001,8002,8003,8004,8005,8006,8007" \
    "hqlite" \
    "results/hqlite/sft_hq_lite_4b" \
    "chartqa_human,chartqa_augmented,charxiv_reasoning,chartqa_pro,chartmuseum"
```

## Expected outcome

- **Baseline hypothesis**: v_hq-lite achieves ≥ v8 (46% CharXiv, 56.83% avg) due to:
  1. Prompt unification (eval format drift eliminated)
  2. Rule template exclusion (no bimodal collapse)
  3. Additional high-quality v9 reasoning samples
  4. **eval_multi_server.py `max_tokens=4096` removed** — unblocks long reasoning at eval time, likely the direct cause of v9.1 44.8% empty-content runaway
- **Risk**: v9 showed CharXiv regression (-2.4pp) when adding its data to v8. v_hq-lite mitigates by prompt fix + rule exclusion, but if the CharXiv drop was caused by chartqa_train dilution rather than bimodal, we would still see a small drop.
- **Upper bound**: Teacher distillation to add natural-length reasoning to scientific_ext samples is running opportunistically in background (targeted at max 500 samples on 9201 only due to 9200 straggler). If successful, those samples will be merged as `sft_hq_augmented.jsonl` for a second training run.

## Go/no-go

- GO (v_hq-lite achieves CharXiv ≥ 46% AND avg ≥ 56.83%): proceed to GRPO
- Partial GO (CharXiv 44-46%): add teacher-distilled augmentation if available, or try Alternative A
- NO-GO (CharXiv < 44%): try Alternative A (`sft_hq_lite_no_chartqa.jsonl`), then revisit if still failing

## Alternative A recipe

```bash
# Same as primary but change --data path and optionally --lora_rank
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 setsid nohup \
  /ex_disk2/mhpark/poc/vllm_nightly_env/bin/accelerate launch \
    --num_processes 8 \
    --config_file scripts/deepspeed_zero3_nooffload_8gpu.yaml \
    train_sft.py \
    --model_path /ex_disk2/mhpark/poc/chartvr/models/qwen3.5-4b \
    --data data/sft_hq_lite_no_chartqa.jsonl \
    --output_dir ckpt/sft_hq_lite_no_cqa_4b_8gpu \
    --use_lora --lora_rank 64 --lora_alpha 128 \
    --max_length 4096 \
    </dev/null > /tmp/train_logs/sft_hq_lite_no_cqa.log 2>&1 & disown
```
