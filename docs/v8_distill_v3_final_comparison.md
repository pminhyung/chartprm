# v8 distill v3 SFT — final comparison

_Generated: 2026-05-14 02:50:59_

## Standard 5-bench (lmms-eval / charxiv judge / chartmuseum judge, think-on)

| Bench | v2 (orig) | v3 (VLMEK-chartqa) | Δ |
|---|---:|---:|---:|
| chartqa_human | 77.28% | 0.00% | -77.28 |
| chartqa_augmented | 83.52% | 0.00% | -83.52 |
| chartqa_pro | 27.57% | 0.00% | -27.57 |
| charxiv_reasoning | 0.00% | 0.00% | +0.00 |
| chartmuseum | 0.00% | 0.00% | +0.00 |
| **AVG5** | **37.67%** | **0.00%** | **-37.67** |

## VLMEK ChartQA-Pro (per-type prompts + ANLS scorer)

| Model | Overall |
|---|---:|
| zeroshot_qwen35_4b | 45.87% |
| v8_distill_v2 SFT (orig) | 43.85% |