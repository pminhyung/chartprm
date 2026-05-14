# SFT + zero-shot Re-eval @ max_tokens=16384 (matches training) — FIXED

_All inference at max_tokens=16384, thinking-on, vLLM no max-model-len. Single Qwen3.6-27B judge on port 9101._

_chartmuseum_extract patched to use FULL content (was last-line, missed multi-line `<answer>\n...\n</answer>`)._

| Model | H | A | Pro (lmms) | Pro (VLMEK) | CXR | CMU | AVG5 (lmms-Pro) | AVG5 (VLMEK-Pro) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| zeroshot | 77.10 | 82.40 | 40.45 | 60.62 | 64.40 | 46.80 | 62.23 | 66.26 |
| v_hq_lite_sft | 85.28 | 91.04 | 33.37 | 47.92 | 49.70 | 37.30 | 59.34 | 62.25 |
| v9_sft | 85.28 | 90.96 | 33.52 | 55.89 | 51.90 | 33.40 | 59.01 | 63.49 |
| v8_distill_v2 | 81.20 | 87.36 | 36.96 | 57.74 | 60.90 | 43.80 | 62.04 | 66.20 |