# VLMEvalKit ChartQA-Pro comparison

_Re-evaluated with VLMEvalKit's per-question-type prompts + ANLS-based scorer (% strip, lenient text)._

| Model | Overall | Factoid | Multi Choice | Conversational | Fact Checking | Hypothetical |
|---|---:|---:|---:|---:|---:|---:|
| row_a_outcome | 53.26% | 51.44% | 63.08% | 48.71% | 61.89% | 44.83% |
| row_b_vapv | 49.84% | 47.32% | 56.54% | 46.40% | 62.30% | 42.93% |
| row4_dapo_vapv | 49.75% | 48.25% | 51.87% | 48.14% | 59.43% | 42.79% |
| row3_dapo_outcome | 48.21% | 46.26% | 50.93% | 46.79% | 57.79% | 44.44% |
| zeroshot_qwen35_4b | 45.87% | 43.67% | 52.34% | 45.60% | 52.60% | 40.19% |
| v9_sft | 45.50% | 42.99% | 49.53% | 46.92% | 53.28% | 40.42% |
| sft_4b | 43.85% | 41.33% | 47.66% | 45.27% | 53.28% | 35.32% |
| v_hq_lite_sft | 40.46% | 37.86% | 48.60% | 34.19% | 53.37% | 39.14% |