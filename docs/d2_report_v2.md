# D2 Pilot Report v2 (Spec A+B+C patched)

## Setup
- Verifier: InternVL2.5-26B (cross-family, NOT_FOUND aware)
- Extractor: Qwen3.6-27B LLM-primary (strict entity validation)
- Tolerance: 10% rel / 1.0 abs → full; ≤25% rel → partial
- N samples: 99, outcome correct rate: 43.4%
- Mean NOT_FOUND rate per sample: 17.5%

## Alignment 4-metric

| Metric | Value | Target | Pass |
|---|---:|---|:---:|
| Perception ROC AUC vs outcome | 0.465 | ≥0.65 | FAIL |
| MC ROC AUC vs outcome         | 0.930 | ≥0.70 | PASS |
| r(perception, mc)             | -0.068         | 0.3-0.7 | FAIL |
| AUC lift (mc+p vs mc)         | +0.001     | ≥+0.02 | FAIL |

passed: 1/4

NOTE: 4-cell distribution이 본 실험의 main decision metric (alignment 통과 못 해도 진행).
