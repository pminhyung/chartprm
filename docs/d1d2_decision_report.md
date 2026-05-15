# D1+D2 Decision Report — 2026-05-15 (open-weights, Spec A+B+C+D patched)

## Setup

- Verifier: **InternVL2.5-26B** (cross-family, NOT_FOUND fallback, 10%/25% tolerance)
- Extractor: **Qwen3.6-27B** (LLM-primary with strict entity validation)
- Baselines: zero-shot Qwen3.5-VL-4B (archived), Chart-R1 7B, ChartGemma 12B
- Sample pool: 100 D2 pilot samples (ChartQA+ReachQA train, pHash-clean)

## Day 1.A — Verifier reliability (CSV-based)

- N queries: 197
- OK (parsed): 192 (97.5%)
- NOT_FOUND : 5 (2.5%)
- Agreement (on parsed): **90.1%**
- Criteria: coverage>=80 AND agreement>=85 (10% rel or 1.0 abs)
- **VERDICT: PASS**

## Day 1.B — Patched pipeline alignment (4-metric vs outcome)

- N samples: 99, outcome correct rate: 43.4%
- Mean NOT_FOUND rate per sample: 17.5%

| Metric | Value | Target | Pass |
|---|---:|---|:---:|
| Perception ROC AUC vs outcome | 0.465 | ≥0.65 | FAIL |
| MC ROC AUC vs outcome         | 0.930 | ≥0.70 | PASS |
| r(perception, mc)             | -0.068 | 0.3-0.7 | FAIL |
| AUC lift (mc+p vs mc)         | +0.001 | ≥+0.02 | FAIL |

- **passed: 1/4**

- NOTE: 4-cell distribution이 main decision metric. 통과 못 해도 Day 2로 진행 (decoupling 자체가 thesis).

## Day 2 — 4-cell distribution (drift × outcome)

_Drift threshold (perception_agg ≥ X → low drift): **0.5**_

| Model | N | Grounded+Correct | Shortcut | CarefulFlawed | Hallucinated | Unknown | Outcome% | DriftWithinCorrect% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| zeroshot_4b | 100 | 38.0% | 3.0% | 54.0% | 5.0% | 0.0% | 41.0% | 7.3% |
| chart_r1_7b | 100 | 37.0% | 5.0% | 41.0% | 3.0% | 14.0% | 42.0% | 11.9% |
| chartgemma_12b | 100 | 31.0% | 5.0% | 36.0% | 3.0% | 25.0% | 36.0% | 13.9% |

### Pattern detected: **B  (drift not universal → MC-only Math-Shepherd pivot)**

## Decision

**MC-only pivot** — perception term 폐기 또는 w_p=0.1 minor weight. Main contribution: Math-Shepherd chart-domain application.

## Reproducibility

**Data**:
- `data/d2_pilot/claims_v2.jsonl` — main 100 sample LLM-primary claims
- `data/d2_pilot/perception_v2.jsonl` — InternVL verifier scores
- `data/d2_pilot/alignment_v2.json` — 4-metric alignment
- `data/d2_pilot/baseline_{zeroshot_4b,chart_r1,chartgemma}_perception.jsonl` — per-baseline perception
- `data/d2_pilot/4cell_distribution.json` — Pattern A/B/C source

**Scripts**:
- `scripts/d2_claim_extract_v2.py` (LLM-primary, multi-host async, resume)
- `scripts/d2_perception_verify_v2.py` (NOT_FOUND fallback, multi-host)
- `scripts/d2_alignment_v2.py` / `scripts/d2_4cell_distribution.py`
- `scripts/d1_verifier_reliability_csv.py` (CSV-based reliability)

**vLLM servers** (10 GPU: 2-11):
- 8100/8101 InternVL2.5-26B verifier (TP=2 × 2 hosts)
- 8200 Qwen3.6-27B extractor (TP=4)
- 8300 Chart-R1 7B, 8301 ChartGemma 12B (TP=1 each)
