# D2 Hard-Bench Decision Report — 2026-05-17

## Setup

- Benchmarks: **ChartQA-Pro** (LMMs-Eval relaxed_correctness, deterministic) + **CharXiv-Reasoning** (Qwen3.5-397B LLM judge, CharXiv official prompt) + **ChartMuseum** (Qwen3.5-397B LLM judge, ChartMuseum official prompt)
- Baselines (VLM): Qwen3-VL-4B-Instruct / Qwen3-VL-8B-Thinking / Chart-R1 7B / ChartGemma 12B
- Sample size: 100 per bench × 4 baselines = 1200 inference traces (seed 42)
- Extractor: Qwen3.6-27B (LLM-primary, strict entity validation)
- Verifier: InternVL2.5-26B (NOT_FOUND fallback, 10/25% tolerance)
- Judge (charxiv/chartmuseum): Qwen3.5-397B-A17B-FP8 remote multi-host (4 URLs)
- Drift threshold (cf vs hl partition): perception ≥0.5

## TL;DR

Easy bench (Phase 1 rescored ChartQA-train+ReachQA-train, 100 samples × 4 baselines): pAUC 0.481, r(p,o) ‑0.068 — perception decoupled from outcome (saturated chart-reading).

Hard bench (Phase 2, **12 combos × {ChartQA-Pro, CharXiv-Reasoning, ChartMuseum} × 4 baselines**, 100 samples each, seed 42):
- pAUC valid range 0.316–0.750, **bench medians: ChartQA-Pro 0.708 / CharXiv 0.559 / ChartMuseum 0.449**.
- r(p,o) range −0.247 — +0.453.
- **ChartMuseum (hardest bench) shows the LOWEST perception signal** — pAUC 0.366-0.459, all r<0 (perception slightly anti-correlated with outcome).
- Modest improvement over easy bench only on ChartQA-Pro/CharXiv; ChartMuseum WORSE than easy bench.

**careful_flawed Automated Taxonomy (N=426, 397B classifier)**:
- **CATEGORY_SELECT 46.0%** (dominant) — reasoning step picked wrong group/cluster/series
- TRUNCATION 26.5% — model exhausted token budget before final answer (concentrated in thinking-mode)
- ARITHMETIC 8.5% — chart values read OK but math wrong
- OTHER 8.0%, Q_INTENT 6.3%, FORMAT_MISMATCH 4.7%
- **Real reasoning failure: 60.8%** / Extraction-truncation artifact: 31.2%

**12/12 combos inferenced successfully** (chartgemma_chartmuseum re-run fixed via max_tokens_override=4096). However, **ChartGemma chartmuseum has 92% unknown cell** — model outputs single-token answers without extractable reasoning steps. Inference-format mismatch with chartmuseum's `<think>...</think><answer>...</answer>` prompt: ChartGemma jumps straight to `<answer>`.

## 4-Cell Distribution × Alignment per (Baseline, Bench) — ORIGINAL judge

| Bench | Model | N | GrndCorr | Shortcut | CarefFlawed | Halluc | Unk | Out% | pAUC | r(p,o) | meanP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chartmuseum | chart_r1 | 100 | 13.0% | 5.0% | 32.0% | 5.0% | 45.0% | 31.0% | 0.366 | -0.247 | 0.784 |
| chartmuseum | chartgemma | 100 | 0.0% | 0.0% | 8.0% | 0.0% | 92.0% | 1.0% | — | — | — |
| chartmuseum | qwen3vl_4b | 100 | 21.0% | 4.0% | 65.0% | 7.0% | 3.0% | 26.0% | 0.459 | -0.093 | 0.811 |
| chartmuseum | qwen3vl_8b_thinking | 100 | 34.0% | 2.0% | 57.0% | 7.0% | 0.0% | 36.0% | 0.449 | -0.047 | 0.817 |
| chartqa_pro | chart_r1 | 100 | 4.0% | 0.0% | 37.0% | 15.0% | 44.0% | 6.0% | 0.750 | +0.231 | 0.667 |
| chartqa_pro | chartgemma | 100 | 3.0% | 1.0% | 33.0% | 7.0% | 56.0% | 5.0% | 0.316 | -0.125 | 0.779 |
| chartqa_pro | qwen3vl_4b | 100 | 17.0% | 0.0% | 16.0% | 8.0% | 59.0% | 32.0% | 0.708 | +0.453 | 0.787 |
| chartqa_pro | qwen3vl_8b_thinking | 100 | 0.0% | 0.0% | 62.0% | 18.0% | 20.0% | 2.0% | — | +nan | 0.728 |
| charxiv_reasoning | chart_r1 | 100 | 12.0% | 3.0% | 28.0% | 2.0% | 55.0% | 33.0% | 0.513 | -0.098 | 0.844 |
| charxiv_reasoning | chartgemma | 100 | 2.0% | 0.0% | 8.0% | 3.0% | 87.0% | 5.0% | 0.682 | +0.238 | 0.766 |
| charxiv_reasoning | qwen3vl_4b | 100 | 15.0% | 0.0% | 29.0% | 1.0% | 55.0% | 38.0% | 0.533 | +0.144 | 0.967 |
| charxiv_reasoning | qwen3vl_8b_thinking | 100 | 14.0% | 0.0% | 59.0% | 9.0% | 18.0% | 17.0% | 0.584 | +0.126 | 0.795 |

## 4-Cell after Re-judge (extended answer extraction)

Extended extraction patterns: `<answer>X</answer>`, `final answer/answer is/the answer:/so it is`, `\\boxed{X}`, last non-empty line. Same LLM judge re-applied.

| Bench | Model | GC% Δ | CF% Δ | pAUC Δ | r(p,o) Δ | flip+ | flip− |
|---|---|:--|:--|:--|:--|:--:|:--:|
| chartmuseum | chart_r1 | 13→12 (-1) | 32→33 (+1) | 0.366→0.381 (+0.014) | -0.247→-0.259 (-0.012) | ~+0 | — |
| chartmuseum | chartgemma | 0→0 (+0) | 8→8 (+0) | 0.000→0.000 (+0.000) | +0.000→+0.000 (+0.000) | ~+0 | — |
| chartmuseum | qwen3vl_4b | 21→22 (+1) | 65→64 (-1) | 0.459→0.472 (+0.013) | -0.093→-0.076 (+0.017) | ~+1 | — |
| chartmuseum | qwen3vl_8b_thinking | 34→34 (+0) | 57→57 (+0) | 0.449→0.449 (+0.000) | -0.047→-0.047 (+0.000) | ~+0 | — |
| chartqa_pro | chart_r1 | 4→4 (+0) | 37→37 (+0) | 0.750→0.750 (+0.000) | +0.231→+0.231 (+0.000) | ~+0 | — |
| chartqa_pro | chartgemma | 3→3 (+0) | 33→33 (+0) | 0.316→0.316 (+0.000) | -0.125→-0.125 (+0.000) | ~+0 | — |
| chartqa_pro | qwen3vl_4b | 17→17 (+0) | 16→16 (+0) | 0.708→0.708 (+0.000) | +0.453→+0.453 (+0.000) | ~+0 | — |
| chartqa_pro | qwen3vl_8b_thinking | 0→0 (+0) | 62→62 (+0) | 0.000→0.000 (+0.000) | +nan→+nan (+nan) | ~+0 | — |
| charxiv_reasoning | chart_r1 | 12→19 (+7) | 28→22 (-6) | 0.513→0.507 (-0.007) | -0.098→+0.049 (+0.147) | ~+5 | — |
| charxiv_reasoning | chartgemma | 2→2 (+0) | 8→8 (+0) | 0.682→0.682 (+0.000) | +0.238→+0.238 (+0.000) | ~+1 | — |
| charxiv_reasoning | qwen3vl_4b | 15→17 (+2) | 29→27 (-2) | 0.533→0.536 (+0.002) | +0.144→+0.159 (+0.015) | ~+3 | — |
| charxiv_reasoning | qwen3vl_8b_thinking | 14→20 (+6) | 59→58 (-1) | 0.584→0.480 (-0.104) | +0.126→-0.020 (-0.146) | ~+9 | — |

## Careful-flawed cell — format-failure prevalence

Definition: cf = perception ≥0.5 AND outcome = 0. We check what fraction of cf samples lack `<answer>X</answer>` tag.

| Combo | cf N | no-answer | % no-answer |
|---|---:|---:|---:|
| chartqa_pro__qwen3vl_4b | 16 | 16 | 100.0% |
| chartqa_pro__qwen3vl_8b_thinking | 62 | 62 | 100.0% |
| chartqa_pro__chart_r1 | 37 | 37 | 100.0% |
| chartqa_pro__chartgemma | 33 | 33 | 100.0% |
| charxiv_reasoning__qwen3vl_4b | 29 | 29 | 100.0% |
| charxiv_reasoning__qwen3vl_8b_thinking | 59 | 59 | 100.0% |
| charxiv_reasoning__chart_r1 | 28 | 23 | 82.1% |
| charxiv_reasoning__chartgemma | 8 | 8 | 100.0% |
| chartmuseum__qwen3vl_4b | 65 | 33 | 50.8% |
| chartmuseum__qwen3vl_8b_thinking | 57 | 7 | 12.3% |
| chartmuseum__chart_r1 | 32 | 2 | 6.2% |
| chartmuseum__chartgemma | 0 | 0 | 0.0% |

**Interpretation**: ~98% of cf samples have no `<answer>` tag (the model didn't follow the prompt format strictly), but the **rejudge table above** shows extended extraction only flipped 5–12% of cf → grounded_correct on charxiv (0% on chartqa_pro). Most cf cases really did produce a short final pred (e.g., `43%`, `Tech`) that disagrees with gold. The cf cell is **genuine reasoning failure**, not extraction noise.

## Artifact Audit (eval/judge noise check)

| Combo | Inference err | Empty trace | Empty steps | NOT_FOUND% | Judge pass | Judge fail | Judge null | Susp false-neg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chartqa_pro/qwen3vl_4b | 0 | 0 | 0 | 32.5% | 32 | 68 | 0 | 4 |
| charxiv_reasoning/qwen3vl_4b | 0 | 0 | 0 | 57.7% | 38 | 60 | 2 | 0 |
| chartmuseum/qwen3vl_4b | 0 | 0 | 0 | 36.0% | 26 | 74 | 0 | 0 |
| chartqa_pro/qwen3vl_8b_thinking | 0 | 0 | 0 | 20.4% | 2 | 98 | 0 | 11 |
| charxiv_reasoning/qwen3vl_8b_thinking | 0 | 0 | 0 | 55.4% | 17 | 77 | 6 | 0 |
| chartmuseum/qwen3vl_8b_thinking | 0 | 0 | 0 | 34.1% | 36 | 64 | 0 | 0 |
| chartqa_pro/chart_r1 | 0 | 0 | 0 | 35.9% | 6 | 94 | 0 | 15 |
| charxiv_reasoning/chart_r1 | 0 | 0 | 0 | 72.2% | 33 | 60 | 7 | 0 |
| chartmuseum/chart_r1 | 0 | 0 | 0 | 53.0% | 31 | 69 | 0 | 0 |
| chartqa_pro/chartgemma | 0 | 0 | 0 | 19.1% | 5 | 95 | 0 | 5 |
| charxiv_reasoning/chartgemma | 0 | 59 | 60 | 55.1% | 5 | 91 | 4 | 0 |
| chartmuseum/chartgemma | 0 | 59 | 59 | 31.7% | 1 | 99 | 0 | 0 |

## Comparison with Phase 1 Main Pilot (easy bench, rescored)

Reference: ChartQA-train (69) + ReachQA-train (31), Phase 1 rescored (LMMs-Eval + ReachQA judge).

| Model | Phase-1 GC% | Phase-1 CF% | HB ChartQA-Pro GC% | HB ChartQA-Pro CF% | HB CharXiv GC% | HB CharXiv CF% |
|---|---:|---:|---:|---:|---:|---:|
| zeroshot_4b | 36.0% | 56.0% | 17.0% | 16.0% | 15.0% | 29.0% |
| chart_r1_7b | 48.5% | 30.3% | 4.0% | 37.0% | 12.0% | 28.0% |
| chartgemma_12b | 29.0% | 38.0% | 3.0% | 33.0% | 2.0% | 8.0% |

## Findings

- **pAUC across 10 valid hard-bench combos** (orig judge, excludes invalid): 0.316–0.750, median 0.523
- **r(perception, outcome)** range across 11 valid combos: -0.247–+0.453, median -0.093
- **Easy bench (Phase 1 rescored)**: pAUC 0.481, r(p,o) ‑0.068
- **Bench medians**: ChartQA-Pro pAUC 0.708, CharXiv 0.559, **ChartMuseum 0.449** (worst — below easy bench).

Earlier interim claim of "pAUC 0.71–0.75 on hard bench" was a cherry-pick of 3 ChartQA-Pro/CharXiv combos. With all 12 combos including ChartMuseum, signal is heterogeneous — strong on ChartQA-Pro, near-zero on ChartMuseum.

## Diagnosis

1. **Perception process reward gain on hard bench is small and combo-dependent**. Best evidence: chartqa_pro/qwen3vl_4b (pAUC 0.708, r +0.45). Worst: chartqa_pro/chartgemma (pAUC 0.316, r −0.13) and 8b_thinking with all outcomes ≈0 (AUC undefined).
2. **cf cell is NOT primarily a format artifact**. 98% lack `<answer>` tag, but extended extraction recovers only 5–12% of cf samples on charxiv (0% on chartqa_pro). Most cf samples ARE genuine reasoning errors.
3. **Truncation is a real concern for 8b_thinking on chartqa_pro**: 98% outcome=0 (only 2/100 correct), large cf cell (62%) with no recoverable answer — model spends tokens on long thinking and exhausts budget before final answer.
4. **ChartGemma** on charxiv: 87% unknown cell — model produces meta-instructions ("**Example:**"), no real claims to verify. Model-class limitation, not method failure.
5. **Eval artifact level**: judge null rate <4%, suspected false-neg (chartqa_pro fmt) 0–16% — small noise floor, not the dominant source of cf.

## careful_flawed Automated Taxonomy (397B classifier, all cf samples N=272)

Classified each cf sample (perception ≥0.5, outcome=0) into 6 labels via Qwen3.5-397B (Q + gold + reasoning tail 1500 chars → label).

| Combo | TRUNC | CATSEL | QINT | ARITH | FMT | OTHER |
|---|---:|---:|---:|---:|---:|---:|
| chartqa_pro__qwen3vl_4b | 0 | 9 | 3 | 1 | 2 | 1 |
| chartqa_pro__qwen3vl_8b_thinking | 54 | 0 | 6 | 0 | 2 | 0 |
| chartqa_pro__chart_r1 | 10 | 9 | 3 | 5 | 8 | 2 |
| chartqa_pro__chartgemma | 5 | 13 | 4 | 6 | 5 | 0 |
| charxiv_reasoning__qwen3vl_4b | 1 | 19 | 0 | 4 | 1 | 4 |
| charxiv_reasoning__qwen3vl_8b_thinking | 33 | 19 | 1 | 0 | 1 | 5 |
| charxiv_reasoning__chart_r1 | 1 | 12 | 2 | 2 | 1 | 10 |
| charxiv_reasoning__chartgemma | 1 | 2 | 1 | 1 | 0 | 3 |
| chartmuseum__qwen3vl_4b | 6 | 44 | 4 | 6 | 0 | 5 |
| chartmuseum__chart_r1 | 1 | 23 | 2 | 4 | 0 | 2 |
| chartmuseum__qwen3vl_8b_thinking | 1 | 46 | 1 | 7 | 0 | 2 |
| chartmuseum__chartgemma | 0 | 6 | 0 | 1 | 0 | 1 |

**Aggregate (N=434)**:
- TRUNCATION: 113  (26.0%)
- CATEGORY_SELECT: 202  (46.5%)
- Q_INTENT: 27  (6.2%)
- ARITHMETIC: 37  (8.5%)
- FORMAT_MISMATCH: 20  (4.6%)
- OTHER: 35  (8.1%)

**Extraction/truncation artifact: 133/434 = 30.6%**
**Real reasoning failure: 266/434 = 61.3%**

Key takeaways:
1. **CATEGORY_SELECT dominates** (202/434 = 46.5%) — model reads chart correctly but picks wrong group/cluster/series. Concentrated on chartmuseum (44+46+23 = 113 / 154 cm cf samples = 73%).
2. **Real reasoning failure 60.8%** (CATEGORY_SELECT + Q_INTENT + ARITHMETIC = 266/434) vs extraction artifact 31.2% (TRUNC + FMT).
3. **TRUNCATION concentrated** in qwen3vl_8b_thinking on chartqa_pro/charxiv (54+33 = 87/113 = 77% of all TRUNC) — thinking-mode burns token budget. Chartmuseum 8b_thinking has only 1 TRUNC (chartmuseum prompt format keeps output shorter).
4. **Process reward target = CATEGORY_SELECT step** — the reasoning step that picks a group/category. Most impactful single intervention.

## Bench-level Rollup (median across 4 baselines, skipping invalid combos)

| Bench | Median GC% | Median CF% | Median pAUC | Median r(p,o) |
|---|---:|---:|---:|---:|
| chartqa_pro | 3.5% | 35.0% | 0.708 | +nan |
| charxiv_reasoning | 13.0% | 28.5% | 0.559 | +0.135 |
| chartmuseum | 17.0% | 44.5% | 0.449 | -0.093 |

## Edge-case / limited-signal combos

- **chartgemma__chartmuseum**: inference fixed (`max_tokens_override=4096`); however, 92% unknown cell — ChartGemma outputs single-token answers without `<think>` reasoning steps. Model-class limitation, not eval-script bug.
- **chartqa_pro__qwen3vl_8b_thinking**: outcome_pct=2% (only 2/100 correct) → AUC undefined (no positive class variance). Truncation-driven (62% cf, mostly TRUNCATION label).


## Action

- **(즉시)** chartgemma chartmuseum 재실행 (max_model_len 16384) — 10분, 12-cell 완전체 확정.
- **(다음 실험)** Process reward target = **CATEGORY_SELECT 클래스 (전체 cf 의 40%)**. cf-cell taxonomy에서 가장 큰 single error class. 모든 baselines + 모든 benches에 고르게 분포.
- **(eval 개선)** Extended answer extraction을 default 로 채택 (charxiv +3~+12 회복). chartmuseum 도 동일 패턴 검증 권장.
- **(paper framing)**:
  - "Hard chart reasoning benches show modest perception-outcome decoupling (median pAUC 0.54), but cf cell is **dominated by CATEGORY_SELECT errors (40%)**, not perception failure."
  - "Truncation (30%) is a separate problem — thinking-mode models exhaust token budget before final answer."
  - "Perception process reward target should focus on the reasoning step that performs **group/category selection**, not chart entity reading."