# v8_distill_v2 SFT vs Zero-shot — ChartQA-Pro (VLMEK) Failure Analysis

**Inputs (read-only, main repo)**
- SFT: `/ex_disk2/mhpark/poc/chartvr/results/v8_distill_v2/sft_4b_vlmek/chartqa_pro_vlmek.jsonl` (1948 rows)
- Zero-shot: `/ex_disk2/mhpark/poc/chartvr/results/v8_distill_v2/zeroshot_qwen35_4b_vlmek/chartqa_pro_vlmek.jsonl` (1948 rows)
- Gold + Year flags: `data/chartqa_pro/data/test-00000-of-00001.parquet`

**Scorer**: `third_party/VLMEvalKit/vlmeval/dataset/utils/chartqapro.py::relaxed_correctness_chartqapro`
- Numeric within 5% relative tolerance; text via ANLS ≥ 0.5; FC/MC strict exact match.
- Prediction extraction = `score_standard.extract_post_think` (last non-empty line of `content`; fall back to `reasoning_content` tail when `content` is empty).
- Conversational gold = `Answer[0]` (matches published per-type 45.27% / 45.60%, vs `Answer[-1]` which collapses to ~1%).

**Reproduction sanity**
| | Recomputed | Published (scored.json) |
|---|---:|---:|
| SFT Overall | 43.56% | 43.85% |
| Zero Overall | 45.21% | 45.87% |

Per-type recomputed matches published within 0.5pp for Factoid, Hypothetical, Fact Checking, Multi Choice, Conversational. The ~0.3–0.7pp aggregate gap is consistent with Year=YES handling on ~53 Factoid samples — does not affect regression/gain decomposition.

---

## 1. Funnel

| Bucket | Count | % of 1948 |
|---|---:|---:|
| Regression (zero=1 & sft=0) | 160 | 8.21% |
| Gain (sft=1 & zero=0) | 125 | 6.42% |
| Both correct | 736 | 37.78% |
| Both wrong | 927 | 47.59% |

Net cell-flip = 125 − 160 = −35 samples → −1.80pp on 1948, aligned with measured Overall Δ = −2.02pp (small remaining gap = ANLS partial-credit on Conversational multi-token answers).

---

## 2. Per-type regression vs gain

| Question Type | n | SFT Δpp | Regression / rate | Gain / rate | Net cases |
|---|---:|---:|---:|---:|---:|
| Factoid | 1081 | −1.89 | 84 (7.77%) | 60 (5.55%) | −24 |
| Multi Choice | 214 | −4.67 | 21 (9.81%) | 11 (5.14%) | −10 |
| Hypothetical | 98 | −4.87 | 10 (10.20%) | 6 (6.12%) | −4 |
| Conversational | 311 | −0.00 | 22 (7.07%) | 22 (7.07%) | 0 |
| Fact Checking | 244 | +1.23 | 23 (9.43%) | 26 (10.66%) | +3 |

Net loss concentrated in Multi Choice (−10 cases) and Factoid (−24). Fact Checking and Conversational are roughly neutral.

---

## 3. Generation-budget evidence (root cause)

| | SFT | Zero-shot | Δ |
|---|---:|---:|---:|
| `finish_reason=length` rate | 39.68% (773/1948) | 35.11% (684/1948) | **+4.57pp** |
| Mean `reasoning_content` chars | 6,691 | 6,289 | +402 (+6.4%) |
| Mean `content` chars | 5 | 6 | ≈0 |

Within the **160 regression cases**, finish_reason distribution:
- **SFT length & zero stop: 134 / 160 = 83.75%** (SFT ran out of budget while zero-shot finished cleanly)
- SFT stop & zero stop: 26 / 160 = 16.25%
- (No cases where both truncated and SFT wins/loses — both-truncated mostly bucket into both-wrong.)

Mirror on the **125 gain cases**: zero length & sft stop = **91 / 125 = 72.8%**. The gain bucket is mostly zero-shot losing its own answer to truncation while SFT happens to commit faster.

Conclusion: most of the SFT↔zero flips on either side are driven by which model happened to fit inside the 8K-token budget on a given sample, not by a difference in chart-reading ability.

---

## 4. SFT failure-mode breakdown (160 regression cases)

| Failure type | Count | % of regression | Dominant qtypes |
|---|---:|---:|---|
| truncated_midthink | 123 | 76.88% | Factoid 66, FC 19, MC 17, Conv 12, Hyp 9 |
| wrong (clean stop, semantic error) | 34 | 21.25% | Factoid 18, Conv 8, MC 4, FC 3, Hyp 1 |
| verbose (correct fact, extra prose blocks ANLS / list-match) | 2 | 1.25% | Conv 2 |
| format_fc | 1 | 0.62% | FC 1 |
| format_mc / unit_added / decimal_precision | 0 | 0% | — |

Definitions
- **truncated_midthink**: `finish_reason=length` and the post-think extraction surfaced a mid-thought sentence fragment (bullet markers, hedge phrases, sentence cut mid-word, or unusually long/non-answer-shape string). The SFT model was still inside `<think>` when the 8K cap fired, so the final answer line was never emitted.
- **wrong**: model committed an answer cleanly (`finish_reason=stop` or a clean-shape short answer despite `length`) but the value/letter/binary is incorrect.
- **format_fc/format_mc**: would-be correct semantics but the SFT output is not in the required exact-match form (FC binary `true/false`, MC single letter `a–d`).
- **verbose**: correct content embedded in extra prose that breaks numeric parsing or list comparison (e.g., gold `'29.1%'`, pred `'Difference: 29.1.'`).

---

## 5. Examples (per failure type)

### 5.1 truncated_midthink (3 of 123)
| sample_id | qtype | gold | SFT pred (`finish=length`) | zero pred |
|---|---|---|---|---|
| chartqa_pro_13 | Factoid | `Police` | `If the question is strict, "between" usually means >40 and <60.` | `Police` |
| chartqa_pro_36 | Factoid | `1991` | `*   The label "19` | `1991` |
| chartqa_pro_37 | Factoid | `200` | `If the question` | `200` |

```
chartqa_pro_1705  Multi Choice  gold='B'  sft_finish=length
  pred_sft: '*   d) 43 vs'    pred_zero: 'b'
chartqa_pro_1725  Multi Choice  gold='D'  sft_finish=length
  pred_sft: 'The 2010 point is halfway between 0 and 10.'   pred_zero: 'd'
chartqa_pro_1498  Fact Checking gold='False' sft_finish=length
  pred_sft: "Let's look at the"   pred_zero: 'False'
```

### 5.2 wrong (3 of 34)
| sample_id | qtype | gold | SFT pred | zero pred |
|---|---|---|---|---|
| chartqa_pro_9 | Factoid | `2013` | `2011` (stop) | `2013` |
| chartqa_pro_220 | Factoid | `Unanswerable` | `14` (stop) | `unanswerable` |
| chartqa_pro_362 | Factoid | `832B` | `563B` (stop) | `$832B` |

### 5.3 verbose (2 of 2)
| sample_id | qtype | gold | SFT pred | zero pred |
|---|---|---|---|---|
| chartqa_pro_1330 | Conversational | `GDP` | `GDP, 2012 forecast, % change on previous year` | `GDP` |
| chartqa_pro_1334 | Conversational | `[God, Angels or Benevolent Spirits, Heaven]` | `[God, Angels or benevolent spirits, Heaven]` | `['God', 'Angels or benevolent spirits', 'Heaven']` |

### 5.4 format_fc (1 of 1)
| sample_id | qtype | gold | SFT pred | zero pred |
|---|---|---|---|---|
| chartqa_pro_1647 | Fact Checking | `False` | `2. China (11)` (length) | `False` |

---

## 6. Gain-side mirror (125 cases)

| Failure side (zero-shot view) | Count | Share |
|---|---:|---:|
| zero finish_reason=length | 91 | 72.8% |
| zero finish_reason=stop (genuine SFT improvement) | 34 | 27.2% |

Gain qtype distribution: Factoid 60, Fact Checking 26, Conversational 22, Multi Choice 11, Hypothetical 6.

Reading: of the 125 cases SFT wins, only ~34 are flips where zero-shot finished cleanly but answered wrong — those represent any "real" reading-ability gain from distillation. The other 91 are zero-shot-only truncations the SFT happened to escape.

---

## 7. Conclusion

**Mechanism of −2.02pp Overall regression**

1. **Reasoning-length inflation under fixed 8K budget.** SFT fine-tunes on 397B teacher distillation traces; the resulting student emits longer thinking traces (avg `reasoning_content` 6,691 vs 6,289 chars; `finish_reason=length` 39.68% vs 35.11%). On 8.21% of samples the SFT model hits the 8K cap mid-think while zero-shot finishes cleanly — these contribute 134 of the 160 regression cases (83.75%).

2. **Mid-think-extracted predictions are noise.** `score_standard.extract_post_think` rescues the empty-`content` cases by taking the last `reasoning_content` line, but the rescued strings are hedge phrases or sentence fragments (`"If the question is strict, …"`, `"*   The label \"19"`, `"Let's look at the"`). These score 0 under ANLS / numeric / exact-match. Within regression, 76.88% (123/160) are classified as `truncated_midthink`.

3. **Genuine semantic regression is small (≤21%).** Only 34/160 = 21.25% of regression cases are clean-stop wrong answers (gold 2013 → pred 2011, gold 832B → pred 563B, gold 2013 → pred 2011, gold Unanswerable → pred 14). The SFT model has not measurably *lost* chart-reading ability; it has lost answer-emission throughput.

4. **Mirror on gain side.** 91/125 = 72.8% of SFT gains are zero-shot self-truncations that SFT escaped. Only ~34 cases represent real distilled-reasoning improvements on samples both models finish. Net "real ability" delta ≈ +34 (SFT real gains) − ~26 (SFT clean-stop semantic regressions) = roughly neutral.

5. **Per-type pattern is consistent with (1).** Multi Choice (−4.67pp) and Hypothetical (−4.87pp) regress most — both demand longer multi-step reasoning where the SFT-style verbose trace is most likely to overrun 8K. Fact Checking (+1.23pp) is the only positive Δ: it is the shortest-answer category and the 8K budget rarely matters, so distillation gains aren't eaten by truncation.

6. **Format/unit issues are not the cause.** Format violations (MC letter, FC binary), unit suffixes, decimal precision, and verbose-but-correct phrasings together account for ≤2% of regression cases. The VLMEK prompt's "single letter / true|false / no units" instructions are followed by both models. The reported `45.27%` Conversational figure is reproducible only when scoring against `Answer[0]` (first turn) rather than `Answer[-1]` — neither model is materially better or worse at multi-turn conversational answering.

**Quantitative attribution of the −2.02pp gap**
- Truncation-driven regression: 123/1948 × −1.0 ≈ −6.31pp gross
- Truncation-driven gain (mirror): 91/1948 × +1.0 ≈ +4.67pp gross
- Net truncation contribution: ≈ −1.64pp (≈ 81% of observed −2.02pp)
- Genuine semantic regression net: 34/1948 − 34/1948 ≈ 0pp (offsetting)
- Format / verbose / other: ~−0.2pp

The −2.02pp Overall delta is overwhelmingly an artifact of an asymmetric mid-think truncation rate at the 8K decoding budget, induced by SFT learning a longer-reasoning style from the 397B teacher.

---

## Artifacts

- Analysis script: `analyze_sft_vs_zeroshot.py`
- Per-case dumps: `sft_vs_zeroshot_regression_cases.jsonl` (160), `sft_vs_zeroshot_gain_cases.jsonl` (125)
- Aggregated metrics: `sft_vs_zeroshot_summary.json`
