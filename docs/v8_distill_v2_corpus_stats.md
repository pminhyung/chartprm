# v8 SFT Distill v2 — Corpus Statistics

_Generated: 2026-05-10 18:20:10_

## Pipeline funnel

| Stage | Count | Note |
|---|---:|---|
| Input rows (`sft_v8_improved_v2`) | 28594 | source v8 corpus |
| Distillation kept | 28575 | Qwen3.5-397B teacher (1st pass + retry) |
| Distill 1st-pass failed | 721 | HTTP timeout (recoverable) |
| Distill retry failed | 19 | timeout > 1800 s — truly unrecoverable |
| Format-compliance build kept | 28575 | drop 0: missing `<answer>` etc. |
| **Quality-filter kept (SFT input)** | **27577** | drop 998: R1-R4 |

## Family distribution

| Family | After distill | After build | After filter | Δ filter (%) |
|---|---:|---:|---:|---:|
| chartmuseum | 12549 | 12549 | 11751 | 6.4% |
| chartqa | 9500 | 9500 | 9500 | 0.0% |
| charxiv | 6526 | 6526 | 6326 | 3.1% |

## Drop reasons

### Format-compliance build
(none)

### Quality filter (R1-R4)
| Reason | Count | Description |
|---|---:|---|
| `R4a_numeric_mismatch` | 610 | answer_type=numeric AND teacher mismatch (>5%) |
| `R4b_mc_value_form_mismatch` | 196 | value-form multichoice AND teacher wrong |
| `R2_cm_answer_body_long` | 143 | ChartMuseum `_clean_answer` > 200 chars |
| `R1_reasoning_too_short` | 47 | len(_teacher_reasoning) < 150 chars |
| `R3_cm_answer_body_multiline` | 2 | newline inside ChartMuseum answer body |

## `assistant_text` length distribution (approx tokens, char/4)

| Stage | n_sampled | p50 | p75 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| after build | 1000 | 714 | 3349 | 8100 | 10156 | 14859 | 18144 |
| after filter | 1000 | 635 | 3515 | 8426 | 11600 | 15978 | 20393 |

## Sample dropped rows (quality filter)

- **`R1_reasoning_too_short`** | `answer_type=numeric`, `mc=no`
  - gold: `141.45`
  - teacher pred: `...`
- **`R1_reasoning_too_short`** | `answer_type=`, `mc=no`
  - gold: `6.29`
  - teacher pred: `...`
- **`R2_cm_answer_body_long`** | `answer_type=comparison`, `mc=no`
  - gold: `Division A-B is the highest (81); it is 69 greater than the lowest (Division C-B`
  - teacher pred: `Division A-B has the highest value. It is visually significantly greater than th`
- **`R2_cm_answer_body_long`** | `answer_type=`, `mc=no`
  - gold: `0.5`
  - teacher pred: `The provided chart does not contain the necessary data to answer the question. I`
- **`R3_cm_answer_body_multiline`** | `answer_type=`, `mc=no`
  - gold: `1.29`
  - teacher pred: `Total growth: -30
Average annual increase: -2.14`
- **`R3_cm_answer_body_multiline`** | `answer_type=`, `mc=no`
  - gold: `0.09`
  - teacher pred: `Total decrease: 0.4
Average annual decrease: 0.1`
- **`R4a_numeric_mismatch`** | `answer_type=numeric`, `mc=no`
  - gold: `26.45`
  - teacher pred: `Cannot be determined from the chart (no numerical values provided)`
- **`R4a_numeric_mismatch`** | `answer_type=numeric`, `mc=no`
  - gold: `223.9`
  - teacher pred: `Cannot be determined from the chart (no numerical values are provided).`
- **`R4b_mc_value_form_mismatch`** | `answer_type=multichoice`, `mc=yes`
  - gold: `219.0`
  - teacher pred: `Division D`
- **`R4b_mc_value_form_mismatch`** | `answer_type=multichoice`, `mc=yes`
  - gold: `Gross Profit + Operating Exp + Tax + Other Income`
  - teacher pred: `32`
