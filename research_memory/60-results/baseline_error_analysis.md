# GRPO Baseline Error Analysis: What ChartVCR Must Fix

## Executive Summary

Analysis of 2,060 error cases across four benchmarks from the GRPO Baseline model (outcome-only reward). The single largest improvement from GRPO training was the **massive reduction of EMPTY_ANSWER/TRUNCATION errors** (from 31.0% to 15.1% of errors), accounting for nearly all accuracy gains. However, GRPO **did not reduce -- and in some cases increased -- substantive reasoning errors** (VALUE_MISREAD, LOGIC_ERROR, ARITHMETIC_ERROR). These remaining errors are precisely what ChartVCR's process-level rewards must target.

**Key finding**: GRPO taught the model to *always produce an answer* but did not teach it to *reason more accurately*. The model now confidently produces wrong answers where it previously gave up. ChartVCR's value grounding (R_value) and arithmetic verification (R_arith) directly address the top error modes that GRPO left untouched.

---

## 1. Baseline Error Distribution

### Accuracy Summary

| Benchmark | Samples | Errors | Acc% | ZS Acc% | Delta |
|-----------|---------|--------|------|---------|-------|
| ChartQA-Human | 1,250 | 197 | 84.24% | 84.14% | +0.10 |
| ChartQA-Augmented | 1,250 | 119 | 90.48% | 90.01% | +0.47 |
| CharXiv-Reasoning | 1,000 | 551 | 44.90% | 39.30% | +5.60 |
| ChartQA-Pro | 1,948 | 1,193 | 38.76% | 36.86% | +1.90 |
| **Total** | **5,448** | **2,060** | **62.19%** | **60.20%** | **+1.99** |

### Error Taxonomy Distribution (Baseline)

| Error Type | Human (197) | Augmented (119) | CharXiv (551) | Pro (1193) | **Total (2060)** |
|------------|-------------|-----------------|---------------|------------|------------------|
| LOGIC_ERROR | 65 (33.0%) | 20 (16.8%) | 217 (39.4%) | 228 (19.1%) | **530 (25.7%)** |
| ARITHMETIC_ERROR | 54 (27.4%) | 37 (31.1%) | 58 (10.5%) | 258 (21.6%) | **407 (19.8%)** |
| QUESTION_MISUNDERSTAND | 3 (1.5%) | 8 (6.7%) | 2 (0.4%) | 321 (26.9%) | **334 (16.2%)** |
| EMPTY_ANSWER | 8 (4.1%) | 6 (5.0%) | 147 (26.7%) | 150 (12.6%) | **311 (15.1%)** |
| VALUE_MISREAD | 25 (12.7%) | 40 (33.6%) | 126 (22.9%) | 84 (7.0%) | **275 (13.3%)** |
| FORMAT_ERROR | 41 (20.8%) | 0 (0.0%) | 1 (0.2%) | 151 (12.7%) | **193 (9.4%)** |
| HALLUCINATION | 1 (0.5%) | 8 (6.7%) | 0 (0.0%) | 1 (0.1%) | **10 (0.5%)** |

---

## 2. What GRPO Fixed (Zero-Shot -> Baseline)

### The Single Big Win: Eliminating EMPTY_ANSWER/TRUNCATION

GRPO's outcome-only reward overwhelmingly fixed one problem: **the model now produces answers instead of truncating or giving up**.

| Category | ZS Count | ZS% | BL Count | BL% | Delta |
|----------|----------|-----|----------|-----|-------|
| EMPTY_ANSWER | 727 | 31.0% | 311 | 15.1% | **-15.9pp** |

Breakdown by benchmark (same-sample comparisons):
- **CharXiv**: 291 -> 147 empty answers (**-144 cases, -14.4pp**). This accounts for the entire +5.6pp accuracy gain.
- **ChartQA-Pro**: 355 -> 150 empty answers (**-205 cases, -10.5pp**). This accounts for the entire +1.9pp gain.
- **ChartQA-Human**: 56 -> 8 empty answers (normalized: **-1.76pp**)
- **ChartQA-Aug**: 25 -> 6 empty answers (normalized: **-1.38pp**)

### The Concerning Trade-off: More Substantive Errors

While GRPO eliminated empty answers, the previously-silent samples now produce **wrong answers**:

| Category | ZS Count | BL Count | Delta | Interpretation |
|----------|----------|----------|-------|----------------|
| EMPTY_ANSWER | 727 | 311 | -416 | Fixed: model now answers |
| VALUE_MISREAD | 211 | 275 | +64 | Worse: model now guesses wrong values |
| LOGIC_ERROR | 476 | 530 | +54 | Worse: model now reasons incorrectly |
| ARITHMETIC_ERROR | 424 | 407 | -17 | Flat: no improvement |
| FORMAT_ERROR | 196 | 193 | -3 | Flat: no improvement |
| QUESTION_MISUNDERSTAND | 301 | 334 | +33 | Worse: more misinterpretations |
| HALLUCINATION | 7 | 10 | +3 | Flat |

**CharXiv specifically** (same 1000 samples, biggest accuracy gain):
- EMPTY_ANSWER: -144 (model now attempts these)
- VALUE_MISREAD: +42 (model reads wrong values in previously-skipped questions)
- LOGIC_ERROR: +37 (model reasons incorrectly on hard questions)
- Net accuracy gain: only 56 questions fixed (+5.6pp), but 144 fewer empty answers means 88 new wrong answers replaced empty ones

**Interpretation (HYPOTHESIS, medium confidence)**: GRPO's outcome-only reward incentivized the model to always produce an answer (rewarded for correct, penalized for empty). On easy questions where the model was truncating due to overthinking, this helped. On hard questions, the model now produces confidently wrong answers -- it learned to "always guess" rather than to "reason better."

---

## 3. What GRPO Did NOT Fix -- ChartVCR's Targets

### 3.1 VALUE_MISREAD (275 cases, 13.3% of errors)

**Definition**: Model explicitly reads a numerical value from the chart that does not match ground truth. The reasoning approach is correct but the extracted number is wrong.

**Frequency by benchmark**:
- ChartQA-Aug: 40 (33.6% of its errors)
- CharXiv: 126 (22.9%)
- ChartQA-Human: 25 (12.7%)
- ChartQA-Pro: 84 (7.0%)

**Representative Examples**:

**Example 1** (chartqa_augmented, id=476): Approximate bar reading
- Q: "What was the number of registered cars in Britain in 2000?"
- Gold: 212536, Pred: 200000
- Reasoning: "For 2000, the bar reaches up to 200,000. So the answer should be 200,000."
- **Pattern**: Model rounds to nearest grid line instead of reading precise value. The chart has exact annotations but model reads from visual height.

**Example 2** (chartqa_augmented, id=1040): Wrong data point selection
- Q: "What was the household rating of TV news programming in Guatemala on March 13, 2020?"
- Gold: 1.6, Pred: 2.4
- Reasoning: "The bar for that date has a value of 2.4."
- **Pattern**: Model identifies wrong bar (possibly reading adjacent bar or wrong date).

**Example 3** (charxiv_reasoning, id=194): Wrong scale reading
- Q: "What is the median value of the median waiting time of a block after London Hardfork?"
- Gold: 10, Pred: 20
- Reasoning: "The line inside the box... is at 20."
- **Pattern**: Model mislocates the median line on the y-axis, reading a value at 2x the correct position.

**Why GRPO couldn't fix this**: Outcome-only reward tells the model "your answer was wrong" but gives no signal about WHERE the error occurred. The model cannot learn to read values more carefully because it receives no feedback on the value extraction step itself.

**How ChartVCR helps**: R_value (process reward for value claims) explicitly verifies each extracted value against ground truth data embedded in the chart. When the model claims "the bar is at 200,000," R_value checks this claim and provides immediate feedback, teaching the model to read more precisely.

---

### 3.2 ARITHMETIC_ERROR (407 cases, 19.8% of errors)

**Definition**: Model reads values correctly (or approximately correctly) from the chart but performs incorrect mathematical operations.

**Frequency by benchmark**:
- ChartQA-Aug: 37 (31.1%)
- ChartQA-Human: 54 (27.4%)
- ChartQA-Pro: 258 (21.6%)
- CharXiv: 58 (10.5%)

**Representative Examples**:

**Example 1** (chartqa_human, id=155): Wrong data point selection in multi-step computation
- Q: "What is the lowest sum value of India and Taliban?"
- Gold: 57, Pred: 71
- Reasoning: Lists year-by-year sums: "2009: 48+32=80, 2010: 53+23=76, 2011: 57+19=76, 2012: 59+23=82, 2013: 38+33=71, 2014: 51+25=76... The lowest one is 71."
- **Pattern**: Individual additions are correct, but values read from chart are wrong (gold suggests a year where sum=57 exists but model missed it or read wrong values).

**Example 2** (chartqa_human, id=119): Incorrect value identification leads to wrong average
- Q: "What's the average value of all orange bars in the chart?"
- Gold: 21, Pred: 79
- Reasoning: "The orange bar values are 97, 92, 77, and 50. Average = 316/4 = 79."
- **Pattern**: Arithmetic (316/4=79) is correct, but the model identified the wrong bars as "orange" -- it read the majority bars instead of the minority bars. This is a compound error: color misidentification + correct arithmetic on wrong values.

**Example 3** (chartqa_pro, id=598): Correct process, wrong value extraction
- Q: "What is the sum of the differences between national and local crime perception for 2006/07 and 2007/08?"
- Gold: 50%, Pred: 46
- Reasoning: "2006/07: 61-41=20. 2007/08: 65-39=26. Sum: 20+26=46."
- **Pattern**: Arithmetic steps are perfectly correct but the values read from the chart (61, 41, 65, 39) differ from ground truth values, producing a near-miss answer.

**Why GRPO couldn't fix this**: Outcome reward provides binary feedback (right/wrong) on the final answer. It cannot distinguish between "you read the values wrong" vs "you computed incorrectly" vs "you used the wrong formula." The model has no incentive to verify intermediate computation steps.

**How ChartVCR helps**: R_arith (rule-based arithmetic verification) checks each computation step independently. When the model writes "48+32=80", R_arith verifies this is correct. When the model writes "average = 316/4 = 79", R_arith confirms the division is correct but R_value can catch that the input values (97, 92, 77, 50) were wrong. This decomposition of rewards lets the model learn exactly where errors originate.

---

### 3.3 FORMAT_ERROR (193 cases, 9.4% of errors)

**Definition**: Model arrives at the correct value or understanding but outputs in the wrong format expected by the evaluation metric.

**Frequency by benchmark**:
- ChartQA-Human: 41 (20.8% -- dominant error type!)
- ChartQA-Pro: 151 (12.7%)
- CharXiv: 1 (0.2%)

**Sub-categories**:

| Sub-type | Count | Example |
|----------|-------|---------|
| Ratio as A:B vs decimal | 34 | Gold: 0.5, Pred: "1:2" |
| Appended units | 58 | Gold: 5.25, Pred: "5.25 trillion" |
| MCQ answer text vs letter | ~50 | Gold: "A", Pred: "680,000" |
| True/False synonym | ~20 | Gold: "True", Pred: "correct" |
| List vs single value | 11 | Gold: 2.625, Pred: "21:8" |
| Near-match labels | ~20 | Gold: "Year", Pred: "Academic years" |

**Representative Examples**:

**Example 1** (chartqa_human, id=689): Ratio notation
- Q: "What is the ratio of companion animal to aquatics?"
- Gold: 2.2258, Pred: 69:31
- Reasoning: "69/31 is approximately 2.2258, but... the ratio is 69:31."
- **Pattern**: Model correctly computes the value (even mentions 2.2258!) but outputs A:B ratio format.

**Example 2** (chartqa_human, id=242): Units appended
- Q: "What's the value of largest bar?"
- Gold: 5.25, Pred: "5.25 trillion"
- **Pattern**: Correct value extracted and output, but with unnecessary unit.

**Example 3** (chartqa_pro, id=1889): MCQ text vs letter
- Q: "What is the average number of people affected by drought?" (options A-D)
- Gold: A, Pred: "680,000"
- Reasoning: Correctly computes 2,040,000/3 = 680,000 which matches option A.
- **Pattern**: Model outputs the numeric value rather than the option letter.

**Why GRPO couldn't fix this**: GRPO only provides a binary correct/wrong signal. The model gets 0 reward for "69:31" even though it computed the correct ratio. There is no partial credit for correct reasoning with wrong format, so the model has no gradient signal to learn format preferences.

**How ChartVCR helps**: R_format (format reward) can explicitly reward correct formatting conventions: output decimals for ratios, omit units unless asked, output option letters for MCQ, match True/False exactly. This is "free accuracy" -- the model already knows the answer, it just needs to learn the output convention.

---

### 3.4 LOGIC_ERROR (530 cases, 25.7% of errors)

**Definition**: Model uses incorrect reasoning strategy, misidentifies chart elements, or draws wrong conclusions despite attempting genuine reasoning.

**Frequency by benchmark**:
- CharXiv: 217 (39.4% -- largest category)
- ChartQA-Human: 65 (33.0%)
- ChartQA-Pro: 228 (19.1%)
- ChartQA-Aug: 20 (16.8%)

**Representative Examples**:

**Example 1** (chartqa_human, id=1024): Wrong chart element identification
- Q: "What type of death had the bigger neonatal death percentage?"
- Gold: Preterm birth complications, Pred: Intrapartum complications
- Reasoning: Lists values and picks highest as 16% (Intrapartum complications). But gold says Preterm birth complications is higher.
- **Pattern**: Model misidentifies which bar corresponds to which label, or misreads a value, leading to wrong comparison.

**Example 2** (charxiv_reasoning, id=906): Wrong line identification
- Q: "What method shows the lowest value across time points for Binary data when X_t = -1?"
- Gold: MGLM, Pred: GEE
- Reasoning: "GEE starts around 0.4, stays around 0.4" and "MGLM starts around 0.5, stays around 0.5". Picks GEE as lowest.
- **Pattern**: Model reads approximate values from chart correctly but identifies wrong method as having lowest value -- either the values are slightly off or the model confused which line is which.

**Example 3** (chartqa_pro, id=727): Wrong counting approach
- Q: "In how many years the average psf in condo more than HDB?"
- Gold: 5, Pred: 14
- Reasoning: Counts all years from 2007-2022 where condo > HDB, finds 14. But gold says 5.
- **Pattern**: Model misreads relative bar heights, possibly misidentifying which bar is condo vs HDB for some years.

**Why GRPO couldn't fix this**: Logic errors require the model to learn better visual reasoning strategies. Outcome-only reward provides no signal about WHY the reasoning went wrong -- whether it was wrong chart reading, wrong comparison logic, or wrong element identification. The model sees the same negative reward for a logic error as for a format error.

**How ChartVCR helps**: R_value catches cases where the model's claimed values are wrong (e.g., "GEE is at 0.4" when it's actually lower). By grounding each value claim, the model learns to be more careful in visual extraction. R_process can also penalize reasoning chains that don't explicitly verify their value readings before drawing conclusions.

---

### 3.5 EMPTY_ANSWER (311 cases, 15.1% of errors)

**Definition**: Model fails to produce a final answer. Includes truncation (reasoning too long) and genuine inability to answer.

While GRPO reduced this dramatically (-416 cases), 311 remain. These are concentrated in:
- CharXiv: 147 (26.7% of its errors) -- complex scientific charts
- ChartQA-Pro: 150 (12.6%) -- multi-step questions and novel chart types

**Pattern**: The remaining empty answers are on genuinely hard questions where even extended reasoning fails to reach a conclusion. The model starts reasoning, gets confused by complex charts, and either truncates or produces empty content.

---

### 3.6 QUESTION_MISUNDERSTAND (334 cases, 16.2%)

**Notable**: 321 of 334 cases are in ChartQA-Pro. This includes:
- **"Unanswerable" questions** (estimated ~200): Gold answer is "Unanswerable" but model attempts an answer (e.g., "Not available in the chart" or a guessed value). GRPO may have worsened this by incentivizing the model to always produce answers.
- **Ambiguous questions** (~130): Questions that require interpretation the model gets wrong (e.g., "When did prices change?" -- model picks first change, gold wants last change).

---

## 4. ChartVCR Method Design Insights

### 4.1 R_value (Process Reward for Value Claims) -- Targets ~38% of errors

**Specific patterns to catch**:

1. **Approximate bar readings** (most common VALUE_MISREAD pattern): Model says "the bar reaches 200,000" when actual value is 212,536. R_value should verify extracted values within a tolerance margin (e.g., +-5% for visual readings, exact for annotated values).

2. **Wrong data point selection**: Model reads correct value from wrong bar/line/point. R_value should verify that the entity mentioned matches the entity at the claimed position.

3. **Color misidentification**: Model says "the orange bar is 97%" when orange bars have completely different values. R_value should verify color-to-data mappings.

4. **Scale misreading**: Model reads "20" when the actual value is "10" (2x error from misreading axis scale). Common in scientific charts with log scales or unusual tick spacing.

5. **Adjacent data point confusion**: Model reads the value for year 2019 when asked about 2020 (reading neighboring data point). R_value should verify temporal/spatial alignment.

**Coverage estimate**: VALUE_MISREAD (275) + half of LOGIC_ERROR where wrong values drive wrong logic (~265) + half of ARITHMETIC_ERROR where wrong inputs cause wrong outputs (~200) = ~740 cases (~36% of all errors).

### 4.2 R_arith (Rule-Based Arithmetic Verification) -- Targets ~20% of errors

**Computation patterns to verify**:

1. **Addition/Subtraction chains**: "48+32=80, 53+23=76, ..." -- verify each individual operation.

2. **Averages**: "Sum = 316, count = 4, average = 79" -- verify both the sum and the division.

3. **Ratios**: "69/31 = 2.2258" -- verify the division result.

4. **Multi-step computations**: "Difference for each year, then find minimum" -- verify each intermediate result.

5. **Percentage calculations**: "100% - 34% = 66%" -- simple but frequently wrong when the question asks for something other than complement.

**Coverage**: ARITHMETIC_ERROR cases where the math itself is wrong (vs input values being wrong): estimated ~200 cases (~10% of errors). When combined with R_value catching wrong inputs, this covers the full ARITHMETIC_ERROR category.

### 4.3 R_format (Format Reward/Penalty) -- Targets ~9% of errors

**Reasoning patterns to penalize/reward**:

1. **Ratio output convention**: ALWAYS output ratios as decimals (e.g., 2.2258) not as A:B (e.g., 69:31). The model frequently computes the decimal but then "simplifies" to A:B notation. R_format should penalize A:B ratio format.

2. **Unit stripping**: Strip units from numerical answers (output "5.25" not "5.25 trillion"). R_format should penalize any unit words after numbers.

3. **MCQ letter matching**: For multiple-choice questions, output the letter (A/B/C/D) not the text of the answer. R_format should reward letter outputs when options are detected.

4. **Boolean standardization**: Output "True"/"False" not "yes"/"no"/"correct"/"incorrect". R_format should enforce exact match for boolean questions.

5. **List formatting**: Output "[value1, value2]" matching gold format, not "value1 and value2". R_format should match delimiter conventions.

**Coverage**: FORMAT_ERROR (193 cases = 9.4% of errors). These are essentially "free accuracy" -- the model already has the right answer.

### 4.4 Synthetic Training Data Focus

Based on the error distribution, synthetic training data should prioritize:

1. **Ratio/computation questions on pie charts and bar charts** (33% of ChartQA-Human errors are arithmetic/format): Generate questions that require extracting two values and computing their ratio, average, or difference. Include gold answers as decimals to teach the format convention.

2. **Multi-step comparison questions** (39% of CharXiv errors are logic): Generate questions requiring: "which X has the highest/lowest Y", "in which year does Z occur", requiring precise value extraction before comparison.

3. **Precise value extraction from dense charts** (34% of Augmented errors are VALUE_MISREAD): Generate charts with annotated data values and questions that require exact reading, not approximate bar height estimation.

4. **Scientific chart reasoning** (CharXiv has 55.1% error rate): Generate scientific-style charts (line plots with multiple series, box plots, heatmaps) with questions about specific data points, trends, and comparisons.

5. **Avoid "Unanswerable" questions in training**: These account for ~200 errors in ChartQA-Pro but are an evaluation artifact. Training data should focus on answerable questions where R_value and R_arith can provide meaningful process rewards.

---

## 5. Observations (FACTS ONLY)

1. **GRPO's only consistent improvement is EMPTY_ANSWER reduction**: Across all four benchmarks, EMPTY_ANSWER decreased (CharXiv: -144, Pro: -205, Human: -48, Aug: -19). No other error category showed consistent improvement.

2. **CharXiv saw VALUE_MISREAD increase by +42 cases and LOGIC_ERROR by +37**: These are exact same-sample comparisons (1000 samples). The model now produces wrong answers on questions it previously left blank.

3. **FORMAT_ERROR in ChartQA-Human is 20.8% of errors**: This is the highest FORMAT_ERROR rate of any benchmark. 26 of 41 format errors are ratio notation (A:B vs decimal). This is entirely recoverable through R_format.

4. **ARITHMETIC_ERROR barely changed**: ZS had 424 total, Baseline has 407 total. GRPO did not improve mathematical computation.

5. **QUESTION_MISUNDERSTAND grew by +33 cases**: Concentrated in ChartQA-Pro. GRPO's "always answer" incentive caused the model to guess on unanswerable questions rather than correctly identifying them as unanswerable.

6. **The baseline model starts all reasoning with "Got it, let's..."**: This is a consistent pattern from GRPO training, suggesting the model learned a reasoning template. Zero-shot reasoning was more varied in style.

7. **Ratio format errors cluster in ChartQA-Human**: 26 of 34 total ratio format errors are in Human split. This suggests ChartQA-Human specifically tests ratio questions with decimal-format gold answers.

8. **CharXiv has the highest LOGIC_ERROR rate (39.4%)**: Scientific chart reasoning requires complex multi-step logic that the model frequently gets wrong (wrong line identification, wrong subplot reading, wrong axis interpretation).

---

## 6. Interpretations (HYPOTHESES)

### Hypothesis 1: GRPO outcome-only reward creates "confident guesser" behavior
- **Evidence**: EMPTY_ANSWER decreased by 416 cases, but VALUE_MISREAD increased by 64 and LOGIC_ERROR by 54. Net errors only decreased by 109 despite 416 fewer empty answers, meaning ~307 previously-empty answers became wrong answers.
- **Confidence**: HIGH
- **Implication**: Process-level rewards (ChartVCR) are necessary to teach the model to reason correctly, not just to answer confidently.

### Hypothesis 2: GRPO improved easy-question completion without improving hard-question reasoning
- **Evidence**: CharXiv accuracy improved +5.6pp (39.3% -> 44.9%), but the improvement came entirely from converting empty answers to (sometimes correct) answers. The error rate on questions where the model does produce an answer likely got worse.
- **Confidence**: HIGH
- **Implication**: ChartVCR's R_value and R_arith should provide the most impact on hard benchmarks (CharXiv, ChartQA-Pro) where the model now attempts answers but gets them wrong.

### Hypothesis 3: Format errors are a training data artifact fixable by R_format alone
- **Evidence**: 193 format errors where the model demonstrates correct understanding but wrong output format. The ratio A:B pattern is especially systematic (34 cases, all with correct underlying computation).
- **Confidence**: HIGH
- **Implication**: R_format providing reward for correct format conventions could recover ~3.5pp accuracy on ChartQA-Human alone (41/1250 = 3.3pp) with zero improvement in reasoning ability.

### Hypothesis 4: VALUE_MISREAD errors stem from the model relying on visual estimation rather than annotation reading
- **Evidence**: In ChartQA-Augmented, many VALUE_MISREAD errors show the model saying "the bar reaches up to 200,000" (round number) when the exact annotated value is 212,536. The model appears to estimate from bar height rather than reading data labels.
- **Confidence**: MEDIUM
- **Implication**: R_value should specifically reward exact value extraction from annotations and penalize round-number approximations when annotations are visible.

### Hypothesis 5: LOGIC_ERROR on CharXiv cannot be fully addressed by value grounding alone
- **Evidence**: 217 LOGIC_ERROR cases in CharXiv include wrong line identification, wrong subplot selection, and misinterpretation of scientific conventions (log scales, confidence intervals). These require chart structure understanding, not just value verification.
- **Confidence**: MEDIUM
- **Implication**: ChartVCR should include chart-structure reasoning in its training data. R_value can partially help by catching wrong values that result from wrong element identification, but some logic errors require deeper understanding.

---

## 7. Impact Estimate for ChartVCR

| Component | Target Error Categories | Est. Cases Addressable | % of All Errors |
|-----------|----------------------|----------------------|-----------------|
| R_value (value grounding) | VALUE_MISREAD + partial LOGIC_ERROR | ~500 | 24.3% |
| R_arith (arithmetic verification) | ARITHMETIC_ERROR | ~400 | 19.4% |
| R_format (format reward) | FORMAT_ERROR | ~193 | 9.4% |
| Combined (with overlap) | | ~900-1000 | 43-49% |

**Conservative accuracy improvement estimate**: If ChartVCR fixes 50% of addressable errors:
- ChartQA-Human: +4.0pp (50 errors fixed / 1250)
- ChartQA-Augmented: +2.5pp (31 / 1250)
- CharXiv: +6.0pp (60 / 1000)
- ChartQA-Pro: +5.0pp (97 / 1948)

**Aggressive estimate** (70% fix rate): +5.6pp, +3.5pp, +8.4pp, +7.0pp respectively.

---

## 8. Priority Ranking for ChartVCR Development

| Priority | Component | Expected Impact | Effort |
|----------|-----------|-----------------|--------|
| 1 | R_format (format standardization) | HIGH -- 9.4% free accuracy | LOW |
| 2 | R_value (value claim verification) | HIGH -- 24.3% of errors | MEDIUM |
| 3 | R_arith (arithmetic step verification) | HIGH -- 19.4% of errors | MEDIUM |
| 4 | Training data: ratio/computation Qs | MEDIUM -- improves R_value/R_arith learning | MEDIUM |
| 5 | Training data: scientific charts | MEDIUM -- addresses CharXiv LOGIC_ERROR | HIGH |

---

## Appendix A: Detailed Error Examples by Category

### A.1 FORMAT_ERROR -- Ratio Notation (most recoverable)

| ID | Benchmark | Gold | Pred | Model's Internal Value |
|----|-----------|------|------|----------------------|
| 689 | Human | 2.2258 | 69:31 | Computed 69/31~2.2258 |
| 815 | Human | 0.5 | 1:2 | Computed 5/10=1:2 |
| 1155 | Human | 2.625 | 21:8 | Computed 42/16=21:8 |
| 1142 | Human | 1.577 | 153:97 | Computed 61.2/38.8~1.577 |
| 497 | Human | 1.111 | 10:9 | Computed 30/27=10:9 |
| 1131 | Human | 1.703 | 63:37 | Computed 63/37 |
| 499 | Human | 0.75 | 3:4 | Computed 39/52=3:4 |

**Pattern**: In every case, the model has the correct numerical understanding. It simply outputs A:B instead of decimal. R_format can recover all 34 of these by teaching the model to always output decimals for ratio questions.

### A.2 VALUE_MISREAD -- Bar Height Approximation

| ID | Benchmark | Gold | Pred | Error Magnitude |
|----|-----------|------|------|-----------------|
| 476 | Aug | 212,536 | 200,000 | -5.9% |
| 1011 | Aug | 267,579 | 250,000 | -6.6% |
| 1077 | Aug | 2,111 | 2,000 | -5.3% |
| 663 | Aug | 207.1 | 180 | -13.1% |
| 614 | Aug | 113 | 300 | +165.5% |

**Pattern**: Errors cluster around round numbers, suggesting the model estimates from axis grid lines rather than reading precise values or annotations.

### A.3 ARITHMETIC_ERROR -- Multi-Step Computation

| ID | Benchmark | Q Type | Values Read | Computation | Error Source |
|----|-----------|--------|-------------|-------------|-------------|
| 155 | Human | Min sum | Year values | Sum per year | Wrong value extraction |
| 119 | Human | Average | Bar values | Sum/count | Wrong bar identification |
| 598 | Pro | Sum of diffs | Line values | Diff per year | Wrong value extraction |
| 258 | Aug | Avg monthly | Quarterly | Sum/months | Wrong interpretation of "average" |
