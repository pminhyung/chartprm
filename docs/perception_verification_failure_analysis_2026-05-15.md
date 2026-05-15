# Perception-Grounded Verification — Failure Mode Analysis & Redesign Brief

_For: research-strategy-advisor (mentor) / Project: ChartVCR_
_Date: 2026-05-15 (after D1+D2 pilot, 4B+9B and 27B+27B both runs)_
_Author: training pipeline owner_

---

## 0. TL;DR

D1/D2 pilot 결과 **Math-Shepherd 스타일 outcome MC rollout은 strong signal** (정답 trajectory 평균 step value 0.73, 오답 0.05, gap 0.68, monotonic, bimodal — 5-check 5/5 PASS). 반면 **perception-grounded verification은 outcome 예측에 거의 lift 없음** (ROC AUC 0.568 ≈ chance, AUC lift over MC-only +0.003, complementarity r 0.111). 27B+27B로 모델 강도를 올려도 4B+9B와 결과 동일 → **모델 capability 문제 아님**.

근본 원인을 데이터 기반으로 분류한 결과 세 가지 issue로 압축됨:

1. **Claim extraction이 spurious entity 다량 생성** (1957건 중 14.3% 직접 spurious 매칭, 더 많은 sentence-fragment 추정) — regex Tier 1의 영문법적 한계.
2. **Per-claim faithfulness와 per-trajectory outcome이 본질적으로 다른 dimension** (r=0.111) — "정답 도달했지만 grounding 약함" 15 sample, "정답 못 갔지만 grounding 강함" 9 sample 발견.
3. **Tolerance band 5%/15% binary saturation** — value claims의 5.8% (113건/1957)가 close-but-failed (15-30% rel diff) 영역, partial credit 사실상 무력 (4.3%).

본 문서는 issue별 verbatim 예시와 재설계 spec(A–D)를 담는다. **모델 강도가 아닌 measurement design을 재설계해야** perception term이 의미를 갖는다고 판단함.

---

## 1. Setting

| 항목 | 값 |
|---|---|
| Policy / Verifier | Qwen3.5-VL-4B + 9B 1차 / **Qwen3.5-VL-27B (Qwen3.6-27B file) 양쪽 2차** |
| Sampling (policy thinking) | temp=1.0, top_p=0.95, top_k=20, min_p=0, presence=1.5, rep=1.1, max=8192 |
| Sampling (verifier) | temp=0.0, max=128 |
| MC continuation | temp=0.7, top_p=0.9, max=2048, K=8 |
| Data | ChartQA-train(100) + ReachQA-train(100), test bench pHash overlap=0 |
| Segmentation | `\n\n` split + merge < 200 tok + split > 560 tok at sentence boundary |
| Claim extractor | Tier 1 regex 4 patterns + Tier 2 LLM fallback (when regex==0 & step has digit) |
| Tolerance | value: 5% rel or 0.5 abs → 1.0; ≤15% rel → 0.5; else 0.0 |

D1 PASS: 6/6 auto-validate (length median 196 tok, p90 552, single-step 5.5%).
D2: 100 sample × 565 steps × K=8 = 4520 MC generations + 1957 claim verifications.

---

## 2. Pilot Result Matrix

| 검증 목표 | Target | 4B+9B | **27B+27B** | 판정 |
|---|---|---:|---:|:---:|
| MC step-value informative (gap, slope, bimodal) | ≥3/5 check | 4/5 | **5/5** | ✓ |
| Claim extraction (Jaccard vs LLM oracle) | ≥0.70 mean / ≥85% recall | 0.461 / 46% | **0.441 / 44%** | ✗ |
| Perception ROC AUC vs outcome | ≥0.65 | 0.581 | **0.568** | ✗ |
| r(perception, mc) | 0.3–0.7 | +0.208 | **+0.111** | ✗ |
| AUC lift (mc+p vs mc only) | ≥+0.02 | +0.001 | **+0.003** | ✗ |
| 종합 alignment | ≥3/4 | 1/4 | **1/4** | ✗ |

→ guide §4.1 decision: `<2/4` → "Verifier 재고". stronger verifier 시도 (4B+9B → 27B+27B) 결과 거의 무변화 → **stronger verifier ≠ 해결책**.

---

## 3. Issue 1 — Claim extraction이 spurious entity 다량 추출

Tier 1 정규식은 step 내 `<entity> <op> <value>` 패턴을 강제 매칭하여 **chart label이 아닌 영문법적 noun/대명사/수식어/문장 단편**까지 entity로 잡아낸다. 분류한 5 subtype + 실제 verbatim:

### 1A. Referential pronouns (n=11, 0.6%) — "It", "the answer", "the value" 등

```
id=reachqa_train_s6_r618 step7 claim0  score=0.0
  entity='It'  value=25.0
  verifier_response='8'
  step_snippet: "Let's write down the steps clearly.
                 Step 1: Read the graph to get the % value for Wood. It is 25%.
                 Step 2: Read values for potential component materials. Steel = 15%, …"
```

```
id=chartqa_train_5982 step4 claim0  score=0.0
  entity='the answer'  value=198.0
  verifier_response='157'
  step_snippet: "Let's double check the exact request: 'Q1 average daily rate... in 2016'.
                 Visual inspection of the line for 2016 (lime green):
                 It passes through a point at Q3 labeled '221'. …"
```

→ "It"이나 "the answer" 같은 anaphor가 entity로 추출됨. Verifier는 chart에서 그런 label을 찾을 수 없어 임의의 number를 반환 — false-negative.

### 1B. Math result terms (n=167, **8.5%** of all claims) — "Diff", "Average", "Ratio", "Sum" 등

```
id=reachqa_train_s3_r681 step10 claim7  score=0.0
  entity='Average'  value=14.0
  verifier_response='75'
  step_snippet: "Value in 2023: Approx 48 - 20 = 28.
                 Value in 2019: Approx 25 - 14 = 11.
                 Difference: 17. Average: 17 / 4 = 4.25."
```

```
id=reachqa_train_s4_r1866 step5 claim1  score=0.0
  entity='Difference'  value=6.0
  verifier_response='20'
  step_snippet: "…If we ignore the weird shape and focus purely on bounding the main population: Box top ≈ 90, Box bottom ≈ 70…"
```

```
id=chartqa_train_3681 step0 claim2  score=0.0
  entity='Ratio'  value=23.6
  verifier_response='19.5'
  step_snippet: "1. **Identify the relevant data points:** Look for the bar labeled 'Female'. The value next to it is …"
```

→ Reasoning step은 계산 결과 ("Difference: 17", "Average: 4.25", "Ratio: 23.6")를 명시하는데, regex가 이 결과값을 chart entity claim으로 잡음. 사실 chart에 그런 entity 자체가 없음. **8.5%로 spurious 중 가장 큰 단일 비중**.

### 1C. Positional descriptors (n=45, **2.3%**) — "Top", "Bottom cell", "Rightmost bar" 등

```
id=reachqa_train_s3_r681 step4 claim4  score=0.0
  entity='Bottom'  value=67.0
  verifier_response='2.5'
  step_snippet: "**Year 2023:** … Top of pink is exactly on the grid line for 20. … Bottom is 20. Value = 4…"
```

```
id=chartqa_train_3126 step2 claim3  score=0.0
  entity='Rightmost bar'  value=None
  verifier_response='Greece'
  step_snippet: "Let me double-check. - Leftmost bar: Green (Ireland) - Second bar: Red (Chile) - Third bar: Purple (Portugal) - Rightmost bar: Blue (Greece)"
```

→ "Bottom", "Rightmost bar"는 chart 내 entity가 아닌 **position descriptor**. Reasoning이 인덱싱 의도로 사용 (e.g. "Rightmost bar는 Greece"), 그러나 verifier에게는 의미 모호.

### 1D. Abstract descriptors (n=13, 0.7%) — "Smaller value", "Bigger value", "Smallest" 등

```
id=chartqa_train_2705 step7 claim1  score=0.0
  entity='smaller value'  value=41.0
  verifier_response='22'
  step_snippet: "Compare 59 (bigger) vs 41 (smaller). Note: technically we should compare absolute magnitudes…"
```

```
id=chartqa_train_815 step0 claim1  score=1.0  ← 우연히 일치
  entity='Smallest'  value=18.0
  verifier_response='18'
  step_snippet: "1. **Identify the segments and their values:** 'Easier': 50% (Dark blue segment) …"
```

→ Comparative adjectives ("smaller", "bigger", "smallest")가 entity로 매칭. 일부 (1.0 score)는 우연히 verifier가 같은 답 반환.

### 1E. Sentence fragment as entity (n=44, **2.2%**) — "Second lowest is Mali", "bers are the 3rd and 4th values" 등

```
id=chartqa_train_1059 step1 claim1  score=1.0  ← 우연히 catch
  entity='Second lowest is Mali'  value=146.58
  verifier_response='146.58'
  step_snippet: "Lowest is Liberia (132.95). Second lowest is Mali (146.58)."
```

```
id=reachqa_train_s1_r318 step2 claim0  score=1.0
  entity='bers are the 3rd and 4th values'  value=85.0
  verifier_response='85'
  step_snippet: "Find the Median: The median is the middle value … Since there are an even number of data points (6 sports), the median will be the average of the two middle numbers."
```

→ Regex가 `pattern1 = "<entity> is/equals <value>"` 매칭으로 `"Second lowest is Mali"`를 entity로 잡음. Mali가 chart label인데 entity로 인식 안 됨. Verifier도 우연히 같은 value 반환 (Liberia/Mali가 두 sample이라 우연일치).

### 1Sum. Direct-spurious 합계

| Subtype | n | % of 1957 claims |
|---|---:|---:|
| 1A pronoun_ref | 11 | 0.6% |
| 1B math_result | 167 | **8.5%** |
| 1C positional | 45 | 2.3% |
| 1D abstract_descriptor | 13 | 0.7% |
| 1E sentence_fragment | 44 | 2.2% |
| **합계 (직접 매칭)** | **280** | **14.3%** |

이 외에도 entity가 chart label인지 확인 안 된 모든 claim — sample inspection 50% spurious rate라 **실제 spurious 비율 30-50% 추정**. Jaccard 0.44 vs LLM oracle 결과와 정합.

---

## 4. Issue 2 — Per-claim accuracy ≠ trajectory outcome (구조적 decoupling)

Per-sample perception_agg와 outcome correctness의 상관 r=0.12, AUC 0.57. 두 signal이 **다른 dimension**을 잡고 있음을 시사. Verbatim 실증:

### 2A. 정답 trajectory + low grounding (n=15) — "정답 맞췄지만 step claim 다수 spurious"

```
id=chartqa_train_2691  p_agg=0.000  mc_agg=0.875  outcome=1
  Q: What is the sum of major threat distribution between India and Indonesia?
  gold: 87 | pred: 87
  → 정답 도달, MC 0.875 (정답 trajectory). 그러나 perception 0.0 — 추출된 모든 claim이 spurious.
```

```
id=chartqa_train_5982  p_agg=0.300  mc_agg=0.650  outcome=1
  Q: What is the Q1 average daily rate in hotels in Toronto in 2016 (in Canadian dollars)?
  gold: 198 | pred: 198
  → 정답이나 perception 0.30 (대부분 claim mismatch).
```

```
id=chartqa_train_6507  p_agg=0.203  mc_agg=0.734  outcome=1
  Q: What is the difference between highest standard of living is getting better or getting worse?
  gold: 19 | pred: 19%
  → 정답이나 perception 0.20.
```

### 2B. 오답 trajectory + high grounding (n=9) — "step claim 거의 정확하지만 최종 답 틀림"

```
id=reachqa_train_s4_r1565  p_agg=0.800  mc_agg=0.000  outcome=0
  Q: composition of marine biodiversity in Oceania, largest 'Fish' = 40%. If …
  gold: combined Coral Reefs + Crustaceans + Marine Mammals = 30% (15+10+5)
  pred: 37.0%
  → step claims (chart entity values)는 정확히 추출. 최종 산술/조합 단계에서 실패.
```

```
id=reachqa_train_s0_r1307  p_agg=0.825  mc_agg=0.000  outcome=0
  Q: continent with largest single-year solar capacity increase between 2010–2020?
  gold: Asia (20 GW between 2019 and 2020)
  pred: 2020
  → step claims grounded, 그러나 question intent에 대한 final answer wrong format.
```

```
id=chartqa_train_892  p_agg=0.857  mc_agg=0.000  outcome=0
  Q: What color does the highest value in Pie chart represent?
  gold: Dark blue
  pred: 58%
  → 모든 chart value claim grounded, 그러나 final answer가 색깔 대신 % 반환.
```

### 2 함의

- Perception ↔ outcome decoupling은 **measurement artifact가 아닌 본질**. Step grounding faithfulness와 final answer correctness는 독립 axis.
- 따라서 "verifier AUC vs outcome ≥0.65"는 **잘못된 통과 기준** [추정]. 우리가 측정하고자 했던 것 ("reasoning이 chart로부터 derivable한가") = step-level faithfulness이지 trajectory success가 아님.
- Paper narrative상 Grounding Drift framing은 **두 dimension을 분리 측정**해야 옳음:
  - **Drift rate** (step claim 중 chart-grounded 비율) — verifier가 직접 측정해야 할 것
  - **Outcome correctness** — 별도 metric
  - 둘이 어떻게 결합하는가 (drift→outcome causation)는 separate analysis.

---

## 5. Issue 3 — Tolerance band saturation (5%/15% binary)

Score distribution: 0.0=51.9%, 0.5=4.3%, 1.0=43.8% → **사실상 binary**.

15-30% rel diff 영역에 113건 claim 존재 — 모두 score=0이지만 "verifier가 chart에서 정확히 가까운 값 잡았는데 strict threshold 못 넘김":

```
id=reachqa_train_s1_r987  entity='Rate'
  claim_value=7.0  verifier_value=5.0  rel_diff=28.6%
  step: "Solar: Start = 35%. Rate = 5%. Factor ≈ (1.05)^10 ≈ 1.63. New Value ≈ 35 * 1.63 ≈ 57"
  → claim에서 "Rate = 7%"라 기재, verifier는 chart에서 "5%" 읽음. 사실 step text 자체에도 5%로 두 번 나옴 — claim의 7.0은 spurious.
```

```
id=chartqa_train_2740  entity='Green'
  claim_value=58.0  verifier_value=72.0  rel_diff=24.1%
id=chartqa_train_2740  entity='Orange'
  claim_value=31.0  verifier_value=23.0  rel_diff=25.8%
  step: "72 > 62 = 62 > 58 > 53 > 46 > 45 > 40. The smallest number is 40, corresponding to Germany. Wait, let me re-check…"
  → policy가 chart을 다른 매핑으로 해석. Entity "Green"/"Orange"는 색깔 — Issue 1C와 1D 복합.
```

```
id=chartqa_train_6083  entity='Chartered'
  claim_value=83.0  verifier_value=101.0  rel_diff=21.7%
  step: "2017: Owned = 39, Chartered = 89. Difference = |39 - 89| = 50. 2018: Owned = 43, Chartered = 80…"
  → Chartered는 valid chart entity, 그러나 policy가 잘못 읽고 (89가 아닌 83), verifier는 다른 연도 값 (101) 반환. 실제 grounding 차이.
```

→ 5%/15% threshold는 chart visual reading noise (특히 hand-drawn 차트, no exact data label)에 너무 strict. **chart QA 영역에서는 10%/25% 또는 unit-aware 권장** [추정].

---

## 6. Issue 4 — Verifier capability bound: stronger model 효과 없음

| Metric | 4B policy + 9B verifier | **27B policy + 27B verifier** | Δ |
|---|---:|---:|---:|
| Outcome correct rate | 35% | 43% | +8pp (policy 강함 자체) |
| Perception ROC AUC vs outcome | 0.581 | **0.568** | **−0.013** |
| AUC lift | +0.001 | +0.003 | +0.002 |
| r(p, m) complementarity | +0.208 | **+0.111** | −0.097 (더 독립적) |

→ Verifier 9B → 27B 변경으로도 perception 신호 거의 변화 없음. **claim extraction quality와 metric design이 bottleneck**이지 verifier capability 아님.

---

## 7. 재설계 권고 (Spec A–D)

### Spec A. Claim Extraction — LLM-primary + entity validation (highest priority)

**Replace** regex Tier 1 → **LLM-primary** (예: Qwen3.5-VL-27B with strict prompt):

```python
EXTRACT_PROMPT = """Given the chart image and the reasoning step below, extract
all NUMERIC or CATEGORICAL claims that this step makes about VISIBLE CHART
ENTITIES. Rules:
1. Entity MUST be a label/category that appears on the chart axes, legend, or
   data labels. Reject pronouns ("it", "the answer"), math operation names
   ("difference", "average", "ratio"), positional descriptors ("rightmost
   bar", "bottom cell"), comparative adjectives ("smaller value"), and
   sentence fragments.
2. If you cannot verify the entity exists on the chart, OMIT that claim.
3. Decompose claims like "Mali is 146.58" into entity="Mali", value=146.58
   (not "Second lowest is Mali"). The entity is the chart label only.

Reasoning step:
{step_text}

Chart: <image>

Output JSON list of {{"entity": str (chart label), "value": number_or_null,
"type": "value"|"categorical", "comparator": null_or_str}}.
If no valid claim, output []."""
```

기대 효과: spurious 30-50% → <10%, Jaccard 0.44 → 0.7+ [추정].

### Spec B. Verifier — NOT_FOUND fallback prompt

**Replace** "what is the value of {entity}?" → reject-aware prompt:

```python
VERIFY_PROMPT = """Looking at this chart, locate the entity labeled exactly
'{entity}' on the chart's axes, legend, or data labels. If you find it,
reply with ONLY its associated numeric value. If '{entity}' does NOT appear
as a chart label (i.e. it's a derived/computed reference or not visible),
reply EXACTLY: NOT_FOUND.
Do not infer or compute. Read directly from the chart."""
```

NOT_FOUND claims → score=null (penalty 없음, perception_agg 계산에서 제외). Spurious claim의 영향력을 0으로 만듦.

### Spec C. Tolerance — chart-domain calibrated

- 5% rel or 0.5 abs → 1.0 → **10% rel or 1.0 abs** → 1.0
- ≤15% rel → 0.5 → **≤25% rel** → 0.5
- 추가: value with unit `%`인 경우 absolute 2pp tolerance 적용 ("14%" vs "16%": rel 14%이지만 chart reading 정밀도상 정합).

### Spec D. Metric 재설계 — Drift rate vs Outcome separation

현재 "perception AUC vs outcome ≥ 0.65"는 잘못된 통과 기준. 진짜 측정하고자 했던 것은 **step-level grounding faithfulness 자체**.

**New metric — Drift Rate (DR)**:
```
DR = 1 - mean_over_chart_grounded_claims(perception_score)
```

평가 시 4×4 분할표 (drift × outcome):

| | low drift | high drift |
|---|---|---|
| correct outcome | grounded reasoning (good) | shortcut reasoning (lucky) |
| wrong outcome | careful but flawed | hallucinated reasoning |

각 cell의 비율을 model 간 비교 (zero-shot vs SFT vs SFT+VAPV). 통과 기준:
- "grounded+correct" cell이 our method에서 ≥ baseline +10pp 증가
- "shortcut" + "hallucinated" cell 합계 감소

이는 outcome lift와 별개로 **reasoning quality 자체**를 측정 — paper §1 framing ("Grounding Drift in Multi-Step Chart Reasoning")과 정합.

### Spec 우선순위

1. **A + B** 먼저 적용 → 30 sample subset 재측정 (≈1h). Jaccard 0.7+ 및 perception score saturation 완화 확인.
2. Saturation 완화되면 **D metric**으로 paper §3 narrative 재구성. Outcome lift는 보조 지표로 강등.
3. **C tolerance**는 micro-adjust, 효과 minor 예상.
4. 위 적용 후에도 4-cell 분할표에서 method 간 차이 없으면 → framing pivot 검토 (Path B).

---

## 8. Reproduction

**Data**:
- `data/d1_pilot/segmented_v3.jsonl` — 200 segmented traces (27B, current)
- `data/d1_pilot_4b/segmented_v3.jsonl` — 200 traces (4B, archived)
- `data/d2_pilot/{mc_results,claims,perception_results}.jsonl` — 100 sample D2 (27B)
- `data/d2_pilot_4b9b/...` — 100 sample D2 (4B+9B, archived)
- `data/d2_pilot/{mc_validate,claim_validation,alignment}.json` — 자동 metric summary

**Scripts (모든 sampling/spec 박힘, reproduce용)**:
- `scripts/d1_generate_traces.py` — single-source SAMPLING dict
- `scripts/d1_segment.py` — target_min/max 인자
- `scripts/d1_auto_validate.py` — 6-check
- `scripts/d2_mc_rollout.py` — K=8 continuation via /v1/completions
- `scripts/d2_claim_extract.py` — regex tier1 + LLM fallback (LLM gather fixed parallel)
- `scripts/d2_perception_verify.py` — image re-query
- `scripts/d2_alignment.py` — 4-metric (sklearn LogisticRegression for lift)

**vLLM launch (TP=4)**:
- `scripts/launch_27b_vllm.sh <port> <gpu_csv> <max_len>` — Qwen3.6-27B
- `scripts/launch_4b_vllm.sh`, `launch_9b_vllm.sh` — 작은 모델 변종

**Reports**:
- `docs/d2_report.md` — 27B alignment summary
- `docs/d2_report_4b9b.md` — 4B+9B alignment summary

---

## 9. 멘토에게 묻는 사항

1. **Spec D (4-cell metric)이 NeurIPS-class reviewer 관점에서 acceptable한가?** Outcome lift framing 폐기는 위험할 수 있음. paper §3 narrative 재구성 시 reviewer attack vector?
2. **Drift rate metric 단독으로 충분한 contribution인가**, 아니면 outcome lift도 weak 보조라도 보여줘야 하는가?
3. **Spec A의 LLM-primary extractor에서 self-consistency 문제** — claim extraction LLM과 perception verifier LLM이 동일 (27B) 시 circular bias. cross-family (예: Llama 또는 GPT-4o-mini) 필요?
4. **Pilot N=100 → Week 2 N=5K scaling 시 drift rate distribution이 유지될 거라는 가정** — 작은 N에서 측정한 4-cell 비율이 statistical 안정성 부족할 가능성. pilot N 확대 권고?

추가 분석 또는 Spec 우선순위 조정 의견 환영합니다.

---

_End of failure analysis brief_
