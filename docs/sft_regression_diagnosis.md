# SFT 성능 하락 종합 진단 리포트

**Date**: 2026-04-11
**Scope**: v8 SFT → v9 → v9.1 → v_hq-lite 전 버전 비교 분석

---

## 1. Executive Summary

v_hq-lite SFT (avg 51.59%)가 v8 SFT (56.05%) 대비 -4.46pp, 역대 best (v8 Process 56.91%) 대비 -5.32pp 하락한 근본 원인을 진단했다.

**핵심 결론**: v_hq-lite 모델은 **reasoning을 하지 않는다**. chartqa_human에서 99.0%의 출력이 "Let me analyze the chart to answer this question. The answer is X." 템플릿이다. 차트를 step-by-step으로 분석하지 않고 답을 추측하기 때문에, numeric extraction 정확도가 86.1% → 72.0%로 추락했다.

---

## 2. 전 버전 성능 비교

| Model | human | augmented | charxiv | pro | museum | **AVG** |
|---|---|---|---|---|---|---|
| v7 zero-shot | 77.84 | 86.32 | 45.40 | 36.09 | 29.90 | **55.11** |
| **v8 SFT** | **83.68** | 88.64 | 44.50 | 35.83 | 27.60 | **56.05** |
| v8 Outcome | 84.48 | 88.96 | 46.00 | 36.40 | 28.30 | **56.83** |
| v8 Process | 83.76 | 88.96 | 45.30 | 36.14 | 30.40 | **56.91** |
| **v9 SFT** | **82.32** | **91.36** | 42.10 | 35.11 | 27.30 | **55.64** |
| v9.1 SFT | 81.76 | 87.92 | 39.50 | 35.99 | 26.40 | **54.31** |
| **v_hq-lite** | **71.20** | **94.16** | 40.80 | 27.98 | 23.80 | **51.59** |

**추세**: human은 83.68 → 82.32 → 81.76 → 71.20으로 지속 하락. augmented만 94.16으로 상승.

---

## 3. Root Cause #1: Reasoning Collapse (CoT 붕괴)

### 3.1 모델 출력의 reasoning 깊이

| Model | reasoning 중앙값 | template 출력% | chartqa_human acc |
|---|---|---|---|
| v8 SFT | 754 chars | **0.0%** | 83.68% |
| v9 SFT | 1377 chars | **0.0%** | 82.32% |
| v_hq-lite | **82 chars** | **99.0%** | 71.20% |

v8과 v9는 실제로 차트를 분석하는 reasoning을 생성한다. v_hq-lite는 99%가 아래 템플릿:
```
Let me analyze the chart to answer this question.
The answer is X.
```

### 3.2 Reasoning 유무에 따른 정확도 (v_hq-lite)

| Benchmark | template% | template acc | real reasoning acc |
|---|---|---|---|
| chartqa_human | 98.8% | 71.3% | 66.7% (n=15) |
| chartqa_augmented | 100.0% | 94.2% | — |
| charxiv_reasoning | 26.0% | 38.1% | 41.8% |
| chartqa_pro | 73.4% | 24.2% | **38.4%** |

chartqa_pro에서 real reasoning이 있을 때 38.4% vs template 24.2% → reasoning이 14.2pp 차이.

### 3.3 Regression 분석 (v8✓ → hq✗)

chartqa_human에서 **208개 regression** (v8 맞고 hq 틀림) vs 52개 improvement:
- numeric 오답: **72.6%** (151건) — 차트 값을 잘못 읽음
- text 오답: 15.9% (33건)
- yes/no 반전: 11.5% (24건)

**v8 reasoning (정답):**
```
1. Identify "Lamb" value: 103.7
2. Identify "Corn" value: 103.13
3. Difference = |103.7 - 103.13| = 0.57
```

**hq-lite (오답):**
```
Let me analyze the chart to answer this question.
The answer is 0.02.
```

---

## 4. Root Cause #2: Training Data의 Reasoning 품질 문제

### 4.1 학습 데이터 reasoning 커버리지

| Dataset | Total | Has Reasoning | No Reasoning (template) | Avg Reasoning Length |
|---|---|---|---|---|
| **v8 sft_30k** | 30,000 | 12,000 (40%) | 18,000 (60%) | 126 chars |
| **v9 sft** | 41,940 | **41,940 (100%)** | **0 (0%)** | — |
| **v_hq-lite** | 43,242 | 25,242 (58%) | 18,000 (42%) | 204 chars |

**공통점**: v8과 hq-lite 모두 동일한 18,000 chartqa_train (reasoning 없음) 포함.

### 4.2 패러독스: v8도 60% template인데 왜 reasoning 출력?

v8 SFT 모델은 60% template 데이터로 학습했음에도 0% template 출력. **이유**:
- v8은 30K samples × 1 epoch = pretrained reasoning 능력 보존
- v_hq-lite는 43K samples × 1 epoch = 43% 더 많은 학습 → pretrained behavior 더 많이 overwrite
- v9는 42K samples이지만 100% reasoning → template signal 없음 → 0% template 출력

**핵심 차이: template signal 밀도 (samples per think-length)**

| Dataset | Template Exposure | Avg Think Length | 결과 |
|---|---|---|---|
| v8 | 18K × ~80 chars | 98 chars | Pretrained reasoning 보존 ✓ |
| v9 | 0 template | 모두 reasoning | Reasoning 학습 ✓ |
| v_hq-lite | 18K × ~80 chars + 25K × ~204 chars | **152 chars** | Reasoning 붕괴 ✗ |

v_hq-lite의 25K reasoning (avg 204 chars)이 template (80 chars)과 길이 차이가 겨우 2.5배 → 모델이 "reasoning = 짧게"로 학습 → mode collapse.

### 4.3 v_hq-lite 추가 데이터의 특성

hq-lite는 v8 대비 **13K 추가 데이터**:
- `plotly_complex` (3,010, NEW): multichoice 37%, hypothetical 질문 — 벤치마크와 무관
- `scientific_ext` (4,859, NEW): 90% numeric, reasoning 346 chars — 품질 양호
- `owid` (+1,734): reasoning 209 chars
- `scientific` (+2,082): reasoning 147 chars
- `synthetic` (+1,468): reasoning 142 chars

**문제**: 추가 데이터가 **단순 수치 연산** 편중 (How much? Compute difference). chartqa_human의 핵심 question types (visual reasoning, complex comparison, average/median 계산) 와 distribution mismatch.

---

## 5. Root Cause #3: Train-Eval Distribution Gap

### 5.1 Question Type 분포 불일치

| Question Type | hq-lite 학습 | chartqa_human eval | GAP |
|---|---|---|---|
| which/who | 13.4% | **23.8%** | -10.4pp ↓ |
| other (complex) | 25.7% | **32.6%** | -6.9pp ↓ |
| how_many | 17.6% | 12.5% | +5.1pp |
| difference | 13.2% | 9.1% | +4.1pp |
| what_value | 10.3% | — | — |

학습 데이터에 **"which" 질문 (entity 식별)이 부족**하고, **complex/interpretive 질문이 부족**.

### 5.2 Answer Type 분포 불일치

| Answer Type | hq-lite 학습 | chartqa_human eval | chartqa_pro eval |
|---|---|---|---|
| Numeric | **77%** | 66.8% | 37.1% |
| Text | **13%** | 24.9% | **59.9%** |
| Yes/No | **3.8%** | 8.3% | 3.0% |

학습 데이터의 **text 답변이 13%인데, chartqa_pro는 60%가 text**. chartqa_pro에서 text 정확도 17.7%의 원인.

### 5.3 Per-type Accuracy 비교 (chartqa_human)

| Answer Type | v8 SFT | v9 SFT | v_hq-lite | Regression |
|---|---|---|---|---|
| Numeric | **86.1%** | 86.2% | 72.0% | **-14.1pp** |
| Text | 74.6% | 70.7% | 69.1% | -5.5pp |
| Yes/No | **91.3%** | 85.6% | 71.2% | **-20.1pp** |

**Numeric 정확도가 가장 큰 하락** — reasoning 없이 차트 값을 정확하게 읽을 수 없음.
**Yes/No 하락도 심각** — 비교/판단 논리가 reasoning 없이 불가능.

---

## 6. Root Cause #4: augmented만 향상된 이유

chartqa_augmented (94.16%, +5.52pp) 가 유일한 향상 벤치마크인 이유:
- augmented는 86.4%가 numeric, 32.9%가 "What is the value of X?" 타입
- 학습 데이터의 numeric 77% 비율과 가장 일치
- **단순 값 읽기는 reasoning 없이도 가능** → template 출력으로도 정답 가능
- v8이 augmented에서 틀렸던 88개 중 상당수가 extraction artifact (모델이 reasoning을 답으로 출력)

---

## 7. 전 버전 하락 추세 분석

### v8 → v9 (-0.41pp): 경미한 하락
- v9는 100% reasoning 커버리지로 template 문제 없음
- 하지만 v9 데이터의 block 구조가 chartqa_human과 다른 분포
- charxiv가 44.5 → 42.1 (block_c scientific data의 노이즈)

### v9 → v9.1 (-1.33pp): 중간 하락
- v9.1은 Block C 재생성 (sci_ranking/numeric/trend/compare)
- Runaway thinking 문제 시작 (charxiv empty 44.8%)
- human 82.32 → 81.76: 데이터 변경보다 runaway의 간접 영향

### v9.1 → v_hq-lite (-2.72pp): 급격한 하락
- Reasoning collapse가 주원인
- 18K chartqa_train template + 25K 짧은 reasoning → mode collapse
- Runaway는 해결 (empty 0-2.4%) 되었지만 reasoning 자체가 사라짐

---

## 8. SOTA 달성 전략

### 8.1 핵심 원칙

1. **Reasoning은 필수**: 100% reasoning coverage → v9의 교훈
2. **Reasoning 품질**: 평균 200자로는 부족. 최소 500자+ (v8 모델 출력 중앙값 754자)
3. **Distribution alignment**: 학습 데이터가 eval 벤치마크의 question/answer type 분포를 커버해야
4. **Template 데이터 근절**: chartqa_train 18K를 reasoning 없이 포함 금지

### 8.2 Option A: v9 SFT + Teacher Distill 혼합 (추천)

**근거**: v9 SFT가 현재까지 최고 SFT 성능 (human 82.32%, avg 55.64%)

1. **Base**: v9 sft_v9.jsonl (41,940, 100% reasoning) 를 그대로 사용
2. **chartqa_train 18K 보강**: 397B teacher에게 reasoning 생성 → chartqa_train에 고품질 reasoning 주입
3. **Teacher distilled data 혼합**: 현재 축적 중인 teacher distill (8.7K) 에서 quality gate 통과한 것만 추가
4. **Target**: 42K base + 최대 18K teacher reasoning → ~50K with 100% reasoning

**예상 효과**: v9 수준(82.3% human) 복원 + teacher reasoning으로 charxiv/pro 개선

### 8.3 Option B: chartqa_train Teacher Reasoning 생성 후 v8 재학습

1. 18K chartqa_train에 대해 397B teacher로 reasoning 생성 (chart image + Q + A → CoT)
2. v8 sft_30k에서 template 18K를 teacher reasoning 버전으로 교체
3. 30K 전부 reasoning 커버리지 → v8의 distribution 장점 유지

**예상 효과**: v8 수준(83.7% human) 유지 + reasoning 품질 향상

### 8.4 Option C: 통합 최적 데이터셋 구축 (High-effort, Highest potential)

1. **Core**: chartqa_train 18K + teacher reasoning (필수)
2. **Supplement**: v9 block 데이터 중 quality gate 통과분 (~20K)
3. **Teacher distill**: 397B thinking mode 고품질 reasoning (~5-10K)
4. **Distribution balancing**:
   - Text answer 비율 25%+ (현재 13%)
   - "Which" question 비율 20%+ (현재 13.4%)
   - Complex reasoning 질문 추가 (trend, comparison, inference)
5. **Quality gate**: reasoning 최소 200자 (현재 중앙값 185자 → 불충분)
6. **Target**: ~45K, 100% reasoning, balanced distribution

### 8.5 GRPO 전략 (SFT 이후)

SFT가 충분히 강한 base를 만들면:
1. v8 Process reward (avg 56.91%)의 성공 재현
2. GRPO는 SFT base의 reasoning 능력을 증폭 — SFT가 template이면 GRPO도 실패
3. **SFT avg 56+ → GRPO로 58-60+ 가능** (v8에서 SFT 56.05 → Process 56.91 달성)

---

## 9. 즉시 실행 Action Items

### Phase 1: chartqa_train 18K Teacher Reasoning (최우선)
```
# 397B teacher에게 18K chartqa_train의 reasoning 생성
# Input: chart image + question + gold answer
# Output: step-by-step reasoning (500+ chars target)
# 예상 시간: 18K × ~2 tok/s → ~3-5h (dual host)
```

### Phase 2: 데이터셋 조합 실험
- **Row A**: v9 sft (42K, 100% reasoning) — baseline 재확인
- **Row B**: v8 sft (30K) + chartqa_train teacher reasoning 교체
- **Row C**: v9 + teacher distilled supplement
- 3개 모두 SFT → eval → 최고 성능 base 선택

### Phase 3: GRPO
- 최고 SFT base로 Process reward GRPO
- v8의 성공 패턴 (SFT → GRPO +0.86pp) 재현

---

## 10. Files & Data

- 학습 데이터: `data/sft_30k.jsonl`, `data/sft_v9.jsonl`, `data/sft_hq_lite_clean.jsonl`
- 평가 결과: `results/v8/row2_sft/`, `results/v9/sft_v9_4b/`, `results/v_hq_lite/`
- Teacher distill: `data/sft_hq_teacher.jsonl` (8,686 samples)
- 이 리포트: `docs/sft_regression_diagnosis.md`
