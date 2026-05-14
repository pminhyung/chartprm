> **SUPERSEDED 2026-04-12** → see `docs/sft_v8_improved_action_guide_v3.md` for the authoritative execution plan. This v2 plan remains for historical context (pool analysis, GOLD subset identification, regression error breakdown).

# ChartVCR v8_improved SFT 데이터 재구축 Plan (v2 — 에러 패턴 targeting)

## Context

`docs/sft_regression_diagnosis.md`에서 v_hq-lite SFT 실패의 근본 원인이 규명되었다 (reasoning collapse, 99% template 출력). 유저 가이드 `v8_improved_balanced` 30K 5-block 구조를 따르되:

1. **개수/비율은 가이드 고정**: Block 1=10K(33%), 2=8K(27%), 3=6K(20%), 4=4K(13%), 5=2K(7%)
2. **선별 방법은 data-driven**으로 더 정확·효율적
3. **이전 모델(v8/v9/v9.1) regression 패턴을 명시적으로 target**해서 SOTA 상회

핵심 hard constraints:
- 30K, 100% reasoning, chartqa≤35%, reasoning median ≥300, text 답변 ≥20%(목표 30%), Y/N ≥5%

---

## 핵심 발견 #1: Teacher Pool 자산 (실측)

기존 distill 누적 자산 inventory (sft_hq_teacher.jsonl + sft_v8_improved_teacher_distill.jsonl, deduped):

| Source | Count | text% | which% | text+which% |
|---|---|---|---|---|
| chartqa_train | **13,367** | 17.6 | 10.9 | **8.3** |
| kaggle_like | **2,082** | 29.4 | 29.4 | **29.4** ⭐ |
| owid | 2,395 | 23.5 | 24.1 | 23.5 |
| synthetic | 2,262 | 18.9 | 21.9 | 18.4 |
| scientific | 2,069 | 24.3 | 10.6 | 0 |
| plotly_complex | 1,235 | 4.0 | 9.6 | 4.0 |
| scientific_ext | 1,204 | 27.4 | 2.4 | 2.2 |
| additional | 1,032 | 23.2 | 30.0 | **23.2** ⭐ |
| worldbank | 437 | 28.2 | 28.2 | 27.2 |
| **Total deduped** | **26,083** | — | — | — |

**모두 reasoning median 1,618 chars (100% ≥300 char gate 통과)**

---

## 핵심 발견 #2: 이전 모델 Regression 패턴 (실측)

v8 ✓ → v9/v9.1 ✗ regression (우리가 회복해야 할 cases):

| Benchmark | Regressions | Text 답변% | which/who% | 대표 에러 |
|---|---|---|---|---|
| chartqa_human | 13 | 38% | 8% | small numeric off |
| **charxiv_reasoning** | **67** | **45%** | **27%** | text entity 식별 |
| **chartqa_pro** | **76** | **59%** | 16% | percentage/average |
| **chartmuseum** | **50** | **74%** | **36%** | text entity 식별 |

### Universally hard samples (3개 SFT 모델 모두 틀림)

| Benchmark | Hard | Text 답변% | which/who% |
|---|---|---|---|
| chartqa_human | 154 | 45% | 18% |
| charxiv_reasoning | 423 | **60%** | **23%** |
| chartqa_pro | 1029 | **73%** | 8% (percentage 17%, average 15%) |
| chartmuseum | 587 | **75%** | **35%** |

### chartqa_human 154 hard samples error pattern 분석

| Error type | Count | % |
|---|---|---|
| wrong_entity (text 답변, 잘못 선택) | 86 | 56% |
| partial_text (text 답변 truncated) | 48 | 31% |
| small_off_by_few (numeric ±1-2) | 14 | 9% |
| decimal_scale_off (60 vs 0.6) | 7 | 5% |

### 결론: 데이터가 target해야 할 능력

1. **Text answer 능력** (regression 60-74%, hard 60-75% — 압도적 1위)
2. **Which/Who entity 식별** (regression 16-36%)
3. **Percentage / decimal scale 처리** (CQA-Pro 17%, 흔한 error)
4. **Average/median 계산** (CQA-Pro 15%)
5. **Extreme finding with full text label** (CharXiv 14%, ChartMuseum 13%)

---

## 핵심 발견 #3: GOLD Subset (text+which+reasoning≥300)

이전 모델 regression을 직접적으로 보완할 수 있는 "이상적" sample 4,147개 식별:

| Source | Gold count |
|---|---|
| chartqa_train teacher | 1,056 |
| plotly_complex (teacher+self ≥300) | 936 |
| owid (teacher+self ≥300) | 656 |
| kaggle_like teacher | **622** ⭐ |
| synthetic | 425 |
| additional | **240** ⭐ |
| worldbank | 129 |
| scientific_ext | 82 |

⭐ kaggle_like / additional은 새로 발견된 high-value source. 이전 plan에서 underused.

---

## 수정된 Block 할당 (regression-targeted)

### 핵심 변경 사항 vs v1 plan

1. **Block 1**: 단순 question type quota → **text+which gold subset 우선** + question type quota 보조
2. **Block 3**: plotly_complex 단독 → **plotly_complex + kaggle_like + additional teacher** 통합 ("Complex/Diverse"의 "Diverse"에 kaggle/additional 포함)
3. **Block 5**: 단순 gap fill → **regression pattern targeted** (100% text 또는 100% which 우선)

### Block 1: ChartQA-train (10K, 33%) — text+which 우선 selection

**Pool**: chartqa_train teacher 13,367

**Selection algorithm**:
```
Phase 1 (priority): text+which gold subset 1,056 → 전부 포함
Phase 2: text 답변 (text+which 제외) ~1,170 → 전부 포함  
Phase 3: which 질문 (text+which 제외) ~327 → 전부 포함
Phase 4 (남은 ~7,447 quota): question type quota 균형
  - multi_step / arithmetic: 3,000 (CQA-Pro percentage/average 대응)
  - extreme (highest/lowest): 1,500 (CharXiv 14% 대응)
  - counting: 1,000
  - comparison/difference: 1,500
  - 기타: 447 (other_complex)
Phase 5: 각 bucket 내 reasoning 길이 내림차순 → 상위 quota 선별
```

**예상 결과**: 10,000 samples
- text 답변: ~2,500 (25%) ⭐
- which/who: ~1,400 (14%)
- 100% teacher reasoning, median ~1,600 chars

### Block 2: Scientific Charts (8K, 27%) — CharXiv 도메인 강화

**Pool**:
- Teacher: scientific_ext (1,204) + scientific (2,069) = 3,273
- Self-gen ≥300 chars: scientific_ext 3,040 + scientific 252 = 3,292
- Filler: block_c v9 (4,905, median 73)

**Selection algorithm (regression-targeted)**:
```
Phase 1: scientific_ext teacher 1,204 전부 (CharXiv style match)
Phase 2: scientific teacher 2,069 전부 (text 답변 24.3%)
Phase 3: scientific_ext self-gen ≥300 3,040 → text 답변 우선 정렬 → 전부
Phase 4: scientific self-gen ≥300 252 → 전부
   = 6,565 누적
Phase 5: 부족 1,435 → block_c v9에서:
   - which/extreme 질문 우선
   - 짧지만 reasoning 있는 것 우선 (template 차단: "Let me analyze" 패턴 reject)
```

**예상 결과**: 8,000 samples
- text 답변: ~2,000 (25%)
- 6,565 ≥300 chars (82%), 100% reasoning

### Block 3: Complex/Diverse Charts (6K, 20%) — kaggle/additional 통합

**Pool**:
- Teacher: plotly_complex (1,235) + **kaggle_like (2,082)** + **additional (1,032)** = 4,349
- Self-gen ≥300: plotly_complex 716
- Filler: block_d 5,929 + block_f 4,000 (text-rich)

**근거**: kaggle_like와 additional은 chart 다양성 + 가장 높은 text+which 비율 (29% / 23%)을 가진 자산이지만 v1 plan에서 underuse 되었음. "Complex/Diverse"의 "Diverse" 영역으로 재해석.

**Selection algorithm**:
```
Phase 1: plotly_complex teacher 1,235 전부
Phase 2: kaggle_like teacher 2,082 전부 (text 29.4%, which 29.4% — best for regression)
Phase 3: additional teacher 1,032 전부 (which 30%)
Phase 4: plotly_complex self-gen ≥300 716 → text 우선
   = 5,065 누적
Phase 5: 부족 935 → block_d/f v9에서:
   - text 답변 우선 (block_f text 74.5%, block_d text 17%)
   - which 질문 우선
   - reasoning ≥40 chars (template 차단)
```

**예상 결과**: 6,000 samples
- text 답변: ~2,400 (40%) ⭐
- which/who: ~1,500 (25%) ⭐
- 5,065 ≥300 chars (84%), 100% reasoning

### Block 4: OWID/Synth/WB (4K, 13%) — 검증된 GT

**Pool**: 
- Teacher: owid 2,395 + synthetic 2,262 + worldbank 437 = 5,094
- 검증 GT: `chartvr_train_final.jsonl` 3,916 (모두 reasoning, GRPO 호환)

**Selection algorithm**:
```
Phase 1: 가이드 비율 (owid 2K + synth 1.5K + wb 0.5K) 적용
Phase 2: 각 source 내에서 text+which 우선 정렬 → 상위 quota
Phase 3: chartvr_train_final.jsonl과 dedup
Phase 4: 부족 시 chartvr_train_final.jsonl에서 보충 (이중 검증 데이터)
```

**예상 결과**: 4,000 samples
- text 답변: ~960 (24%)
- which/who: ~1,000 (25%)
- 100% teacher reasoning, GRPO 호환 (CSV 동반)

### Block 5: Balance 보충 (2K, 7%) — Regression Targeting

**2-pass process**:

**Pass 1**: Block 1-4 (28K) 조립 후 정량 측정:
```python
- text 답변% (목표 30%, 미달 시 gap 계산)
- which/who 질문% (목표 15%)
- yes_no 질문% (목표 8%)  
- percentage 질문% (CQA-Pro 17% 대응)
- average/median 질문% (CQA-Pro 15% 대응)
```

**Pass 2**: Gap targeting (regression 패턴 기반)
```
Pool (Block 1-4에서 사용 안 한 자산):
  - GOLD subset 잔량 (text+which+reasoning≥300): 위 4,147 - Block 사용분
  - Teacher 잔량: kaggle/additional 잔량, sci_ext text 잔량
  - Self-gen ≥300 잔량: plotly/owid/sci_ext text answer

Selection priority (gap에 따라 동적 조정):
  1. text 답변 + which 질문 (regression 가장 큰 패턴)
  2. text 답변 + percentage/average 질문 (CQA-Pro 32% 대응)
  3. text 답변 + extreme 질문 (CharXiv/Museum 대응)
  4. yes_no 답변 (Y/N 5% 충족)
```

**예상 결과**: 2,000 samples, ~80% text 답변, ~50% which 질문 → 전체 30K text 답변 30%+ 보장

---

## Final 30K 예상 분포

| 항목 | Block 1 | Block 2 | Block 3 | Block 4 | Block 5 | Total |
|---|---|---|---|---|---|---|
| Samples | 10,000 | 8,000 | 6,000 | 4,000 | 2,000 | **30,000** |
| Teacher reasoning | 10,000 | 3,273 | 4,349 | 4,000 | ~1,200 | **22,822 (76%)** |
| Self-gen ≥300 | 0 | 3,292 | 716 | 0 | ~600 | **4,608 (15%)** |
| Self-gen <300 (filler) | 0 | 1,435 | 935 | 0 | ~200 | **2,570 (9%)** |
| **≥300 chars** | 10,000 | 6,565 | 5,065 | 4,000 | 1,800 | **27,430 (91%)** |
| text 답변 | ~2,500 | ~2,000 | ~2,400 | ~960 | ~1,600 | **~9,460 (32%)** ✓ |
| which/who | ~1,400 | ~600 | ~1,500 | ~1,000 | ~1,000 | **~5,500 (18%)** ✓ |
| Y/N 답변 | ~200 | ~50 | ~300 | ~400 | ~600 | **~1,550 (5%)** ✓ |

**ChartQA 비율**: 33% ✓ | **Reasoning coverage**: 100% ✓ | **신규 distill**: 0 ✓

---

## 구현할 스크립트

### `scripts/assemble_v8_improved_v2.py` (NEW)

기존 `assemble_v8_improved.py`(v1) 폐기하고 새로 작성. 핵심 함수:

```python
# 데이터 로딩
def load_all_pools():
    """teacher (2 files merged), v8 rest, v9, hq-lite, chartvr_train_final"""

# 정규화 + 분류
def dedup_key(s): ...        # image_path 정규화 + question[:80]
def classify_question(q): ... # 가이드 8 카테고리
def classify_answer(a): ...   # numeric/text/yes_no
def is_text(a), is_which(q), is_percentage(q), is_average(q): ...

# Quality gates
def passes_quality_gate(s):
    """reasoning 길이 ≥40 + template 패턴 차단 + <think> 또는 reasoning_steps 존재"""
def reasoning_length(s): ...

# Block selection (regression-targeted)
def select_block1_chartqa(teacher_pool, target=10000):
    """Phase 1: text+which gold (1056) → Phase 2: text → Phase 3: which → Phase 4: quota"""

def select_block2_scientific(teacher_pool, selfgen_300, filler_pool, target=8000):
    """sci_ext teacher → scientific teacher → sci_ext self-gen ≥300 → block_c filler"""

def select_block3_complex(teacher_pool, selfgen_300, filler_pool, target=6000):
    """plotly + kaggle + additional teacher → plotly self-gen ≥300 → block_d/f text-rich filler"""

def select_block4_owid_synth_wb(teacher_pool, gt_pool, target=4000):
    """owid 2K + synth 1.5K + wb 0.5K from teacher, prioritize text+which"""

def select_block5_balance(blocks_1_4, remaining_pool, target=2000):
    """2-pass: measure gaps → fill with regression-targeted samples"""

# Verification
def verify_dataset(final):
    """CHECK 1-9: 규모, reasoning coverage, median ≥300, template <5%,
    text ≥30%, which ≥15%, Y/N ≥5%, chartqa ≤35%, image exists, answer match"""

# Output format
def convert_to_messages_format(s):
    """train_sft.py 호환 messages 필드 (system + user + assistant with <think>/<answer>)"""
```

### `train_sft.py` line 73-76 수정 (안전장치 추가)

```python
# OLD: template fallback
if reasoning:
    response = f"<think>\n{reasoning}\n</think>\n<answer>{ans}</answer>"
else:
    response = f"<think>\nLet me analyze...\n</think>\n<answer>{ans}</answer>"

# NEW: fail-fast (template 사고 방지)
if not reasoning and not assistant_text:
    raise ValueError(f"Sample missing reasoning, template fallback disabled. Q: {q[:80]}")
```

---

## 검증 프로토콜 (CHECK 1-9 + Regression Coverage Check)

### Standard CHECKs

1. 규모 28K-32K
2. 100% reasoning coverage (no template)
3. Reasoning median ≥300 chars
4. Template-shaped (`Let me analyze.*answer is`) <5%
5. Text 답변 ≥20% (목표 30%)
6. Y/N 답변 ≥5%
7. chartqa_train ≤35%
8. 이미지 50개 랜덤 존재 확인
9. `<answer>` 내용이 gold와 ≥98% relaxed match

### 추가 CHECK 10: Regression Coverage

이전 모델 regression patterns에 대한 학습 신호 충분성 검증:

```python
def check_regression_coverage(final):
    text_count = sum(1 for s in final if is_text(s['answer']))
    which_count = sum(1 for s in final if is_which(s['question']))
    text_and_which = sum(1 for s in final if is_text(s['answer']) and is_which(s['question']))
    percentage_count = sum(1 for s in final if is_percentage(s['question']))
    average_count = sum(1 for s in final if is_average(s['question']))
    
    # Regression target coverage
    assert text_count >= 7500,    f"text 답변 {text_count} <7500 (25%)"
    assert which_count >= 4500,   f"which 질문 {which_count} <4500 (15%)"
    assert text_and_which >= 2500,f"text+which 교집합 {text_and_which} <2500 (8%)"
    assert percentage_count >= 1500, f"percentage 질문 부족"
    assert average_count >= 1500,    f"average 질문 부족"
```

### 추가 CHECK 11: Source Distribution Sanity

```python
src_dist = Counter(s['source'] for s in final)
# kaggle_like teacher 사용량 확인 (regression target source)
assert src_dist['kaggle_like'] >= 1500, "kaggle_like underused"
# additional teacher 사용량 확인
assert src_dist['additional'] >= 800, "additional underused"
```

---

## 실행 순서 + 타임라인

```
Day 1 (1.5h):
  ├── [40min] scripts/assemble_v8_improved_v2.py 작성 (regression-targeted selection)
  ├── [10min] dry-run으로 분포 확인 (--dry_run --stats)
  ├── [10min] CHECK 1-11 통과 확인
  ├── [10min] 실제 assemble 실행 → data/sft_v8_improved.jsonl
  └── [20min] 최종 검증 + 분포 리포트 작성

Day 1 추가 (5h):
  └── train_sft.py + LoRA r=64 학습 (8 GPU × 1 epoch)

Day 2 (1-2h):
  ├── 8 GPU vLLM 서버 기동 → eval_multi_server.py
  └── 5-bench 결과 → results/sft_v8_improved/ + 분석

Total: ~8h, 신규 API $0
```

---

## 성공 기준 + 실패 시 대응

| Benchmark | 최소 | 목표 | 실패 |
|---|---|---|---|
| CQA-H | 83.0 | 85+ | <80 |
| CQA-A | 89.0 | 91+ | <86 |
| CharXiv | 45.0 | **49+** | <43 |
| CQA-Pro | 36.0 | **40+** | <35 |
| ChartMus | 29.0 | **34+** | <27 |
| **AVG** | **56.4** | **61+** | **<54** |

CharXiv/Pro/Museum 목표를 가이드보다 1-2pp 높게 잡은 이유: 이번 데이터가 regression pattern (text/which)을 명시적으로 target하므로 charxiv/pro/museum에서 추가 향상 기대.

**실패 시 진단**:
- text 답변 정확도가 안 오르면 → kaggle_like/additional teacher 비중을 더 늘려 재학습
- which/who 정확도가 안 오르면 → Block 5 quota 확대 (2K→3K)
- numeric만 잘 되면 → percentage/average filler 부족, Block 1 quota 재조정
- 전체 하락 → 25K 축소 시도 (overfit 방지)

---

## Critical Files

**Read-only (참고)**:
- `data/sft_hq_teacher.jsonl` (9,526 samples, teacher pool 1)
- `data/sft_v8_improved_teacher_distill.jsonl` (4,156 samples, teacher pool 2 — 신규)
- `data/sft_30k.jsonl` (v8 base)
- `data/sft_v9.jsonl` (v9 supplement)
- `data/sft_hq_lite_clean.jsonl` (hq-lite extra)
- `data/charts_v2/chartvr_train_final.jsonl` (Block 4 GT)
- `results/v8/row2_sft/` `results/v9/sft_v9_4b/` `results/v9_1/sft_v9_1_4b/` (regression 분석용)

**기존 활용**:
- `chartvr/extraction.py::extract_answer, relaxed_accuracy` (검증용)
- `chartvr/prompts.py::EVAL_SYSTEM_PROMPT` (messages 변환용)

**작성**:
- `scripts/assemble_v8_improved_v2.py` (NEW, 메인)
- `data/sft_v8_improved.jsonl` (output)

**수정** (안전장치):
- `train_sft.py` line 73-76 (template fallback → fail-fast)

**Cleanup**:
- `scripts/assemble_v8_improved.py` (v1 deprecated)
- `data/sft_v8_improved_need_teacher.jsonl` (no longer needed)

---

## 가이드 대비 핵심 차이 (data-driven 최적화)

| 가이드 (pseudocode) | 본 plan (data-driven) |
|---|---|
| Block 1: question type quota만 고정 | **Adaptive quota + text+which gold subset 우선** (1,056 직접 식별) |
| Block 1: 18K → 신규 distill | **이미 보유한 teacher 13,367에서 직접 선별** ($0) |
| Block 2: 부족 시 새 차트 생성 (matplotlib) | **scientific_ext/scientific teacher + self-gen ≥300 6,565 + block_c filler** ($0) |
| Block 3: plotly_complex 단독 | **plotly + kaggle_like + additional teacher 통합** (text+which 비율 최고 source 활용) |
| Block 4: 부족 시 teacher 생성 | **teacher 5,094 + chartvr_train_final 3,916으로 충분** ($0) |
| Block 5: 단순 distribution gap fill | **이전 모델 regression pattern (text/which/percentage/average) 명시 target** |
| Quality gate: reasoning 길이만 | **reasoning + template 패턴 차단 + tag 검증 + answer 일치 4중 검증** |
| 22h, $60-80 | **2h, $0** (assemble + verify) |

**Regression-targeting의 의의**: 
- 단순히 v_hq-lite 실패를 복구하는 것이 아니라, **v8/v9/v9.1이 모두 틀렸던 1,300+ universally hard cases의 패턴**을 학습 데이터에 명시 반영.
- text/which 능력이 v8 SFT 대비 강해지면 CharXiv/CQA-Pro/ChartMuseum에서 v8 base 성능을 상회 가능.
- 이것이 단순 "30K 재구성"이 아닌 "**SOTA 상회를 위한 targeted 데이터 설계**"의 핵심.
