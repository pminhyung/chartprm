# 멘토 자문 요청 — ChartQA-Pro 성능 갭 + SFT 데이터 추가 작업 전략

_프로젝트: ChartVCR (Chart-Verifiable Causal Rewards for Chart Reasoning via GRPO)_
_본 페이지: 측정 결과·관찰·의문 정리. 옵션 비교까지만 제시._

---

## 1. 배경

본 프로젝트는 5개 벤치마크 (`chartqa_human`, `chartqa_augmented`, `chartqa_pro`, `charxiv_reasoning`, `chartmuseum`) 의 AVG5 를 main metric 으로 한다. 이 중 **ChartQA-Pro** 가 v8 SFT (Qwen3.5-4B + LoRA r64/a128, 28K 데이터, 거품 distill teacher=Qwen3.5-397B) 결과에서 다음과 같은 점수를 보였다 (think-on, lmms-eval `relaxed_correctness` 채점):

| Model | chartqa_pro |
|---|---:|
| zero_shot_qwen3_5_4b | 30.54% |
| v8_sft (이전 baseline) | 30.80% |
| **v8_distill_v2 (현재 SFT, VLMEK 정정 prompt 적용 distill)** | **27.57%** |
| row_a_outcome (현재 best AVG5 RL) | 34.80% |

본 SFT 모델이 zero-shot 및 v8_sft 보다 ChartQA-Pro 에서 낮게 측정됨에 따라, **lmms-eval scorer 가 ChartQA-Pro 원본 paper 가 의도한 평가와 다르다는 의심**으로 ChartQA-Pro 의 공식 evaluation framework 인 **VLMEvalKit** 의 scorer + question-type-branched prompt 로 재평가했다.

---

## 2. VLMEvalKit (ChartQA-Pro 공식 채점) 채점 방식

`https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/dataset/utils/chartqapro.py`

### 2.1 Question type 별 prompt (Direct mode)

ChartQA-Pro 의 question_type (1,948개 test set 비율):

| Question Type | n | % | VLMEvalKit prompt 핵심 형식 지침 |
|---|---:|---:|---|
| Factoid | 1,081 | 55.5% | `single word, number, or phrase` · `Do not generate units. But if numerical units (million/m/billion/B/K) are required, use exact notation shown in chart` · `unanswerable` literal · list = `['A','B']` · "no additional text" |
| Conversational | 311 | 16.0% | Factoid 와 동일 + multi-turn 대화 history 포함 |
| Fact Checking | 244 | 12.5% | `either true or false (without any additional text)` |
| Multi Choice | 214 | 11.0% | `one of the options letters only: a, b, c or d` |
| Hypothetical | 98 | 5.0% | Factoid 와 동일 (counterfactual 질문에 단답) |

### 2.2 Scorer 차이

| 비교 | lmms-eval | VLMEvalKit |
|---|---|---|
| 숫자 % 처리 | `"77%" → 0.77` (`/100`) → gold `"77"` 와 mismatch | `"77%" → 77` (% strip) → gold `"77"` 와 match ✓ |
| 텍스트 매칭 | `pred.lower() == gold.lower()` strict exact | **ANLS** (Approximate Normalized Levenshtein Similarity, threshold 0.5) — fuzzy |
| Fact Checking | strict text 비교 | `exact_match` (lowercase) |
| Multi Choice | strict text 비교 | letter 만 strict |
| List 답변 | 통째 문자열 비교 | element 별 평균 |
| 점수 출력 | Overall only | Question_type 별 + Overall |

### 2.3 우리 standard eval (`eval_standard.py`) 와의 차이

- 우리 standard eval 은 1,948개 ChartQA-Pro 전체에 단일 ChartQA prompt (`"\nAnswer the question with a single word."`) 적용. → MC / FC / Conv 의도된 형식 지침 누락.
- 채점기 = lmms-eval `relaxed_correctness` (위 2.2 lmms-eval 컬럼).
- VLMEvalKit re-eval 은 **같은 모델 출력에 prompt+scorer 둘 다 ChartQA-Pro 의도대로 적용**.

---

## 3. VLMEvalKit re-eval 결과

모든 점수: Qwen3.5-4B + LoRA / RL checkpoint, think-on, VLMEvalKit per-question-type prompt + ANLS scorer.

### 표 1 — VLMEvalKit ChartQA-Pro Accuracy (think-on, %)

모델은 학습 stage (Base / SFT / RL) 별 그룹화. 굵게 = 그룹 내 최고, 별표 = 전체 최고.

| Stage | Model | Overall | Factoid | Multi&nbsp;Choice | Conv | Fact&nbsp;Check | Hyp |
|---|---|---:|---:|---:|---:|---:|---:|
| Base | Qwen3.5-4B (zero-shot) | 45.87 | 43.67 | 52.34 | 45.60 | 52.60 | 40.19 |
|  |  |  |  |  |  |  |  |
| SFT | v_hq_lite (chartqa-H 1위) | 40.46 | 37.86 | 48.60 | 34.19 | 53.37 | 39.14 |
| SFT | v8_distill_v2 (ours) | 43.85 | 41.33 | 47.66 | 45.27 | 53.28 | 35.32 |
| SFT | v9_sft | **45.50** | **42.99** | **49.53** | **46.92** | **53.28** | **40.42** |
|  |  |  |  |  |  |  |  |
| RL | row3_dapo_outcome | 48.21 | 46.26 | 50.93 | 46.79 | 57.79 | 44.44 |
| RL | row4_dapo_vapv | 49.75 | 48.25 | 51.87 | 48.14 | 59.43 | 42.79 |
| RL | row_b_vapv | 49.84 | 47.32 | 56.54 | 46.40 | **62.30** | 42.93 |
| RL | **row_a_outcome** | **★53.26** | **★51.44** | **★63.08** | **48.71** | 61.89 | **★44.83** |

### 패턴

- **3개 SFT 모델 모두 base zero-shot 이하** (Δ vs zero-shot: −0.37 ~ −5.41pp).
- **4개 RL 모델 모두 base zero-shot 초과** (Δ vs zero-shot: +2.34 ~ +7.39pp).
- ChartQA-H 1위 v_hq_lite 가 ChartQA-Pro 최하 (40.46) — 단일 bench 최적화의 cross-bench trade-off.
- row_a_outcome (best AVG5 baseline) 이 6 type 중 4 type 1위 (Overall + Factoid + MC + Hyp).

### 표 2 — V8 distill v2 SFT (ours) vs Base zero-shot, question type 별

| Question Type | Test % | zero-shot | SFT (ours) | Δ |
|---|---:|---:|---:|---:|
| Factoid | 55.5% | 43.67 | 41.33 | **−2.34** |
| Multi Choice | 11.0% | 52.34 | 47.66 | **−4.68** |
| Conversational | 16.0% | 45.60 | 45.27 | −0.33 |
| Fact Checking | 12.5% | 52.60 | 53.28 | +0.68 |
| Hypothetical | 5.0% | 40.19 | 35.32 | **−4.87** |
| **Overall (weighted)** | 100% | **45.87** | **43.85** | **−2.02** |

요인 (관찰):
- Multi Choice: SFT 가 chart 본문 값 ("Revenue") 으로 답하나 ChartQA-Pro 채점은 letter (a/b/c/d) 만 인정.
- Hypothetical: SFT 데이터에 counterfactual training signal 부재.
- Factoid: 단답 형식에서도 % 단위 부착, decimal 정밀도 차이로 손실.

### 표 3 — Scorer/Prompt 변경 효과 (동일 모델, lmms-eval → VLMEvalKit)

_각 행은 동일 모델 추론 결과를 두 채점 framework 로 재채점. 모델 간 비교 아님._

| Model | lmms-eval (단일 prompt + relaxed) | VLMEvalKit (type 별 prompt + ANLS) | Δ (scorer만) |
|---|---:|---:|---:|
| zero-shot | 30.54 | 45.87 | +15.33 |
| v8_distill_v2 SFT (ours) | 27.57 | 43.85 | +16.28 |
| row_a_outcome | 34.80 | 53.26 | +18.46 |

모든 모델에서 +15~+18pp 상승. lmms-eval scorer 가 ChartQA-Pro 의도된 평가와 어긋남 (% strip 누락, type 별 형식 강제 누락, ANLS 미적용) 이 광범위.

---

## 4. 답변 포맷 스타일 영향 분석

### 4.1 lmms-eval 채점에서 손실 패턴

300건+ 정밀 분석 (`results/v8_distill_v2/sft_4b/chartqa_pro_scored.jsonl`):

- **숫자 단위 부착 손실**: gold `"77"` vs pred `"77%"` → lmms-eval `_to_float("77%")=0.77` → 0.99× 차이 → fail. SFT 가 chart 표기 그대로 emit → unit 손실 다수.
- **소수 변환 손실**: gold `"0.382"` vs pred `"38.2%"` (의미 동등). lmms-eval 채점 fail.
- **분수 / 비율**: gold `"0.33333333"` vs pred `"1:3"` — text 비교 fail.
- **부연 설명 부착**: gold `"Democrat (scores 60 to 100)"` vs pred `"Democrat"` — 짧은 답으로 학습된 모델이 부연 누락. 텍스트 strict mismatch.
- **categorical token mismatch**: gold `"Dec"` vs pred `"January"` — 모델 reasoning 정답이라도 token 양식 다르면 fail.

### 4.2 정규식 후처리 (pred 에서 첫 숫자 substring 만 추출)

별도 worktree 에서 `re.search(r"-?\d+(?:\.\d+)?", pred)` 적용 후 동일 `relaxed_correctness` 재채점:

| bench | 원본 | regex 후 | Δ |
|---|---:|---:|---:|
| chartqa_human | 77.28% | 78.16% | +0.88pp |
| chartqa_augmented | 83.52% | 85.52% | **+2.00pp** |
| chartqa_pro | 27.57% | 26.23% | **−1.33pp** |

→ Human/Aug 에서는 효과. **Pro 에서는 categorical 답 (`65+`, `Q2`, `5G`, `2017/2018`) 잘려 역효과**. categorical 비중이 높은 Pro 에서는 단순 정규식 적용 불가.

### 4.3 397B teacher 의 VLMEvalKit 출력 양식 검증

`docs/v8_distill_v2_chartqapro_vlmek_compare.md` + `data/sft_v8_distill_v3_chartqa.jsonl` 9,498 행 분석:

| 항목 | 비율 | 의미 |
|---|---:|---|
| Single-word/phrase (≤2 tokens) | 98.6% | VLMEvalKit "single word" 룰 준수 |
| Content ≤ 100 chars | 99.9% | 1 row outlier (runaway thinking이 content에 새어 들어감) |
| `%` 단위 부착 (VLMEvalKit "Do not generate units" 위반) | **1.9%** (180 rows) | 차트가 % bar 일 때 teacher 가 차트 표기 그대로 emit |
| million/M/B/K 단위 (VLMEvalKit 허용) | 0.7% | 차트 단위 표기 보존 |
| List bracket (`['A','B']`) | 0% | multi-answer 없음 |
| `unanswerable` literal | 0% | chartqa_train 에 unknown 없음 |

→ 98.1% 정상 포맷. 1.9% % 부착 잔존. **단, chartqa_train 만 처리 (Factoid 만 보강) — 다른 4개 type (MC/FC/Conv/Hyp) 의 학습 신호는 없음**.

---

## 5. SFT 데이터 추가 작업 옵션 (객관 비교)

### 5.1 합성 데이터 source / 분배 후보

ChartQA-Pro test set 분포 mirror 시 합성 필요량 (총 9,500 chartqa-aligned 동결 가정):

| Type | Test % | Target rows (9,500 mirror) | 후보 source | 합성 비용 |
|---|---:|---:|---|---|
| Factoid | 55.5% | 5,275 | chartqa_train (real) 그대로 downsample | 없음 (재사용) |
| Conversational | 16.0% | 1,520 | owid (paragraph context rich) + kaggle_like | 397B 합성 ~1,500 calls |
| Fact Checking | 12.5% | 1,188 | chartqa_train (real) + owid | 397B 합성 ~1,250 calls + validation pair |
| Multi Choice | 11.0% | 1,045 | plotly_complex (`answer_type=multichoice`, 1,107 rows in v8) | re-distill ~1,100 calls (letter 형식 강제) |
| Hypothetical | 5.0% | 475 | scientific (numeric trend) + owid | 397B 합성 ~500 calls |

### 5.2 합성 시 corpus 영향

| 시나리오 | total corpus | chartqa-aligned 비중 | 다른 source 비중 변화 |
|---|---:|---:|---|
| 현재 v3 (chartqa_train Factoid only) | 27,575 | 9,500 (34%) | 변화 없음 |
| 총량 동결 redistribute | 27,575 | 9,500 (34%) | 변화 없음 |
| 합성 추가 (+5,600) | 33,175 | 15,100 (45%) | chartmuseum/charxiv 비중 ↓ |

### 5.3 잠재적 reviewer 공격면

1. ChartQA-Pro test 분포 mirror 시 → "test-set fitting" 의심.
2. 397B teacher 가 ChartQA-Pro test 도 사전 학습에 봤을 가능성 (검증 불가) → teacher-generated synthesis 의 contamination 의심.
3. AVG5 = (H + A + Pro + CXR + CMU) / 5. ChartQA-Pro 단일 bench 만 보강 시 AVG5 부풀림 의심.
4. 합성 FC/Conv/Hyp 데이터 정합성 검증 부담 (validation pair 추가 → 397B call 2×).
5. main contribution (CVR rewards in GRPO) 가 SFT 데이터 엔지니어링 비중에 가려질 우려.

### 5.4 합성 안 함 (v3 결과로 마무리) 시나리오

- ChartQA-Pro 의 비-Factoid 44.5% 비중에 학습 신호 없는 상태로 SFT v3 학습 진행 중.
- 결과는 v8_distill_v2 (Factoid 보강 전) 대비 chartqa_pro 만 단독 측정 가능.
- 다른 4 bench 영향 = v2 와 v3 차이 (chartqa_train 9,500 rows 의 prompt+answer 변화) 에 한정.

---

## 6. 현재 진행 상태

- v3 SFT (chartqa_train 9,498 rows VLMEvalKit Factoid prompt 로 재 distill, 다른 source 변경 없음, 총 27,575 rows): **학습 진행 중 (step 10/431, ~35min 경과, ETA ~24h)**.
- 학습 후 자동 chain: merge LoRA → vLLM 8 server → standard eval 5 bench (lmms-eval scorer) + VLMEvalKit ChartQA-Pro eval → `docs/v8_distill_v3_final_comparison.md` 자동 생성.
- 7개 prior ckpt 의 VLMEvalKit ChartQA-Pro 비교 점수 미완 (vLLM swap timeout 으로 6 ckpt skip). 재시도 위해서는 vLLM swap 안정화 패치 필요.

---

**SFT 데이터 추가버전 작업 액션에 대한 여부와 전략이 필요한 상황.**
