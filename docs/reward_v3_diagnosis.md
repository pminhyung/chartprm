# ChartVCR v8 진단 및 개선 전략 보고서

---

## 1. 현재 결과 요약

| Benchmark | Zero-shot | SFT (R2) | Outcome (R3) | Process v2 (R4) | Δ(R4-R3) |
|-----------|-----------|----------|-------------|-----------------|----------|
| CQA-H | 80.0 | 83.7 | 84.5 | 83.8 | -0.7 |
| CQA-A | 85.0 | 88.6 | 89.0 | 89.0 | 0.0 |
| CharXiv | 43.0 | 44.5 | 46.0 | 45.3 | -0.7 |
| CQA-Pro | 35.0 | 35.8 | 36.4 | 36.1 | -0.3 |
| ChartMus | 29.0 | 27.6 | 28.3 | 30.4 | +2.1 |
| **AVG** | 54.4 | 56.0 | 56.8 | 56.9 | +0.1 |

v7(SFT 없음)에서 process reward +2.7pp → v8(SFT 위)에서 +0.1pp. 효과 거의 소멸.

---

## 2. Reward 공식 문제 진단

### 2.1 Additive 공식의 reward inflation (v2 핵심 결함)

**v2 공식**: `min(1.0, r_acc + 0.3 * r_proc)`

실측 (Row 4 학습 로그 19,576 reward calls 분석):

| 지표 | Row 3 (outcome) | Row 4 (process v2) |
|------|-----------------|-------------------|
| Wrong answer mean reward | 0.553 | **0.692** |
| Wrong answers ≥ 0.9 | 0% | **31.3%** |
| Wrong answers = 1.0 | 0% | **21.1%** |
| Correct-Wrong gap | **0.447** | **0.308** |
| Within-group std | **0.126** | **0.104** (-17.7%) |
| Zero-variance groups | 23.9% | **38.9%** (+63%) |

**Root cause chain**:
```
r_proc 높고 일정 (mean 0.77, std 0.19)
  → additive: wrong_reward = min(1.0, r_acc + 0.3*0.77) → 많은 wrong이 1.0 도달
  → correct-wrong gap 31% 축소
  → within-group variance 17.7% 감소 → zero-variance group 63% 증가
  → DAPO 학습 signal 파괴
  → frac_correct: Row 3 +6.3pp vs Row 4 +4.5pp (process가 오히려 악화)
```

### 2.2 v3 수정: Multiplicative 공식

**v3 공식**: `r_acc + 0.3 * (1.0 - r_acc) * r_proc_v2`

| 방식 | Wrong mean | Gap | Wrong at 1.0 |
|------|-----------|-----|-------------|
| Outcome-only | 0.553 | 0.447 | 0% |
| v2 (additive) | 0.692 | **0.308** | **21.1%** |
| v3 (multiplicative) | 0.287 | **0.713** | **0%** |

v3는 수학적으로 wrong answer가 1.0에 도달 불가능. Row 4b로 학습 진행 중 (step ~250/1631).

### 2.3 Process reward 함수 개선 (`compute_process_reward_v2`)

| 파라미터 | v1 | v2 | 영향 |
|----------|----|----|------|
| Grounding threshold | > 0.8 (~20%) | > 0.9 (~10%) | 35.5% 샘플에 영향 |
| Arithmetic default | 0.5 | **0.0** | 10.3% 샘플의 공짜 보너스 제거 |
| No CSV fallback | 0.5 | **0.0** | |
| Parse failures | correct += 1 | **skip** | |
| r_proc mean | 0.729 | 0.674 | 낮아짐 |
| r_proc std | 0.186 | 0.205 | 약간 증가 |

---

## 3. 학습 과정 트러블슈팅 이력

### 3.1 이미지 해상도 문제 (Qwen VL token mismatch)

TRL GRPOTrainer에서 배치 내 이미지 크기가 다르면 Qwen VL의 동적 tiling으로 인해
`ValueError: Image features and image tokens do not match` 발생.

| 시도 | 설정 | 결과 |
|------|------|------|
| 1차 | 원본 크기 (max 1024 resize) | step ~110 crash (tokens:559, features:558) |
| 2차 | 28px 배수 정렬 | step ~110 crash |
| 3차 | 32px 배수 정렬 | step ~119 crash (tokens:560, features:558) |
| 4차 | **960×960 고정** | Row 3 완주 ✅, Row 4 완주 ✅, **Row 4b step 148 crash** (tokens:901, features:900) |
| 5차 | **448×448 고정** | Row 4b step 193+ 통과, 안정 ✅ |

**원인**: Qwen VL이 이미지 크기에 따라 동적으로 tile 수 결정. 동일 크기라도 일부 시드에서 비결정적.
**최종 해결**: 448×448 (patch_size=16, 448/16=28, merge_size=2 → 항상 784 visual tokens).
**평가에는 영향 없음**: eval은 vLLM이 이미지를 자체 처리.

### 3.2 기타 학습 이슈

| 이슈 | 원인 | 해결 |
|------|------|------|
| `SFTConfig: unexpected keyword 'max_seq_length'` | TRL 버전 차이 | `max_length`로 변경 |
| `generation_batch_size not divisible` | GPU 수 × batch와 num_gen의 LCM 필요 | `_lcm()` 적용 |
| TRL vLLM client 404 | 일반 vLLM 서버 사용 | `trl vllm-serve` 사용 |
| GPU zombie 프로세스 | TRL vLLM 종료 후 GPU 메모리 미해제 | PID 기반 kill |

---

## 4. 에러 유형 분석 (5개 벤치마크, 2,828 wrong samples)

### 4.1 전체 에러 분포 (Row 3 기준)

| 에러 유형 | 비율 | 주요 벤치마크 |
|-----------|------|-------------|
| **FORMAT_ERROR (truncation)** | **30.2%** | CQA-H 42.8%, ChartMuseum 77.4%, CQA-Pro 63.8% |
| LOGIC_ERROR | 25.8% | ChartMuseum 43.0%, CharXiv 34.1% |
| ENTITY_CONFUSION | 24.0% | CQA-Pro 33.7%, ChartMuseum 18.8% |
| VALUE_MISREAD | 7.5% | CQA-A 33.3% |
| CALCULATION_ERROR | 7.5% | CQA-H 14.9% |
| TEMPORAL_ERROR | 4.0% | 고르게 분포 |
| APPROXIMATION_ERROR | 1.0% | 무시 가능 |

### 4.2 Truncation이 #1 실패 원인

Wrong answer의 reasoning 길이 vs correct:

| Benchmark | Correct (tokens) | Wrong (tokens) | 배율 |
|-----------|-----------------|----------------|------|
| CQA-H | 197 | 859 | 4.3x |
| CharXiv | 443 | 1,414 | 3.2x |
| ChartMuseum | 520 | 1,517 | 2.9x |
| CQA-Pro | - | - | ~3-5x |

**패턴**: 모델이 불확실하면 reasoning을 과도하게 길게 작성 → max_completion_length(4096) 초과 → `<answer>` 태그 미생성 → 답 추출 실패.
**ChartMuseum wrong의 77.4%**, **CQA-Pro wrong의 63.8%**가 truncation.

### 4.3 Process reward(v2)가 개선한/악화시킨 에러

| 에러 유형 | Row 3→Row 4 변화 |
|-----------|-----------------|
| FORMAT_ERROR | -25건 (소폭 개선) |
| CALCULATION_ERROR | -14건 (개선) |
| **VALUE_MISREAD** | **+37건 (악화)** |
| ENTITY_CONFUSION | +3건 (변화 없음) |

Process reward가 계산 에러는 줄였지만, value misread를 악화시킴 — process reward가 "숫자를 많이 언급하면 보상"하므로 잘못된 숫자를 자신있게 사용하는 패턴 강화.

### 4.4 텍스트 vs 수치 답변 정확도 격차

| Benchmark | 수치 정확도 | 텍스트 정확도 | 격차 |
|-----------|-----------|-------------|------|
| CQA-Pro | 51.7% | 28.4% | -23.3pp |
| CharXiv | 50.7% | 41.1% | -9.6pp |
| ChartMuseum | 35.1% | 25.8% | -9.3pp |

텍스트 답변이 모든 벤치마크에서 수치보다 현저히 낮음.

### 4.5 Format compliance

| Benchmark | `<answer>` 사용률 | Accuracy |
|-----------|----------------|----------|
| CQA-H | 93% | 84% |
| CharXiv | 67% | 45% |
| ChartMuseum | 53% | 30% |

어려운 벤치마크일수록 format compliance 하락 → 답 추출 실패 → 정확도 추가 하락.

---

## 5. 학습 데이터 vs 벤치마크 품질 비교

### 5.1 차트 이미지 품질 gap

| 항목 | 우리 학습 데이터 | 벤치마크 |
|------|---------------|---------|
| **해상도** | 88.7%가 ~990×590 (단일) | CQA-H 736×548 ~ ChartMuseum 5104×4800 |
| **차트 타입** | ~90% bar chart (아래 참조) | scatter, dot, multi-panel, infographic, map 등 다양 |
| **시각 스타일** | matplotlib default 단일 | editorial, 학술, infographic, branded 다양 |
| **텍스트 밀도** | 최소 라벨 (Alpha, Beta...) | 제목, 각주, 출처, 값 annotation 풍부 |
| **복잡도** | 단일 패널, 단순 구조 | multi-panel (CharXiv ~40%), 복합 레이아웃 |
| **색상** | matplotlib 기본 color cycle | 전문 팔레트, 시맨틱 컬러 코딩 |

### 5.2 Synthetic 차트 타입 버그 (심각)

**학습 데이터의 37.2% (1,819개 synthetic 샘플)의 차트 타입이 잘못됨.**

`pie_151.png`, `donut_185.png`, `funnel_198.png`, `radar_102.png`, `violin_101.png` 등
→ **전부 bar chart로 렌더링**됨. 제목만 "Pie 151", "Radar 102".

차트 생성 파이프라인(`chart_qa_data_pipeline.py`)의 `_generate_synthetic_data` 함수가
30개 차트 타입을 선언했지만, 실제 렌더링은 대부분 `render_chart_with_labels`의 bar chart 분기로 빠짐.

**결과**: 의도한 차트 타입 다양성(pie, donut, radar, violin, treemap 등)이 전혀 존재하지 않음.
학습 데이터는 사실상 **bar chart + line chart** 두 종류만 존재.

### 5.3 답변 타입 불일치

| | GRPO 학습 | CQA-H | CharXiv | ChartMuseum | CQA-Pro |
|---|---|---|---|---|---|
| 수치 답변 | **100%** | 68% | 45% | 27% | 30% |
| 텍스트 답변 | **0%** | 32% | 55% | 73% | 70% |

**주의**: 모델 입력은 학습/평가 모두 동일 (이미지 + 질문, CSV 미포함).
이 불일치는 "모델이 학습에서 텍스트 답변 패턴을 경험하지 못함"을 의미.
Reward signal도 수치에서만 continuous → 텍스트 답변의 reasoning 개선에 기여 불가.

---

## 6. 학습 dynamics 분석

### 6.1 Reward 궤적 (Training metrics log, 61,602 rows)

**Row 3 (outcome)**:
- reward_mean: 0.765 → 0.802 (+3.7pp, 완만 상승)
- frac_correct: 0.454 → 0.512 (+5.8pp)
- 포화 시점: ~step 151 이후 개선 둔화

**Row 4 (process v2)**:
- reward_mean: 0.855 → 0.872 (+1.7pp, 거의 평탄)
- frac_correct: 0.475 → 0.520 (+4.5pp, Row 3보다 적음)
- r_proc_mean: **0.773 → 0.784 (전 학습 동안 거의 불변)**
- 포화 시점: ~step 1051

### 6.2 DAPO 그룹 내 variance (핵심 학습 signal)

| 지표 | Row 3 | Row 4 | 차이 |
|------|-------|-------|------|
| Mean within-group std | **0.126** | 0.104 | -17.7% |
| Zero-variance groups | 23.9% | **38.9%** | +63% |
| Good-signal groups (std>0.1) | **48.3%** | 41.1% | -15% |
| Strong-signal groups (std>0.3) | 11.8% | 10.4% | -12% |

Row 4의 process reward가 DAPO의 그룹 내 variance를 줄여서 학습 signal을 악화시킴.

### 6.3 CSV 크기별 r_proc 분포

| 데이터 소스 | 평균 CSV 크기 | v1 grounding | v2 grounding |
|------------|-------------|-------------|-------------|
| OWID | 25.2 values | 0.533 | 0.480 |
| Synthetic | 10.3 values | 0.716 | 0.651 |
| Scientific | 31.3 values | **0.782** | 0.716 |
| WorldBank | 40.4 values | 0.557 | 0.512 |

Scientific 소스의 grounding이 가장 높음 (47.2%가 완벽 매칭) — CSV 값 수가 많아서 reasoning의 숫자가 우연히 매칭될 확률 높음.

---

## 7. 구체적 개선안 (근거 기반)

### 7.1 즉시 적용 가능 (재학습 불필요)

**Post-processing 답 추출 개선** — ~200건 복구 가능:
- 숫자 내 공백 처리: "451 176.9" → "451176.9" (22건)
- 텍스트 부분 매칭: "Democrat" vs "Democrat (scores 60 to 100)" (157건)
- 퍼센트 형식 통일: 0.6과 60 모두 수용 (25건)

### 7.2 Reward 개선 (v3 이후, 근거: §2, §6)

**A. Conciseness reward** (근거: FORMAT_ERROR 30.2%, truncation이 #1 실패)
- Reasoning 길이에 soft penalty: think_tokens > 1500이면 감점
- 예: `penalty = -0.1 * max(0, (think_tokens - 1500) / 1500)`
- 기대 효과: truncation 30% → 10~15%, 특히 ChartMuseum/CQA-Pro에서 +3-5pp

**B. Value misread penalty** (근거: process reward가 VALUE_MISREAD +37건 악화)
- 현재 grounding은 "reasoning에 CSV 숫자가 있으면 보상" → 틀린 숫자도 보상
- 개선: final answer의 숫자가 CSV에 없으면 penalty
- 기대 효과: VALUE_MISREAD 감소

### 7.3 데이터 개선 (근거: §5)

**A. 차트 타입 다양성 확보** (근거: synthetic 37.2% 잘못된 렌더링)
- `chart_qa_data_pipeline.py`의 차트 타입별 렌더링 수정
- Pie, donut, scatter, heatmap, radar 등 실제 구현
- 기대 효과: 다양한 차트에서의 시각 이해력 향상

**B. 텍스트 답변 학습 데이터** (근거: 텍스트 정확도 -9~23pp 격차)
- GRPO 데이터에 "which country", "what color", "Yes/No" 유형 추가 (~2K)
- `relaxed_text_match`를 continuous scoring으로 개선 (부분 매칭 0.5 등)
- 기대 효과: ChartMuseum +3-5pp, CQA-Pro +2-3pp

**C. 이미지 다양성** (근거: 해상도 단일화 990×590)
- 다양한 figsize/dpi로 차트 생성
- 학술 스타일(seaborn), 에디토리얼 스타일 추가
- Multi-panel 차트 추가 (CharXiv 대응)
- 기대 효과: CharXiv +2-3pp

**D. 질문 다양성** (근거: 98.7% 수치, 템플릿 고정)
- Entity identification 질문 ("Which country has the highest...?")
- Comparison 질문 ("Is A greater than B?")
- Trend 질문 ("What is the overall trend?")
- 기대 효과: ENTITY_CONFUSION, LOGIC_ERROR 감소

### 7.4 학습 전략 개선 (근거: §6.1 포화)

**A. Hard-sample curriculum** (근거: frac_correct 50%에서 포화)
- 모델이 < 50% 정답률인 샘플에 집중
- num_generations 4 → 8로 증가하여 within-group variance 확보
- 기대 효과: 포화 지점 이후 추가 개선

---

## 8. 구조적 한계 요약

| 한계 | 영향도 | v3로 해결? | 근거 섹션 |
|------|--------|-----------|----------|
| Reward inflation (additive) | 매우 높음 | ✅ v3로 해결 | §2.1 |
| Reward signal 범위 (수치만) | 매우 높음 | ❌ | §5.3 |
| r_proc floor effect | 높음 | △ 부분적 | §6.3 |
| Truncation / 과도한 reasoning | 높음 | ❌ | §4.2 |
| 차트 타입 다양성 부족 | 높음 | ❌ | §5.2 |
| Rule verifier 60% 미검출 | 중간 | ❌ | §4.3 |
| Format compliance 하락 | 낮음 | ❌ | §4.5 |

**핵심 인사이트**: v3는 "공식 설계 오류"를 수정하지만, "무엇을 보상하는가"의 범위와 "무엇으로 학습하는가"의 품질은 별도 대응 필요.

---

## 9. 현재 진행 상태

- **Row 4b** (conditional_v3, multiplicative): 학습 진행 중
- GPU 4: TRL vLLM, GPUs 5-7: 학습 (3 GPUs)
- 이미지: 448×448 고정
- Row 4 결과는 유지 (v2→v3 비교용)

## 10. 파일 변경 요약

| 파일 | 변경 |
|------|------|
| `code/rewards/rule_verifier_fast.py` | `compute_process_reward_v2()` 추가 |
| `train_grpo_dapo.py` | `reward_conditional_v3()` 추가, argparse/fn_map, 이미지 448×448 |
| `chartvr/extraction.py` | `_is_numeric_answer()`, `relaxed_text_match()` 추가 |
